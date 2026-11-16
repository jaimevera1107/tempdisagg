"""Revision analysis: how much past estimates change when a new period arrives.

Why
---
Each new low-frequency figure changes the estimated parameters and, through
the distribution filter, the high-frequency values already published. Users
of official statistics see these revisions as a cost, so statistical offices
weigh them when choosing a method. Quilis (2018, Table 5) finds that simple
methods (Denton, Fernandez) revise least and dynamic models most.

Procedure (real-time simulation with the final data)
----------------------------------------------------
For the last ``n_vintages`` observed periods ``T_1 < ... < T_V``, fit each
candidate with the targets up to ``T_{k-1}`` and up to ``T_k`` and compare the
two estimates on every high-frequency period they share. The indicators are
cut at the end of the last target period, or one low-frequency period later
with ``extrapolation=True`` (the indicator usually arrives before the
benchmark; Quilis' "with extrapolation" columns). With extrapolation, the
shared periods include the extrapolated ones, whose revision is driven by
the arrival of the benchmark itself.

Revisions are measured on levels, always in percent of the absolute earlier
estimate, and on period-on-period changes: growth rates in percentage points
when both estimates are strictly positive, first differences otherwise (the
``change_units`` column of the summary says which). Periods before the first
target (retropolation) are excluded by default: every vintage shares them, and
a model that is unstable when going back in time would dominate the table.

This isolates the revisions caused by new data and re-estimation. Revisions
of the inputs themselves (benchmark or seasonal-adjustment vintages) need
real-time vintages, which the user has to supply.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tempdisagg.evaluation.backtest import Candidate
from tempdisagg.exceptions import InputValidationError, TempDisaggError

#: Minimum number of targets in the first (smallest) vintage.
MIN_TRAINING_TARGETS: int = 3


@dataclass(frozen=True)
class RevisionResult:
    """Outcome of :func:`revision_analysis`.

    Attributes
    ----------
    revisions : pd.DataFrame
        Long table: ``candidate``, ``vintage`` (period added), ``period``,
        ``previous``, ``current``, ``level_revision`` (percent of
        ``|previous|``), ``change_revision`` (see ``change_units``),
        ``change_units`` and ``latest`` (period belongs to the most recent
        target period of the earlier vintage).
    extrapolation : bool
        Whether the indicators ran one period beyond the targets.
    failures : dict
        Candidate -> messages for vintages that could not be fitted.
    """

    revisions: pd.DataFrame
    extrapolation: bool
    failures: dict[str, list[str]] = field(default_factory=dict)

    def summary(self) -> pd.DataFrame:
        """Mean absolute revisions per candidate, sorted by change revision.

        Columns: number of vintage steps, mean absolute level revision, mean
        absolute change revision, the same restricted to the most recent
        low-frequency period of the earlier vintage (where revisions
        concentrate), and the largest absolute level revision.
        """
        if self.revisions.empty:
            return pd.DataFrame()
        rows = {}
        for name, block in self.revisions.groupby("candidate", sort=False):
            latest = block[block["latest"].astype(bool)]
            rows[name] = {
                "n_vintages": int(block["vintage"].nunique()),
                "change_units": "/".join(sorted(block["change_units"].unique())),
                "MAR_level": float(block["level_revision"].abs().mean()),
                "MAR_change": float(block["change_revision"].abs().mean()),
                "MAR_level_latest": float(latest["level_revision"].abs().mean()),
                "MAR_change_latest": float(latest["change_revision"].abs().mean()),
                "max_level": float(block["level_revision"].abs().max()),
            }
        table = pd.DataFrame.from_dict(rows, orient="index")
        return table.sort_values("MAR_change")


def _cut(indicators: pd.Series | pd.DataFrame | None, last_low: pd.Period
         ) -> pd.Series | pd.DataFrame | None:
    """Keep the indicator periods that belong to ``last_low`` or earlier."""
    if indicators is None:
        return None
    index = indicators.index
    if not isinstance(index, pd.PeriodIndex):
        raise InputValidationError("revision_analysis requires period-indexed indicators.")
    return indicators[np.asarray(index.asfreq(last_low.freqstr) <= last_low)]


def _changes(values: pd.Series, growth: bool) -> pd.Series:
    return 100.0 * values.pct_change() if growth else values.diff()


def revision_analysis(
    y_low: pd.Series,
    candidates: Mapping[str, Candidate],
    n_vintages: int = 5,
    extrapolation: bool = False,
    high_freq: str | None = None,
    include_retropolated: bool = False,
) -> RevisionResult:
    """Revisions caused by the arrival of each of the last ``n_vintages`` periods.

    Parameters
    ----------
    y_low : pd.Series
        Low-frequency target with a PeriodIndex (final values).
    candidates : mapping of str to Candidate
        Models to compare, each with its indicators.
    n_vintages : int, default 5
        Number of new periods (revision steps) to simulate.
    extrapolation : bool, default False
        Let the indicators run one low-frequency period beyond the targets.
        Candidates without indicators cannot extrapolate and are reported in
        ``failures``.
    high_freq : str, optional
        Output frequency for candidates without indicators.
    include_retropolated : bool, default False
        Also compare the periods before the first target.
    """
    if not isinstance(y_low, pd.Series) or not isinstance(y_low.index, pd.PeriodIndex):
        raise InputValidationError("revision_analysis requires y_low with a PeriodIndex.")
    if not candidates:
        raise InputValidationError("At least one candidate is required.")
    if isinstance(n_vintages, bool) or not isinstance(n_vintages, int) or n_vintages < 1:
        raise InputValidationError("n_vintages must be a positive integer.")
    observed = y_low.dropna()
    if len(observed) - n_vintages < MIN_TRAINING_TARGETS:
        raise InputValidationError(
            f"{len(observed)} targets leave fewer than {MIN_TRAINING_TARGETS} for the "
            f"first vintage with n_vintages={n_vintages}."
        )

    ends = observed.index[-(n_vintages + 1):]
    first = observed.index[0]
    rows: list[dict[str, object]] = []
    failures: dict[str, list[str]] = {name: [] for name in candidates}
    for name, candidate in candidates.items():
        if extrapolation and candidate.indicators is None:
            failures[name].append("cannot extrapolate without indicators")
            continue
        estimates: dict[pd.Period, pd.Series] = {}
        for end in ends:
            cutoff = end + 1 if extrapolation else end
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    fit = candidate.model.fit(y_low.loc[:end],
                                              _cut(candidate.indicators, cutoff), high_freq)
                    estimates[end] = fit.values
                except TempDisaggError as exc:
                    failures[name].append(f"{end}: {exc}")
        for previous_end, current_end in zip(ends[:-1], ends[1:], strict=True):
            if previous_end not in estimates or current_end not in estimates:
                continue
            previous, current = estimates[previous_end], estimates[current_end]
            common = previous.index.intersection(current.index)
            if not include_retropolated:
                common = common[np.asarray(common.asfreq(first.freqstr) >= first)]
            old, new = previous.loc[common], current.loc[common]
            growth = bool(np.all(old.to_numpy() > 0) and np.all(new.to_numpy() > 0))
            with np.errstate(divide="ignore", invalid="ignore"):
                level = 100.0 * (new - old) / old.abs()
            change = _changes(new, growth) - _changes(old, growth)
            units = "growth pp" if growth else "difference"
            latest = np.asarray(common.asfreq(previous_end.freqstr) == previous_end)
            for i, period in enumerate(common):
                rows.append({
                    "candidate": name, "vintage": current_end, "period": period,
                    "previous": float(old.iloc[i]), "current": float(new.iloc[i]),
                    "level_revision": float(level.iloc[i]),
                    "change_revision": float(change.iloc[i]),
                    "change_units": units,
                    "latest": bool(latest[i]),
                })
    columns = ["candidate", "vintage", "period", "previous", "current", "level_revision",
               "change_revision", "change_units", "latest"]
    table = pd.DataFrame(rows, columns=columns)
    return RevisionResult(table, extrapolation, {k: v for k, v in failures.items() if v})
