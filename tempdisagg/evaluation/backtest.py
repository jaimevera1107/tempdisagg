"""Out-of-sample backtesting by holding out low-frequency periods.

Why
---
Every constrained method reproduces the low-frequency data exactly, so
in-sample agreement cannot rank methods. Without the true high-frequency
series, the only out-of-sample evidence available is the ability to
*extrapolate*: estimate with the targets up to ``T - 1``, extrapolate the
high-frequency series over period ``T`` using the indicators, aggregate it and
compare with the actual target of ``T``.

Procedure (rolling origin, one step ahead)
------------------------------------------
For each of the last ``holdout`` observed periods ``T``:

1. set the target of ``T`` and of every later period to NaN;
2. fit each candidate (model + its indicators);
3. aggregate the estimate over ``T`` with the model's conversion rule;
4. record the error ``prediction - actual``.

Limits
------
The backtest measures one-step-ahead extrapolation of the aggregate. It says
nothing direct about the shape of the series *within* each period, which is
not observable. When a known high-frequency series exists, an artificial
experiment (aggregate it with :func:`~tempdisagg.temporal_aggregate`,
disaggregate, compare period by period) is the stronger test. With few
low-frequency observations, differences between candidates are noisy: look at
all metrics and at the per-origin errors, not only at the ranking.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from tempdisagg.core.aggregation import Conversion
from tempdisagg.exceptions import InputValidationError, TempDisaggError
from tempdisagg.model import TemporalDisaggregator

#: Minimum number of training targets required before the first origin.
MIN_TRAINING_TARGETS: int = 3


@dataclass(frozen=True)
class Candidate:
    """A model and the indicators it uses.

    Parameters
    ----------
    model : TemporalDisaggregator
    indicators : pd.Series or pd.DataFrame, optional
        Must cover the held-out periods (otherwise they cannot be predicted).
    """

    model: TemporalDisaggregator
    indicators: pd.Series | pd.DataFrame | None = None


@dataclass(frozen=True)
class BacktestResult:
    """Outcome of :func:`backtest`.

    Attributes
    ----------
    actual : pd.Series
        Held-out targets.
    predictions : pd.DataFrame
        Predicted aggregates (rows: held-out periods, columns: candidates).
    failures : dict
        Candidate -> list of messages for origins that could not be fitted.
    warnings : dict
        Candidate -> number of statistical warnings raised during the fits.
    """

    actual: pd.Series
    predictions: pd.DataFrame
    failures: dict[str, list[str]] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)

    @property
    def errors(self) -> pd.DataFrame:
        """``prediction - actual`` per origin and candidate."""
        return self.predictions.sub(self.actual, axis=0)

    def summary(self) -> pd.DataFrame:
        """Metrics per candidate, sorted by RMSE.

        Columns: number of evaluated origins, mean error (bias), MAE, RMSE and
        MAPE (``nan`` if an actual value is zero).
        """
        errors = self.errors
        actual = self.actual.to_numpy()
        rows = {}
        for name in errors.columns:
            err = errors[name].to_numpy()
            ok = np.isfinite(err)
            if not ok.any():
                rows[name] = (0, np.nan, np.nan, np.nan, np.nan)
                continue
            mape = (float(np.mean(np.abs(err[ok] / actual[ok]))) * 100.0
                    if np.all(actual[ok] != 0) else np.nan)
            rows[name] = (int(ok.sum()), float(np.mean(err[ok])), float(np.mean(np.abs(err[ok]))),
                          float(np.sqrt(np.mean(err[ok] ** 2))), mape)
        table = pd.DataFrame.from_dict(
            rows, orient="index", columns=["n_origins", "bias", "MAE", "RMSE", "MAPE_%"])
        return table.sort_values("RMSE")


def backtest(
    y_low: pd.Series,
    candidates: Mapping[str, Candidate],
    holdout: int = 3,
    high_freq: str | None = None,
) -> BacktestResult:
    """Run a rolling one-step-ahead backtest of the low-frequency aggregate.

    Parameters
    ----------
    y_low : pd.Series
        Low-frequency target with a PeriodIndex.
    candidates : mapping of str to Candidate
        Models to compare (they must share the same conversion rule).
    holdout : int, default 3
        Number of final observed periods used as origins.
    high_freq : str, optional
        Output frequency for candidates without indicators.

    Returns
    -------
    BacktestResult
    """
    if not isinstance(y_low, pd.Series) or not isinstance(y_low.index, pd.PeriodIndex):
        raise InputValidationError("backtest requires y_low with a PeriodIndex.")
    if not candidates:
        raise InputValidationError("At least one candidate is required.")
    conversions = {candidate.model.conversion for candidate in candidates.values()}
    if len(conversions) != 1:
        raise InputValidationError("All candidates must use the same conversion rule.")
    conversion = conversions.pop()
    observed = y_low.dropna()
    if isinstance(holdout, bool) or not isinstance(holdout, int) or holdout < 1:
        raise InputValidationError("holdout must be a positive integer.")
    if len(observed) - holdout < MIN_TRAINING_TARGETS:
        raise InputValidationError(
            f"Too few observations: {len(observed)} targets leave fewer than "
            f"{MIN_TRAINING_TARGETS} for training with holdout={holdout}."
        )

    origins = observed.index[-holdout:]
    predictions = pd.DataFrame(index=origins, columns=list(candidates), dtype=float)
    failures: dict[str, list[str]] = {name: [] for name in candidates}
    warning_counts: dict[str, int] = {name: 0 for name in candidates}
    for origin in origins:
        training = y_low.where(y_low.index < origin)
        for name, candidate in candidates.items():
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    result = candidate.model.fit(training, candidate.indicators, high_freq)
                    predictions.loc[origin, name] = _aggregate(result.values, origin, conversion)
                except (TempDisaggError, KeyError) as exc:
                    failures[name].append(f"{origin}: {exc}")
            warning_counts[name] += len(caught)
    actual = observed.loc[origins].rename("actual")
    return BacktestResult(actual, predictions, {k: v for k, v in failures.items() if v},
                          warning_counts)


def _aggregate(values: pd.Series, period: pd.Period, conversion: Conversion) -> float:
    """Aggregate the high-frequency estimate over one low-frequency period."""
    index = values.index
    members = pd.period_range(period.asfreq(index.freq, how="start"),
                              period.asfreq(index.freq, how="end"), freq=index.freq)
    if not members.isin(index).all():
        raise KeyError(f"the estimate does not cover all sub-periods of {period}")
    subset = values.loc[members].to_numpy()
    if conversion is Conversion.SUM:
        return float(subset.sum())
    if conversion is Conversion.AVERAGE:
        return float(subset.mean())
    return float(subset[0] if conversion is Conversion.FIRST else subset[-1])
