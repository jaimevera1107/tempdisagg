"""Additional linear constraints on the high-frequency series.

Besides the low-frequency target, a disaggregation may have to honour other
exact information about the high-frequency series ``y``:

* values known directly (e.g. two months taken from administrative records);
* totals at an intermediate frequency (some quarters, when the target is
  annual and the output monthly);
* any linear relation ``sum_t a_t y_t = b``.

Each piece of information is a :class:`LinearConstraint`. The constraints are
appended as rows of the aggregation matrix ``C`` and enter the estimators
directly (one-step): GLS methods treat them as additional exact
observations, Denton methods as additional KKT constraints. This keeps the
estimators optimal, unlike estimating first and reconciling afterwards.

Before estimation, :func:`independent_rows` verifies the stacked system:
constraints that contradict each other raise
:class:`~tempdisagg.exceptions.InfeasibleConstraintError`; constraints that are
implied by earlier ones (consistent but redundant) are dropped with a
:class:`~tempdisagg.exceptions.RedundantConstraintWarning`, because redundant
rows make ``V = C Sigma C'`` singular.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tempdisagg.core.aggregation import Conversion
from tempdisagg.exceptions import InfeasibleConstraintError, InputValidationError

#: Relative tolerance of the linear-dependence test.
DEPENDENCE_TOLERANCE: float = 1e-10
#: Relative tolerance of the consistency test for dependent rows.
CONSISTENCY_TOLERANCE: float = 1e-8


@dataclass(frozen=True)
class LinearConstraint:
    """Exact constraint ``sum_t coefficients[t] * y[t] = value``.

    Parameters
    ----------
    coefficients : pd.Series
        Weights indexed by high-frequency labels (the labels of the output:
        Periods for period-indexed input, integers for arrays,
        ``(index, grain)`` tuples for the long format).
    value : float
        Right-hand side.
    label : str, default "constraint"
        Name used in messages and in ``result.data.low_index``.
    """

    coefficients: pd.Series
    value: float
    label: str = "constraint"

    def __post_init__(self) -> None:
        if not isinstance(self.coefficients, pd.Series) or self.coefficients.empty:
            raise InputValidationError(f"{self.label}: coefficients must be a non-empty Series.")
        if not self.coefficients.index.is_unique:
            raise InputValidationError(f"{self.label}: coefficient labels must be unique.")
        weights = pd.to_numeric(self.coefficients, errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(weights)) or not np.any(weights):
            raise InputValidationError(
                f"{self.label}: coefficients must be finite and not all zero.")
        if not np.isfinite(float(self.value)):
            raise InputValidationError(f"{self.label}: value must be finite.")
        object.__setattr__(self, "value", float(self.value))


def known_values(series: pd.Series, label: str = "known") -> list[LinearConstraint]:
    """Constraints fixing individual high-frequency values.

    ``series`` is indexed by high-frequency labels; NaN entries are ignored.
    """
    observed = series.dropna()
    return [
        LinearConstraint(pd.Series([1.0], index=[period]), float(value), f"{label}[{period}]")
        for period, value in observed.items()
    ]


def aggregate_constraints(
    series: pd.Series,
    high_freq: str,
    conversion: Conversion | str = Conversion.SUM,
    label: str = "aggregate",
) -> list[LinearConstraint]:
    """Constraints for totals observed at an intermediate frequency.

    Parameters
    ----------
    series : pd.Series
        Values with a PeriodIndex (e.g. quarterly), NaN entries ignored.
    high_freq : str
        Frequency of the output (e.g. ``"M"``).
    conversion : Conversion or str, default "sum"
        How each value relates to its high-frequency sub-periods.
    """
    if not isinstance(series.index, pd.PeriodIndex):
        raise InputValidationError("aggregate_constraints requires a PeriodIndex.")
    rule = Conversion.parse(conversion)
    constraints = []
    for period, value in series.dropna().items():
        members = pd.period_range(period.asfreq(high_freq, how="start"),
                                  period.asfreq(high_freq, how="end"), freq=high_freq)
        if len(members) < 2:
            raise InputValidationError("high_freq must be higher than the series frequency.")
        weights = np.zeros(len(members))
        if rule is Conversion.SUM:
            weights[:] = 1.0
        elif rule is Conversion.AVERAGE:
            weights[:] = 1.0 / len(members)
        elif rule is Conversion.FIRST:
            weights[0] = 1.0
        else:
            weights[-1] = 1.0
        constraints.append(LinearConstraint(pd.Series(weights, index=members), float(value),
                                            f"{label}[{period}]"))
    return constraints


def independent_rows(
    matrix: np.ndarray, values: np.ndarray, labels: Sequence[str]
) -> tuple[np.ndarray, list[int]]:
    """Return the indices of linearly independent rows, checking consistency.

    Rows are processed in order (earlier rows have priority) with modified
    Gram-Schmidt. A row that is (numerically) a combination of accepted rows
    is dropped if its value equals the implied combination of their values,
    otherwise the system is contradictory.

    Returns
    -------
    keep : np.ndarray
        Indices of independent rows.
    dropped : list of int
        Indices of consistent redundant rows.

    Raises
    ------
    InfeasibleConstraintError
        If a dependent row contradicts the accepted ones.
    """
    basis: list[np.ndarray] = []
    accepted: list[int] = []
    dropped: list[int] = []
    for index, row in enumerate(matrix):
        residual = row.astype(float).copy()
        for vector in basis:
            residual -= (vector @ residual) * vector
        norm = float(np.linalg.norm(residual))
        if norm > DEPENDENCE_TOLERANCE * max(float(np.linalg.norm(row)), 1e-300):
            basis.append(residual / norm)
            accepted.append(index)
            continue
        # Dependent row: express it through accepted rows and compare values.
        sub = matrix[accepted]
        combination, *_ = np.linalg.lstsq(sub.T, row, rcond=None)
        implied = float(combination @ values[accepted])
        scale = max(1.0, abs(float(values[index])), float(np.max(np.abs(values[accepted]))))
        if abs(implied - float(values[index])) > CONSISTENCY_TOLERANCE * scale:
            raise InfeasibleConstraintError(
                f"Constraint '{labels[index]}' contradicts the others: they imply "
                f"{implied:.10g}, but its value is {float(values[index]):.10g}."
            )
        dropped.append(index)
    return np.asarray(accepted, dtype=int), dropped
