"""Temporal aggregation rules and the aggregation (conversion) matrix.

The aggregation matrix ``C`` maps a high-frequency vector ``y`` to its
low-frequency counterpart, ``C @ y = y_low``. Every temporal disaggregation
method in this library is constrained by that identity.
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import pandas as pd

from tempdisagg.exceptions import InputValidationError

#: Group code used for high-frequency observations that belong to no
#: observed low-frequency period (extrapolation or retropolation).
UNCONSTRAINED_GROUP: int = -1


class Conversion(str, Enum):
    """Relationship between the high- and low-frequency series."""

    SUM = "sum"
    AVERAGE = "average"
    FIRST = "first"
    LAST = "last"

    @classmethod
    def parse(cls, value: "Conversion | str") -> "Conversion":
        """Return the enum member for ``value``.

        Raises
        ------
        InputValidationError
            If ``value`` is not a supported conversion.
        """
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            valid = ", ".join(member.value for member in cls)
            raise InputValidationError(
                f"Unknown conversion '{value}'. Valid options: {valid}."
            ) from exc


def _group_weights(size: int, conversion: Conversion) -> np.ndarray:
    """Return the aggregation weights for one low-frequency group."""
    if conversion is Conversion.SUM:
        return np.ones(size)
    if conversion is Conversion.AVERAGE:
        return np.full(size, 1.0 / size)
    weights = np.zeros(size)
    if conversion is Conversion.FIRST:
        weights[0] = 1.0
    else:
        weights[-1] = 1.0
    return weights


def build_aggregation_matrix(
    group_codes: np.ndarray,
    n_groups: int,
    conversion: Conversion | str,
) -> np.ndarray:
    """Build the aggregation matrix from group membership codes.

    Parameters
    ----------
    group_codes : np.ndarray
        Integer array of length ``n_high``. Entry ``i`` is the position of the
        low-frequency period that high-frequency observation ``i`` belongs to,
        or :data:`UNCONSTRAINED_GROUP` if it belongs to none. Observations are
        assumed to be in chronological order.
    n_groups : int
        Number of low-frequency observations (rows of the matrix).
    conversion : Conversion or str
        Aggregation rule.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(n_groups, n_high)``. Unconstrained observations
        have an all-zero column.

    Raises
    ------
    InputValidationError
        If a group has no members or codes are out of range.
    """
    rule = Conversion.parse(conversion)
    codes = np.asarray(group_codes, dtype=int)
    if codes.ndim != 1:
        raise InputValidationError("group_codes must be one-dimensional.")
    if n_groups < 1:
        raise InputValidationError("At least one low-frequency group is required.")
    if np.any((codes < UNCONSTRAINED_GROUP) | (codes >= n_groups)):
        raise InputValidationError("group_codes contains out-of-range values.")

    matrix = np.zeros((n_groups, codes.size))
    for group in range(n_groups):
        members = np.flatnonzero(codes == group)
        if members.size == 0:
            raise InputValidationError(
                f"Low-frequency group {group} has no high-frequency members."
            )
        matrix[group, members] = _group_weights(members.size, rule)
    return matrix


def temporal_aggregate(
    series: pd.Series,
    low_freq: str,
    conversion: Conversion | str = Conversion.SUM,
    require_complete: bool = True,
) -> pd.Series:
    """Aggregate a high-frequency series to a lower frequency.

    This is the inverse operation of temporal disaggregation (``ta`` in the
    R package): applying it to a disaggregated series returns the original
    low-frequency series.

    Parameters
    ----------
    series : pd.Series
        High-frequency series indexed by a regular :class:`pd.PeriodIndex`.
    low_freq : str
        Target pandas frequency alias, e.g. ``"Y"`` or ``"Q"``.
    conversion : Conversion or str, default "sum"
        Aggregation rule.
    require_complete : bool, default True
        If True, low-frequency periods with missing sub-periods are dropped
        instead of being aggregated from partial data.

    Returns
    -------
    pd.Series
        Low-frequency series indexed by a :class:`pd.PeriodIndex`.

    Raises
    ------
    InputValidationError
        If the index is not a PeriodIndex or contains missing values.
    """
    if not isinstance(series.index, pd.PeriodIndex):
        raise InputValidationError("series must be indexed by a pandas PeriodIndex.")
    if series.isna().any():
        raise InputValidationError("series contains missing values.")
    rule = Conversion.parse(conversion)

    low_index = series.index.asfreq(low_freq)
    grouped = series.groupby(low_index, sort=True)
    aggregators = {
        Conversion.SUM: grouped.sum,
        Conversion.AVERAGE: grouped.mean,
        Conversion.FIRST: grouped.first,
        Conversion.LAST: grouped.last,
    }
    result = aggregators[rule]()

    if require_complete:
        counts = grouped.size()
        high_freq = series.index.freq
        expected = pd.Series(
            [
                len(pd.period_range(p.start_time, p.end_time, freq=high_freq))
                for p in counts.index
            ],
            index=counts.index,
        )
        result = result[counts == expected]
    result.index = pd.PeriodIndex(result.index, freq=low_freq)
    result.name = series.name
    return result
