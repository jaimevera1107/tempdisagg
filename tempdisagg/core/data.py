"""Alignment of low-frequency targets with high-frequency indicators.

:class:`DataAligner` converts the supported input formats into a validated
:class:`DisaggregationData`. Format-specific code only translates labels into
a :mod:`calendar <tempdisagg.preprocessing.calendar>`; completion, padding and
the aggregation constraints are built by
:class:`~tempdisagg.preprocessing.grid.GridBuilder`, identically for every
format.

Indicators are optional: methods that only distribute (``uniform``) or smooth
(``denton``, ``denton-cholette`` with a constant, or regressions on a constant)
need none.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

import numpy as np
import pandas as pd

from tempdisagg.core.aggregation import Conversion
from tempdisagg.core.constraints import LinearConstraint
from tempdisagg.exceptions import InputValidationError


class RowKind(str, Enum):
    """Origin of each row of the constraint system."""

    TARGET = "target"          # observed low-frequency value
    PADDING = "padding"        # artificial target added by padding
    CONSTRAINT = "constraint"  # additional user constraint


@dataclass(frozen=True)
class DisaggregationData:
    """Validated inputs for one temporal disaggregation problem.

    Attributes
    ----------
    y_low : np.ndarray
        Low-frequency targets, shape ``(n_low,)`` (observed and, with
        padding, artificial).
    indicators : np.ndarray
        High-frequency indicators, shape ``(n_high, k)``; ``k`` may be zero.
    aggregation_matrix : np.ndarray
        ``C`` of shape ``(n_low, n_high)`` such that ``C @ y = y_low``.
    high_index, low_index : pd.Index
        Labels of the high-frequency grid and of the targets.
    indicator_names : tuple of str
    conversion : Conversion
    artificial_low : np.ndarray
        True for targets created by padding.
    artificial_high : np.ndarray
        True for grid periods absent from the input (completion or padding);
        they are removed from the returned series.
    completed_indicators : np.ndarray
        ``(n_high, k)`` mask of indicator values created by completion/padding.
    row_kind : np.ndarray
        :class:`RowKind` value of each constraint row.
    """

    y_low: np.ndarray
    indicators: np.ndarray
    aggregation_matrix: np.ndarray
    high_index: pd.Index
    low_index: pd.Index
    indicator_names: tuple[str, ...]
    conversion: Conversion
    artificial_low: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    artificial_high: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    completed_indicators: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]
    row_kind: np.ndarray = field(default=None, repr=False)  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n_low, n_high = self.aggregation_matrix.shape
        defaults = {
            "artificial_low": np.zeros(n_low, dtype=bool),
            "artificial_high": np.zeros(n_high, dtype=bool),
            "completed_indicators": np.zeros(self.indicators.shape, dtype=bool),
            "row_kind": np.full(n_low, RowKind.TARGET.value),
        }
        for name, default in defaults.items():
            if getattr(self, name) is None:
                object.__setattr__(self, name, default)
        if self.y_low.shape != (n_low,):
            raise InputValidationError("y_low does not match the aggregation matrix rows.")
        if self.indicators.ndim != 2 or self.indicators.shape[0] != n_high:
            raise InputValidationError("indicators do not match the aggregation matrix columns.")
        if len(self.high_index) != n_high or len(self.low_index) != n_low:
            raise InputValidationError("Index lengths do not match the data.")
        if len(self.indicator_names) != self.indicators.shape[1]:
            raise InputValidationError("indicator_names must have one entry per column.")
        if n_low < 1:
            raise InputValidationError("At least one observed low-frequency value is required.")
        if not np.any(self.row_kind == RowKind.TARGET.value):
            raise InputValidationError(
                "At least one observed (non-artificial) target is required."
            )

    @property
    def target_rows(self) -> np.ndarray:
        """Mask of rows that are observed low-frequency targets."""
        return np.asarray(self.row_kind == RowKind.TARGET.value)

    @property
    def n_low(self) -> int:
        """Number of low-frequency targets (observed + artificial)."""
        return int(self.y_low.size)

    @property
    def n_high(self) -> int:
        """Number of high-frequency periods in the grid."""
        return int(self.aggregation_matrix.shape[1])

    @property
    def has_indicators(self) -> bool:
        """Whether at least one indicator column is available."""
        return self.indicators.shape[1] > 0


# ----------------------------------------------------------------------
# Validation helpers
# ----------------------------------------------------------------------
def _numeric_frame(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    """Coerce to float; reject text and infinite values (NaN allowed)."""
    try:
        numeric = frame.apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise InputValidationError(f"{label} must be numeric: {exc}") from exc
    if np.isinf(numeric.to_numpy()).any():
        raise InputValidationError(f"{label} contains infinite values.")
    return numeric


def _numeric_series(series: pd.Series, label: str) -> pd.Series:
    return _numeric_frame(series.to_frame(), label).iloc[:, 0]


def _to_period_index(index: pd.Index, label: str) -> pd.PeriodIndex:
    """Convert a Period or Datetime index to a sorted, unique PeriodIndex."""
    if isinstance(index, pd.PeriodIndex):
        period_index = index
    elif isinstance(index, pd.DatetimeIndex):
        freq = index.freq or pd.infer_freq(index)
        if freq is None:
            raise InputValidationError(
                f"Cannot infer the frequency of {label}; use a PeriodIndex."
            )
        period_index = index.to_period(freq)
    else:
        raise InputValidationError(f"{label} must have a PeriodIndex or DatetimeIndex.")
    if not period_index.is_unique or not period_index.is_monotonic_increasing:
        raise InputValidationError(f"{label} index must be unique and sorted.")
    return period_index


def _indicator_frame(indicators: pd.Series | pd.DataFrame | None) -> pd.DataFrame | None:
    if indicators is None:
        return None
    if isinstance(indicators, pd.Series):
        name = indicators.name if indicators.name is not None else "indicator"
        return indicators.to_frame(name=str(name))
    if isinstance(indicators, pd.DataFrame):
        return indicators.rename(columns=str)
    raise InputValidationError("indicators must be a pandas Series or DataFrame.")


class DataAligner:
    """Build :class:`DisaggregationData` from the supported input formats.

    Parameters
    ----------
    conversion : Conversion or str, default "sum"
    completion : CompletionSettings or bool, default True
        Completion of the time grid (see
        :mod:`tempdisagg.preprocessing.settings`).
    padding : PaddingSettings, int or (before, after), optional
        Artificial padding around the observed targets (off by default).
    """

    def __init__(
        self,
        conversion: Conversion | str = Conversion.SUM,
        completion: object = True,
        padding: object = None,
    ) -> None:
        # Imported here to avoid a circular import (preprocessing uses this module).
        from tempdisagg.preprocessing.settings import as_completion, as_padding

        self.conversion: Conversion = Conversion.parse(conversion)
        self.completion = as_completion(completion)  # type: ignore[arg-type]
        self.padding = as_padding(padding)  # type: ignore[arg-type]

    def _builder(self, calendar: object):
        from tempdisagg.preprocessing.grid import GridBuilder

        return GridBuilder(
            calendar, self.conversion, self.completion, self.padding  # type: ignore[arg-type]
        )

    # ------------------------------------------------------------------
    # Pandas (calendar-aware) interface
    # ------------------------------------------------------------------
    def from_series(
        self,
        y_low: pd.Series,
        indicators: pd.Series | pd.DataFrame | None = None,
        high_freq: str | None = None,
        constraints: Sequence[LinearConstraint] = (),
        weights: pd.Series | None = None,
    ) -> DisaggregationData:
        """Align period-indexed series.

        Parameters
        ----------
        y_low : pd.Series
            Low-frequency target with a Period/Datetime index; NaN marks
            periods to estimate without a constraint.
        indicators : pd.Series or pd.DataFrame, optional
            High-frequency indicators. Not needed by methods that only
            distribute or smooth; then ``high_freq`` gives the output frequency.
        high_freq : str, optional
            High-frequency alias (``"Q"``, ``"M"``, ``"D"``); required when
            ``indicators`` is None.
        constraints : sequence of LinearConstraint, optional
            Additional exact constraints (labels: high-frequency Periods).
        weights : pd.Series, optional
            Aggregation weights per high-frequency Period (e.g. a denominator,
            to disaggregate a rate with ``conversion="average"``).
        """
        from tempdisagg.preprocessing.calendar import PeriodCalendar

        if not isinstance(y_low, pd.Series):
            raise InputValidationError("y_low must be a pandas Series.")
        low_index = _to_period_index(y_low.index, "y_low")
        targets = pd.Series(_numeric_series(y_low, "y_low").to_numpy(), index=low_index)
        frame = _indicator_frame(indicators)
        if frame is not None and frame.shape[1] > 0:
            high_index = _to_period_index(frame.index, "indicators")
            values = _numeric_frame(frame, "indicators").set_axis(high_index)
            freq = high_index.freq
        else:
            if high_freq is None:
                raise InputValidationError(
                    "Without indicators, high_freq (e.g. 'Q', 'M', 'D') is required."
                )
            freq = pd.PeriodIndex([], freq=high_freq).freq
            values = pd.DataFrame(index=pd.PeriodIndex([], freq=freq))
        calendar = PeriodCalendar(low_index.freq, freq)
        if weights is not None:
            weights = _numeric_series(weights, "weights").set_axis(
                _to_period_index(weights.index, "weights"))
        return self._builder(calendar).build(
            targets, values, tuple(values.columns), constraints, weights)

    # ------------------------------------------------------------------
    # NumPy interface
    # ------------------------------------------------------------------
    def from_arrays(
        self,
        y_low: Sequence[float] | np.ndarray,
        indicators: np.ndarray | None,
        frequency_ratio: int,
        n_high: int | None = None,
        constraints: Sequence[LinearConstraint] = (),
        weights: Sequence[float] | np.ndarray | None = None,
    ) -> DisaggregationData:
        """Align arrays with ``frequency_ratio`` sub-periods per period.

        High-frequency row ``i`` belongs to low-frequency period
        ``i // frequency_ratio``. Rows beyond ``len(y_low) * frequency_ratio``
        are extrapolated. Without indicators, ``n_high`` sets the output length
        (default: ``len(y_low) * frequency_ratio``).
        """
        from tempdisagg.preprocessing.calendar import RatioCalendar

        calendar = RatioCalendar(frequency_ratio)
        y_values = np.asarray(np.ravel(y_low), dtype=float)
        if np.isinf(y_values).any():
            raise InputValidationError("y_low contains infinite values.")
        targets = pd.Series(y_values, index=pd.RangeIndex(y_values.size))
        if indicators is not None:
            matrix = np.asarray(indicators, dtype=float)
            matrix = matrix.reshape(-1, 1) if matrix.ndim == 1 else matrix
            if matrix.ndim != 2:
                raise InputValidationError("indicators must be 1-D or 2-D.")
            names = tuple(f"x{i + 1}" for i in range(matrix.shape[1]))
            values = _numeric_frame(pd.DataFrame(matrix, columns=list(names)), "indicators")
        else:
            length = n_high if n_high is not None else y_values.size * frequency_ratio
            values = pd.DataFrame(index=pd.RangeIndex(length))
        if len(values) < y_values.size * frequency_ratio and len(values.columns):
            raise InputValidationError("indicators are shorter than len(y_low) * frequency_ratio.")
        weight_series = None
        if weights is not None:
            weight_series = pd.Series(np.asarray(weights, dtype=float))
        return self._builder(calendar).build(
            targets, values, tuple(values.columns), constraints, weight_series)

    # ------------------------------------------------------------------
    # Long format (Index, Grain, y, X...)
    # ------------------------------------------------------------------
    def from_long_format(
        self,
        frame: pd.DataFrame,
        index_col: str = "Index",
        grain_col: str = "Grain",
        y_col: str = "y",
        x_cols: Sequence[str] | None = None,
        constraints: Sequence[LinearConstraint] = (),
        weights_col: str | None = None,
    ) -> DisaggregationData:
        """Align the long format of tempdisagg 0.2.x.

        Each row is one high-frequency period; ``y_col`` holds the
        low-frequency value repeated within its group (or NaN). ``x_cols`` is
        optional: pass no indicator columns for methods that only distribute
        or smooth. Missing grains of a group are completed (see
        :class:`~tempdisagg.preprocessing.settings.CompletionSettings`).
        """
        from tempdisagg.preprocessing.calendar import GrainCalendar

        x_cols = list(x_cols or [])
        required = [index_col, grain_col, y_col, *x_cols] + ([weights_col] if weights_col else [])
        missing = [col for col in required if col not in frame.columns]
        if missing:
            raise InputValidationError(f"Missing required columns: {missing}.")
        if frame.duplicated([index_col, grain_col]).any():
            raise InputValidationError("Duplicate (index, grain) pairs found.")
        ordered = frame.sort_values([index_col, grain_col]).reset_index(drop=True)
        target_values = _numeric_series(ordered[y_col], "y")

        targets = {}
        for label, rows in ordered.groupby(index_col, sort=True).groups.items():
            values = target_values.loc[rows].dropna()
            if values.nunique() > 1:
                raise InputValidationError(
                    f"Group {label} has inconsistent target values: {values.unique()}."
                )
            targets[label] = float(values.iloc[0]) if len(values) else np.nan
        target_series = pd.Series(targets, dtype=float)

        calendar = GrainCalendar(ordered[grain_col].unique(), target_series.index)
        high = pd.MultiIndex.from_frame(ordered[[index_col, grain_col]])
        values = _numeric_frame(ordered[x_cols], "indicators").set_axis(high) if x_cols \
            else pd.DataFrame(index=high)
        weight_series = (_numeric_series(ordered[weights_col], "weights").set_axis(high)
                         if weights_col else None)
        return self._builder(calendar).build(
            target_series, values, tuple(map(str, x_cols)), constraints, weight_series)
