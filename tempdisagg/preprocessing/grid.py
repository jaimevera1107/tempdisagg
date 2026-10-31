"""Build a complete, regular time grid and the constraint system.

:class:`GridBuilder` is the single place where inputs become a
:class:`~tempdisagg.core.data.DisaggregationData`. Steps:

1. **Targets.** Observed low-frequency values; periods without one stay
   unconstrained (never imputed).
2. **Padding** (optional). Artificial low-frequency periods before the first
   and after the last observed target, with artificial targets
   (``target_method``) or unconstrained.
3. **Partial periods.** A targeted period whose sub-periods are not all
   observed is completed, constrained only over its observed sub-periods, or
   rejected (``partial_periods``).
4. **Grid.** All high-frequency labels between the first and the last label
   that is observed or required. Interior gaps are included: the residual
   models assume consecutive periods.
5. **Indicators and weights.** Reindexed to the grid; missing values are
   interpolated inside and extended at the edges.
6. **Constraint rows.** One row per targeted period (weighted by the
   conversion rule and optional aggregation weights), then one row per
   additional :class:`~tempdisagg.core.constraints.LinearConstraint`. The
   stacked system is checked for contradictions and redundancies.

With completion disabled, any added label or missing value raises an error
(strict mode). Every artificial value is recorded and reported with an
:class:`~tempdisagg.exceptions.ArtificialDataWarning`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from tempdisagg.core.aggregation import Conversion
from tempdisagg.core.constraints import LinearConstraint, independent_rows
from tempdisagg.core.data import DisaggregationData, RowKind
from tempdisagg.exceptions import (
    ArtificialDataWarning,
    InputValidationError,
    MissingTargetWarning,
    RedundantConstraintWarning,
)
from tempdisagg.preprocessing.calendar import Calendar
from tempdisagg.preprocessing.extension import extend, fill_missing
from tempdisagg.preprocessing.settings import (
    CompletionSettings,
    EdgeMethod,
    PaddingSettings,
    PartialPeriodPolicy,
    TargetPaddingMethod,
)


@dataclass(frozen=True)
class _TargetRow:
    """One low-frequency target before it becomes a row of C."""

    label: object
    value: float
    kind: RowKind
    partial: bool


@dataclass(frozen=True)
class GridBuilder:
    """Turn labelled targets, indicators and constraints into ``DisaggregationData``."""

    calendar: Calendar
    conversion: Conversion
    completion: CompletionSettings
    padding: PaddingSettings

    def build(
        self,
        targets: pd.Series,
        indicators: pd.DataFrame,
        names: tuple[str, ...],
        constraints: Sequence[LinearConstraint] = (),
        weights: pd.Series | None = None,
    ) -> DisaggregationData:
        """Build the data object.

        Parameters
        ----------
        targets : pd.Series
            Low-frequency values indexed by low labels (NaN = unknown).
        indicators : pd.DataFrame
            High-frequency indicators indexed by high labels; may have zero
            columns (no indicator). With zero rows, the grid spans the targets.
        names : tuple of str
            Indicator names.
        constraints : sequence of LinearConstraint
            Additional exact constraints on the high-frequency series.
        weights : pd.Series, optional
            Aggregation weights per high-frequency period (e.g. a denominator
            for rates). With ``sum``: ``sum w_t y_t``; with ``average``:
            ``sum w_t y_t / sum w_t``.
        """
        observed = targets.dropna()
        if observed.empty:
            raise InputValidationError("At least one observed low-frequency value is required.")
        if weights is not None and self.conversion in (Conversion.FIRST, Conversion.LAST):
            raise InputValidationError("Aggregation weights apply to 'sum' and 'average' only.")

        low_labels = self._low_labels(targets, indicators, observed)
        real_index = self._real_index(indicators, targets, low_labels)
        rows = self._target_rows(low_labels, observed, real_index)

        padded = [label for label in self._padded_labels(observed) if label in low_labels]
        grid, real = self._grid(low_labels, real_index, rows, padded)
        matrix, completed = self._fill(grid, indicators, "indicators")
        weight_values = self._weights(grid, weights)

        constraint_matrix, values, kinds, labels = self._constraint_system(
            grid, real, rows, constraints, weight_values)
        self._report(grid, real, completed, rows, low_labels, observed)

        return DisaggregationData(
            y_low=values,
            indicators=matrix,
            aggregation_matrix=constraint_matrix,
            high_index=grid,
            low_index=self._label_index(labels, low_labels),
            indicator_names=names,
            conversion=self.conversion,
            artificial_low=kinds == RowKind.PADDING.value,
            artificial_high=~real,
            completed_indicators=completed,
            row_kind=kinds,
        )

    # ------------------------------------------------------------------
    # Labels and targets
    # ------------------------------------------------------------------
    def _low_labels(self, targets: pd.Series, indicators: pd.DataFrame,
                    observed: pd.Series) -> pd.Index:
        """Contiguous low labels covering data and padding."""
        candidates = list(targets.index)
        if len(indicators.index):
            candidates += list(self.calendar.parents(indicators.index))
        first, last = min(candidates), max(candidates)
        if self.padding.before:
            first = min(first, self.calendar.shift(observed.index.min(), -self.padding.before))
        if self.padding.after:
            last = max(last, self.calendar.shift(observed.index.max(), self.padding.after))
        return self.calendar.low_range(first, last)

    def _real_index(self, indicators: pd.DataFrame, targets: pd.Series,
                    low_labels: pd.Index) -> pd.Index:
        """High-frequency labels present in the input.

        Without indicator rows (pandas input without indicators), the periods
        spanned by the targets are the output and count as present.
        """
        if len(indicators.index):
            return indicators.index
        span = self.calendar.low_range(targets.index.min(), targets.index.max())
        labels = [label for low in span for label in self.calendar.subperiods(low)]
        return self.calendar.high_index(labels)

    def _target_rows(self, low_labels: pd.Index, observed: pd.Series,
                     real_index: pd.Index) -> list[_TargetRow]:
        """Observed and padding targets with their partial-period status."""
        padding_values = self._padding_targets(low_labels, observed)
        rows = []
        for label in low_labels:
            if label in observed.index:
                members = self.calendar.subperiods(label)
                present = members.isin(real_index)
                partial = not present.all()
                if partial:
                    partial = self._partial_policy(label, int(present.sum()), len(members))
                rows.append(_TargetRow(label, float(observed[label]), RowKind.TARGET, partial))
            elif label in padding_values.index:
                rows.append(_TargetRow(label, float(padding_values[label]),
                                       RowKind.PADDING, False))
        return rows

    def _partial_policy(self, label: object, present: int, expected: int) -> bool:
        """Apply the partial-period policy; return True if constrained partially."""
        policy = self.completion.partial_periods
        if policy is PartialPeriodPolicy.ERROR:
            raise InputValidationError(
                f"Low-frequency period {label} has {present} of {expected} sub-periods "
                "(partial_periods='error')."
            )
        if policy is PartialPeriodPolicy.PARTIAL:
            if present == 0:
                raise InputValidationError(
                    f"Low-frequency period {label} has no observed sub-period, so a "
                    "partial constraint is impossible."
                )
            return True
        if not self.completion.enabled:
            raise InputValidationError(
                f"The time grid is incomplete: period {label} has {present} of {expected} "
                "sub-periods. Enable completion, use partial_periods='partial', or "
                "provide them."
            )
        return False

    def _padded_labels(self, observed: pd.Series) -> list:
        before = [self.calendar.shift(observed.index.min(), -k)
                  for k in range(1, self.padding.before + 1)]
        after = [self.calendar.shift(observed.index.max(), k)
                 for k in range(1, self.padding.after + 1)]
        return before + after

    def _padding_targets(self, low_labels: pd.Index, observed: pd.Series) -> pd.Series:
        """Artificial targets of padded periods (empty without target padding)."""
        method = self.padding.target_method
        if not self.padding.active or method is TargetPaddingMethod.NONE:
            return pd.Series(dtype=float)
        positions = np.asarray([low_labels.get_loc(label) for label in observed.index])
        known = observed.to_numpy(dtype=float)
        result = {}
        padded = [label for label in self._padded_labels(observed) if label in low_labels]
        for label in padded:
            position = low_labels.get_loc(label)
            after = position > positions.max()
            if method is TargetPaddingMethod.REPEAT:
                result[label] = known[-1] if after else known[0]
            else:
                result[label] = float(extend(positions, known, np.array([position]),
                                             EdgeMethod.LINEAR, self.padding.fit_window)[0])
        return pd.Series(result, dtype=float)

    # ------------------------------------------------------------------
    # Grid
    # ------------------------------------------------------------------
    def _grid(self, low_labels: pd.Index, real_index: pd.Index, rows: list[_TargetRow],
              padded: list) -> tuple[pd.Index, np.ndarray]:
        """Regular grid from the first to the last observed/required label.

        Required: all sub-periods of fully constrained periods and of padded
        periods (with or without artificial targets). Additional constraints
        must refer to periods inside this grid; they never extend it, so a
        constraint cannot silently trigger a long extrapolation.
        """
        pieces = [self.calendar.subperiods(label) for label in low_labels]
        candidates = self.calendar.high_index([lab for piece in pieces for lab in piece])
        full = [row.label for row in rows if not row.partial] + padded
        parents = self.calendar.parents(candidates)
        keep = np.asarray(candidates.isin(real_index)) | np.asarray(parents.isin(full))
        if not keep.any():
            raise InputValidationError("No high-frequency periods to estimate.")
        first, last = np.flatnonzero(keep)[[0, -1]]
        grid = candidates[first:last + 1]
        real = np.asarray(grid.isin(real_index))
        if not self.completion.enabled and not real.all():
            missing = list(grid[~real][:5])
            raise InputValidationError(
                f"The time grid is incomplete ({int((~real).sum())} high-frequency "
                f"periods missing, e.g. {missing}). Enable completion or provide them."
            )
        return grid, real

    # ------------------------------------------------------------------
    # Indicators and weights
    # ------------------------------------------------------------------
    def _fill(self, grid: pd.Index, frame: pd.DataFrame, label: str
              ) -> tuple[np.ndarray, np.ndarray]:
        """Reindex columns to the grid and fill missing values."""
        if frame.shape[1] == 0:
            return np.empty((len(grid), 0)), np.zeros((len(grid), 0), dtype=bool)
        raw = frame.reindex(grid).to_numpy(dtype=float)
        missing = np.isnan(raw)
        if missing.any() and not self.completion.enabled:
            raise InputValidationError(
                f"{label} contain {int(missing.sum())} missing values. Enable completion "
                "or clean them before fitting."
            )
        method, window = self.completion.edge_method, self.completion.fit_window
        if self.padding.active:
            method, window = self.padding.indicator_method, self.padding.fit_window
        n_parents = max(1, len(set(self.calendar.parents(grid))))
        group_size = max(1, int(round(len(grid) / n_parents)))
        filled = np.column_stack([
            fill_missing(raw[:, j], method, window * group_size) for j in range(raw.shape[1])
        ])
        return filled, missing

    def _weights(self, grid: pd.Index, weights: pd.Series | None) -> np.ndarray | None:
        if weights is None:
            return None
        if not weights.index.is_unique:
            raise InputValidationError("Aggregation weights must have unique labels.")
        values, _ = self._fill(grid, weights.to_frame("weight"), "aggregation weights")
        result = values[:, 0]
        if not np.all(np.isfinite(result)) or np.any(result <= 0):
            raise InputValidationError("Aggregation weights must be finite and positive.")
        return result

    # ------------------------------------------------------------------
    # Constraint system
    # ------------------------------------------------------------------
    def _constraint_system(
        self, grid: pd.Index, real: np.ndarray, rows: list[_TargetRow],
        constraints: Sequence[LinearConstraint], weights: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        parents = np.asarray(self.calendar.parents(grid), dtype=object)
        matrix_rows, values, kinds, labels = [], [], [], []
        for row in rows:
            members = np.flatnonzero(parents == row.label)
            if row.partial:
                members = members[real[members]]
            matrix_rows.append(self._target_weights(row, grid, members, weights))
            values.append(row.value)
            kinds.append(row.kind.value)
            labels.append(row.label)

        position = {label: i for i, label in enumerate(grid)}
        for constraint in constraints:
            vector = np.zeros(len(grid))
            for label, weight in constraint.coefficients.items():
                if label not in position:
                    raise InputValidationError(
                        f"Constraint '{constraint.label}' refers to {label!r}, which is not "
                        "a high-frequency period of the output."
                    )
                vector[position[label]] = float(weight)
            matrix_rows.append(vector)
            values.append(constraint.value)
            kinds.append(RowKind.CONSTRAINT.value)
            labels.append(constraint.label)

        matrix = np.vstack(matrix_rows)
        values_array = np.asarray(values, dtype=float)
        kinds_array = np.asarray(kinds)
        if constraints:
            keep, dropped = independent_rows(matrix, values_array, [str(x) for x in labels])
            if dropped:
                warnings.warn(
                    f"{len(dropped)} redundant constraint(s) dropped (implied by the others): "
                    + ", ".join(str(labels[i]) for i in dropped[:5]),
                    RedundantConstraintWarning, stacklevel=5,
                )
            matrix, values_array, kinds_array = matrix[keep], values_array[keep], kinds_array[keep]
            labels = [labels[i] for i in keep]
        return matrix, values_array, kinds_array, labels

    def _target_weights(self, row: _TargetRow, grid: pd.Index, members: np.ndarray,
                        weights: np.ndarray | None) -> np.ndarray:
        """Row of C for one target, following the conversion rule."""
        vector = np.zeros(len(grid))
        if self.conversion in (Conversion.FIRST, Conversion.LAST):
            expected = self.calendar.subperiods(row.label)
            edge = expected[0] if self.conversion is Conversion.FIRST else expected[-1]
            target = [i for i in members if grid[i] == edge]
            if not target:
                raise InputValidationError(
                    f"Period {row.label}: the {self.conversion.value} sub-period ({edge}) is "
                    "not observed, so the target cannot be applied."
                )
            vector[target[0]] = 1.0
            return vector
        base = np.ones(members.size) if weights is None else weights[members]
        if self.conversion is Conversion.AVERAGE:
            base = base / base.sum()
        vector[members] = base
        return vector

    @staticmethod
    def _label_index(labels: list, low_labels: pd.Index) -> pd.Index:
        """Index of the constraint rows (object dtype when constraints are mixed in)."""
        if all(label in low_labels for label in labels):
            if isinstance(low_labels, pd.PeriodIndex):
                return pd.PeriodIndex(labels, freq=low_labels.freq)
            return pd.Index(labels)
        return pd.Index(labels, dtype=object)

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------
    @staticmethod
    def _report(grid: pd.Index, real: np.ndarray, completed: np.ndarray,
                rows: list[_TargetRow], low_labels: pd.Index, observed: pd.Series) -> None:
        """Warn about artificial values and unconstrained interior periods."""
        messages = []
        if (~real).any():
            messages.append(f"{int((~real).sum())} high-frequency period(s) added to the grid")
        filled_real = completed[real].sum() if completed.size else 0
        if filled_real:
            messages.append(f"{int(filled_real)} missing indicator value(s) filled")
        n_padding = sum(row.kind is RowKind.PADDING for row in rows)
        if n_padding:
            messages.append(f"{n_padding} artificial low-frequency target(s) added by padding")
        n_partial = sum(row.partial for row in rows)
        if n_partial:
            messages.append(f"{n_partial} period(s) constrained over observed sub-periods only")
        if messages:
            text = "Artificial data: " + "; ".join(messages) + ". See result.preprocessing."
            warnings.warn(text, ArtificialDataWarning, stacklevel=5)
        targeted = {row.label for row in rows}
        inside = low_labels[(low_labels >= observed.index.min()) &
                            (low_labels <= observed.index.max())]
        unconstrained = [label for label in inside if label not in targeted]
        if unconstrained:
            warnings.warn(
                f"{len(unconstrained)} low-frequency period(s) have no target value; their "
                "high-frequency values are estimated without an aggregation constraint.",
                MissingTargetWarning, stacklevel=5,
            )
