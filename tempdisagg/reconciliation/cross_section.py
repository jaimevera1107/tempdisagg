"""Cross-sectional reconciliation of a system of disaggregated series.

Problem
-------
``S`` series (e.g. departments) have been disaggregated individually and
satisfy their own temporal constraints ``C_s y_s = a_s``. A high-frequency
total ``z`` (e.g. the national series) must also hold:
``sum_s y_{s,t} = z_t`` for every period. Individually consistent estimates
generally violate it.

Method (two-step reconciliation; Di Fonzo and Marini, 2011)
-----------------------------------------------------------
Keep the preliminary estimates ``p_s`` from step one and find the closest
system satisfying **both** sets of constraints:

    minimise  sum_s || D_h W_s (y_s - p_s) ||^2
    subject to  C_s y_s = a_s  (all s),   sum_s y_s = z,

with ``W_s = I`` (additive) or ``diag(1/p_s)`` (proportional, preserves
relative movements and keeps zero estimates fixed), and ``D_h`` the identity
(``h = 0``, minimal adjustments) or the first difference without initial
condition (``h = 1``, movement preservation, multivariate Denton-Cholette).

The KKT system is sparse and solved by sparse LU.

Consistency
-----------
The two constraint sets overlap: for every low-frequency period ``g`` in which
all series have a target, ``sum_s a_s(g)`` must equal ``C_g z``. A mismatch
means the inputs contradict each other and raises
:class:`~tempdisagg.exceptions.InfeasibleConstraintError` with the periods and
discrepancies. When consistent, one temporal row per such period is redundant
and is removed before solving.

Reference: Di Fonzo, T. and Marini, M. (2011). Simultaneous and two-step
reconciliation of systems of time series: methodological and practical issues.
Journal of the Royal Statistical Society C, 60(2), 143-164.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import splu

from tempdisagg.core.aggregation import Conversion
from tempdisagg.exceptions import InfeasibleConstraintError, InputValidationError, NumericalError
from tempdisagg.results import DisaggregationResult

#: Relative tolerance of the consistency check between totals.
CONSISTENCY_TOLERANCE: float = 1e-8


class AdjustmentCriterion(str, Enum):
    """Scale in which adjustments are penalised."""

    ADDITIVE = "additive"
    PROPORTIONAL = "proportional"


@dataclass(frozen=True)
class CrossSectionalResult:
    """Reconciled system.

    Attributes
    ----------
    values : pd.DataFrame
        Reconciled high-frequency series (columns: series).
    preliminary : pd.DataFrame
        Step-one estimates.
    temporal_error : float
        Largest violation of the series' own low-frequency constraints.
    contemporaneous_error : float
        Largest violation of ``sum_s y_s = z``.
    redundant_rows : int
        Temporal rows removed because they are implied by the others.
    """

    values: pd.DataFrame
    preliminary: pd.DataFrame
    temporal_error: float
    contemporaneous_error: float
    redundant_rows: int

    @property
    def adjustments(self) -> pd.DataFrame:
        """``values - preliminary``."""
        return self.values - self.preliminary


class CrossSectionalReconciler:
    """Reconcile disaggregated series with a high-frequency total.

    Parameters
    ----------
    criterion : {"proportional", "additive"}, default "proportional"
    h : {0, 1}, default 1
        0 minimises the size of the adjustments; 1 preserves their movement.
    """

    def __init__(self, criterion: AdjustmentCriterion | str = AdjustmentCriterion.PROPORTIONAL,
                 h: int = 1) -> None:
        try:
            self.criterion = AdjustmentCriterion(criterion)
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        if h not in (0, 1) or isinstance(h, bool):
            raise InputValidationError("h must be 0 or 1.")
        self.h = int(h)

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def reconcile_results(
        self, results: Mapping[str, DisaggregationResult], total: pd.Series
    ) -> CrossSectionalResult:
        """Reconcile fitted models with a high-frequency total.

        Uses each result's estimate as preliminary series and its observed
        targets (with the fit's conversion rule) as temporal constraints.
        """
        if not results:
            raise InputValidationError("At least one result is required.")
        conversions = {result.data.conversion for result in results.values()}
        if len(conversions) != 1:
            raise InputValidationError("All results must share the same conversion.")
        preliminary = pd.DataFrame({name: r.values for name, r in results.items()})
        targets = pd.DataFrame({
            name: pd.Series(r.data.y_low[r.data.target_rows],
                            index=r.data.low_index[r.data.target_rows])
            for name, r in results.items()
        })
        return self.reconcile(preliminary, targets, total, conversions.pop())

    def reconcile(
        self,
        preliminary: pd.DataFrame,
        low_targets: pd.DataFrame,
        total: pd.Series,
        conversion: Conversion | str = Conversion.SUM,
    ) -> CrossSectionalResult:
        """Reconcile preliminary estimates.

        Parameters
        ----------
        preliminary : pd.DataFrame
            High-frequency estimates, PeriodIndex, one column per series.
        low_targets : pd.DataFrame
            Low-frequency targets, PeriodIndex, same columns (NaN allowed).
        total : pd.Series
            High-frequency total (PeriodIndex); NaN periods are not constrained.
        conversion : Conversion or str
            Temporal aggregation rule of the targets.
        """
        rule = Conversion.parse(conversion)
        prelim, targets, totals = self._validate(preliminary, low_targets, total)
        n, n_series = prelim.shape
        index = prelim.index
        low_freq = targets.index.freq
        parents = index.asfreq(low_freq)

        temporal_rows, temporal_values, dropped = self._temporal_rows(
            prelim, targets, totals, parents, rule)
        observed_total = np.flatnonzero(~np.isnan(totals))
        contemporaneous = sparse.hstack(
            [sparse.eye(n, format="csr")[observed_total]] * n_series, format="csr")

        penalty = sparse.block_diag(
            [self._penalty(prelim.iloc[:, s].to_numpy()) for s in range(n_series)], format="csc")
        constraints = sparse.vstack([temporal_rows, contemporaneous], format="csc")
        rhs_constraints = np.concatenate([temporal_values, totals[observed_total]])
        p = prelim.to_numpy().T.reshape(-1)  # series-major stacking

        scale = penalty.diagonal().mean() if penalty.nnz else 1.0
        penalty = penalty / scale
        m = constraints.shape[0]
        kkt = sparse.bmat([[penalty, constraints.T], [constraints, None]], format="csc")
        rhs = np.concatenate([penalty @ p, rhs_constraints])
        try:
            solution = splu(kkt).solve(rhs)
        except RuntimeError as exc:
            raise NumericalError(
                "The reconciliation system is singular (constraints do not identify the "
                "adjustments; with h=1 each series needs at least one temporal target)."
            ) from exc
        y = solution[: n * n_series]
        if not np.all(np.isfinite(y)) or m == 0:
            raise NumericalError("The reconciliation produced non-finite values.")

        values = pd.DataFrame(y.reshape(n_series, n).T, index=index, columns=prelim.columns)
        temporal_error = self._temporal_error(values, targets, parents, rule)
        sums = values.to_numpy().sum(axis=1)
        contemporaneous_error = float(np.max(np.abs(sums[observed_total]
                                                    - totals[observed_total]))) \
            if observed_total.size else 0.0
        return CrossSectionalResult(values, prelim, temporal_error, contemporaneous_error,
                                    dropped)

    # ------------------------------------------------------------------
    # Building blocks
    # ------------------------------------------------------------------
    @staticmethod
    def _validate(preliminary: pd.DataFrame, low_targets: pd.DataFrame, total: pd.Series
                  ) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        for name, obj in (("preliminary", preliminary), ("low_targets", low_targets),
                          ("total", total)):
            if not isinstance(obj.index, pd.PeriodIndex):
                raise InputValidationError(f"{name} must have a PeriodIndex.")
        if list(preliminary.columns) != list(low_targets.columns):
            raise InputValidationError("preliminary and low_targets must have the same columns.")
        prelim = preliminary.astype(float)
        if prelim.isna().any().any() or not np.all(np.isfinite(prelim.to_numpy())):
            raise InputValidationError("preliminary estimates must be finite.")
        totals = total.reindex(prelim.index).to_numpy(dtype=float)
        targets = low_targets.astype(float)
        return prelim, targets, totals

    def _penalty(self, p: np.ndarray) -> sparse.csr_matrix:
        n = p.size
        operator = sparse.eye(n, format="csr")
        if self.h == 1:
            operator = (sparse.eye(n, format="csr") - sparse.eye(n, k=-1, format="csr"))[1:]
        if self.criterion is AdjustmentCriterion.PROPORTIONAL:
            if np.any(p == 0):
                raise InputValidationError(
                    "The proportional criterion requires non-zero preliminary estimates; "
                    "use criterion='additive'.")
            operator = operator @ sparse.diags(1.0 / p)
        return (operator.T @ operator).tocsr()

    @staticmethod
    def _row_weights(members: np.ndarray, rule: Conversion) -> np.ndarray:
        weights = np.zeros(members.size)
        if rule is Conversion.SUM:
            weights[:] = 1.0
        elif rule is Conversion.AVERAGE:
            weights[:] = 1.0 / members.size
        elif rule is Conversion.FIRST:
            weights[0] = 1.0
        else:
            weights[-1] = 1.0
        return weights

    def _temporal_rows(self, prelim: pd.DataFrame, targets: pd.DataFrame, totals: np.ndarray,
                       parents: pd.PeriodIndex, rule: Conversion
                       ) -> tuple[sparse.csr_matrix, np.ndarray, int]:
        n, n_series = prelim.shape
        rows, cols, data, values = [], [], [], []
        row_count = dropped = 0
        mismatches = []
        for period in targets.index:
            members = np.flatnonzero(parents == period)
            if members.size == 0:
                continue
            column_targets = targets.loc[period].to_numpy()
            all_observed = np.all(~np.isnan(column_targets))
            weights = self._row_weights(members, rule)
            redundant_series = None
            if all_observed and np.all(~np.isnan(totals[members])):
                implied = float(weights @ totals[members])
                declared = float(np.sum(column_targets))
                scale = max(1.0, abs(implied), abs(declared))
                if abs(implied - declared) > CONSISTENCY_TOLERANCE * scale:
                    mismatches.append(f"{period}: series sum {declared:.10g} vs total "
                                      f"{implied:.10g}")
                redundant_series = n_series - 1  # implied by the others and the total
            for s in range(n_series):
                if np.isnan(column_targets[s]) or s == redundant_series:
                    dropped += int(s == redundant_series)
                    continue
                rows.extend([row_count] * members.size)
                cols.extend((s * n + members).tolist())
                data.extend(weights.tolist())
                values.append(column_targets[s])
                row_count += 1
        if mismatches:
            raise InfeasibleConstraintError(
                "The series' low-frequency targets do not add up to the total: "
                + "; ".join(mismatches[:5]))
        matrix = sparse.csr_matrix((data, (rows, cols)), shape=(row_count, n * n_series))
        return matrix, np.asarray(values, dtype=float), dropped

    def _temporal_error(self, values: pd.DataFrame, targets: pd.DataFrame,
                        parents: pd.PeriodIndex, rule: Conversion) -> float:
        worst = 0.0
        for period in targets.index:
            members = np.flatnonzero(parents == period)
            if members.size == 0:
                continue
            weights = self._row_weights(members, rule)
            for column in values.columns:
                target = targets.loc[period, column]
                if not np.isnan(target):
                    worst = max(worst, abs(float(weights @ values[column].to_numpy()[members])
                                           - float(target)))
        return worst
