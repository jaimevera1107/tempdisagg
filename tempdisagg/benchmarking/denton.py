"""Denton and Denton-Cholette benchmarking, and uniform distribution.

Denton methods adjust a high-frequency indicator ``x`` so that it satisfies
the low-frequency constraint while keeping its movement:

* additive criterion:      minimise ``|| D (y - x) ||^2``
* proportional criterion:  minimise ``|| D diag(1/x) (y - x) ||^2``

subject to ``C y = y_l``. ``D`` is the ``h``-th difference operator.
The original Denton (1971) keeps the initial-condition rows of ``D``
(implicitly assuming ``y_0 = x_0``), which distorts the start of the series;
Denton-Cholette drops them.

The problem is solved through its KKT system

    [ A   C' ] [ y      ]   [ A x ]
    [ C   0  ] [ lambda ] = [ y_l ],    A = D' D,

which is non-singular whenever the constraints pin down the null space of
``D`` (for ``h = 1``, the constant vector). No regulariser or
pseudo-inverse is used, so the solution is the exact minimiser
(see issue #4 of tempdisagg 0.2.13).

``D`` is banded and ``C`` has one non-zero per column, so the KKT matrix is
sparse; it is factorised with a sparse LU, which keeps daily problems with
thousands of periods fast. The reported condition number is the 1-norm
estimate ``||K||_1 ||K^{-1}||_1`` (Higham's estimator).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from scipy import linalg, sparse
from scipy.sparse.linalg import LinearOperator, onenormest, splu

from tempdisagg.core.aggregation import Conversion
from tempdisagg.core.data import DisaggregationData
from tempdisagg.core.linalg import difference_matrix
from tempdisagg.estimation.base import DentonOutcome, MethodOutput
from tempdisagg.estimation.diagnostics import DiagnosticLog
from tempdisagg.exceptions import (
    IllConditionedWarning,
    IndicatorSignWarning,
    InputValidationError,
    NumericalError,
)

#: Relative tolerance used to verify the aggregation constraint after solving.
CONSTRAINT_TOLERANCE: float = 1e-8
#: Condition number above which the KKT system is reported as ill-conditioned.
ILL_CONDITION_THRESHOLD: float = 1e12
#: Above this length the difference operator is built directly in sparse form.
DENSE_DIFFERENCE_LIMIT: int = 2000


def _sparse_difference(n: int, order: int, drop_initial: bool) -> sparse.csr_matrix:
    """Sparse ``h``-th difference operator (same rows as :func:`difference_matrix`)."""
    first = sparse.eye(n, format="csr") - sparse.eye(n, k=-1, format="csr")
    matrix = sparse.eye(n, format="csr")
    for _ in range(order):
        matrix = first @ matrix
    return matrix[order:] if drop_initial else matrix


def _condition_estimate(matrix: sparse.csc_matrix, factorisation: object) -> float:
    """1-norm condition number estimate using the existing LU factors."""
    size = matrix.shape[0]
    inverse = LinearOperator(
        (size, size),
        matvec=factorisation.solve,  # type: ignore[attr-defined]
        rmatvec=lambda v: factorisation.solve(v, trans="T"),  # type: ignore[attr-defined]
        dtype=float,
    )
    norm = float(abs(matrix).sum(axis=0).max())
    return norm * float(onenormest(inverse))


class DentonVariant(str, Enum):
    """Treatment of the initial condition."""

    ORIGINAL = "original"    # Denton (1971): keeps the initial-condition rows
    CHOLETTE = "cholette"    # Dagum and Cholette (2006): drops them


class DentonCriterion(str, Enum):
    """Quantity whose movement is preserved."""

    ADDITIVE = "additive"
    PROPORTIONAL = "proportional"


class DentonBenchmarker:
    """Denton / Denton-Cholette benchmarking of a single indicator.

    Parameters
    ----------
    variant : DentonVariant or str, default "cholette"
    criterion : DentonCriterion or str, default "proportional"
    order : int, default 1
        Differencing order ``h`` (0 penalises levels, 1 changes, 2 curvature).
    """

    def __init__(
        self,
        variant: DentonVariant | str = DentonVariant.CHOLETTE,
        criterion: DentonCriterion | str = DentonCriterion.PROPORTIONAL,
        order: int = 1,
    ) -> None:
        try:
            self.variant: DentonVariant = DentonVariant(variant)
            self.criterion: DentonCriterion = DentonCriterion(criterion)
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        if not isinstance(order, (int, np.integer)) or order < 0:
            raise InputValidationError("order (h) must be a non-negative integer.")
        self.order: int = int(order)

    def disaggregate(self, data: DisaggregationData) -> MethodOutput:
        """Benchmark the indicator (or a constant) to the low-frequency data."""
        log = DiagnosticLog()
        indicator = self._indicator(data, log)
        n_high = data.n_high
        drop_initial = self.variant is DentonVariant.CHOLETTE
        if drop_initial and self.order >= n_high:
            raise InputValidationError("order (h) must be smaller than the series length.")

        difference = sparse.csr_matrix(
            difference_matrix(n_high, self.order, drop_initial=drop_initial)
            if n_high <= DENSE_DIFFERENCE_LIMIT
            else _sparse_difference(n_high, self.order, drop_initial)
        )
        if self.criterion is DentonCriterion.PROPORTIONAL:
            difference = difference @ sparse.diags(1.0 / indicator)
        penalty = (difference.T @ difference).tocsc()
        diagonal = penalty.diagonal()
        # Rescaling the penalty leaves the minimiser unchanged and keeps the
        # KKT system balanced when the indicator has a large scale.
        scale = float(diagonal.mean()) if np.any(diagonal) else 1.0
        penalty = penalty / scale

        n_low = data.n_low
        if drop_initial:
            self._check_identification(data.aggregation_matrix, indicator)
        matrix = sparse.csr_matrix(data.aggregation_matrix)
        kkt = sparse.bmat([[penalty, matrix.T], [matrix, None]], format="csc")
        rhs = np.concatenate([penalty @ indicator, data.y_low])
        try:
            factorisation = splu(kkt)
        except RuntimeError as exc:
            raise NumericalError(
                "The Denton system is singular: the constraints do not identify "
                f"the series (h={self.order} may be too large for {n_low} "
                "low-frequency observations)."
            ) from exc
        solution = factorisation.solve(rhs)
        if not np.all(np.isfinite(solution)):
            raise NumericalError("The Denton system produced non-finite values.")

        values = solution[:n_high]
        self._check_constraint(data.aggregation_matrix @ values, data.y_low)
        condition_number = _condition_estimate(kkt, factorisation)
        if condition_number > ILL_CONDITION_THRESHOLD:
            log.warn(IllConditionedWarning,
                     f"The Denton KKT system is ill-conditioned ({condition_number:.2e}).")
        movement = difference @ (values - indicator)
        return MethodOutput(
            values=values,
            fitted_low=data.aggregation_matrix @ indicator,
            diagnostics=log,
            denton=DentonOutcome(float(movement @ movement), condition_number),
        )

    def _check_identification(self, matrix: np.ndarray, indicator: np.ndarray) -> None:
        """Verify that the constraints pin down the null space of the penalty.

        Without the initial-condition rows, ``D_h`` annihilates polynomials of
        degree < h (times ``x`` under the proportional criterion). The KKT
        system is non-singular iff ``C N`` has full column rank, where the
        columns of ``N`` span that null space.
        """
        if self.order == 0:
            return
        n_high = matrix.shape[1]
        time_index = np.linspace(-1.0, 1.0, n_high)
        basis = np.vander(time_index, self.order, increasing=True)
        if self.criterion is DentonCriterion.PROPORTIONAL:
            basis = basis * indicator[:, np.newaxis]
        projected = matrix @ np.linalg.qr(basis)[0]
        singular_values = np.linalg.svd(projected, compute_uv=False)
        if singular_values.size < self.order or (
            singular_values[-1] <= 1e-10 * max(singular_values[0], np.finfo(float).tiny)
        ):
            raise NumericalError(
                f"Denton-Cholette with h={self.order} is not identified: the "
                f"{matrix.shape[0]} low-frequency constraint(s) cannot determine the "
                f"{self.order}-dimensional set of series the penalty ignores. "
                "Lower h or use variant='original'."
            )

    def _indicator(self, data: DisaggregationData, log: DiagnosticLog) -> np.ndarray:
        """Return the single indicator, or a constant if none is given."""
        if data.indicators.shape[1] > 1:
            raise InputValidationError(
                "Denton methods benchmark exactly one indicator; got "
                f"{data.indicators.shape[1]}. Use a regression method for "
                "several indicators."
            )
        if not data.has_indicators:
            return np.ones(data.n_high)
        indicator = data.indicators[:, 0].astype(float)
        if self.criterion is DentonCriterion.PROPORTIONAL and np.any(indicator == 0.0):
            raise InputValidationError(
                "The proportional criterion requires a non-zero indicator; "
                "use criterion='additive'."
            )
        if self.criterion is DentonCriterion.PROPORTIONAL and (
            indicator.min() < 0.0 < indicator.max()
        ):
            log.warn(
                IndicatorSignWarning,
                "The indicator changes sign; proportional benchmarking of a "
                "sign-changing series distorts ratios near zero.",
            )
        return indicator

    @staticmethod
    def _check_constraint(aggregated: np.ndarray, target: np.ndarray) -> None:
        """Verify that the solution reproduces the low-frequency data."""
        scale = max(float(np.max(np.abs(target))), 1.0)
        if np.max(np.abs(aggregated - target)) > CONSTRAINT_TOLERANCE * scale:
            raise NumericalError("The Denton solution violates the aggregation constraint.")


class UniformDistributor:
    """Spread each low-frequency value evenly over its sub-periods.

    Only defined for ``sum`` and ``average`` conversions, and only for
    high-frequency periods covered by an observed low-frequency value: a
    uniform split has no basis for extrapolation.
    """

    def disaggregate(self, data: DisaggregationData) -> MethodOutput:
        """Return ``C' (C C')^{-1} y_l``."""
        if not np.all(data.target_rows | data.artificial_low):
            raise InputValidationError(
                "The uniform method does not support additional constraints; use "
                "'denton-cholette' with criterion='additive' and h=0 instead."
            )
        if data.conversion not in (Conversion.SUM, Conversion.AVERAGE):
            raise InputValidationError(
                "The uniform method is only defined for 'sum' and 'average' conversions."
            )
        matrix = data.aggregation_matrix
        if np.any(~matrix.any(axis=0)):
            raise InputValidationError(
                "The uniform method cannot extrapolate: every high-frequency "
                "period must belong to an observed low-frequency period."
            )
        gram = matrix @ matrix.T
        values = matrix.T @ linalg.solve(gram, data.y_low, assume_a="pos")
        return MethodOutput(
            values=values, fitted_low=data.y_low.copy(), diagnostics=DiagnosticLog()
        )
