"""Exact non-negative reconciliation of a disaggregated series.

Problem
-------
Given an estimate ``y_hat``, find the closest non-negative series that still
satisfies every constraint:

    minimise ||y - y_hat||^2   subject to   C y = b,  y >= 0.

Algorithms
----------
*Separable case* (each period in at most one constraint row, the usual
temporal disaggregation): the problem splits by row. For weights ``c_i > 0``
the solution is ``y_i = max(0, y_hat_i + lambda c_i)``, where ``lambda`` is
bracketed by bisection and then computed in closed form on the active set.

*General case* (overlapping rows, e.g. quarters inside years or known values):
Dykstra's alternating projections between the affine set ``{C y = b}`` and
the orthant converge to the exact projection onto their intersection. The
iterate is then *polished*: with the zero set ``A`` fixed, the equality-
constrained least-squares problem on the free set ``F`` is solved in closed
form and the KKT conditions are verified,

    y_F = y_hat_F + C_F' lambda >= 0,
    mu_A = -(y_hat_A + C_A' lambda) >= 0,

which certifies optimality. If the sets do not intersect (the constraints
admit no non-negative solution), :class:`InfeasibleConstraintError` is raised.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import linalg

from tempdisagg.exceptions import InfeasibleConstraintError, InputValidationError, NumericalError
from tempdisagg.results import DisaggregationResult

#: Iterations used to bracket the active set (separable case).
BISECTION_ITERATIONS: int = 200
#: Maximum Dykstra iterations (general case).
MAX_DYKSTRA_ITERATIONS: int = 20000
#: Relative tolerance for feasibility and KKT checks.
TOLERANCE: float = 1e-9


class NonNegativeReconciler:
    """Project a disaggregated series onto ``{C y = b, y >= 0}``."""

    def reconcile(self, result: DisaggregationResult) -> pd.Series:
        """Return a non-negative version of ``result.values``.

        All constraints of the fit (targets, padding targets and additional
        constraints) are preserved.

        Raises
        ------
        InfeasibleConstraintError
            If no non-negative series satisfies the constraints.
        """
        values = self.project(result.full_values.to_numpy(), result.data.aggregation_matrix,
                              result.data.y_low)
        full = pd.Series(values, index=result.full_values.index, name="estimate_non_negative")
        return full[~result.data.artificial_high]

    def project(self, y_hat: np.ndarray, aggregation_matrix: np.ndarray,
                y_low: np.ndarray) -> np.ndarray:
        """Array version of :meth:`reconcile`."""
        y_hat = np.asarray(y_hat, dtype=float)
        matrix = np.asarray(aggregation_matrix, dtype=float)
        targets = np.asarray(y_low, dtype=float)
        if matrix.shape != (targets.size, y_hat.size):
            raise InputValidationError("Shapes of y_hat, matrix and targets do not match.")
        if np.all(y_hat >= 0):
            return y_hat.copy()
        separable = np.all((matrix != 0).sum(axis=0) <= 1) and np.all(matrix >= 0)
        if separable:
            return self._project_separable(y_hat, matrix, targets)
        return self._project_general(y_hat, matrix, targets)

    # ------------------------------------------------------------------
    # Separable case
    # ------------------------------------------------------------------
    def _project_separable(self, y_hat: np.ndarray, matrix: np.ndarray,
                           targets: np.ndarray) -> np.ndarray:
        result = np.maximum(y_hat, 0.0)  # periods outside every row are just clipped
        for row, target in enumerate(targets):
            members = np.flatnonzero(matrix[row] != 0)
            result[members] = self._project_group(y_hat[members], matrix[row, members],
                                                  float(target))
        return result

    @staticmethod
    def _project_group(values: np.ndarray, weights: np.ndarray, target: float) -> np.ndarray:
        if target < 0:
            raise InfeasibleConstraintError(
                f"Low-frequency value {target} is negative; a non-negative series "
                "cannot reproduce it."
            )

        def weighted_sum(lam: float) -> float:
            return float(weights @ np.maximum(values + lam * weights, 0.0))

        lower = float(np.min(-values / weights)) - 1.0
        upper = 1.0
        while weighted_sum(upper) < target:
            upper *= 2.0
        for _ in range(BISECTION_ITERATIONS):
            middle = 0.5 * (lower + upper)
            if weighted_sum(middle) < target:
                lower = middle
            else:
                upper = middle
        lam = upper
        active = values + lam * weights > 0
        if active.any() and target > 0:
            lam = (target - weights[active] @ values[active]) / (weights[active] @ weights[active])
        return np.maximum(values + lam * weights, 0.0)

    # ------------------------------------------------------------------
    # General case
    # ------------------------------------------------------------------
    def _project_general(self, y_hat: np.ndarray, matrix: np.ndarray,
                         targets: np.ndarray) -> np.ndarray:
        try:
            gram = linalg.cho_factor(matrix @ matrix.T)
        except linalg.LinAlgError as exc:
            raise NumericalError("The constraint rows are linearly dependent.") from exc

        def to_affine(point: np.ndarray) -> np.ndarray:
            return point - matrix.T @ linalg.cho_solve(gram, matrix @ point - targets)

        scale = max(1.0, float(np.max(np.abs(y_hat))), float(np.max(np.abs(targets))))
        x = y_hat.copy()
        p = np.zeros_like(x)
        q = np.zeros_like(x)
        for iteration in range(1, MAX_DYKSTRA_ITERATIONS + 1):
            z = to_affine(x + p)
            p = x + p - z
            x = np.maximum(z + q, 0.0)
            q = z + q - x
            if iteration % 25 == 0:
                polished = self._polish(y_hat, matrix, targets, x, scale)
                if polished is not None:
                    return polished
                if np.max(np.abs(x - z)) <= TOLERANCE * scale and iteration > 1000:
                    break
        gap = float(np.max(np.abs(matrix @ np.maximum(x, 0.0) - targets)))
        polished = self._polish(y_hat, matrix, targets, x, scale)
        if polished is not None:
            return polished
        raise InfeasibleConstraintError(
            "No non-negative series satisfies the constraints "
            f"(remaining constraint gap {gap:.3g})."
        )

    @staticmethod
    def _polish(y_hat: np.ndarray, matrix: np.ndarray, targets: np.ndarray,
                iterate: np.ndarray, scale: float) -> np.ndarray | None:
        """Exact solution for the active set suggested by ``iterate`` (KKT-verified)."""
        free = iterate > TOLERANCE * scale
        if not free.any():
            return None
        sub = matrix[:, free]
        try:
            with warnings.catch_warnings():
                # Ill-conditioned candidates are rejected by the KKT check below.
                warnings.simplefilter("ignore", linalg.LinAlgWarning)
                lam = linalg.solve(sub @ sub.T, targets - sub @ y_hat[free], assume_a="pos")
        except (linalg.LinAlgError, ValueError):
            return None
        candidate = np.zeros_like(y_hat)
        candidate[free] = y_hat[free] + sub.T @ lam
        multipliers = -(y_hat[~free] + matrix[:, ~free].T @ lam)
        tolerance = 1e-7 * scale
        feasible = (np.all(candidate[free] >= -tolerance)
                    and np.max(np.abs(matrix @ candidate - targets)) <= tolerance)
        if feasible and np.all(multipliers >= -tolerance):
            return np.maximum(candidate, 0.0)
        return None
