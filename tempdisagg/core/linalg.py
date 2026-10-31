"""Small, numerically careful linear-algebra helpers.

No pseudo-inverses are used anywhere in the library: every matrix that must be
inverted is either triangular, positive definite (Cholesky) or a square
system that is solved directly. Failures raise :class:`NumericalError`
instead of silently returning a least-norm answer.
"""

from __future__ import annotations

import numpy as np
from scipy import linalg

from tempdisagg.exceptions import NumericalError


def inverse_lag_polynomial(n: int, rho: float) -> np.ndarray:
    """Return the inverse of ``(I - rho * L)`` for an ``n``-period sample.

    ``L`` is the lag operator with a zero initial condition. The inverse is
    lower triangular with entries ``rho ** (i - j)`` for ``i >= j``. With
    ``rho = 1`` it is the cumulative-sum matrix, i.e. the inverse of the
    first-difference matrix.
    """
    if n < 1:
        raise ValueError("n must be positive.")
    rows = np.arange(n)[:, None]
    cols = np.arange(n)[None, :]
    exponents = rows - cols
    lower = exponents >= 0
    return np.where(lower, np.power(float(rho), np.where(lower, exponents, 0)), 0.0)


def difference_matrix(n: int, order: int = 1, drop_initial: bool = False) -> np.ndarray:
    """Return the ``order``-th difference operator for ``n`` periods.

    Parameters
    ----------
    n : int
        Number of periods.
    order : int, default 1
        Differencing order ``h``; ``0`` returns the identity.
    drop_initial : bool, default False
        If True, drop the first ``order`` rows, which encode the initial
        condition (Denton-Cholette). If False, keep them (original Denton).
    """
    if order < 0:
        raise ValueError("order must be non-negative.")
    if drop_initial and order >= n:
        raise ValueError("order must be smaller than the number of periods.")
    first_difference = np.eye(n) - np.eye(n, k=-1)
    matrix = np.linalg.matrix_power(first_difference, order)
    return matrix[order:] if drop_initial else matrix


def cholesky_lower(matrix: np.ndarray, label: str) -> np.ndarray:
    """Return the lower Cholesky factor or raise :class:`NumericalError`."""
    try:
        return linalg.cholesky(matrix, lower=True, check_finite=True)
    except (linalg.LinAlgError, ValueError) as exc:
        raise NumericalError(f"{label} is not positive definite.") from exc


def solve_lower(factor: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve ``factor @ x = rhs`` for a lower-triangular ``factor``."""
    return linalg.solve_triangular(factor, rhs, lower=True, check_finite=False)


def log_determinant_from_cholesky(factor: np.ndarray) -> float:
    """Return ``log|A|`` given the Cholesky factor of ``A``."""
    return float(2.0 * np.sum(np.log(np.diag(factor))))
