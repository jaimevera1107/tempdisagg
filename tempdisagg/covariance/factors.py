"""Structured square-root factors of residual covariance matrices.

Every residual model in this library has a covariance of the form

    Sigma = T T',      T = F_1 F_2 ... F_m S,

where each ``F_k = (I - phi_k L)^{-1}`` is the inverse of a first-order lag
polynomial with zero initial condition (``phi = 1`` is the cumulative sum,
i.e. the inverse of the first difference) and
``S = diag(s_0, s, s, ..., s)`` rescales the first innovation (needed for the
stationary AR(1) start-up).

Working with ``T`` instead of the dense ``Sigma`` changes the cost of one
likelihood evaluation from O(n^2 n_low) to O(n n_low^2): products with ``T``
and ``T'`` are recursive filters (:func:`scipy.signal.lfilter`), so monthly-to-
daily problems with thousands of periods remain fast. The dense matrix is still
available through :meth:`LowerTriangularFactor.dense` for verification; the
test-suite checks that both routes agree.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import lfilter


@dataclass(frozen=True)
class LowerTriangularFactor:
    """Factor ``T = F_1 ... F_m S`` of an ``n x n`` covariance matrix.

    Attributes
    ----------
    n : int
        Dimension.
    ar_coefficients : tuple of float
        ``phi_k`` of each inverse lag polynomial ``(I - phi_k L)^{-1}``.
        Zero coefficients are allowed (identity filter).
    first_scale : float
        ``s_0``, scale of the first innovation.
    scale : float
        ``s``, scale of the remaining innovations.
    """

    n: int
    ar_coefficients: tuple[float, ...] = ()
    first_scale: float = 1.0
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError("n must be positive.")
        if any(abs(phi) > 1.0 for phi in self.ar_coefficients):
            raise ValueError("Filter coefficients must satisfy |phi| <= 1.")
        if self.first_scale <= 0.0 or self.scale <= 0.0:
            raise ValueError("Scales must be positive.")

    # ------------------------------------------------------------------
    # Products
    # ------------------------------------------------------------------
    def apply(self, matrix: np.ndarray) -> np.ndarray:
        """Return ``T @ matrix`` (``matrix`` has ``n`` rows)."""
        result = self._scale_rows(np.asarray(matrix, dtype=float))
        for phi in self._active_coefficients():
            result = lfilter([1.0], [1.0, -phi], result, axis=0)
        return result

    def apply_transpose(self, matrix: np.ndarray) -> np.ndarray:
        """Return ``T' @ matrix``.

        ``F'`` is upper triangular, i.e. the same recursion run backwards in
        time, implemented by reversing the rows before and after filtering.
        """
        result = np.asarray(matrix, dtype=float)
        for phi in self._active_coefficients():
            result = lfilter([1.0], [1.0, -phi], result[::-1], axis=0)[::-1]
        return self._scale_rows(result)

    def dense(self) -> np.ndarray:
        """Return ``T`` as a dense matrix (O(n^2) memory; for checks only)."""
        return self.apply(np.eye(self.n))

    def covariance_diagonal(self) -> np.ndarray:
        """Return ``diag(T T')`` in O(n).

        ``T`` is lower triangular with entries ``h_{i-j} s_j`` where ``h`` is
        the impulse response of the filter chain, so
        ``diag_i = s^2 sum_{k<=i} h_k^2 + (s_0^2 - s^2) h_i^2``.
        """
        impulse = np.zeros(self.n)
        impulse[0] = 1.0
        for phi in self._active_coefficients():
            impulse = lfilter([1.0], [1.0, -phi], impulse)
        squared = impulse ** 2
        return self.scale ** 2 * np.cumsum(squared) + (
            self.first_scale ** 2 - self.scale ** 2
        ) * squared

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _active_coefficients(self) -> tuple[float, ...]:
        return tuple(phi for phi in self.ar_coefficients if phi != 0.0)

    def _scale_rows(self, matrix: np.ndarray) -> np.ndarray:
        if self.first_scale == 1.0 and self.scale == 1.0:
            return matrix
        scaled = matrix * self.scale
        scaled[0] = matrix[0] * self.first_scale
        return scaled
