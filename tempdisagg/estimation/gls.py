"""Generalised least squares on the low-frequency equation.

Model
-----
High frequency:  ``y = X b + u``,  ``Var(u) = s^2 Sigma``  (``n`` periods).
Low frequency:   ``y_l = C y = X_l b + u_l``, ``X_l = C X``,
``Var(u_l) = s^2 V``, ``V = C Sigma C'``  (``n_l`` observations).

Estimation
----------
With the Cholesky factorisation ``V = L L'`` the equation is whitened,
``L^{-1} y_l = L^{-1} X_l b + e`` with ``Var(e) = s^2 I``, and solved by a QR
decomposition (no explicit inverse). Statistics:

* ``rss = e' e`` (weighted residual sum of squares, ``u_l' V^{-1} u_l``)
* ``tss`` weighted total sum of squares, always centred on the aggregated
  constant ``C 1`` (projection on ``L^{-1} C 1``), with or without an
  intercept in the model (convention of the R package)
* ``s2_ml = rss / n_l``           (ML estimator; ``s_2`` in R)
* ``s2_gls = rss / (n_l - k)``    (unbiased; ``s_2_gls`` in R)
* ``Var(b) = s2_gls (X_l' V^{-1} X_l)^{-1}``
* concentrated Gaussian log-likelihood

      logL = -n_l/2 (log(2 pi) + 1 + log s2_ml) - 1/2 log|V|,

  which is invariant to rescaling ``y_l`` or ``V`` and is the objective of the
  ``maxlog`` estimators of ``rho``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tempdisagg.core.linalg import (
    cholesky_lower,
    log_determinant_from_cholesky,
    solve_lower,
)
from tempdisagg.exceptions import SingularDesignError

#: Relative threshold on the R diagonal (of the column-equilibrated design)
#: below which the design is singular.
RANK_TOLERANCE: float = 1e-10
#: rss below this fraction of the whitened sum of squares is an exact fit.
EXACT_FIT_TOLERANCE: float = 1e-24


@dataclass(frozen=True)
class GLSFit:
    """Result of one GLS regression on low-frequency data.

    Attributes
    ----------
    coefficients : np.ndarray
        ``b``, shape ``(k,)``.
    residuals : np.ndarray
        ``u_l = y_l - X_l b``, shape ``(n_l,)``.
    whitened_residuals : np.ndarray
        ``e = L^{-1} u_l``; i.i.d. ``N(0, s^2)`` under the model.
    rss, tss : float
        Weighted residual / total sums of squares (TSS centred on the
        whitened intercept when one is present).
    sigma2_ml, sigma2_gls : float
        ``rss / n_l`` and ``rss / (n_l - k)``.
    log_likelihood : float
        Concentrated log-likelihood (``-inf``-safe: ``nan`` on exact fit).
    coefficient_covariance : np.ndarray
        ``sigma2_gls (X_l' V^{-1} X_l)^{-1}``.
    cholesky_factor : np.ndarray
        ``L`` with ``V = L L'``.
    r_inverse : np.ndarray
        ``R^{-1}`` from the QR of the whitened design (reused for prediction
        variances).
    condition_number_v : float
        2-norm condition number of ``V``.
    condition_number_design : float
        2-norm condition number of the whitened design after scaling each
        column to unit length. Scaling makes it a pure collinearity measure,
        independent of units (Belsley, Kuh and Welsch, 1980).
    exact_fit : bool
        Whether the residuals vanish (``rss`` negligible).
    """

    coefficients: np.ndarray
    residuals: np.ndarray
    whitened_residuals: np.ndarray
    rss: float
    tss: float
    sigma2_ml: float
    sigma2_gls: float
    log_likelihood: float
    coefficient_covariance: np.ndarray
    cholesky_factor: np.ndarray
    r_inverse: np.ndarray
    condition_number_v: float
    condition_number_design: float
    exact_fit: bool

    @property
    def n_obs(self) -> int:
        """Number of low-frequency observations."""
        return int(self.residuals.size)

    @property
    def n_coefficients(self) -> int:
        """Number of estimated coefficients."""
        return int(self.coefficients.size)


def fit_gls(
    y_low: np.ndarray,
    regressors_low: np.ndarray,
    covariance_low: np.ndarray,
    centering: np.ndarray | None = None,
) -> GLSFit:
    """Estimate ``y_low = regressors_low b + u`` with ``Var(u) = s^2 V`` by GLS.

    Parameters
    ----------
    y_low : np.ndarray
        Shape ``(n_l,)``.
    regressors_low : np.ndarray
        ``X_l``, shape ``(n_l, k)`` with ``1 <= k < n_l``.
    covariance_low : np.ndarray
        ``V``, shape ``(n_l, n_l)``, symmetric positive definite.
    centering : np.ndarray, optional
        Low-frequency vector used to centre the TSS, normally the aggregated
        constant ``C 1``. Without it the TSS is uncentred.

    Raises
    ------
    SingularDesignError
        If ``k == 0``, ``n_l <= k`` or the whitened design is rank deficient.
    NumericalError
        If ``V`` is not positive definite.
    """
    n_obs, n_coef = regressors_low.shape
    if n_coef == 0:
        raise SingularDesignError("At least one regressor is required.")
    if n_obs <= n_coef:
        raise SingularDesignError(
            f"{n_obs} low-frequency observations cannot identify {n_coef} "
            "coefficients (at least one residual degree of freedom is needed)."
        )

    factor = cholesky_lower(covariance_low, "The low-frequency covariance V")
    y_white = solve_lower(factor, y_low)
    x_white = solve_lower(factor, regressors_low)

    # Equilibrate columns so that rank and conditioning do not depend on units
    # (e.g. an intercept next to an indicator measured in billions).
    column_norms = np.linalg.norm(x_white, axis=0)
    if np.any(column_norms == 0.0) or not np.all(np.isfinite(column_norms)):
        raise SingularDesignError(
            "A regressor is identically zero after aggregation (check for all-zero "
            "indicators)."
        )
    q_matrix, r_scaled = np.linalg.qr(x_white / column_norms)
    r_diag = np.abs(np.diag(r_scaled))
    if r_diag.min() <= RANK_TOLERANCE * r_diag.max():
        raise SingularDesignError(
            "The aggregated regressors are collinear: check for duplicated "
            "indicators, or an intercept combined with an indicator that is "
            "constant within the sample."
        )
    scaled_coefficients = np.linalg.solve(r_scaled, q_matrix.T @ y_white)
    coefficients = scaled_coefficients / column_norms
    # R of the unscaled design is R_scaled diag(norms); its inverse:
    r_inverse = np.linalg.solve(r_scaled, np.eye(n_coef)) / column_norms[:, np.newaxis]
    singular_values = np.linalg.svd(r_scaled, compute_uv=False)

    whitened_residuals = y_white - x_white @ coefficients
    rss = float(whitened_residuals @ whitened_residuals)
    scale = float(y_white @ y_white)
    exact_fit = rss <= EXACT_FIT_TOLERANCE * max(scale, np.finfo(float).tiny) or scale == 0.0

    sigma2_ml = rss / n_obs
    sigma2_gls = rss / (n_obs - n_coef)
    if exact_fit:
        log_likelihood = float("nan")
    else:
        log_likelihood = -0.5 * (
            n_obs * (np.log(2.0 * np.pi) + 1.0 + np.log(sigma2_ml))
            + log_determinant_from_cholesky(factor)
        )

    return GLSFit(
        coefficients=coefficients,
        residuals=y_low - regressors_low @ coefficients,
        whitened_residuals=whitened_residuals,
        rss=rss,
        tss=_total_sum_of_squares(y_white, factor, centering),
        sigma2_ml=sigma2_ml,
        sigma2_gls=sigma2_gls,
        log_likelihood=float(log_likelihood),
        coefficient_covariance=sigma2_gls * (r_inverse @ r_inverse.T),
        cholesky_factor=factor,
        r_inverse=r_inverse,
        condition_number_v=float(np.linalg.cond(covariance_low)),
        condition_number_design=float(singular_values[0] / singular_values[-1]),
        exact_fit=bool(exact_fit),
    )


def _total_sum_of_squares(
    y_white: np.ndarray, factor: np.ndarray, centering: np.ndarray | None
) -> float:
    """Weighted TSS centred on the whitened ``centering`` vector (if any)."""
    if centering is None:
        return float(y_white @ y_white)
    constant = solve_lower(factor, centering)
    centred = y_white - constant * (constant @ y_white) / (constant @ constant)
    return float(centred @ centred)
