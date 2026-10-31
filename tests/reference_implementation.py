"""Literal, dense transcription of the textbook formulas.

This module is deliberately naive: it builds every matrix explicitly and uses
``numpy.linalg.inv``, following Sax and Steiner (2013, Table 2) and
Chow and Lin (1971) line by line. It is slow and numerically fragile, but easy
to audit. The test-suite checks that the optimised library (structured
factors, Cholesky/QR, sparse KKT) reproduces it.
"""

from __future__ import annotations

import numpy as np


def lag(n: int) -> np.ndarray:
    """Lag operator L with zero initial condition."""
    return np.eye(n, k=-1)


def sigma_chow_lin(rho: float, n: int, form: str = "correlation") -> np.ndarray:
    """Sigma_CL = rho^|i-j| (/(1 - rho^2) for the covariance form)."""
    matrix = rho ** np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    return matrix / (1 - rho ** 2) if form == "covariance" else matrix


def sigma_litterman(rho: float, n: int) -> np.ndarray:
    """Sigma_L = (Delta' H' H Delta)^{-1}; Fernandez is rho = 0."""
    delta = np.eye(n) - lag(n)
    h_matrix = np.eye(n) - rho * lag(n)
    return np.linalg.inv(delta.T @ h_matrix.T @ h_matrix @ delta)


def gls_disaggregation(y_l: np.ndarray, x: np.ndarray, c: np.ndarray, sigma: np.ndarray) -> dict:
    """Chow-Lin BLUE with explicit inverses (Sax and Steiner, eqs. 3-6)."""
    v = c @ sigma @ c.T
    v_inv = np.linalg.inv(v)
    x_l = c @ x
    beta = np.linalg.inv(x_l.T @ v_inv @ x_l) @ x_l.T @ v_inv @ y_l
    p = x @ beta
    distribution = sigma @ c.T @ v_inv
    u_l = y_l - c @ p
    y_hat = p + distribution @ u_l
    rss = float(u_l @ v_inv @ u_l)
    n_l, k = x_l.shape
    s2_ml = rss / n_l
    s2 = rss / (n_l - k)
    logl = -n_l / 2 * (np.log(2 * np.pi) + 1 + np.log(s2_ml)) - 0.5 * np.log(np.linalg.det(v))
    vcov_beta = s2 * np.linalg.inv(x_l.T @ v_inv @ x_l)
    # Goldberger BLUP prediction-error covariance.
    q = x - distribution @ x_l
    prediction_cov = s2 * (sigma - distribution @ c @ sigma) + q @ vcov_beta @ q.T
    return {
        "y_hat": y_hat, "beta": beta, "rss": rss, "logl": logl,
        "vcov_beta": vcov_beta, "prediction_var": np.diag(prediction_cov),
        "prediction_cov": prediction_cov,
    }


def denton(y_l: np.ndarray, x: np.ndarray, c: np.ndarray, h: int,
           proportional: bool, cholette: bool) -> np.ndarray:
    """Denton via the dense KKT system (original Denton: closed form)."""
    n = x.size
    delta = np.eye(n) - lag(n)
    d = np.linalg.matrix_power(delta, h)
    if cholette:
        d = d[h:]
    if proportional:
        d = d @ np.diag(1 / x)
    a = d.T @ d
    if not cholette:
        a_inv = np.linalg.inv(a)
        return x + a_inv @ c.T @ np.linalg.inv(c @ a_inv @ c.T) @ (y_l - c @ x)
    n_l = c.shape[0]
    kkt = np.block([[a, c.T], [c, np.zeros((n_l, n_l))]])
    return np.linalg.solve(kkt, np.concatenate([a @ x, y_l]))[:n]
