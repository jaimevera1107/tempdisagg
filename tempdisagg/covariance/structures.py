"""Residual models of the regression-based (GLS) methods.

Each structure maps the autoregressive parameter ``rho`` to a square-root
factor ``T`` of the high-frequency residual covariance ``Sigma = T T'``
(up to the scale ``sigma^2``), see :mod:`tempdisagg.covariance.factors`.

=========================  ======================================  =========================
Structure                  High-frequency residual process         ``T``
=========================  ======================================  =========================
WhiteNoiseCovariance       u_t = e_t                               I
AR1Covariance              u_t = rho u_{t-1} + e_t, stationary     (I - rho L)^{-1} S
RandomWalkAR1Covariance    (1 - L)(1 - rho L) u_t = e_t            (I-L)^{-1} (I-rho L)^{-1}
DynamicStructure           y_t = rho y_{t-1} + x_t b + e_t         (I - rho L)^{-1}
=========================  ======================================  =========================

For the stationary AR(1), ``S = diag(1 / sqrt(1 - rho^2), 1, ..., 1)`` gives
the covariance ``rho^|i-j| / (1 - rho^2)`` (Quilis form). Multiplying ``T``
by ``sqrt(1 - rho^2)`` gives the correlation ``rho^|i-j|`` (Ecotrim form).
The two differ only by a rho-dependent scalar, which cancels in the
concentrated likelihood but not in the residual sum of squares; that scalar
is exactly what separates the ``minrss-ecotrim`` and ``minrss-quilis``
estimators of the R package.

References
----------
Chow and Lin (1971); Fernandez (1981); Litterman (1983); Santos Silva and
Cardoso (2001); Sax and Steiner (2013), Table 2.
"""

from __future__ import annotations

from enum import Enum
from typing import Protocol, runtime_checkable

import numpy as np
from scipy.signal import lfilter

from tempdisagg.covariance.factors import LowerTriangularFactor


class CovarianceForm(str, Enum):
    """Scaling of the stationary AR(1) matrix."""

    CORRELATION = "correlation"   # rho^|i-j|              (Ecotrim)
    COVARIANCE = "covariance"     # rho^|i-j| / (1-rho^2)  (Quilis)


def _check_rho(rho: float, label: str) -> None:
    if not -1.0 < rho < 1.0:
        raise ValueError(f"{label} requires |rho| < 1; got {rho}.")


@runtime_checkable
class ResidualStructure(Protocol):
    """Interface shared by every residual structure."""

    #: Whether the structure depends on ``rho``.
    uses_rho: bool
    #: Whether :meth:`transform_regressors` modifies the regressors.
    transforms_regressors: bool

    def factor(self, rho: float, n: int) -> LowerTriangularFactor:
        """Return ``T`` such that ``Sigma = T T'``."""

    def transform_regressors(self, regressors: np.ndarray, rho: float) -> np.ndarray:
        """Return the regressors actually entering the model."""


class WhiteNoiseCovariance:
    """Independent residuals (OLS): ``Sigma = I``.

    The distribution step then spreads each low-frequency residual uniformly
    within its period, so OLS also satisfies the aggregation constraint.
    """

    uses_rho: bool = False
    transforms_regressors: bool = False

    def factor(self, rho: float, n: int) -> LowerTriangularFactor:
        """Return the identity factor."""
        return LowerTriangularFactor(n)

    def transform_regressors(self, regressors: np.ndarray, rho: float) -> np.ndarray:
        """Return the regressors unchanged."""
        return regressors


class AR1Covariance:
    """Stationary AR(1) residuals (Chow and Lin, 1971).

    Parameters
    ----------
    form : CovarianceForm, default CORRELATION
        Matrix scaling; matters only for the minimum-RSS criterion.
    """

    uses_rho: bool = True
    transforms_regressors: bool = False

    def __init__(self, form: CovarianceForm | str = CovarianceForm.CORRELATION) -> None:
        self.form: CovarianceForm = CovarianceForm(form)

    def factor(self, rho: float, n: int) -> LowerTriangularFactor:
        """Return ``(I - rho L)^{-1} S`` with the scaling of :attr:`form`."""
        _check_rho(rho, "AR(1)")
        innovation_scale = np.sqrt(1.0 - rho ** 2)
        if self.form is CovarianceForm.COVARIANCE:
            return LowerTriangularFactor(n, (rho,), 1.0 / innovation_scale, 1.0)
        return LowerTriangularFactor(n, (rho,), 1.0, innovation_scale)

    def transform_regressors(self, regressors: np.ndarray, rho: float) -> np.ndarray:
        """Return the regressors unchanged."""
        return regressors

    @staticmethod
    def dense_covariance(rho: float, n: int, form: CovarianceForm) -> np.ndarray:
        """Closed-form matrix, used to verify the factor in the tests."""
        lags = np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
        matrix = np.power(float(rho), lags)
        if CovarianceForm(form) is CovarianceForm.COVARIANCE:
            matrix = matrix / (1.0 - rho ** 2)
        return matrix


class RandomWalkAR1Covariance:
    """Random walk with AR(1) innovations (Litterman, 1983).

    ``(1 - L)(1 - rho L) u_t = e_t`` with zero initial conditions, so
    ``Sigma = (Delta' H' H Delta)^{-1}``, ``H = I - rho L``,
    ``Delta = I - L``. Fernandez (1981) is ``rho = 0``.
    """

    uses_rho: bool = True
    transforms_regressors: bool = False

    def factor(self, rho: float, n: int) -> LowerTriangularFactor:
        """Return ``(I - L)^{-1} (I - rho L)^{-1}``."""
        _check_rho(rho, "Litterman")
        return LowerTriangularFactor(n, (1.0, rho))

    def transform_regressors(self, regressors: np.ndarray, rho: float) -> np.ndarray:
        """Return the regressors unchanged."""
        return regressors


class DynamicStructure:
    """Dynamic Chow-Lin model (Santos Silva and Cardoso, 2001). Experimental.

    The model ``y_t = rho y_{t-1} + x_t' b + e_t`` (white-noise ``e``) solves to
    ``y = H^{-1} X b + H^{-1} e_1 (rho y_0) + H^{-1} e`` with ``H = I - rho L``.
    Regressors are filtered by ``H^{-1}`` and, if ``rho != 0`` and
    ``include_truncation_remainder``, augmented with ``rho^t`` (the
    "truncation remainder"), whose coefficient is the unknown initial value
    ``y_0``. The residual covariance is ``H^{-1} H^{-T}``.

    Parameters
    ----------
    include_truncation_remainder : bool, default True
    """

    uses_rho: bool = True
    transforms_regressors: bool = True

    def __init__(self, include_truncation_remainder: bool = True) -> None:
        self.include_truncation_remainder: bool = include_truncation_remainder

    def factor(self, rho: float, n: int) -> LowerTriangularFactor:
        """Return ``(I - rho L)^{-1}``."""
        _check_rho(rho, "The dynamic model")
        return LowerTriangularFactor(n, (rho,))

    def transform_regressors(self, regressors: np.ndarray, rho: float) -> np.ndarray:
        """Filter regressors and append the truncation remainder."""
        n = regressors.shape[0]
        filtered = lfilter([1.0], [1.0, -rho], regressors, axis=0) if rho != 0.0 else regressors
        if rho == 0.0 or not self.include_truncation_remainder:
            return filtered
        remainder = np.power(float(rho), np.arange(1, n + 1)).reshape(-1, 1)
        return np.hstack([filtered, remainder])


class ADLStructure:
    """ADL(1,1) model of Proietti (2006). Experimental.

    The high-frequency model is the autoregressive distributed lag

        y_t = rho y_{t-1} + x_t' b0 + x_{t-1}' b1 + e_t,   e_t white noise,

    which contains the dynamic Chow-Lin model (``b1 = 0``) and, through the
    common-factor restriction ``b1 = -rho b0``, the static Chow-Lin model.
    Written as ``H y = Z g + e`` with ``H = I - rho L`` and
    ``Z = [X, L X]`` (the constant is not lagged: its lag is collinear with
    it), the solution ``y = H^{-1} Z g + H^{-1} e`` is a GLS regression on
    filtered regressors with residual covariance ``H^{-1} H^{-T}``, exactly as
    in :class:`DynamicStructure`.

    Initial conditions. ``y_0`` enters through the truncation remainder
    ``rho^t`` (as in the dynamic model and in R). The lag ``x_0`` of the
    first period is not observed; it is backcast as ``x_1`` (zero first
    difference). Quilis (2018) backcasts it with an AR model instead; the
    difference only affects the first period, ``b1 (x_0 - x_1)``, and decays
    as ``rho^t``.

    Parameters
    ----------
    include_truncation_remainder : bool, default True
    """

    uses_rho: bool = True
    transforms_regressors: bool = True

    #: Suffix appended to the name of a lagged indicator.
    LAG_SUFFIX: str = "(-1)"

    def __init__(self, include_truncation_remainder: bool = True) -> None:
        self.include_truncation_remainder: bool = include_truncation_remainder

    def factor(self, rho: float, n: int) -> LowerTriangularFactor:
        """Return ``(I - rho L)^{-1}``."""
        _check_rho(rho, "The ADL model")
        return LowerTriangularFactor(n, (rho,))

    @staticmethod
    def lagged_columns(regressors: np.ndarray) -> np.ndarray:
        """Positions of the columns that are lagged (every non-constant one)."""
        return np.flatnonzero(np.ptp(regressors, axis=0) > 0.0)

    def transform_regressors(self, regressors: np.ndarray, rho: float) -> np.ndarray:
        """Append lagged indicators, filter by ``H^{-1}``, add the remainder."""
        columns = self.lagged_columns(regressors)
        lagged = np.empty((regressors.shape[0], columns.size))
        if columns.size:
            lagged[1:] = regressors[:-1, columns]
            lagged[0] = regressors[0, columns]  # backcast x_0 = x_1
        stacked = np.hstack([regressors, lagged])
        n = stacked.shape[0]
        filtered = lfilter([1.0], [1.0, -rho], stacked, axis=0) if rho != 0.0 else stacked
        if rho == 0.0 or not self.include_truncation_remainder:
            return filtered
        remainder = np.power(float(rho), np.arange(1, n + 1)).reshape(-1, 1)
        return np.hstack([filtered, remainder])

    def transformed_names(
        self, names: tuple[str, ...], regressors: np.ndarray, rho: float, n_columns: int
    ) -> tuple[str, ...]:
        """Names of the columns returned by :meth:`transform_regressors`."""
        lagged = tuple(f"{names[j]}{self.LAG_SUFFIX}" for j in self.lagged_columns(regressors))
        result = (*names, *lagged)
        if n_columns > len(result):
            result = (*result, "(truncation remainder)")
        return result
