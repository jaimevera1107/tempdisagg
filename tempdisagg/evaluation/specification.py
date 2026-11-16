"""Specification test for the ADL(1,1) model of Proietti (2006).

Common-factor restriction
-------------------------
The static Chow-Lin model ``y_t = c + x_t' b + u_t`` with AR(1) residuals
``u_t = rho u_{t-1} + e_t`` can be rewritten as

    y_t = rho y_{t-1} + (1 - rho) c + x_t' b - rho x_{t-1}' b + e_t,

an ADL(1,1) model whose lag coefficients satisfy ``b1 = -rho b0``. Fitting the
unrestricted ADL(1,1) (``"proietti-*"``) and testing ``g = b1 + rho b0 = 0``
therefore checks whether the dynamics of the target are fully captured by
autocorrelated residuals (Chow-Lin) or need a genuine distributed-lag
relation with the indicators (Quilis, 2018, section 6).

The statistic is a Wald test **conditional on the estimated rho**:

    W = g' (J V_b J')^{-1} g,     J = [rho I, I] on the (b0, b1) block,

reported as ``F = W / q`` against ``F(q, n_l - k)``, with ``q`` the number of
lagged indicators. It ignores the sampling uncertainty of rho, so its size can
depart from the nominal level, especially with few low-frequency observations
(the Monte Carlo test in ``tests/test_specification.py`` documents it). Read
it as a diagnostic, not as a formal decision rule.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from tempdisagg.covariance.structures import ADLStructure
from tempdisagg.exceptions import InputValidationError
from tempdisagg.results import DisaggregationResult


@dataclass(frozen=True)
class CommonFactorTest:
    """Outcome of :func:`common_factor_test`.

    Attributes
    ----------
    restriction : pd.Series
        ``b1 + rho b0`` per indicator (zero under Chow-Lin).
    standard_errors : pd.Series
        Standard errors of ``restriction`` (conditional on rho).
    f_statistic, p_value : float
        Joint test of ``restriction = 0`` against ``F(q, df)``.
    df_numerator, df_denominator : int
    rho : float
        Autoregressive parameter used.
    """

    restriction: pd.Series
    standard_errors: pd.Series
    f_statistic: float
    p_value: float
    df_numerator: int
    df_denominator: int
    rho: float

    def summary(self) -> str:
        """Short text report."""
        table = pd.DataFrame({"b1 + rho b0": self.restriction,
                              "Std. Error": self.standard_errors})
        return "\n".join([
            "Common-factor test (H0: Chow-Lin restriction b1 = -rho b0), conditional on rho",
            table.to_string(float_format=lambda v: f"{v:.6g}"),
            f"F({self.df_numerator}, {self.df_denominator}) = {self.f_statistic:.4f}, "
            f"p-value = {self.p_value:.4g}   (rho = {self.rho:.4f})",
        ])


def common_factor_test(result: DisaggregationResult) -> CommonFactorTest:
    """Test the Chow-Lin restriction on a fitted ``proietti-*`` model.

    Raises
    ------
    InputValidationError
        If ``result`` does not come from a Proietti (ADL) model, has no lagged
        indicator, or rho is not identified.
    """
    if result.regression is None or result.rho is None or not result.method.name.startswith(
            "proietti"):
        raise InputValidationError("common_factor_test requires a fitted 'proietti-*' model.")
    rho = float(result.rho.value)
    if not np.isfinite(rho):
        raise InputValidationError("rho is not identified; the test is undefined.")
    regression = result.regression
    names = list(regression.coefficients.index)
    suffix = ADLStructure.LAG_SUFFIX
    lagged = [name for name in names if name.endswith(suffix)]
    if not lagged:
        raise InputValidationError("The model has no lagged indicator to test.")
    current = [name[: -len(suffix)] for name in lagged]

    coefficients = regression.coefficients
    covariance = regression.covariance
    jacobian = pd.DataFrame(0.0, index=lagged, columns=names)
    for lag_name, name in zip(lagged, current, strict=True):
        jacobian.loc[lag_name, lag_name] = 1.0
        jacobian.loc[lag_name, name] = rho
    restriction = jacobian.to_numpy() @ coefficients.to_numpy()
    variance = jacobian.to_numpy() @ covariance.to_numpy() @ jacobian.to_numpy().T
    q = len(lagged)
    wald = float(restriction @ np.linalg.solve(variance, restriction))
    df = regression.df_resid
    f_statistic = wald / q
    index = pd.Index(current, name="indicator")
    return CommonFactorTest(
        restriction=pd.Series(restriction, index=index, name="b1 + rho b0"),
        standard_errors=pd.Series(np.sqrt(np.diag(variance)), index=index, name="std_error"),
        f_statistic=f_statistic,
        p_value=float(stats.f.sf(f_statistic, q, df)),
        df_numerator=q,
        df_denominator=df,
        rho=rho,
    )
