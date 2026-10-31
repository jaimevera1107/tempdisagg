"""Result objects: the estimate plus every statistic behind it.

A fit returns a :class:`DisaggregationResult`. Users who only need the series
read ``result.values``; analysts and statisticians find the full estimation
record in composed, immutable objects:

====================  =====================================================
Attribute             Content
====================  =====================================================
``values``            high-frequency estimate
``method``            the exact specification used (all settings explicit)
``regression``        :class:`RegressionStatistics` (GLS methods)
``rho``               :class:`RhoEstimate` (methods with an AR parameter)
``benchmark``         :class:`BenchmarkStatistics` (Denton methods)
``prediction_se``     standard error of each high-frequency value
``diagnostics``       :class:`NumericalDiagnostics` (conditioning, warnings)
``preprocessing``     :class:`PreprocessingRecord` (artificial values)
====================  =====================================================

``values`` contains only the periods present in the input. Periods added by
completion or padding are in ``full_values``.

Formulas are documented in ``docs/statistics.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from tempdisagg.core.data import DisaggregationData
from tempdisagg.preprocessing.settings import CompletionSettings, PaddingSettings

#: Significance thresholds and markers (R convention).
SIGNIFICANCE_CODES: tuple[tuple[float, str], ...] = (
    (0.001, "***"), (0.01, "**"), (0.05, "*"), (0.1, "."),
)
#: Default confidence level for intervals.
DEFAULT_LEVEL: float = 0.95


def _check_level(level: float) -> None:
    if not 0.0 < level < 1.0:
        raise ValueError("level must lie strictly between 0 and 1.")


def _stars(p_value: float) -> str:
    for threshold, marker in SIGNIFICANCE_CODES:
        if p_value < threshold:
            return marker
    return ""


@dataclass(frozen=True)
class RegressionStatistics:
    """GLS estimation record of a regression-based method.

    Attributes
    ----------
    coefficients, standard_errors : pd.Series
        ``b`` and ``sqrt(diag(Var(b)))`` with ``Var(b) = s2_gls (X_l'V^{-1}X_l)^{-1}``.
    covariance : pd.DataFrame
        ``Var(b)``.
    sigma2_ml, sigma2_gls : float
        ``rss / n_l`` (R: ``s_2``) and ``rss / (n_l - k)`` (R: ``s_2_gls``).
    rss, tss : float
        Weighted residual and total sums of squares.
    r_squared, adj_r_squared : float
        ``1 - rss/tss`` and its degrees-of-freedom adjustment (``nan`` if
        ``tss = 0``).
    log_likelihood : float
        Concentrated Gaussian log-likelihood.
    aic, bic : float
        Information criteria as defined by the R package:
        ``log(sigma2_ml) + 2 k / n_l`` and ``log(sigma2_ml) + k log(n_l) / n_l``.
        They depend on the normalisation of ``Sigma`` and are only comparable
        between models with the same residual structure.
    aic_loglik, bic_loglik : float
        Likelihood-based criteria ``-2 logL + 2 p`` and ``-2 logL + p log(n_l)``
        with ``p = k + 1 (+1 if rho is estimated)``. They include ``log|V|`` and
        are comparable across residual structures (e.g. Chow-Lin vs Litterman).
    n_obs, n_coefficients : int
        ``n_l`` and ``k``.
    whitened_residuals : np.ndarray
        ``L^{-1} u_l``; approximately i.i.d. under a correct model.
    n_artificial_obs : int
        Padding targets included in ``n_obs``. When positive, inference treats
        artificial values as data and is not valid in the usual sense.
    """

    coefficients: pd.Series
    standard_errors: pd.Series
    covariance: pd.DataFrame
    sigma2_ml: float
    sigma2_gls: float
    rss: float
    tss: float
    r_squared: float
    adj_r_squared: float
    log_likelihood: float
    aic: float
    bic: float
    aic_loglik: float
    bic_loglik: float
    n_obs: int
    n_coefficients: int
    whitened_residuals: np.ndarray = field(repr=False)
    n_artificial_obs: int = 0

    @property
    def df_resid(self) -> int:
        """Residual degrees of freedom ``n_l - k``."""
        return self.n_obs - self.n_coefficients

    @property
    def t_values(self) -> pd.Series:
        """Coefficient t statistics."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return self.coefficients / self.standard_errors

    @property
    def p_values(self) -> pd.Series:
        """Two-sided p-values, Student t with ``df_resid`` degrees of freedom."""
        return pd.Series(
            2.0 * stats.t.sf(np.abs(self.t_values.to_numpy()), self.df_resid),
            index=self.coefficients.index, name="p_value",
        )

    @property
    def residual_autocorrelation(self) -> float:
        """Lag-1 autocorrelation of the whitened residuals (≈ 0 if well specified)."""
        residuals = self.whitened_residuals - self.whitened_residuals.mean()
        denominator = float(residuals @ residuals)
        if denominator == 0.0 or residuals.size < 3:
            return float("nan")
        return float(residuals[1:] @ residuals[:-1] / denominator)

    def confidence_intervals(self, level: float = DEFAULT_LEVEL) -> pd.DataFrame:
        """Student-t confidence intervals for the coefficients."""
        _check_level(level)
        quantile = stats.t.ppf(0.5 + level / 2.0, self.df_resid)
        half = quantile * self.standard_errors
        return pd.DataFrame({
            f"{(1 - level) / 2:.1%}": self.coefficients - half,
            f"{0.5 + level / 2:.1%}": self.coefficients + half,
        })

    def table(self) -> pd.DataFrame:
        """Coefficient table (estimate, std. error, t, p, significance)."""
        p_values = self.p_values
        table = pd.DataFrame({
            "Estimate": self.coefficients,
            "Std. Error": self.standard_errors,
            "t value": self.t_values,
            "Pr(>|t|)": p_values,
        })
        table[""] = [_stars(p) for p in p_values]
        return table


@dataclass(frozen=True)
class RhoEstimate:
    """How the autoregressive parameter was obtained.

    Attributes
    ----------
    value : float
        ``rho`` used (``nan`` if not identified).
    method : str
        ``"maxlog"``, ``"minrss"`` or ``"fixed"``.
    estimated, truncated, identified : bool
        Whether it was estimated, replaced by the truncation bound, and
        identified by the data.
    unconstrained_value : float or None
        Optimum before truncation.
    at_bound : bool
        Optimum on the boundary of the search interval.
    objective : float or None
        Criterion at the optimum (negative log-likelihood or RSS).
    profile : pd.Series or None
        Criterion over the search grid (index: rho); plot it to inspect
        flatness or multimodality.
    local_minima : tuple of float
        Grid locations of local minima.
    n_evaluations : int
        Criterion evaluations performed.
    """

    value: float
    method: str
    estimated: bool
    truncated: bool
    identified: bool
    unconstrained_value: float | None = None
    at_bound: bool = False
    objective: float | None = None
    profile: pd.Series | None = field(default=None, repr=False)
    local_minima: tuple[float, ...] = ()
    n_evaluations: int = 0


@dataclass(frozen=True)
class BenchmarkStatistics:
    """Record of a Denton benchmark.

    Attributes
    ----------
    variant, criterion : str
    h : int
    penalty : float
        Minimised movement-preservation criterion.
    indicator_discrepancy : pd.Series
        ``y_l - C x``: what the benchmark had to distribute.
    """

    variant: str
    criterion: str
    h: int
    penalty: float
    indicator_discrepancy: pd.Series


@dataclass(frozen=True)
class NumericalDiagnostics:
    """Numerical health of the estimation.

    Attributes
    ----------
    aggregation_error : float
        ``max |C y_hat - y_l|``.
    condition_number : float or None
        Of ``V`` (regressions) or of the KKT system (Denton).
    design_condition_number : float or None
        Of the whitened design (regressions).
    exact_fit : bool
        Zero low-frequency residuals.
    warnings : tuple of str
        Every warning raised during the fit.
    """

    aggregation_error: float
    condition_number: float | None = None
    design_condition_number: float | None = None
    exact_fit: bool = False
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class PreprocessingRecord:
    """Every artificial value created before estimation.

    Attributes
    ----------
    completion, padding : settings used.
    added_periods : pd.Index
        High-frequency periods absent from the input (removed from ``values``).
    filled_indicators : pd.DataFrame
        Boolean mask, over the returned periods, of indicator values that were
        missing in the input and filled.
    artificial_targets : pd.Series
        Padding targets (label -> value).
    """

    completion: CompletionSettings
    padding: PaddingSettings
    added_periods: pd.Index
    filled_indicators: pd.DataFrame = field(repr=False)
    artificial_targets: pd.Series

    @property
    def n_filled_values(self) -> int:
        """Indicator values filled on returned periods."""
        return int(self.filled_indicators.to_numpy().sum())


@dataclass(frozen=True)
class DisaggregationResult:
    """Outcome of a temporal disaggregation (see module docstring)."""

    values: pd.Series
    method: Any
    data: DisaggregationData = field(repr=False)
    fitted_low: pd.Series = field(repr=False)
    diagnostics: NumericalDiagnostics = field(repr=False)
    full_values: pd.Series = field(repr=False)
    preprocessing: PreprocessingRecord = field(repr=False)
    regression: RegressionStatistics | None = field(default=None, repr=False)
    rho: RhoEstimate | None = None
    benchmark: BenchmarkStatistics | None = field(default=None, repr=False)
    prediction_se: pd.Series | None = field(default=None, repr=False)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------
    @property
    def method_name(self) -> str:
        """Canonical name of the method used."""
        return str(self.method.name)

    @property
    def coefficients(self) -> pd.Series | None:
        """Shortcut to ``regression.coefficients``."""
        return None if self.regression is None else self.regression.coefficients

    @property
    def residuals(self) -> pd.Series:
        """Low-frequency residuals ``y_l - fitted_low`` (observed targets only)."""
        observed = self.data.target_rows
        return pd.Series(self.data.y_low[observed] - self.fitted_low.to_numpy(),
                         index=self.data.low_index[observed], name="residuals")

    def aggregation_error(self) -> float:
        """Largest absolute violation of ``C y_hat = y_l``."""
        return self.diagnostics.aggregation_error

    def prediction_interval(self, level: float = DEFAULT_LEVEL) -> pd.DataFrame:
        """Pointwise intervals ``y_hat ± t * se`` (regression methods only).

        Conditional on ``rho`` and on Gaussian residuals; they do not reflect
        the sampling uncertainty of an estimated ``rho``.
        """
        if self.prediction_se is None or self.regression is None:
            raise ValueError("Prediction intervals require a regression-based method.")
        _check_level(level)
        quantile = stats.t.ppf(0.5 + level / 2.0, self.regression.df_resid)
        return pd.DataFrame({
            "estimate": self.values,
            "lower": self.values - quantile * self.prediction_se,
            "upper": self.values + quantile * self.prediction_se,
        })

    def to_frame(self) -> pd.DataFrame:
        """Estimate, standard error, indicators and data flags per returned period."""
        real = ~self.data.artificial_high
        frame = pd.DataFrame(self.data.indicators[real], index=self.data.high_index[real],
                             columns=list(self.data.indicator_names))
        frame.insert(0, "estimate", self.values.to_numpy())
        if self.prediction_se is not None:
            frame.insert(1, "std_error", self.prediction_se.to_numpy())
        frame["constrained"] = self.data.aggregation_matrix.any(axis=0)[real]
        frame["indicator_filled"] = self.data.completed_indicators[real].any(axis=1) \
            if self.data.has_indicators else False
        return frame

    def summary(self) -> str:
        """R-style text summary."""
        data = self.data
        n_artificial_low = int(data.artificial_low.sum())
        n_constraints = int(np.sum(data.row_kind == "constraint"))
        lines = [f"'{self.method_name}' disaggregation with '{data.conversion.value}' conversion",
                 f"{int(data.target_rows.sum())} low-freq. obs. converted to "
                 f"{len(self.values)} high-freq. obs."]
        if n_constraints:
            lines.append(f"Additional exact constraints: {n_constraints}")
        prep = self.preprocessing
        if len(prep.added_periods) or prep.n_filled_values or n_artificial_low:
            lines.append(
                f"Preprocessing: {len(prep.added_periods)} period(s) added, "
                f"{prep.n_filled_values} indicator value(s) filled, "
                f"{n_artificial_low} artificial target(s) (padding "
                f"{prep.padding.before} before / {prep.padding.after} after)."
            )
        residuals = self.residuals.to_numpy()
        quantiles = np.quantile(residuals, [0.0, 0.25, 0.5, 0.75, 1.0])
        lines += ["", "Residuals (low frequency):",
                  pd.DataFrame([quantiles], columns=["Min", "1Q", "Median", "3Q", "Max"])
                  .to_string(index=False, float_format=lambda v: f"{v:.5g}")]
        if self.regression is not None:
            reg = self.regression
            lines += ["", "Coefficients:",
                      reg.table().to_string(float_format=lambda v: f"{v:.6g}"),
                      "---", "Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1",
                      "",
                      f"Residual std. error (GLS): {np.sqrt(reg.sigma2_gls):.6g} on "
                      f"{reg.df_resid} degrees of freedom",
                      f"Multiple R-squared: {reg.r_squared:.4f}   "
                      f"Adjusted R-squared: {reg.adj_r_squared:.4f}",
                      f"Log-likelihood: {reg.log_likelihood:.4f}   AIC: {reg.aic:.4f}   "
                      f"BIC: {reg.bic:.4f}   (likelihood-based AIC: {reg.aic_loglik:.4f})",
                      f"Lag-1 autocorrelation of whitened residuals: "
                      f"{reg.residual_autocorrelation:.4f}"]
        if self.rho is not None:
            rho = self.rho
            status = "fixed" if not rho.estimated else f"estimated by {rho.method}"
            if rho.truncated:
                status += f", truncated (unconstrained {rho.unconstrained_value:.6f})"
            if rho.at_bound:
                status += ", ON BOUNDARY"
            if not rho.identified:
                status += ", NOT IDENTIFIED"
            lines.append(f"AR1-Parameter: {rho.value:.6f} ({status})")
        if self.benchmark is not None:
            bench = self.benchmark
            lines += ["", f"Denton {bench.variant}, {bench.criterion}, h = {bench.h}; "
                          f"penalty = {bench.penalty:.6g}"]
        diag = self.diagnostics
        lines.append(f"Max. aggregation error: {diag.aggregation_error:.3e}")
        if diag.warnings:
            lines += ["", "Warnings:"] + [f"  - {message}" for message in diag.warnings]
        return "\n".join(lines)
