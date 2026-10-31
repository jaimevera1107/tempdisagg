"""Regression-based temporal disaggregation (the Chow-Lin family).

All GLS methods share this engine. A method is defined by its residual
structure (:mod:`tempdisagg.covariance`), by how ``rho`` is obtained and by
whether a constant is included.

Algorithm
---------
1. Build the regressors ``X`` (optional intercept + indicators), possibly
   transformed by the structure (dynamic model).
2. For a given ``rho`` obtain ``T`` with ``Sigma = T T'`` and compute
   ``W = C T`` (``n_l x n``) by backward filtering, so ``V = W W'``.
3. Estimate ``b`` by GLS (:func:`tempdisagg.estimation.gls.fit_gls`).
4. Distribute the low-frequency residuals with the BLUE

       y_hat = X b + Sigma C' V^{-1} u_l = X b + T W' V^{-1} u_l,

   which satisfies ``C y_hat = y_l`` exactly.
5. Prediction variance (Goldberger's BLUP formula, conditional on ``rho``):

       Var(y_hat - y) = s^2 [ Sigma - G V^{-1} G' ]
                        + Q Var(b) Q',      G = Sigma C',  Q = X - G V^{-1} X_l.

   Only the diagonal is computed, in O(n n_l^2). It ignores the sampling
   uncertainty of ``rho`` (as is standard), so intervals are conditional.

``rho`` handling
----------------
* Collinearity and exact fit are independent of ``rho`` (``V`` is
  non-singular), so they are checked once before any search.
* With an exact fit, ``u_l = 0`` and the estimate does not depend on ``rho``:
  ``rho`` is reported as not identified instead of returning an arbitrary
  optimum.
* An estimated ``rho`` below ``truncated_rho`` is replaced by it
  (R: ``truncated.rho``; ``-1`` disables truncation).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.linalg import cho_solve

from tempdisagg.core.data import DisaggregationData
from tempdisagg.core.linalg import solve_lower
from tempdisagg.covariance.factors import LowerTriangularFactor
from tempdisagg.covariance.structures import ResidualStructure
from tempdisagg.estimation.base import MethodOutput, RhoOutcome
from tempdisagg.estimation.diagnostics import DiagnosticLog
from tempdisagg.estimation.gls import GLSFit, fit_gls
from tempdisagg.estimation.rho import (
    RhoEstimator,
    RhoMethod,
    RhoOptimizerSettings,
    RhoSearchResult,
)
from tempdisagg.exceptions import (
    ExactFitWarning,
    IllConditionedWarning,
    InputValidationError,
    MultipleOptimaWarning,
    RhoBoundaryWarning,
    RhoNotIdentifiedWarning,
)

#: Name of the intercept coefficient (as in R).
INTERCEPT_NAME: str = "(Intercept)"
#: Name of the extra regressor of the dynamic model.
TRUNCATION_REMAINDER_NAME: str = "(truncation remainder)"
#: ``truncated_rho`` value that disables truncation (as in R).
NO_TRUNCATION: float = -1.0
#: Condition number of V above which it is reported as ill-conditioned.
ILL_CONDITION_THRESHOLD: float = 1e12
#: Scaled design condition number above which collinearity is reported
#: (about six significant digits lost in the coefficients).
COLLINEARITY_THRESHOLD: float = 1e6
#: ``rho`` used for checks that do not depend on ``rho``.
REFERENCE_RHO: float = 0.0
#: A secondary local maximum of the log-likelihood within this many units of
#: the global one is reported (half the 95% chi-square(1) critical value: the
#: two optima are not distinguishable by a likelihood-ratio test).
COMPETING_LOGLIK_MARGIN: float = 3.841 / 2.0
#: Relative RSS margin for reporting a competing minimum-RSS optimum.
COMPETING_RSS_MARGIN: float = 0.05


class RegressionDisaggregator:
    """GLS disaggregation with a configurable residual structure.

    Parameters
    ----------
    structure : ResidualStructure
        Residual covariance model.
    rho_method : RhoMethod
        ``maxlog``, ``minrss`` or ``fixed``.
    fixed_rho : float
        ``rho`` when ``rho_method`` is ``fixed``.
    truncated_rho : float
        Lower bound applied to an estimated ``rho``; ``-1`` disables it.
    include_intercept : bool
        Whether to add a constant regressor.
    optimizer : RhoOptimizerSettings
        Numerical settings of the ``rho`` search.
    reporting_structure : ResidualStructure, optional
        Structure used for the final fit once ``rho`` is fixed. It must differ
        from ``structure`` only by a rho-dependent scalar, so estimates,
        standard errors and the likelihood are unchanged; only the scale of
        ``sigma^2``, RSS and TSS follows it. Chow-Lin uses it to report in the
        stationary-covariance units of the R package.
    """

    def __init__(
        self,
        structure: ResidualStructure,
        rho_method: RhoMethod,
        fixed_rho: float,
        truncated_rho: float,
        include_intercept: bool,
        optimizer: RhoOptimizerSettings,
        reporting_structure: ResidualStructure | None = None,
    ) -> None:
        if not -1.0 < fixed_rho < 1.0:
            raise InputValidationError("fixed_rho must lie strictly between -1 and 1.")
        if not -1.0 <= truncated_rho < 1.0:
            raise InputValidationError("truncated_rho must lie in [-1, 1).")
        self.structure: ResidualStructure = structure
        self.rho_method: RhoMethod = RhoMethod(rho_method)
        self.fixed_rho: float = float(fixed_rho)
        self.truncated_rho: float = float(truncated_rho)
        self.include_intercept: bool = bool(include_intercept)
        self.optimizer: RhoOptimizerSettings = optimizer
        self.reporting_structure: ResidualStructure = reporting_structure or structure

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def disaggregate(self, data: DisaggregationData) -> MethodOutput:
        """Estimate the high-frequency series by GLS."""
        log = DiagnosticLog()
        regressors, names = self._design(data)
        centering = data.aggregation_matrix @ np.ones(data.n_high)

        def evaluate(rho: float, structure: ResidualStructure | None = None
                     ) -> tuple[GLSFit, LowerTriangularFactor, np.ndarray, np.ndarray]:
            structure = structure or self.structure
            transformed = structure.transform_regressors(regressors, rho)
            factor = structure.factor(rho, data.n_high)
            # W = C T, obtained as (T' C')' with backward filters.
            weights = factor.apply_transpose(data.aggregation_matrix.T).T
            fit = fit_gls(
                data.y_low, data.aggregation_matrix @ transformed, weights @ weights.T,
                centering,
            )
            return fit, factor, weights, transformed

        # rho-independent checks: raise on collinearity, detect exact fit.
        reference_rho = self.fixed_rho if self.rho_method is RhoMethod.FIXED else REFERENCE_RHO
        reference = evaluate(reference_rho if self.structure.uses_rho else 0.0)
        rho_outcome = self._resolve_rho(evaluate, reference[0], log)
        # The final fit uses the reporting structure (same estimate, R's scaling).
        final_rho = (rho_outcome.value if rho_outcome is not None
                     and np.isfinite(rho_outcome.value) else
                     (reference_rho if self.structure.uses_rho else 0.0))
        fit, factor, weights, transformed = evaluate(final_rho, self.reporting_structure)

        prediction = transformed @ fit.coefficients
        v_inverse_residuals = cho_solve((fit.cholesky_factor, True), fit.residuals)
        values = prediction + factor.apply(weights.T @ v_inverse_residuals)
        if transformed.shape[1] > len(names):
            names = (*names, TRUNCATION_REMAINDER_NAME)

        self._check_conditioning(fit, log)
        return MethodOutput(
            values=values,
            fitted_low=data.aggregation_matrix @ prediction,
            diagnostics=log,
            coefficient_names=names,
            gls=fit,
            rho=rho_outcome,
            prediction_variance=self._prediction_variance(
                fit, factor, weights, transformed, data.aggregation_matrix @ transformed
            ),
        )

    # ------------------------------------------------------------------
    # Steps
    # ------------------------------------------------------------------
    def _design(self, data: DisaggregationData) -> tuple[np.ndarray, tuple[str, ...]]:
        """Assemble the high-frequency regressor matrix."""
        columns = [data.indicators]
        names: tuple[str, ...] = data.indicator_names
        if self.include_intercept:
            columns.insert(0, np.ones((data.n_high, 1)))
            names = (INTERCEPT_NAME, *names)
        regressors = np.hstack(columns)
        if regressors.shape[1] == 0:
            raise InputValidationError(
                "Regression methods need at least one regressor: supply an "
                "indicator or set include_intercept=True. For disaggregation "
                "without indicators use 'denton-cholette' or 'uniform'."
            )
        return regressors, names

    def _resolve_rho(
        self,
        evaluate: Callable[[float], tuple[GLSFit, LowerTriangularFactor, np.ndarray, np.ndarray]],
        reference: GLSFit,
        log: DiagnosticLog,
    ) -> RhoOutcome | None:
        """Determine ``rho`` and record diagnostics."""
        if not self.structure.uses_rho:
            if reference.exact_fit:
                log.warn(ExactFitWarning, "The low-frequency regression fits exactly.")
            return None

        if reference.exact_fit:
            log.warn(
                ExactFitWarning,
                "The low-frequency regression fits exactly: the estimate does not "
                "depend on rho, which is therefore not identified; standard "
                "errors are zero and the log-likelihood is unbounded.",
            )
            if self.rho_method is RhoMethod.FIXED:
                return RhoOutcome(self.fixed_rho, self.rho_method)
            return RhoOutcome(float("nan"), self.rho_method, identified=False)

        if self.rho_method is RhoMethod.FIXED:
            return RhoOutcome(self.fixed_rho, self.rho_method)

        if self.rho_method is RhoMethod.MAXLOG:
            def criterion(rho: float) -> float:
                return -evaluate(rho)[0].log_likelihood
        else:
            def criterion(rho: float) -> float:
                return evaluate(rho)[0].rss

        search = RhoEstimator(self.optimizer).estimate(criterion)
        self._report_search(search, log)
        if self.truncated_rho > NO_TRUNCATION and search.rho < self.truncated_rho:
            return RhoOutcome(
                self.truncated_rho, self.rho_method, truncated=True,
                unconstrained_value=search.rho, search=search, identified=search.identified,
            )
        return RhoOutcome(
            search.rho, self.rho_method, unconstrained_value=search.rho,
            search=search, identified=search.identified,
        )

    def _report_search(self, search: RhoSearchResult, log: DiagnosticLog) -> None:
        """Translate search diagnostics into warnings."""
        lower, upper = self.optimizer.bounds
        if not search.identified:
            log.warn(RhoNotIdentifiedWarning,
                     "The rho criterion is flat over the search interval.")
        elif search.at_bound:
            log.warn(
                RhoBoundaryWarning,
                f"Estimated rho = {search.rho:.6f} lies on the boundary of "
                f"[{lower}, {upper}]; the optimum may lie outside the interval.",
            )
        competing = self._competing_optima(search)
        if competing:
            locations = ", ".join(f"{rho:.3f}" for rho in competing)
            log.warn(
                MultipleOptimaWarning,
                f"The rho criterion has competing local optima at {locations} "
                f"(global optimum {search.rho:.3f}); inspect result.rho.profile.",
            )

    def _competing_optima(self, search: RhoSearchResult) -> tuple[float, ...]:
        """Admissible local optima whose criterion is close to the global one.

        Optima below an active truncation bound are ignored because they can
        never be selected.
        """
        lower = self.truncated_rho if self.truncated_rho > NO_TRUNCATION else -np.inf
        if self.rho_method is RhoMethod.MAXLOG:
            margin = COMPETING_LOGLIK_MARGIN
        else:
            margin = COMPETING_RSS_MARGIN * abs(search.objective)
        competing = []
        for rho in search.local_minima:
            value = search.profile[int(np.argmin(np.abs(search.grid - rho)))]
            far = abs(rho - search.rho) > 2.0 * (search.grid[1] - search.grid[0])
            if rho >= lower and far and value - search.objective <= margin:
                competing.append(rho)
        return tuple(competing)

    @staticmethod
    def _check_conditioning(fit: GLSFit, log: DiagnosticLog) -> None:
        if fit.condition_number_design > COLLINEARITY_THRESHOLD:
            log.warn(
                IllConditionedWarning,
                f"The regressors are nearly collinear (scaled condition number "
                f"{fit.condition_number_design:.2e}); coefficients are imprecise.",
            )
        if fit.condition_number_v > ILL_CONDITION_THRESHOLD:
            log.warn(
                IllConditionedWarning,
                f"V = C Sigma C' is ill-conditioned (condition number "
                f"{fit.condition_number_v:.2e}).",
            )

    @staticmethod
    def _prediction_variance(
        fit: GLSFit,
        factor: LowerTriangularFactor,
        weights: np.ndarray,
        regressors: np.ndarray,
        regressors_low: np.ndarray,
    ) -> np.ndarray:
        """Diagonal of ``Var(y_hat - y)`` (see module docstring)."""
        # G = Sigma C' = T W'  (n x n_l);  L^{-1} G' gives diag(G V^{-1} G').
        covariance_columns = factor.apply(weights.T)
        whitened = solve_lower(fit.cholesky_factor, covariance_columns.T)
        distributed = np.sum(whitened ** 2, axis=0)
        # Q = X - G V^{-1} X_l, and Var(b) = s2 R^{-1} R^{-T}.
        whitened_x = solve_lower(fit.cholesky_factor, regressors_low)
        q_matrix = regressors - whitened.T @ whitened_x
        parameter = np.sum((q_matrix @ fit.r_inverse) ** 2, axis=1)
        variance = fit.sigma2_gls * (factor.covariance_diagonal() - distributed + parameter)
        return np.maximum(variance, 0.0)  # remove tiny negative rounding
