"""High-level interface.

Two levels of use:

* **Automation** — :func:`disaggregate` returns only the high-frequency
  series, for pipelines that do not inspect statistics::

      series = disaggregate(y_annual, indicators, method="chow-lin-maxlog")

* **Analysis** — :class:`TemporalDisaggregator` returns a
  :class:`~tempdisagg.results.DisaggregationResult` with the full estimation
  record (coefficients, inference, ``rho`` profile, diagnostics)::

      model = TemporalDisaggregator("chow-lin-maxlog", include_intercept=False)
      result = model.fit(y_annual, indicators)
      print(result.summary())

Both accept an R preset name plus overrides of its settings, or a method
specification object (:mod:`tempdisagg.methods`), and two preprocessing
options: ``completion`` (on by default: completes the time grid) and
``padding`` (off by default: artificial periods around the sample).

Indicators are optional. Methods that only distribute or smooth
(``uniform``, ``denton-cholette``, ``denton``, or a regression on a constant)
run without them; pass ``high_freq`` to set the output frequency.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from tempdisagg.core.aggregation import Conversion
import warnings

from tempdisagg.core.constraints import LinearConstraint
from tempdisagg.core.data import DataAligner, DisaggregationData
from tempdisagg.estimation.base import MethodOutput
from tempdisagg.exceptions import (
    ArtificialDataWarning,
    IneffectiveParameterWarning,
    InputValidationError,
)
from tempdisagg.methods import MethodSpec, make_method
from tempdisagg.preprocessing.settings import (
    CompletionSettings,
    PaddingSettings,
    as_completion,
    as_padding,
)
from tempdisagg.results import (
    BenchmarkStatistics,
    DisaggregationResult,
    NumericalDiagnostics,
    PreprocessingRecord,
    RegressionStatistics,
    RhoEstimate,
)


class TemporalDisaggregator:
    """Disaggregate a low-frequency series to a higher frequency.

    Parameters
    ----------
    method : str or MethodSpec, default "chow-lin-maxlog"
        R preset name (:func:`tempdisagg.available_methods`) or a
        specification such as ``ChowLin(include_intercept=False)``.
    conversion : {"sum", "average", "first", "last"}, default "sum"
        How the high-frequency series aggregates to the low-frequency one.
    completion : bool or CompletionSettings, default True
        Complete the time grid and fill missing indicator values (recorded in
        ``result.preprocessing``). ``False`` gives strict mode (errors).
    padding : int, (before, after) or PaddingSettings, optional
        Artificial low-frequency periods around the observed targets, to damp
        end-point distortions. Removed from the returned series.
    **overrides
        Settings of the method to change, e.g. ``include_intercept=False``,
        ``truncated_rho=-1``, ``fixed_rho=0.8``, ``criterion="additive"``,
        ``h=2``. Call :meth:`describe` to list them.
    """

    def __init__(
        self,
        method: str | MethodSpec = "chow-lin-maxlog",
        conversion: Conversion | str = Conversion.SUM,
        *,
        completion: CompletionSettings | bool = True,
        padding: PaddingSettings | int | tuple[int, int] | None = None,
        **overrides: Any,
    ) -> None:
        self.method: MethodSpec = make_method(method, **overrides)
        self.conversion: Conversion = Conversion.parse(conversion)
        self.completion: CompletionSettings = as_completion(completion)
        self.padding: PaddingSettings = as_padding(padding)
        self._estimator = self.method.build()

    def _aligner(self) -> DataAligner:
        return DataAligner(self.conversion, self.completion, self.padding)

    def describe(self) -> pd.DataFrame:
        """Return every setting of the configured method."""
        return self.method.describe()

    # ------------------------------------------------------------------
    # Entry points
    # ------------------------------------------------------------------
    def fit(
        self,
        y_low: pd.Series,
        indicators: pd.Series | pd.DataFrame | None = None,
        high_freq: str | None = None,
        *,
        constraints: Sequence[LinearConstraint] = (),
        aggregation_weights: pd.Series | None = None,
    ) -> DisaggregationResult:
        """Fit period-indexed pandas inputs (calendar aware).

        Parameters
        ----------
        y_low : pd.Series
            Low-frequency target (Period/Datetime index; NaN = unknown).
        indicators : pd.Series or pd.DataFrame, optional
            Not needed by methods that only distribute or smooth.
        high_freq : str, optional
            Output frequency when ``indicators`` is None.
        constraints : sequence of LinearConstraint, optional
            Additional exact information (known values, intermediate totals),
            see :mod:`tempdisagg.core.constraints`.
        aggregation_weights : pd.Series, optional
            Weights per high-frequency period. With ``conversion="average"``
            the target is ``sum w_t y_t / sum w_t`` (e.g. a rate weighted by
            its denominator); with ``"sum"`` it is ``sum w_t y_t``.
        """
        data = self._aligner().from_series(y_low, indicators, high_freq,
                                           constraints, aggregation_weights)
        return self.fit_data(data)

    def fit_arrays(
        self,
        y_low: Sequence[float] | np.ndarray,
        indicators: np.ndarray | None = None,
        frequency_ratio: int = 4,
        n_high: int | None = None,
        *,
        constraints: Sequence[LinearConstraint] = (),
        aggregation_weights: Sequence[float] | np.ndarray | None = None,
    ) -> DisaggregationResult:
        """Fit plain arrays with ``frequency_ratio`` sub-periods per period.

        Constraint labels are row positions (0, 1, ...).
        """
        data = self._aligner().from_arrays(y_low, indicators, frequency_ratio, n_high,
                                           constraints, aggregation_weights)
        return self.fit_data(data)

    def fit_long_format(
        self,
        frame: pd.DataFrame,
        index_col: str = "Index",
        grain_col: str = "Grain",
        y_col: str = "y",
        x_cols: Sequence[str] | None = None,
        *,
        constraints: Sequence[LinearConstraint] = (),
        weights_col: str | None = None,
    ) -> DisaggregationResult:
        """Fit the long format of tempdisagg 0.2.x (``x_cols`` optional).

        Constraint labels are ``(index, grain)`` tuples.
        """
        data = self._aligner().from_long_format(frame, index_col, grain_col, y_col, x_cols,
                                                constraints, weights_col)
        return self.fit_data(data)

    def fit_data(self, data: DisaggregationData) -> DisaggregationResult:
        """Fit a pre-aligned :class:`DisaggregationData`."""
        if data.conversion is not self.conversion:
            raise InputValidationError("data was aligned with a different conversion.")
        if getattr(self.method, "family", "") == "benchmarking" and data.has_indicators \
                and self.method.name == "uniform":
            warnings.warn("The uniform method ignores the indicators.",
                          IneffectiveParameterWarning, stacklevel=3)
        output = self._estimator.disaggregate(data)
        if output.gls is not None and data.artificial_low.any():
            output.diagnostics.warn(
                ArtificialDataWarning,
                f"Regression statistics include {int(data.artificial_low.sum())} artificial "
                "(padding) targets; standard errors and tests treat them as data.",
            )
        output.diagnostics.emit(stacklevel=4)
        return _assemble_result(self.method, data, output, self.completion, self.padding)


def disaggregate(
    y_low: pd.Series,
    indicators: pd.Series | pd.DataFrame | None = None,
    method: str | MethodSpec = "chow-lin-maxlog",
    conversion: Conversion | str = Conversion.SUM,
    high_freq: str | None = None,
    *,
    completion: CompletionSettings | bool = True,
    padding: PaddingSettings | int | tuple[int, int] | None = None,
    constraints: Sequence[LinearConstraint] = (),
    aggregation_weights: pd.Series | None = None,
    **overrides: Any,
) -> pd.Series:
    """Return only the disaggregated series (automation entry point).

    Same arguments as :class:`TemporalDisaggregator` and
    :meth:`TemporalDisaggregator.fit`. Statistical warnings are still emitted.
    """
    model = TemporalDisaggregator(method, conversion, completion=completion,
                                  padding=padding, **overrides)
    return model.fit(y_low, indicators, high_freq, constraints=constraints,
                     aggregation_weights=aggregation_weights).values


# ----------------------------------------------------------------------
# Result assembly (pure functions of the method output)
# ----------------------------------------------------------------------
def _assemble_result(
    spec: MethodSpec,
    data: DisaggregationData,
    output: MethodOutput,
    completion: CompletionSettings,
    padding: PaddingSettings,
) -> DisaggregationResult:
    real = ~data.artificial_high
    observed = data.target_rows
    padding_rows = data.artificial_low
    full_values = pd.Series(output.values, index=data.high_index, name="estimate")
    error = float(np.max(np.abs(data.aggregation_matrix @ output.values - data.y_low)))
    gls = output.gls
    diagnostics = NumericalDiagnostics(
        aggregation_error=error,
        condition_number=(gls.condition_number_v if gls is not None
                          else output.denton.condition_number if output.denton else None),
        design_condition_number=gls.condition_number_design if gls is not None else None,
        exact_fit=bool(gls.exact_fit) if gls is not None else False,
        warnings=output.diagnostics.messages(),
    )
    rho = _rho_estimate(output)
    preprocessing = PreprocessingRecord(
        completion=completion,
        padding=padding,
        added_periods=data.high_index[~real],
        filled_indicators=pd.DataFrame(data.completed_indicators[real],
                                       index=data.high_index[real],
                                       columns=list(data.indicator_names)),
        artificial_targets=pd.Series(data.y_low[padding_rows],
                                     index=data.low_index[padding_rows],
                                     name="artificial_target"),
    )
    return DisaggregationResult(
        values=full_values[real],
        method=spec,
        data=data,
        fitted_low=pd.Series(output.fitted_low[observed], index=data.low_index[observed],
                             name="fitted"),
        diagnostics=diagnostics,
        full_values=full_values,
        preprocessing=preprocessing,
        regression=_regression_statistics(spec, output, rho, int(padding_rows.sum())),
        rho=rho,
        benchmark=_benchmark_statistics(spec, data, output),
        prediction_se=(None if output.prediction_variance is None else pd.Series(
            np.sqrt(output.prediction_variance[real]), index=data.high_index[real],
            name="std_error")),
    )


def _rho_estimate(output: MethodOutput) -> RhoEstimate | None:
    outcome = output.rho
    if outcome is None:
        return None
    search = outcome.search
    return RhoEstimate(
        value=outcome.value,
        method=outcome.method.value,
        estimated=outcome.estimated,
        truncated=outcome.truncated,
        identified=outcome.identified,
        unconstrained_value=outcome.unconstrained_value,
        at_bound=bool(search.at_bound) if search else False,
        objective=search.objective if search else None,
        profile=(pd.Series(search.profile, index=pd.Index(search.grid, name="rho"),
                           name="criterion") if search else None),
        local_minima=search.local_minima if search else (),
        n_evaluations=search.n_evaluations if search else 0,
    )


def _regression_statistics(
    spec: MethodSpec, output: MethodOutput, rho: RhoEstimate | None, n_artificial: int
) -> RegressionStatistics | None:
    gls = output.gls
    if gls is None:
        return None
    names = list(output.coefficient_names)
    n_obs, n_coef = gls.n_obs, gls.n_coefficients
    n_params = n_coef + 1 + int(rho is not None and rho.estimated and rho.identified)
    with np.errstate(divide="ignore", invalid="ignore"):
        r_squared = 1.0 - gls.rss / gls.tss if gls.tss > 0 else float("nan")
        # Conventions of the R package (verified against tempdisagg 1.2.0):
        # adjusted R2 always uses (n - 1) / (n - k); AIC/BIC use log(s2_ml).
        adj_r_squared = 1.0 - (1.0 - r_squared) * (n_obs - 1) / (n_obs - n_coef)
        log_s2 = np.log(gls.sigma2_ml) if gls.sigma2_ml > 0 else float("nan")
    log_likelihood = gls.log_likelihood
    return RegressionStatistics(
        coefficients=pd.Series(gls.coefficients, index=names, name="coefficient"),
        standard_errors=pd.Series(np.sqrt(np.diag(gls.coefficient_covariance)),
                                  index=names, name="std_error"),
        covariance=pd.DataFrame(gls.coefficient_covariance, index=names, columns=names),
        sigma2_ml=gls.sigma2_ml,
        sigma2_gls=gls.sigma2_gls,
        rss=gls.rss,
        tss=gls.tss,
        r_squared=float(r_squared),
        adj_r_squared=float(adj_r_squared),
        log_likelihood=log_likelihood,
        aic=float(log_s2 + 2.0 * n_coef / n_obs),
        bic=float(log_s2 + n_coef * np.log(n_obs) / n_obs),
        aic_loglik=-2.0 * log_likelihood + 2.0 * n_params,
        bic_loglik=-2.0 * log_likelihood + np.log(n_obs) * n_params,
        n_obs=n_obs,
        n_coefficients=n_coef,
        whitened_residuals=gls.whitened_residuals,
        n_artificial_obs=n_artificial,
    )


def _benchmark_statistics(
    spec: MethodSpec, data: DisaggregationData, output: MethodOutput
) -> BenchmarkStatistics | None:
    if output.denton is None:
        return None
    return BenchmarkStatistics(
        variant=str(spec.variant.value),  # type: ignore[attr-defined]
        criterion=str(spec.criterion.value),  # type: ignore[attr-defined]
        h=int(spec.h),  # type: ignore[attr-defined]
        penalty=output.denton.penalty,
        indicator_discrepancy=pd.Series(
            (data.y_low - output.fitted_low)[data.target_rows],
            index=data.low_index[data.target_rows], name="discrepancy"),
    )
