"""Statistical properties every method must satisfy."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tempdisagg import (
    InputValidationError,
    SingularDesignError,
    TemporalDisaggregator,
    available_methods,
)
from tempdisagg.core.linalg import difference_matrix
from tests.conftest import make_quarterly_problem, quiet_fit

CONVERSIONS = ("sum", "average", "first", "last")
SINGLE_INDICATOR_METHODS = ("denton", "denton-cholette")


def _inputs_for(method: str, indicators: pd.DataFrame) -> pd.DataFrame | None:
    if method == "uniform":
        return None
    if method in SINGLE_INDICATOR_METHODS:
        return indicators.iloc[:, :1]
    return indicators


@pytest.mark.parametrize("method", available_methods())
@pytest.mark.parametrize("conversion", CONVERSIONS)
def test_every_method_reproduces_the_low_frequency_data(method: str, conversion: str) -> None:
    annual, indicators, truth = make_quarterly_problem(n_years=20)
    annual = truth.groupby(truth.index.asfreq("Y")).agg(
        {"sum": "sum", "average": "mean", "first": "first", "last": "last"}[conversion]
    )
    annual.index = pd.PeriodIndex(annual.index, freq="Y")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = TemporalDisaggregator(method, conversion=conversion)
    if method == "uniform" and conversion in ("first", "last"):
        with pytest.raises(InputValidationError):
            model.fit(annual, high_freq="Q")
        return
    result = quiet_fit(model, annual, _inputs_for(method, indicators), high_freq="Q")
    assert result.aggregation_error() < 1e-7 * float(np.abs(annual).max())


@pytest.mark.parametrize("method", ["chow-lin-maxlog", "chow-lin-minrss-ecotrim",
                                    "litterman-maxlog", "dynamic-maxlog"])
def test_rho_is_invariant_to_units(method: str) -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        model = TemporalDisaggregator(method, truncated_rho=-1)
    base = quiet_fit(model, annual, indicators)
    scaled = quiet_fit(model, annual * 1000.0, indicators * 1000.0)
    assert scaled.rho.value == pytest.approx(base.rho.value, abs=1e-5)
    np.testing.assert_allclose(scaled.values.to_numpy(), base.values.to_numpy() * 1000.0,
                               rtol=1e-5)


def test_fernandez_equals_litterman_with_zero_rho() -> None:
    annual, indicators, _ = make_quarterly_problem()
    fernandez = TemporalDisaggregator("fernandez").fit(annual, indicators)
    litterman = TemporalDisaggregator("litterman-fixed", fixed_rho=0.0).fit(annual, indicators)
    np.testing.assert_allclose(fernandez.values, litterman.values, rtol=1e-12)


def test_litterman_differs_from_chow_lin() -> None:
    """Regression test: 0.2.x implemented Litterman as a plain AR(1)."""
    annual, indicators, _ = make_quarterly_problem()
    litterman = TemporalDisaggregator("litterman-fixed", fixed_rho=0.5).fit(annual, indicators)
    chow_lin = TemporalDisaggregator("chow-lin-fixed", fixed_rho=0.5).fit(annual, indicators)
    assert np.max(np.abs(litterman.values - chow_lin.values)) > 1e-3


def test_denton_cholette_is_the_constrained_minimiser() -> None:
    """Regression test for issue #4: no identity added to the penalty."""
    annual, indicators, _ = make_quarterly_problem(n_years=12, n_indicators=1)
    model = TemporalDisaggregator("denton-cholette", criterion="additive")
    result = model.fit(annual, indicators)
    x = indicators.iloc[:, 0].to_numpy()
    y = result.values.to_numpy()
    difference = difference_matrix(len(y), 1, drop_initial=True)

    def objective(series: np.ndarray) -> float:
        residual = difference @ (series - x)
        return float(residual @ residual)

    matrix = result.data.aggregation_matrix
    _, _, vt = np.linalg.svd(matrix)
    null_space = vt[matrix.shape[0]:].T  # directions that keep C y fixed
    rng = np.random.default_rng(0)
    for _ in range(50):
        candidate = y + null_space @ rng.normal(0.0, 0.5, null_space.shape[1])
        assert objective(candidate) >= objective(y) - 1e-9


def test_denton_issue_4_reference_value() -> None:
    """Issue #4 example: the exact optimum is 324/11 = 29.4545..., not 35.81."""
    from tempdisagg.core.data import DataAligner

    data = DataAligner("sum").from_arrays([40.0, 4.0], np.ones(8), frequency_ratio=4)
    result = TemporalDisaggregator("denton-cholette", criterion="additive").fit_data(data)
    values = result.values.to_numpy()
    difference = difference_matrix(8, 1, drop_initial=True)
    roughness = float((difference @ values) @ (difference @ values))
    assert roughness == pytest.approx(324.0 / 11.0, rel=1e-12)


def test_fast_approximates_additive_denton_cholette() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=20, n_indicators=1)
    fast = TemporalDisaggregator("fast", include_intercept=False).fit(annual, indicators)
    scaled_indicator = indicators * fast.coefficients.iloc[0]
    denton = TemporalDisaggregator("denton-cholette", criterion="additive").fit(
        annual, scaled_indicator
    )
    np.testing.assert_allclose(fast.values, denton.values, rtol=1e-5)


def test_multiple_indicators_recover_true_coefficients() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=200, n_indicators=2)
    result = TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators)
    np.testing.assert_allclose(result.coefficients[["x1", "x2"]], [1.5, 0.5], atol=0.1)
    assert result.regression.p_values["x1"] < 0.01


def test_truncation_flag() -> None:
    annual, indicators, _ = make_quarterly_problem()
    result = TemporalDisaggregator("chow-lin-maxlog", truncated_rho=0.998).fit(annual, indicators)
    assert result.rho.truncated and result.rho.value == pytest.approx(0.998)
    assert result.rho.unconstrained_value < 0.998


def test_extrapolation_is_returned_and_constrained_part_holds() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=20, extra_quarters=5)
    result = TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators)
    assert len(result.values) == 85
    assert result.aggregation_error() < 1e-7 * float(annual.max())
    assert np.isfinite(result.values.iloc[-5:]).all()


def test_ols_satisfies_the_constraint() -> None:
    annual, indicators, _ = make_quarterly_problem()
    result = TemporalDisaggregator("ols").fit(annual, indicators)
    assert result.aggregation_error() < 1e-8 * float(annual.max())


def test_collinear_indicators_raise() -> None:
    annual, indicators, _ = make_quarterly_problem()
    indicators["x3"] = indicators["x1"] * 2.0
    with pytest.raises(SingularDesignError):
        TemporalDisaggregator("chow-lin-fixed").fit(annual, indicators)


def test_regression_without_regressors_raises() -> None:
    annual, _, _ = make_quarterly_problem()
    model = TemporalDisaggregator("chow-lin-fixed", include_intercept=False)
    with pytest.raises(InputValidationError):
        model.fit(annual, high_freq="Q")


def test_denton_rejects_several_indicators() -> None:
    annual, indicators, _ = make_quarterly_problem()
    with pytest.raises(InputValidationError, match="exactly one"):
        TemporalDisaggregator("denton-cholette").fit(annual, indicators)


def test_ensemble_was_removed_with_explanation() -> None:
    with pytest.raises(InputValidationError, match="not identified"):
        TemporalDisaggregator("ensemble")


def test_legacy_names_warn_and_map() -> None:
    with pytest.warns(DeprecationWarning):
        model = TemporalDisaggregator("chow-lin-opt")
    assert model.method.name == "chow-lin-maxlog"


def test_summary_is_rendered() -> None:
    annual, indicators, _ = make_quarterly_problem()
    text = TemporalDisaggregator().fit(annual, indicators).summary()
    assert "Coefficients" in text and "AR1-Parameter" in text and "R-squared" in text


def test_invalid_parameters_fail_at_construction() -> None:
    with pytest.raises(InputValidationError):
        TemporalDisaggregator("chow-lin-fixed", fixed_rho=1.0)
    with pytest.raises(InputValidationError):
        TemporalDisaggregator("denton", h=-1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        TemporalDisaggregator("chow-lin-maxlog")  # no spurious warnings
