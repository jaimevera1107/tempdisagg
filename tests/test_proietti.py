"""Proietti ADL(1,1) model and the common-factor test."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tempdisagg import (
    ExperimentalMethodWarning,
    InputValidationError,
    Proietti,
    TemporalDisaggregator,
    common_factor_test,
)
from tempdisagg.core.data import DataAligner
from tests import reference_implementation as ref

MONTE_CARLO_REPLICATIONS: int = 150


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def _fit(preset: str, data, **overrides):
    """Fit a preset on aligned data without the experimental warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return TemporalDisaggregator(preset, **overrides).fit_data(data)


def _adl_data(rng: np.random.Generator, n_low: int = 30, rho: float = 0.6,
              b0: float = 1.0, b1: float = 0.4, conversion: str = "sum"):
    n = 4 * n_low
    x = 20 + np.cumsum(rng.normal(0.2, 1.0, n))
    y = np.empty(n)
    y[0] = 5 + b0 * x[0]
    for t in range(1, n):
        y[t] = rho * y[t - 1] + 2.0 + b0 * x[t] + b1 * x[t - 1] + rng.normal()
    weights = {"sum": np.ones(4), "average": np.full(4, 0.25)}[conversion]
    y_low = y.reshape(n_low, 4) @ weights
    return DataAligner(conversion).from_arrays(y_low, x, 4), x, y


def _chow_lin_data(rng: np.random.Generator, n_low: int = 30, rho: float = 0.7):
    n = 4 * n_low
    x = 20 + np.cumsum(rng.normal(0.2, 1.0, n))
    u = np.empty(n)
    u[0] = rng.normal(0, 1 / np.sqrt(1 - rho ** 2))
    for t in range(1, n):
        u[t] = rho * u[t - 1] + rng.normal()
    y = 5 + 1.5 * x + u
    return DataAligner("sum").from_arrays(y.reshape(n_low, 4).sum(axis=1), x, 4)


@pytest.mark.parametrize("rho", [0.0, 0.4, -0.3])
@pytest.mark.parametrize("conversion", ["sum", "average"])
def test_proietti_matches_dense_reference(rho: float, conversion: str) -> None:
    data, x, _ = _adl_data(np.random.default_rng(3), n_low=16, conversion=conversion)
    model = _quiet(TemporalDisaggregator, "proietti-fixed", conversion, fixed_rho=rho)
    result = model.fit_data(data)
    design, sigma = ref.adl_design(np.column_stack([np.ones(len(x)), x]), rho)
    expected = ref.gls_disaggregation(data.y_low, design, data.aggregation_matrix, sigma)
    np.testing.assert_allclose(result.values, expected["y_hat"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(result.coefficients, expected["beta"], rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(result.prediction_se ** 2, expected["prediction_var"],
                               rtol=1e-6, atol=1e-8)
    assert result.aggregation_error() < 1e-8 * float(np.abs(data.y_low).max())


def test_coefficient_names_and_experimental_warning() -> None:
    data, _, _ = _adl_data(np.random.default_rng(4))
    with pytest.warns(ExperimentalMethodWarning):
        model = TemporalDisaggregator("proietti-maxlog")
    result = _quiet(model.fit_data, data)
    assert list(result.coefficients.index) == [
        "(Intercept)", "x1", "x1(-1)", "(truncation remainder)"]
    assert result.method_name == "proietti-maxlog"


def test_constant_columns_are_not_lagged() -> None:
    data, x, _ = _adl_data(np.random.default_rng(5))
    dummy = (np.arange(len(x)) >= 60).astype(float)
    aligned = DataAligner("sum").from_arrays(data.y_low, np.column_stack([x, dummy]), 4)
    result = _fit("proietti-fixed", aligned, fixed_rho=0.0)
    names = list(result.coefficients.index)
    assert "x1(-1)" in names and "x2(-1)" in names and "(Intercept)(-1)" not in names


def test_proietti_recovers_adl_parameters() -> None:
    data, _, _ = _adl_data(np.random.default_rng(6), n_low=200)
    result = _fit("proietti-maxlog", data)
    assert result.rho.value == pytest.approx(0.6, abs=0.1)
    np.testing.assert_allclose(result.coefficients[["x1", "x1(-1)"]], [1.0, 0.4], atol=0.15)


def test_spec_settings() -> None:
    spec = _quiet(Proietti, rho_method="fixed", fixed_rho=0.3, include_intercept=False)
    assert spec.name == "proietti-fixed"
    model = _quiet(TemporalDisaggregator, spec)
    data, _, _ = _adl_data(np.random.default_rng(7))
    result = model.fit_data(data)
    assert "(Intercept)" not in result.coefficients.index
    assert result.rho.value == pytest.approx(0.3)


# ----------------------------------------------------------------------
# Common-factor test
# ----------------------------------------------------------------------
def test_common_factor_rejects_a_true_adl() -> None:
    data, _, _ = _adl_data(np.random.default_rng(8), n_low=40)
    test = common_factor_test(_fit("proietti-maxlog", data))
    assert test.p_value < 0.01
    assert "Common-factor test" in test.summary()


def test_common_factor_restriction_value() -> None:
    data, _, _ = _adl_data(np.random.default_rng(9))
    result = _fit("proietti-fixed", data, fixed_rho=0.5)
    test = common_factor_test(result)
    b = result.coefficients
    assert test.restriction["x1"] == pytest.approx(b["x1(-1)"] + 0.5 * b["x1"])
    assert test.df_numerator == 1 and test.df_denominator == result.regression.df_resid


def test_common_factor_size_under_chow_lin() -> None:
    """Under Chow-Lin the test should reject about 5% of the time.

    Measured: 5.5% to 9% for 20 to 80 annual observations (the test is
    conditional on the estimated rho, hence slightly oversized).
    """
    rng = np.random.default_rng(10)
    rejections = 0
    for _ in range(MONTE_CARLO_REPLICATIONS):
        data = _chow_lin_data(rng, n_low=40)
        result = _fit("proietti-maxlog", data)
        rejections += common_factor_test(result).p_value < 0.05
    assert rejections / MONTE_CARLO_REPLICATIONS < 0.15


def test_common_factor_requires_a_proietti_fit() -> None:
    data, _, _ = _adl_data(np.random.default_rng(11))
    result = TemporalDisaggregator("chow-lin-maxlog").fit_data(data)
    with pytest.raises(InputValidationError, match="proietti"):
        common_factor_test(result)


def test_swisspharma_runs() -> None:
    from tempdisagg.datasets import load_swisspharma

    sales, trade = load_swisspharma()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TemporalDisaggregator("proietti-maxlog").fit(sales, trade[["exports"]])
    assert result.aggregation_error() < 1e-6 * float(sales.max())
    assert isinstance(common_factor_test(result).restriction, pd.Series)
