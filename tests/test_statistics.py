"""Statistical validity: coverage, invariances and internal consistency."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from tempdisagg import TemporalDisaggregator
from tempdisagg.core.data import DataAligner
from tests.conftest import make_quarterly_problem

MONTE_CARLO_REPLICATIONS: int = 300
TRUE_RHO: float = 0.7
TRUE_BETA: tuple[float, float] = (5.0, 1.5)


def _simulate_chow_lin(rng: np.random.Generator, n_low: int = 30, ratio: int = 4):
    """Draw from the Chow-Lin model: y = b0 + b1 x + u, u stationary AR(1)."""
    n = n_low * ratio
    x = 20 + np.cumsum(rng.normal(0.2, 1.0, n))
    u = np.empty(n)
    u[0] = rng.normal(0, 1 / np.sqrt(1 - TRUE_RHO ** 2))
    for t in range(1, n):
        u[t] = TRUE_RHO * u[t - 1] + rng.normal()
    y = TRUE_BETA[0] + TRUE_BETA[1] * x + u
    y_low = y.reshape(n_low, ratio).sum(axis=1)
    return DataAligner("sum").from_arrays(y_low, x, frequency_ratio=ratio), y


def test_prediction_intervals_have_nominal_coverage() -> None:
    """With the true rho, 95% intervals must cover ~95% of the true values."""
    rng = np.random.default_rng(123)
    model = TemporalDisaggregator("chow-lin-fixed", fixed_rho=TRUE_RHO)
    covered, total = 0, 0
    for _ in range(MONTE_CARLO_REPLICATIONS):
        data, truth = _simulate_chow_lin(rng)
        band = model.fit_data(data).prediction_interval(0.95)
        covered += int(np.sum((band["lower"] <= truth) & (truth <= band["upper"])))
        total += truth.size
    assert covered / total == pytest.approx(0.95, abs=0.015)


def test_coefficient_intervals_have_nominal_coverage() -> None:
    rng = np.random.default_rng(321)
    model = TemporalDisaggregator("chow-lin-fixed", fixed_rho=TRUE_RHO)
    hits = 0
    for _ in range(MONTE_CARLO_REPLICATIONS):
        data, _ = _simulate_chow_lin(rng)
        interval = model.fit_data(data).regression.confidence_intervals(0.95).iloc[1]
        hits += int(interval.iloc[0] <= TRUE_BETA[1] <= interval.iloc[1])
    # Binomial(300, 0.95): +/- 0.04 is about 3 standard deviations.
    assert hits / MONTE_CARLO_REPLICATIONS == pytest.approx(0.95, abs=0.04)


def test_maxlog_rho_is_consistent() -> None:
    rng = np.random.default_rng(7)
    estimates = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(40):
            data, _ = _simulate_chow_lin(rng, n_low=120)
            estimates.append(TemporalDisaggregator("chow-lin-maxlog").fit_data(data).rho.value)
    assert np.median(estimates) == pytest.approx(TRUE_RHO, abs=0.15)


def test_inference_quantities_are_consistent() -> None:
    annual, indicators, _ = make_quarterly_problem()
    result = TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators)
    reg = result.regression
    np.testing.assert_allclose(reg.t_values, reg.coefficients / reg.standard_errors)
    np.testing.assert_allclose(reg.standard_errors ** 2, np.diag(reg.covariance))
    expected_p = 2 * stats.t.sf(np.abs(reg.t_values), reg.df_resid)
    np.testing.assert_allclose(reg.p_values, expected_p)
    assert reg.sigma2_gls == pytest.approx(reg.rss / reg.df_resid)
    assert reg.sigma2_ml == pytest.approx(reg.rss / reg.n_obs)
    n_params = reg.n_coefficients + 2  # sigma^2 and an estimated rho
    assert reg.aic_loglik == pytest.approx(-2 * reg.log_likelihood + 2 * n_params)
    # R convention (verified against tempdisagg 1.2.0)
    assert reg.aic == pytest.approx(np.log(reg.sigma2_ml) + 2 * reg.n_coefficients / reg.n_obs)
    assert reg.bic == pytest.approx(np.log(reg.sigma2_ml)
                                    + reg.n_coefficients * np.log(reg.n_obs) / reg.n_obs)
    assert 0.0 <= reg.r_squared <= 1.0
    interval = reg.confidence_intervals(0.9)
    assert (interval.iloc[:, 0] < reg.coefficients).all()
    assert (reg.coefficients < interval.iloc[:, 1]).all()


def test_profile_optimum_matches_reported_rho() -> None:
    annual, indicators, _ = make_quarterly_problem()
    result = TemporalDisaggregator("chow-lin-maxlog", truncated_rho=-1).fit(annual, indicators)
    profile = result.rho.profile
    assert result.rho.objective <= profile.min() + 1e-9
    assert abs(profile.idxmin() - result.rho.value) <= 2 * (profile.index[1] - profile.index[0])


def test_adding_a_constant_to_an_indicator_is_absorbed_by_the_intercept() -> None:
    annual, indicators, _ = make_quarterly_problem()
    base = TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators)
    shifted = TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators + 1000.0)
    np.testing.assert_allclose(shifted.values, base.values, rtol=1e-7)


def test_column_order_does_not_matter() -> None:
    annual, indicators, _ = make_quarterly_problem()
    base = TemporalDisaggregator("litterman-maxlog").fit(annual, indicators)
    swapped = TemporalDisaggregator("litterman-maxlog").fit(annual, indicators[["x2", "x1"]])
    np.testing.assert_allclose(swapped.values, base.values, rtol=1e-8)


def test_prediction_se_is_zero_where_only_aggregates_matter() -> None:
    """With 'first' conversion the constrained sub-period is known exactly."""
    annual, indicators, truth = make_quarterly_problem(n_years=20)
    first = truth.groupby(truth.index.asfreq("Y")).first()
    first.index = pd.PeriodIndex(first.index, freq="Y")
    result = TemporalDisaggregator("chow-lin-fixed", "first").fit(first, indicators)
    constrained = result.data.aggregation_matrix.any(axis=0)
    assert np.allclose(result.prediction_se[constrained], 0.0, atol=1e-7)
    assert (result.prediction_se[~constrained] > 0).all()


def test_extrapolation_uncertainty_grows_with_horizon() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=20, extra_quarters=8)
    result = TemporalDisaggregator("fernandez").fit(annual, indicators)
    tail = result.prediction_se.iloc[-8:].to_numpy()
    assert np.all(np.diff(tail) > 0)


def test_whitened_residuals_are_uncorrelated_under_correct_model() -> None:
    rng = np.random.default_rng(99)
    values = []
    for _ in range(60):
        data, _ = _simulate_chow_lin(rng, n_low=60)
        result = TemporalDisaggregator("chow-lin-fixed", fixed_rho=TRUE_RHO).fit_data(data)
        values.append(result.regression.residual_autocorrelation)
    assert abs(np.mean(values)) < 0.1
