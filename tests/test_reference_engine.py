"""The optimised engine must reproduce the literal textbook formulas."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tempdisagg import TemporalDisaggregator
from tempdisagg.core.data import DataAligner
from tests import reference_implementation as ref

CASES = [  # (preset, reference sigma builder, fixed rho)
    ("chow-lin-fixed", lambda rho, n: ref.sigma_chow_lin(rho, n), 0.6),
    ("chow-lin-fixed", lambda rho, n: ref.sigma_chow_lin(rho, n), -0.4),
    ("litterman-fixed", ref.sigma_litterman, 0.5),
    ("fernandez", ref.sigma_litterman, 0.0),
    ("ols", lambda rho, n: np.eye(n), 0.0),
]


def _problem(seed: int, conversion: str, n_low: int = 14, ratio: int = 4, extra: int = 3):
    rng = np.random.default_rng(seed)
    n = n_low * ratio + extra
    x = np.column_stack([50 + np.cumsum(rng.normal(0.3, 1, n)), rng.normal(10, 2, n)])
    y_l = rng.normal(100, 15, n_low) + (x[: n_low * ratio, 0].reshape(n_low, ratio).sum(1))
    data = DataAligner(conversion).from_arrays(y_l, x, frequency_ratio=ratio)
    return data, x, y_l


@pytest.mark.parametrize("preset, sigma_builder, rho", CASES)
@pytest.mark.parametrize("conversion", ["sum", "average", "first", "last"])
@pytest.mark.parametrize("seed", [1, 2])
def test_gls_methods_match_reference(preset, sigma_builder, rho, conversion, seed) -> None:
    data, x, y_l = _problem(seed, conversion)
    overrides = {"fixed_rho": rho} if preset.endswith("fixed") else {}
    result = TemporalDisaggregator(preset, conversion, **overrides).fit_data(data)

    design = np.column_stack([np.ones(len(x)), x])
    expected = ref.gls_disaggregation(y_l, design, data.aggregation_matrix,
                                      sigma_builder(rho, len(x)))
    np.testing.assert_allclose(result.values, expected["y_hat"], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(result.coefficients, expected["beta"], rtol=1e-8)
    np.testing.assert_allclose(result.regression.covariance.to_numpy(),
                               expected["vcov_beta"], rtol=1e-6)
    np.testing.assert_allclose(result.prediction_se ** 2, expected["prediction_var"],
                               rtol=1e-6, atol=1e-8)
    assert result.regression.log_likelihood == pytest.approx(expected["logl"], rel=1e-9)


def test_prediction_covariance_vanishes_on_aggregates() -> None:
    """C Var(y_hat - y) C' = 0 because C y_hat = C y = y_l exactly."""
    data, x, y_l = _problem(5, "sum", extra=0)
    design = np.column_stack([np.ones(len(x)), x])
    expected = ref.gls_disaggregation(y_l, design, data.aggregation_matrix,
                                      ref.sigma_chow_lin(0.7, len(x)))
    aggregated = data.aggregation_matrix @ expected["prediction_cov"] @ data.aggregation_matrix.T
    assert np.max(np.abs(aggregated)) < 1e-6 * np.max(np.abs(expected["prediction_cov"]))


@pytest.mark.parametrize("form, preset", [("correlation", "chow-lin-minrss-ecotrim"),
                                          ("covariance", "chow-lin-minrss-quilis")])
@pytest.mark.parametrize("seed", [3, 7, 11])
def test_minrss_optimum_matches_reference(form: str, preset: str, seed: int) -> None:
    data, x, y_l = _problem(seed, "sum")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TemporalDisaggregator(preset, truncated_rho=-1).fit_data(data)
    design = np.column_stack([np.ones(len(x)), x])

    def rss(rho: float) -> float:
        sigma = ref.sigma_chow_lin(rho, len(x), form)
        return ref.gls_disaggregation(y_l, design, data.aggregation_matrix, sigma)["rss"]

    # The reported objective is the reference criterion at the reported rho ...
    assert result.rho.objective == pytest.approx(rss(result.rho.value), rel=1e-7)
    # ... and no point of a fine reference grid does better.
    grid = np.linspace(-0.999, 0.999, 400)
    assert result.rho.objective <= min(rss(r) for r in grid) * (1 + 1e-9)


@pytest.mark.parametrize("seed", [4, 8, 12])
def test_maxlog_optimum_matches_reference(seed: int) -> None:
    data, x, y_l = _problem(seed, "average")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TemporalDisaggregator("litterman-maxlog", "average",
                                       truncated_rho=-1).fit_data(data)
    design = np.column_stack([np.ones(len(x)), x])

    def logl(rho: float) -> float:
        sigma = ref.sigma_litterman(rho, len(x))
        return ref.gls_disaggregation(y_l, design, data.aggregation_matrix, sigma)["logl"]

    assert -result.rho.objective == pytest.approx(logl(result.rho.value), rel=1e-9)
    grid = np.linspace(-0.999, 0.999, 400)
    assert -result.rho.objective >= max(logl(r) for r in grid) - 1e-9


@pytest.mark.parametrize("variant", ["original", "cholette"])
@pytest.mark.parametrize("criterion", ["additive", "proportional"])
@pytest.mark.parametrize("h", [0, 1, 2])
def test_denton_matches_reference(variant: str, criterion: str, h: int) -> None:
    data, x, y_l = _problem(6, "sum", extra=0)
    data = DataAligner("sum").from_arrays(y_l, x[:, :1], frequency_ratio=4)
    result = TemporalDisaggregator("denton-cholette", variant=variant,
                                   criterion=criterion, h=h).fit_data(data)
    expected = ref.denton(y_l, x[:, 0], data.aggregation_matrix, h,
                          criterion == "proportional", variant == "cholette")
    np.testing.assert_allclose(result.values, expected, rtol=1e-8, atol=1e-8)
