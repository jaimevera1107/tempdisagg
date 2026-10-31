"""Robustness: degenerate, extreme and malformed data.

Contract checked here: for any input, the library either returns a finite
estimate that satisfies the aggregation constraint, or raises a subclass of
:class:`TempDisaggError`. It never returns NaN/inf silently and never leaks
an unrelated exception (IndexError, LinAlgError, ZeroDivisionError, ...).
"""

from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd
import pytest

from tempdisagg import (
    ExactFitWarning,
    IllConditionedWarning,
    IndicatorSignWarning,
    InputValidationError,
    LinearConstraint,
    NumericalError,
    PaddingSettings,
    RhoBoundaryWarning,
    RhoOptimizerSettings,
    SingularDesignError,
    TempDisaggError,
    TemporalDisaggregator,
    available_methods,
)
from tempdisagg.core.data import DataAligner
from tempdisagg.methods import ChowLin
from tests.conftest import make_quarterly_problem

STRESS_CASES: int = 300


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


# ----------------------------------------------------------------------
# Degenerate designs
# ----------------------------------------------------------------------
def test_all_zero_indicator_is_singular() -> None:
    annual, indicators, _ = make_quarterly_problem()
    indicators["x2"] = 0.0
    with pytest.raises(SingularDesignError):
        TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators)


def test_constant_indicator_with_intercept_is_singular() -> None:
    annual, indicators, _ = make_quarterly_problem()
    indicators["x2"] = 7.0
    with pytest.raises(SingularDesignError):
        TemporalDisaggregator("litterman-maxlog").fit(annual, indicators)


def test_constant_indicator_without_intercept_is_fine() -> None:
    annual, indicators, _ = make_quarterly_problem()
    result = TemporalDisaggregator("chow-lin-maxlog", include_intercept=False).fit(
        annual, indicators[["x1"]].assign(x1=7.0))
    assert result.aggregation_error() < 1e-8 * float(annual.max())


def test_more_coefficients_than_observations() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=3, n_indicators=3)
    with pytest.raises(SingularDesignError, match="degree of freedom"):
        TemporalDisaggregator("chow-lin-fixed").fit(annual, indicators)


def test_minimal_degrees_of_freedom_works() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=3, n_indicators=1)
    result = TemporalDisaggregator("chow-lin-fixed").fit(annual, indicators)
    assert result.regression.df_resid == 1


def test_nearly_collinear_indicators_warn_or_fail_cleanly() -> None:
    annual, indicators, _ = make_quarterly_problem()
    indicators["x2"] = indicators["x1"] * (1 + 1e-11 * np.arange(len(indicators)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            result = TemporalDisaggregator("chow-lin-fixed").fit(annual, indicators)
        except SingularDesignError:
            return
    assert any(issubclass(w.category, IllConditionedWarning) for w in caught)
    assert np.isfinite(result.values).all()


# ----------------------------------------------------------------------
# Degenerate targets
# ----------------------------------------------------------------------
def test_exact_fit_is_reported_and_rho_not_identified() -> None:
    annual, indicators, _ = make_quarterly_problem()
    exact = (2.0 + 3.0 * indicators["x1"]).groupby(indicators.index.asfreq("Y")).sum()
    exact.index = pd.PeriodIndex(exact.index, freq="Y")
    exact = exact.iloc[: len(annual)]
    with pytest.warns(ExactFitWarning):
        result = TemporalDisaggregator("chow-lin-maxlog").fit(exact, indicators)
    assert not result.rho.identified and np.isnan(result.rho.value)
    np.testing.assert_allclose(result.coefficients, [2.0, 3.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result.values, 2.0 + 3.0 * indicators["x1"], rtol=1e-8)
    assert result.diagnostics.exact_fit


def test_all_zero_target() -> None:
    annual, indicators, _ = make_quarterly_problem()
    zeros = annual * 0.0
    result = _quiet(TemporalDisaggregator("chow-lin-maxlog").fit, zeros, indicators)
    np.testing.assert_allclose(result.values, 0.0, atol=1e-9)


@pytest.mark.parametrize("scale", [1e-9, 1e9])
def test_extreme_scales(scale: float) -> None:
    annual, indicators, _ = make_quarterly_problem()
    base = TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators)
    scaled = TemporalDisaggregator("chow-lin-maxlog").fit(annual * scale, indicators * scale)
    assert scaled.rho.value == pytest.approx(base.rho.value, abs=1e-5)
    np.testing.assert_allclose(scaled.values / scale, base.values, rtol=1e-6)


def test_negative_target_values_are_allowed() -> None:
    annual, indicators, _ = make_quarterly_problem()
    result = TemporalDisaggregator("fernandez").fit(annual - 2 * annual.max(), indicators)
    assert result.aggregation_error() < 1e-8 * float(annual.abs().max() * 3)


# ----------------------------------------------------------------------
# rho at the boundary / truncation
# ----------------------------------------------------------------------
def test_boundary_estimate_is_flagged() -> None:
    annual, indicators, _ = make_quarterly_problem()
    settings = RhoOptimizerSettings(bounds=(-0.2, 0.2), grid_size=21)
    with pytest.warns(RhoBoundaryWarning):
        result = TemporalDisaggregator(ChowLin(optimizer=settings, truncated_rho=-1)).fit(
            annual, indicators)
    assert result.rho.at_bound
    assert any("RhoBoundaryWarning" in message for message in result.diagnostics.warnings)


# ----------------------------------------------------------------------
# Malformed inputs
# ----------------------------------------------------------------------
@pytest.mark.parametrize("bad_value", [np.inf, -np.inf])
def test_infinite_indicator_rejected(bad_value: float) -> None:
    annual, indicators, _ = make_quarterly_problem()
    indicators.iloc[3, 1] = bad_value
    with pytest.raises(InputValidationError):
        TemporalDisaggregator().fit(annual, indicators)


def test_missing_indicator_rejected_in_strict_mode() -> None:
    annual, indicators, _ = make_quarterly_problem()
    indicators.iloc[3, 1] = np.nan
    with pytest.raises(InputValidationError):
        TemporalDisaggregator(completion=False).fit(annual, indicators)


def test_infinite_target_rejected() -> None:
    annual, indicators, _ = make_quarterly_problem()
    annual.iloc[2] = np.inf
    with pytest.raises(InputValidationError):
        TemporalDisaggregator().fit(annual, indicators)


def test_text_indicator_rejected() -> None:
    annual, indicators, _ = make_quarterly_problem()
    indicators = indicators.astype(object)
    indicators.iloc[0, 0] = "n/a"
    with pytest.raises((InputValidationError, ValueError)):
        TemporalDisaggregator().fit(annual, indicators)


def test_unsorted_or_duplicated_index_rejected() -> None:
    annual, indicators, _ = make_quarterly_problem()
    with pytest.raises(InputValidationError):
        TemporalDisaggregator().fit(annual, indicators.iloc[::-1])
    with pytest.raises(InputValidationError):
        TemporalDisaggregator().fit(annual, pd.concat([indicators, indicators.iloc[-1:]]))


def test_same_frequency_rejected() -> None:
    quarters = pd.period_range("2000Q1", periods=40, freq="Q")
    series = pd.Series(np.arange(40.0), index=quarters)
    with pytest.raises(InputValidationError, match="higher"):
        TemporalDisaggregator().fit(series, series.rename("x"))


def test_all_targets_missing_rejected() -> None:
    annual, indicators, _ = make_quarterly_problem()
    with pytest.raises(InputValidationError):
        _quiet(TemporalDisaggregator().fit, annual * np.nan, indicators)


# ----------------------------------------------------------------------
# Denton edge cases
# ----------------------------------------------------------------------
def test_denton_proportional_rejects_zero_indicator() -> None:
    annual, indicators, _ = make_quarterly_problem(n_indicators=1)
    indicators.iloc[5, 0] = 0.0
    with pytest.raises(InputValidationError, match="non-zero"):
        TemporalDisaggregator("denton-cholette").fit(annual, indicators)


def test_denton_proportional_warns_on_sign_change() -> None:
    annual, indicators, _ = make_quarterly_problem(n_indicators=1)
    indicators = indicators - indicators.mean()
    indicators[indicators == 0] = 1e-3
    with pytest.warns(IndicatorSignWarning):
        TemporalDisaggregator("denton-cholette").fit(annual, indicators)


def test_denton_unidentified_system_raises() -> None:
    data = DataAligner("sum").from_arrays([10.0], np.ones(4), frequency_ratio=4, n_high=4)
    with pytest.raises(NumericalError):
        TemporalDisaggregator("denton-cholette", criterion="additive", h=2).fit_data(data)


# ----------------------------------------------------------------------
# Scale of the problem
# ----------------------------------------------------------------------
def test_daily_disaggregation_is_fast() -> None:
    rng = np.random.default_rng(0)
    days = pd.period_range("2015-01-01", "2019-12-31", freq="D")
    indicator = pd.Series(500 + np.cumsum(rng.normal(0, 1, len(days))), index=days, name="x")
    truth = 2 * indicator + np.cumsum(rng.normal(0, 0.5, len(days)))
    monthly = truth.groupby(days.asfreq("M")).sum()
    monthly.index = pd.PeriodIndex(monthly.index, freq="M")
    for method in ("chow-lin-maxlog", "litterman-maxlog", "denton-cholette"):
        start = time.perf_counter()
        result = _quiet(TemporalDisaggregator(method).fit, monthly, indicator)
        assert time.perf_counter() - start < 30.0
        assert result.aggregation_error() < 1e-8 * float(monthly.max())


# ----------------------------------------------------------------------
# Randomised stress test of the contract
# ----------------------------------------------------------------------
def _random_problem(rng: np.random.Generator):
    n_low = int(rng.integers(1, 30))
    ratio = int(rng.integers(2, 13))
    extra = int(rng.integers(0, 2 * ratio))
    k = int(rng.integers(0, 4))
    scale = 10.0 ** rng.uniform(-6, 6)
    n = n_low * ratio + extra
    indicators = scale * (rng.normal(1, 1, (n, k)).cumsum(axis=0) + rng.uniform(-5, 50, k))
    if k and rng.random() < 0.1:
        indicators[:, 0] = 0.0                          # null column
    if k > 1 and rng.random() < 0.1:
        indicators[:, 1] = indicators[:, 0]             # duplicated column
    y_low = scale * rng.normal(50, 20, n_low)
    if rng.random() < 0.2 and n_low > 2:
        y_low[rng.integers(0, n_low)] = np.nan          # unobserved period
    if k and rng.random() < 0.3:
        holes = rng.integers(0, n, size=int(rng.integers(1, 4)))
        indicators[holes, int(rng.integers(0, k))] = np.nan  # missing indicator values
    conversion = str(rng.choice(["sum", "average", "first", "last"]))
    return y_low, (indicators if k else None), ratio, n, conversion


def test_random_inputs_respect_the_contract() -> None:
    rng = np.random.default_rng(2026)
    methods = available_methods()
    outcomes = {"ok": 0, "library_error": 0}
    for _ in range(STRESS_CASES):
        y_low, indicators, ratio, n, conversion = _random_problem(rng)
        method = str(rng.choice(methods))
        padding = (int(rng.integers(0, 3)), int(rng.integers(0, 3)))
        constraints = []
        if rng.random() < 0.3:  # a few known high-frequency values
            for position in rng.choice(len(y_low) * ratio, size=int(rng.integers(1, 3)),
                                       replace=False):
                guess = np.nanmean(y_low) / ratio * rng.uniform(0.5, 1.5)
                constraints.append(LinearConstraint(pd.Series([1.0], index=[int(position)]),
                                                    float(guess)))
        target_rule = str(rng.choice(["linear", "repeat", "none"]))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = TemporalDisaggregator(
                    method, conversion,
                    padding=PaddingSettings(*padding, target_method=target_rule))
                result = model.fit_arrays(y_low, indicators, frequency_ratio=ratio, n_high=n,
                                          constraints=constraints)
        except TempDisaggError:
            outcomes["library_error"] += 1
            continue
        values = result.values.to_numpy()
        assert np.isfinite(values).all(), (method, conversion)
        assert len(values) == n, (method, conversion, padding)
        scale = max(np.nanmax(np.abs(y_low)), 1e-300)
        assert result.aggregation_error() <= 1e-6 * scale, (method, conversion)
        if result.prediction_se is not None:
            assert np.isfinite(result.prediction_se).all()
        outcomes["ok"] += 1
    assert outcomes["ok"] > STRESS_CASES / 3  # most random problems are solvable
