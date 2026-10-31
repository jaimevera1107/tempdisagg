"""Every method is configurable, and its configuration is explicit."""

from __future__ import annotations

import dataclasses
import warnings

import numpy as np
import pandas as pd
import pytest

from tempdisagg import (
    OLS,
    ChowLin,
    Denton,
    DynamicChowLin,
    Fernandez,
    IneffectiveParameterWarning,
    InputValidationError,
    Litterman,
    RhoOptimizerSettings,
    TemporalDisaggregator,
    Uniform,
    available_methods,
    disaggregate,
    make_method,
)
from tests.conftest import make_quarterly_problem

REGRESSION_PRESETS = [name for name in available_methods()
                      if name not in ("denton", "denton-cholette", "uniform")]


@pytest.mark.parametrize("name", available_methods())
def test_preset_names_round_trip(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert make_method(name).name == name


@pytest.mark.parametrize("name", REGRESSION_PRESETS)
def test_every_regression_method_has_intercept_switch(name: str) -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with_constant = TemporalDisaggregator(name).fit(annual, indicators)
        without = TemporalDisaggregator(name, include_intercept=False).fit(annual, indicators)
    assert "(Intercept)" in with_constant.coefficients.index
    assert "(Intercept)" not in without.coefficients.index
    assert with_constant.method.include_intercept and not without.method.include_intercept
    assert without.aggregation_error() < 1e-6 * float(annual.max())


def test_intercept_default_matches_r() -> None:
    for spec in (ChowLin(), Litterman(), Fernandez(), DynamicChowLin(), OLS()):
        assert spec.include_intercept is True


def test_benchmarking_methods_reject_intercept_with_explanation() -> None:
    with pytest.raises(InputValidationError, match="no regression"):
        TemporalDisaggregator("denton-cholette", include_intercept=True)
    with pytest.raises(InputValidationError, match="Valid parameters"):
        TemporalDisaggregator("uniform", h=2)


def test_unknown_parameter_lists_valid_ones() -> None:
    with pytest.raises(InputValidationError, match="truncated_rho"):
        TemporalDisaggregator("chow-lin-maxlog", truncation=0.0)


def test_spec_object_equals_string_with_overrides() -> None:
    annual, indicators, _ = make_quarterly_problem()
    by_name = TemporalDisaggregator("litterman-minrss", include_intercept=False)
    by_spec = TemporalDisaggregator(Litterman(rho_method="minrss", include_intercept=False))
    assert by_name.method == by_spec.method
    np.testing.assert_array_equal(by_name.fit(annual, indicators).values,
                                  by_spec.fit(annual, indicators).values)


def test_name_follows_settings() -> None:
    assert ChowLin(rho_method="minrss", covariance_form="covariance").name == \
        "chow-lin-minrss-quilis"
    assert ChowLin(rho_method="fixed", fixed_rho=0.99999).name == "fast"
    assert Denton(variant="original").name == "denton"


def test_specs_are_immutable() -> None:
    spec = ChowLin()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.include_intercept = False  # type: ignore[misc]


def test_ineffective_parameters_warn() -> None:
    with pytest.warns(IneffectiveParameterWarning, match="fixed_rho"):
        ChowLin(rho_method="maxlog", fixed_rho=0.3)
    with pytest.warns(IneffectiveParameterWarning, match="truncated_rho"):
        Litterman(rho_method="fixed", truncated_rho=0.2)
    with pytest.warns(IneffectiveParameterWarning, match="covariance_form"):
        ChowLin(rho_method="maxlog", covariance_form="covariance")


@pytest.mark.parametrize("bad", [
    {"fixed_rho": 1.0}, {"fixed_rho": -1.0}, {"truncated_rho": 1.0},
    {"truncated_rho": -1.5}, {"include_intercept": "False"}, {"rho_method": "ml"},
])
def test_invalid_values_are_rejected(bad: dict) -> None:
    with pytest.raises((InputValidationError, ValueError)):
        ChowLin(**bad)


@pytest.mark.parametrize("bad", [{"h": -1}, {"h": 1.5}, {"criterion": "ratio"},
                                 {"variant": "x"}])
def test_invalid_denton_values_are_rejected(bad: dict) -> None:
    with pytest.raises(InputValidationError):
        Denton(**bad)


def test_optimizer_settings_are_configurable_and_used() -> None:
    annual, indicators, _ = make_quarterly_problem()
    narrow = RhoOptimizerSettings(bounds=(0.1, 0.2), grid_size=11)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = TemporalDisaggregator(ChowLin(optimizer=narrow)).fit(annual, indicators)
    assert 0.1 <= result.rho.value <= 0.2
    assert len(result.rho.profile) == 11
    with pytest.raises(InputValidationError):
        RhoOptimizerSettings(bounds=(0.5, 0.2))


def test_describe_lists_settings_and_r_arguments() -> None:
    table = TemporalDisaggregator("fernandez", include_intercept=False).describe()
    assert table.loc["include_intercept", "value"] is False or \
        table.loc["include_intercept", "value"] == False  # noqa: E712
    assert "R argument" in table.columns
    assert "optimizer.bounds" in ChowLin().describe().index


def test_uniform_and_dynamic_specs() -> None:
    assert Uniform().name == "uniform"
    spec = DynamicChowLin(include_truncation_remainder=False, rho_method="fixed", fixed_rho=0.4)
    annual, indicators, _ = make_quarterly_problem()
    result = TemporalDisaggregator(spec).fit(annual, indicators)
    assert "(truncation remainder)" not in result.coefficients.index


def test_automation_function_returns_series_only() -> None:
    annual, indicators, _ = make_quarterly_problem()
    series = disaggregate(annual, indicators, method="fernandez", include_intercept=False)
    assert isinstance(series, pd.Series) and len(series) == len(indicators)


def test_result_records_full_configuration() -> None:
    annual, indicators, _ = make_quarterly_problem()
    result = TemporalDisaggregator("chow-lin-maxlog", truncated_rho=-1).fit(annual, indicators)
    assert result.method.truncated_rho == -1
    assert result.method_name == "chow-lin-maxlog"
