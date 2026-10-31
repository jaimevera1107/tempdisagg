"""Completion of the time grid, padding, and methods without indicators."""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tempdisagg import (
    ArtificialDataWarning,
    CompletionSettings,
    IneffectiveParameterWarning,
    InputValidationError,
    PaddingSettings,
    TempDisaggModel,
    TemporalDisaggregator,
    disaggregate,
)
from tempdisagg.preprocessing.settings import as_padding
from tests.conftest import make_quarterly_problem

EXAMPLE_FILE: Path = (Path(__file__).resolve().parents[1] / "examples" / "data"
                     / "legacy_long_format.csv")


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


# ----------------------------------------------------------------------
# Completion (default)
# ----------------------------------------------------------------------
def test_missing_subperiod_of_a_constrained_year_is_completed() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=12)
    with pytest.warns(ArtificialDataWarning, match="1 high-frequency period"):
        result = TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators.iloc[1:])
    assert list(result.preprocessing.added_periods) == [indicators.index[0]]
    assert len(result.values) == len(indicators) - 1            # artificial period removed
    assert len(result.full_values) == len(indicators)
    assert result.aggregation_error() < 1e-8 * float(annual.max())  # full-year constraint kept


def test_interior_gap_is_completed_and_interpolated_linearly() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=12)
    gappy = indicators.drop(indicators.index[[10, 11]])
    result = _quiet(TemporalDisaggregator("litterman-maxlog").fit, annual, gappy)
    filled = result.data.indicators[10:12, 0]
    expected = np.interp([10, 11], [9, 12], indicators["x1"].iloc[[9, 12]])
    np.testing.assert_allclose(filled, expected)
    assert set(result.preprocessing.added_periods) == set(indicators.index[[10, 11]])


def test_missing_indicator_values_are_filled_and_flagged() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=12)
    indicators.iloc[20, 1] = np.nan
    with pytest.warns(ArtificialDataWarning, match="1 missing indicator value"):
        result = TemporalDisaggregator().fit(annual, indicators)
    assert result.preprocessing.n_filled_values == 1
    assert result.to_frame()["indicator_filled"].sum() == 1
    assert np.isfinite(result.values).all()


def test_edge_methods() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=12, n_indicators=1)
    indicators.iloc[:2, 0] = np.nan
    constant = _quiet(TemporalDisaggregator(
        completion=CompletionSettings(edge_method="constant")).fit, annual, indicators)
    assert constant.data.indicators[0, 0] == indicators.iloc[2, 0]
    linear = _quiet(TemporalDisaggregator().fit, annual, indicators)
    assert linear.data.indicators[0, 0] != indicators.iloc[2, 0]


def test_targets_are_never_invented_by_completion() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=12)
    annual.iloc[5] = np.nan
    result = _quiet(TemporalDisaggregator().fit, annual, indicators)
    assert annual.index[5] not in result.data.low_index
    assert result.data.n_low == 11


def test_partial_last_period_without_target_is_not_padded_out() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=12, extra_quarters=2)
    result = _quiet(TemporalDisaggregator().fit, annual, indicators)
    assert len(result.preprocessing.added_periods) == 0
    assert len(result.values) == len(indicators)


def test_legacy_example_file_runs_with_completion() -> None:
    frame = pd.read_csv(EXAMPLE_FILE, sep=";")
    frame["y"] = frame.groupby("Index")["y"].transform("first")  # file has mixed targets
    result = _quiet(TemporalDisaggregator("chow-lin-maxlog", include_intercept=False)
                    .fit_long_format, frame, x_cols=["X"])
    assert len(result.values) == len(frame)                  # added grains are removed
    assert len(result.preprocessing.added_periods) == 4      # 2010-Q1, 2025-Q2..Q4
    assert result.aggregation_error() < 1e-6 * frame["y"].max()


# ----------------------------------------------------------------------
# Padding
# ----------------------------------------------------------------------
def test_padding_targets_follow_the_requested_rule() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=15)
    linear = _quiet(TemporalDisaggregator(padding=PaddingSettings(before=1, after=2)).fit,
                    annual, indicators)
    targets = linear.preprocessing.artificial_targets
    assert len(targets) == 3
    positions = np.arange(len(annual))
    slope, intercept = np.polyfit(positions[-3:], annual.to_numpy()[-3:], 1)
    np.testing.assert_allclose(targets.iloc[1:], intercept + slope * positions[-1] +
                               slope * np.array([1, 2]))
    repeat = _quiet(TemporalDisaggregator(padding=PaddingSettings(1, 1, target_method="repeat"))
                    .fit, annual, indicators)
    np.testing.assert_allclose(repeat.preprocessing.artificial_targets,
                               [annual.iloc[0], annual.iloc[-1]])


def test_padding_is_removed_from_output_and_constraints_hold() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=15)
    result = _quiet(TemporalDisaggregator(padding=2).fit, annual, indicators)
    assert result.values.index.equals(indicators.index)
    assert len(result.full_values) == len(indicators) + 16
    real = ~result.data.artificial_low
    aggregated = result.data.aggregation_matrix[real] @ result.full_values.to_numpy()
    np.testing.assert_allclose(aggregated, annual.to_numpy(), rtol=1e-10)


def test_padding_with_targets_changes_the_ends() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=15)
    plain = TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators)
    padded = _quiet(TemporalDisaggregator("chow-lin-maxlog", padding=1).fit, annual, indicators)
    assert np.max(np.abs(padded.values.iloc[-4:] - plain.values.iloc[-4:])) > 1e-6


@pytest.mark.parametrize("method", ["chow-lin-maxlog", "litterman-maxlog", "fernandez",
                                    "denton-cholette"])
def test_unconstrained_padding_has_no_effect_in_sample(method: str) -> None:
    """Why padding needs artificial targets: these estimators ignore free periods."""
    annual, indicators, _ = make_quarterly_problem(n_years=15, n_indicators=1)
    plain = _quiet(TemporalDisaggregator(method).fit, annual, indicators)
    padded = _quiet(TemporalDisaggregator(
        method, padding=PaddingSettings(2, 2, target_method="none")).fit, annual, indicators)
    np.testing.assert_allclose(padded.values, plain.values, rtol=1e-7, atol=1e-7)


def test_regression_statistics_flag_artificial_targets() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=15)
    with pytest.warns(ArtificialDataWarning, match="Regression statistics include 2"):
        result = TemporalDisaggregator(padding=1).fit(annual, indicators)
    assert result.regression.n_artificial_obs == 2
    assert len(result.residuals) == len(annual)


def test_padding_arguments() -> None:
    assert as_padding(2) == PaddingSettings(2, 2)
    assert as_padding((0, 3)) == PaddingSettings(0, 3)
    for bad in ({"before": -1}, {"after": 1.5}, {"target_method": "mean"},
                {"fit_window": 1}):
        with pytest.raises(InputValidationError):
            PaddingSettings(**bad)


# ----------------------------------------------------------------------
# Methods without indicators
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["uniform", "denton-cholette", "denton",
                                    "chow-lin-maxlog", "fernandez", "litterman-maxlog"])
def test_methods_run_without_indicators(method: str) -> None:
    annual, _, _ = make_quarterly_problem(n_years=15)
    series = _quiet(disaggregate, annual, method=method, high_freq="Q")
    assert len(series) == 60 and np.isfinite(series).all()
    totals = series.groupby(series.index.asfreq("Y")).sum().to_numpy()
    np.testing.assert_allclose(totals, annual.to_numpy(), rtol=1e-9)


def test_denton_without_indicator_is_smooth() -> None:
    annual, _, _ = make_quarterly_problem(n_years=15)
    uniform = disaggregate(annual, method="uniform", high_freq="Q")
    smooth = disaggregate(annual, method="denton-cholette", high_freq="Q")

    def roughness(series: pd.Series) -> float:
        return float(np.sum(np.diff(series.to_numpy()) ** 2))

    assert roughness(smooth) < roughness(uniform)


def test_long_format_without_indicator_columns() -> None:
    frame = pd.DataFrame({"Index": np.repeat(np.arange(2000, 2010), 12),
                          "Grain": np.tile(np.arange(1, 13), 10),
                          "y": np.repeat(np.linspace(1200, 2000, 10), 12)})
    result = TemporalDisaggregator("denton-cholette").fit_long_format(frame)
    assert len(result.values) == 120


def test_legacy_wrapper_without_indicator() -> None:
    frame = pd.DataFrame({"Index": np.repeat(np.arange(2000, 2010), 4),
                          "Grain": np.tile(np.arange(1, 5), 10),
                          "y": np.repeat(np.linspace(100, 190, 10), 4)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        model = TempDisaggModel(method="uniform", X_col=None).fit(frame)
    np.testing.assert_allclose(model.predict()[:4, 0], 25.0)


def test_uniform_warns_when_given_indicators() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=10)
    with pytest.warns(IneffectiveParameterWarning, match="ignores"):
        TemporalDisaggregator("uniform").fit(annual, indicators)


def test_regression_without_indicator_needs_intercept() -> None:
    annual, _, _ = make_quarterly_problem(n_years=10)
    with pytest.raises(InputValidationError, match="regressor"):
        TemporalDisaggregator("chow-lin-fixed", include_intercept=False).fit(annual, high_freq="Q")


def test_missing_high_freq_without_indicators_is_explained() -> None:
    annual, _, _ = make_quarterly_problem(n_years=10)
    with pytest.raises(InputValidationError, match="high_freq"):
        TemporalDisaggregator("uniform").fit(annual)


# ----------------------------------------------------------------------
# Partial periods
# ----------------------------------------------------------------------
def _first_year_missing_q1():
    annual, indicators, truth = make_quarterly_problem(n_years=12)
    return annual, indicators.iloc[1:], truth


def test_partial_policy_constrains_observed_subperiods_only() -> None:
    annual, indicators, _ = _first_year_missing_q1()
    partial_total = annual.copy()
    partial_total.iloc[0] = annual.iloc[0] * 0.75  # target covers Q2-Q4 only
    model = TemporalDisaggregator(completion=CompletionSettings(partial_periods="partial"))
    result = _quiet(model.fit, partial_total, indicators)
    first_year = result.values.loc[result.values.index.year == annual.index[0].year]
    assert len(first_year) == 3
    assert first_year.sum() == pytest.approx(partial_total.iloc[0], rel=1e-10)
    assert len(result.preprocessing.added_periods) == 0


def test_complete_policy_constrains_the_full_period() -> None:
    annual, indicators, _ = _first_year_missing_q1()
    result = _quiet(TemporalDisaggregator().fit, annual, indicators)
    full_first = result.full_values.loc[result.full_values.index.year == annual.index[0].year]
    assert full_first.sum() == pytest.approx(annual.iloc[0], rel=1e-10)


def test_error_policy_rejects_partial_periods() -> None:
    annual, indicators, _ = _first_year_missing_q1()
    model = TemporalDisaggregator(completion=CompletionSettings(partial_periods="error"))
    with pytest.raises(InputValidationError, match="partial_periods='error'"):
        model.fit(annual, indicators)


def test_partial_first_conversion_needs_the_first_subperiod() -> None:
    annual, indicators, truth = make_quarterly_problem(n_years=12)
    first = truth.groupby(truth.index.asfreq("Y")).first()
    first.index = pd.PeriodIndex(first.index, freq="Y")
    model = TemporalDisaggregator("chow-lin-fixed", "first",
                                  completion=CompletionSettings(partial_periods="partial"))
    with pytest.raises(InputValidationError, match="first sub-period"):
        _quiet(model.fit, first, indicators.iloc[1:])
