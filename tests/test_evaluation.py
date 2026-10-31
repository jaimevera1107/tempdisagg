"""Backtesting, ratios, break indicators and cross-sectional reconciliation."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tempdisagg import (
    Candidate,
    CrossSectionalReconciler,
    InfeasibleConstraintError,
    InputValidationError,
    SingularDesignError,
    TemporalDisaggregator,
    backtest,
    disaggregate_ratio,
    impulse,
    level_shift,
    temporal_aggregate,
)


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def _monthly(seed: int, n_years: int = 14, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2010-01", periods=12 * n_years, freq="M")
    x = pd.Series(100 + np.cumsum(rng.normal(0.2, 1, len(months))), index=months, name="x")
    truth = scale * (2 * x + np.cumsum(rng.normal(0, 1, len(months))))
    return x, truth


# ----------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------
def test_backtest_prefers_the_informative_indicator() -> None:
    x, truth = _monthly(1)
    annual = temporal_aggregate(truth, "Y")
    noise = pd.Series(np.random.default_rng(9).normal(100, 30, len(x)), index=x.index, name="n")
    result = backtest(annual, {
        "good": Candidate(TemporalDisaggregator(), x),
        "noise": Candidate(TemporalDisaggregator(), noise),
    }, holdout=4)
    table = result.summary()
    assert table.index[0] == "good"
    assert (table["n_origins"] == 4).all()
    assert result.errors.shape == (4, 2)


def test_backtest_predictions_match_manual_refit() -> None:
    x, truth = _monthly(2)
    annual = temporal_aggregate(truth, "Y")
    model = TemporalDisaggregator("fernandez")
    result = backtest(annual, {"f": Candidate(model, x)}, holdout=1)
    origin = annual.index[-1]
    manual = model.fit(annual.where(annual.index < origin), x).values
    expected = manual[manual.index.asfreq("Y") == origin].sum()
    assert result.predictions.loc[origin, "f"] == pytest.approx(expected)


def test_backtest_without_indicators_and_failures_are_recorded() -> None:
    x, truth = _monthly(3)
    annual = temporal_aggregate(truth, "Y")
    short = x.loc[: "2021-12"]  # does not cover the last held-out years
    result = _quiet(backtest, annual, {
        "denton": Candidate(TemporalDisaggregator("denton-cholette")),
        "short": Candidate(TemporalDisaggregator(), short),
    }, holdout=3, high_freq="M")
    assert result.summary().loc["denton", "n_origins"] == 3
    assert "short" in result.failures


def test_backtest_validation() -> None:
    x, truth = _monthly(4, n_years=5)
    annual = temporal_aggregate(truth, "Y")
    with pytest.raises(InputValidationError, match="Too few"):
        backtest(annual, {"a": Candidate(TemporalDisaggregator(), x)}, holdout=4)
    with pytest.raises(InputValidationError, match="same conversion"):
        backtest(annual, {"a": Candidate(TemporalDisaggregator(), x),
                          "b": Candidate(TemporalDisaggregator(conversion="average"), x)})


# ----------------------------------------------------------------------
# Ratios
# ----------------------------------------------------------------------
def test_ratio_of_sums_is_preserved() -> None:
    x, employed = _monthly(5)
    employed = employed + 1000
    informal = employed * (0.5 + 0.05 * np.sin(np.arange(len(employed)) / 6))
    result = disaggregate_ratio(
        temporal_aggregate(informal, "Y"), temporal_aggregate(employed, "Y"),
        TemporalDisaggregator(), TemporalDisaggregator(),
        (informal + 3).rename("i"), x)
    assert result.low_frequency_error < 1e-12
    assert result.ratio.between(0.3, 0.7).all()


def test_ratio_requires_sum_conversion() -> None:
    x, truth = _monthly(6)
    annual = temporal_aggregate(truth, "Y")
    with pytest.raises(InputValidationError, match="sum"):
        disaggregate_ratio(annual, annual, TemporalDisaggregator(conversion="average"),
                           TemporalDisaggregator(), x, x)


# ----------------------------------------------------------------------
# Break indicators
# ----------------------------------------------------------------------
def test_level_shift_captures_a_break() -> None:
    x, truth = _monthly(7)
    shift = level_shift(x.index, "2016-01")
    broken = truth + 400 * shift
    annual = temporal_aggregate(broken, "Y")
    with_dummy = TemporalDisaggregator().fit(annual, pd.concat([x, shift], axis=1))
    without = TemporalDisaggregator().fit(annual, x)
    assert with_dummy.regression.p_values[shift.name] < 0.01
    assert np.abs(with_dummy.values - broken).mean() < np.abs(without.values - broken).mean()


def test_indicator_helpers_validation() -> None:
    index = pd.period_range("2010-01", periods=24, freq="M")
    assert impulse(index, ["2010-05"]).sum() == 1
    with pytest.raises(InputValidationError):
        level_shift(index, "2009-01")
    with pytest.raises(InputValidationError):
        impulse(index, ["2030-01"])


def test_constant_dummy_with_intercept_is_singular() -> None:
    x, truth = _monthly(8)
    annual = temporal_aggregate(truth, "Y")
    dummy = pd.Series(1.0, index=x.index, name="d")
    with pytest.raises(SingularDesignError):
        TemporalDisaggregator().fit(annual, pd.concat([x, dummy], axis=1))


# ----------------------------------------------------------------------
# Cross-sectional reconciliation
# ----------------------------------------------------------------------
def _system(n_series: int = 4):
    rng = np.random.default_rng(10)
    months = pd.period_range("2015-01", periods=72, freq="M")
    truths = {f"d{i}": pd.Series(60 + 15 * i + np.cumsum(rng.normal(0.1, 1, 72)), index=months)
              for i in range(n_series)}
    total = sum(truths.values())
    results = {name: TemporalDisaggregator().fit(
        temporal_aggregate(series, "Y"), (series + rng.normal(0, 3, 72)).rename("x"))
        for name, series in truths.items()}
    return results, total, truths


@pytest.mark.parametrize("criterion, h", [("proportional", 1), ("proportional", 0),
                                          ("additive", 1), ("additive", 0)])
def test_reconciliation_meets_both_constraint_sets(criterion: str, h: int) -> None:
    results, total, _ = _system()
    before = sum(r.values for r in results.values())
    assert np.max(np.abs(before - total)) > 1.0
    out = CrossSectionalReconciler(criterion, h).reconcile_results(results, total)
    assert out.temporal_error < 1e-8 * float(total.max()) * 12
    assert out.contemporaneous_error < 1e-8 * float(total.max())
    assert out.redundant_rows == 6


def test_reconciliation_is_minimal_when_already_consistent() -> None:
    results, _, _ = _system()
    consistent_total = sum(r.values for r in results.values())
    out = CrossSectionalReconciler().reconcile_results(results, consistent_total)
    assert np.max(np.abs(out.adjustments.to_numpy())) < 1e-8


def test_reconciliation_rejects_inconsistent_totals() -> None:
    results, total, _ = _system()
    with pytest.raises(InfeasibleConstraintError, match="do not add up"):
        CrossSectionalReconciler().reconcile_results(results, total * 1.01)


def test_reconciliation_with_partial_total() -> None:
    results, total, _ = _system()
    partial = total.copy()
    partial.iloc[:30] = np.nan  # national monthly series only from mid-2017
    out = CrossSectionalReconciler().reconcile_results(results, partial)
    sums = out.values.sum(axis=1)
    np.testing.assert_allclose(sums.iloc[30:], partial.iloc[30:], rtol=1e-10)
