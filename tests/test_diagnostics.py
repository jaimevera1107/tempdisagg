"""Conformity, revision analysis, rho profile plot and ratio padding."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tempdisagg import (
    Candidate,
    InputValidationError,
    PaddingSettings,
    TemporalDisaggregator,
    conformity,
    revision_analysis,
    temporal_aggregate,
)
from tempdisagg.datasets import load_swisspharma


def _quiet(func):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func()


def _monthly(seed: int = 1, n_years: int = 12):
    rng = np.random.default_rng(seed)
    months = pd.period_range("2010-01", periods=12 * n_years, freq="M")
    x = pd.Series(100 + np.cumsum(rng.normal(0.3, 1, len(months))), index=months, name="x")
    truth = 2 * x + np.cumsum(rng.normal(0, 1, len(months)))
    return x, truth, temporal_aggregate(truth, "Y")


# ----------------------------------------------------------------------
# Conformity
# ----------------------------------------------------------------------
def test_conformity_matches_manual_computation() -> None:
    x, _, annual = _monthly()
    result = TemporalDisaggregator().fit(annual, x)
    table = conformity(result)
    y = result.values.to_numpy()
    xs = x.to_numpy()
    growth_y, growth_x = 100 * (y[1:] / y[:-1] - 1), 100 * (xs[1:] / xs[:-1] - 1)
    row = table.loc["x"]
    assert row["change"] == "growth %"
    assert row["corr_levels"] == pytest.approx(np.corrcoef(y, x)[0, 1])
    assert row["corr_changes"] == pytest.approx(np.corrcoef(growth_y, growth_x)[0, 1])
    assert row["volatility_ratio"] == pytest.approx(
        np.std(growth_y, ddof=1) / np.std(growth_x, ddof=1))


def test_conformity_of_denton_with_its_own_indicator_is_near_perfect() -> None:
    x, _, annual = _monthly()
    result = TemporalDisaggregator("denton-cholette").fit(annual, x)
    assert conformity(result).loc["x", "corr_changes"] > 0.95


def test_conformity_uses_differences_for_non_positive_series() -> None:
    x, _, annual = _monthly()
    result = TemporalDisaggregator().fit(annual - annual.mean(), x)
    assert conformity(result).loc["x", "change"] == "difference"


def test_conformity_needs_an_indicator() -> None:
    _, _, annual = _monthly()
    result = TemporalDisaggregator("denton-cholette").fit(annual, high_freq="M")
    with pytest.raises(InputValidationError, match="indicators"):
        conformity(result)
    x, _, _ = _monthly()
    assert "x" in conformity(result, x).index


# ----------------------------------------------------------------------
# Revision analysis
# ----------------------------------------------------------------------
def test_revisions_match_manual_refits() -> None:
    x, _, annual = _monthly()
    model = TemporalDisaggregator("fernandez")
    result = revision_analysis(annual, {"f": Candidate(model, x)}, n_vintages=1)
    previous = model.fit(annual.iloc[:-1], x.loc[: str(annual.index[-2])]).values
    current = model.fit(annual, x).values
    common = previous.index
    expected = 100 * (current.loc[common] - previous) / previous.abs()
    np.testing.assert_allclose(result.revisions["level_revision"], expected.to_numpy())
    summary = result.summary()
    assert summary.loc["f", "n_vintages"] == 1
    assert summary.loc["f", "change_units"] == "growth pp"


def test_denton_revises_less_than_the_dynamic_model() -> None:
    sales, trade = load_swisspharma()
    candidates = {
        "denton": Candidate(TemporalDisaggregator("denton-cholette"), trade[["exports"]]),
        "dynamic": Candidate(_quiet(lambda: TemporalDisaggregator("dynamic-maxlog")),
                             trade[["exports"]]),
    }
    summary = revision_analysis(sales, candidates, n_vintages=6).summary()
    assert summary.loc["denton", "MAR_change"] < summary.loc["dynamic", "MAR_change"]


def test_extrapolation_mode() -> None:
    x, _, annual = _monthly()
    candidates = {"cl": Candidate(TemporalDisaggregator(), x),
                  "no indicator": Candidate(TemporalDisaggregator("denton-cholette"))}
    result = revision_analysis(annual, candidates, n_vintages=3, extrapolation=True,
                               high_freq="M")
    assert "no indicator" in result.failures
    revised = result.revisions
    # The extrapolated year of each earlier vintage is revised when its total arrives.
    assert revised["period"].max() == x.index[-1]
    assert result.extrapolation


def test_retropolated_periods_are_excluded_by_default() -> None:
    sales, trade = load_swisspharma()
    candidates = {"cl": Candidate(TemporalDisaggregator(), trade[["exports"]])}
    default = revision_analysis(sales, candidates, n_vintages=2).revisions
    included = revision_analysis(sales, candidates, n_vintages=2,
                                 include_retropolated=True).revisions
    assert default["period"].min() == pd.Period("1975Q1", "Q")
    assert included["period"].min() == pd.Period("1972Q1", "Q")


def test_revision_analysis_validation() -> None:
    x, _, annual = _monthly(n_years=5)
    candidates = {"cl": Candidate(TemporalDisaggregator(), x)}
    with pytest.raises(InputValidationError, match="first vintage"):
        revision_analysis(annual, candidates, n_vintages=4)
    with pytest.raises(InputValidationError):
        revision_analysis(annual, candidates, n_vintages=0)


# ----------------------------------------------------------------------
# rho profile plot
# ----------------------------------------------------------------------
def test_plot_rho_profile() -> None:
    matplotlib = pytest.importorskip("matplotlib")
    matplotlib.use("Agg")
    from tempdisagg.plotting import plot_rho_profile

    sales, trade = load_swisspharma()
    result = TemporalDisaggregator().fit(sales, trade[["exports"]])
    ax = plot_rho_profile(result)
    assert ax.get_ylabel() == "log-likelihood"
    assert len(ax.lines) >= 1
    fixed = TemporalDisaggregator("chow-lin-fixed").fit(sales, trade[["exports"]])
    with pytest.raises(ValueError, match="profile"):
        plot_rho_profile(fixed)


# ----------------------------------------------------------------------
# Ratio padding (IMF-style extrapolation)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("method", ["ratio", "ratio_linear"])
def test_ratio_padding_target_definition(method: str) -> None:
    x, _, annual = _monthly()
    train = annual.iloc[:-1]
    padding = PaddingSettings(after=1, target_method=method)
    result = _quiet(lambda: TemporalDisaggregator("denton-cholette", padding=padding)
                    .fit(train, x))
    years = x.index.asfreq("Y")
    indicator_totals = x.groupby(years).sum()
    ratios = (train / indicator_totals.reindex(train.index)).to_numpy()
    if method == "ratio":
        projected = ratios[-1]
    else:
        positions = np.arange(len(ratios))
        slope, intercept = np.polyfit(positions[-3:], ratios[-3:], 1)
        projected = intercept + slope * len(ratios)
    expected = projected * indicator_totals.iloc[-1]
    assert result.preprocessing.artificial_targets.iloc[0] == pytest.approx(expected)
    # The year being extrapolated is returned (its indicator is observed).
    assert result.values.index[-1] == x.index[-1]
    assert result.values.loc[str(annual.index[-1])].sum() == pytest.approx(expected)


def test_ratio_padding_needs_one_indicator() -> None:
    x, _, annual = _monthly()
    both = pd.concat([x, x.rename("x2") * 1.1 + 3], axis=1)
    with pytest.raises(InputValidationError, match="exactly one indicator"):
        _quiet(lambda: TemporalDisaggregator(padding=PaddingSettings(
            after=1, target_method="ratio")).fit(annual.iloc[:-1], both))
