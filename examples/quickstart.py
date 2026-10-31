"""Quick-start examples for tempdisagg 1.0.

Run from the repository root:  python examples/quickstart.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tempdisagg import (
    Candidate,
    CrossSectionalReconciler,
    InputValidationError,
    PaddingSettings,
    disaggregate,
    NonNegativeReconciler,
    SeriesSplicer,
    TemporalDisaggregator,
    aggregate_constraints,
    backtest,
    known_values,
    temporal_aggregate,
)

DATA_FILE: Path = Path(__file__).resolve().parent / "data" / "legacy_long_format.csv"
SEED: int = 42


def example_synthetic_two_indicators() -> None:
    """Annual -> quarterly with two indicators and two extrapolated quarters."""
    rng = np.random.default_rng(SEED)
    quarters = pd.period_range("2000Q1", periods=4 * 20 + 2, freq="Q")
    indicators = pd.DataFrame(
        {
            "exports": 100 + np.cumsum(rng.normal(0.5, 1.0, len(quarters))),
            "imports": 80 + np.cumsum(rng.normal(0.3, 1.0, len(quarters))),
        },
        index=quarters,
    )
    noise = np.zeros(len(quarters))
    for t in range(1, len(quarters)):
        noise[t] = 0.7 * noise[t - 1] + rng.normal(0.0, 1.0)
    truth = 5 + 1.2 * indicators["exports"] + 0.4 * indicators["imports"] + noise
    annual = temporal_aggregate(truth.iloc[:80], "Y", "sum")

    result = TemporalDisaggregator("chow-lin-maxlog").fit(annual, indicators)
    print(result.summary())
    print(result.values.tail(6))


def example_configuration_and_automation() -> None:
    """Per-method settings, and the series-only entry point."""
    rng = np.random.default_rng(SEED)
    months = pd.period_range("2015-01", periods=96, freq="M")
    indicator = pd.Series(50 + np.cumsum(rng.normal(0.2, 1.0, 96)), index=months, name="x")
    annual = temporal_aggregate(3 * indicator + rng.normal(0, 2, 96).cumsum(), "Y", "sum")

    model = TemporalDisaggregator("fernandez", include_intercept=False)
    print(model.describe())
    result = model.fit(annual, indicator)
    print(result.regression.table())
    print(result.prediction_interval(0.9).head(3))

    series = disaggregate(annual, indicator, method="litterman-maxlog")
    print("Automated run:", len(series), "monthly values")


def example_legacy_long_format() -> None:
    """The 0.2.x long format: incomplete years are completed automatically."""
    frame = pd.read_csv(DATA_FILE, sep=";")
    model = TemporalDisaggregator("chow-lin-maxlog", include_intercept=False)
    try:
        model.fit_long_format(frame, x_cols=["X"])
    except InputValidationError as error:
        # The shipped file has different targets within the same year;
        # 0.2.x silently kept the first value. The library refuses to guess.
        print("Rejected:", error)

    cleaned = frame.copy()
    cleaned["y"] = cleaned.groupby("Index")["y"].transform("first")  # explicit choice
    result = model.fit_long_format(cleaned, x_cols=["X"])
    print(result.summary())
    print("Added (artificial) periods:", list(result.preprocessing.added_periods))
    adjusted = NonNegativeReconciler().reconcile(result)
    print("Negative values before/after reconciliation:",
          int((result.values < 0).sum()), int((adjusted < 0).sum()))


def example_without_indicator_and_padding() -> None:
    """Distribution/smoothing without indicators, and padding."""
    years = pd.period_range("2010", periods=12, freq="Y")
    annual = pd.Series(np.linspace(1200, 2300, 12) ** 1.02, index=years)
    smooth = disaggregate(annual, method="denton-cholette", high_freq="M")
    even = disaggregate(annual, method="uniform", high_freq="M")
    print("Denton-Cholette vs uniform, first months:",
          smooth.head(3).round(1).tolist(), even.head(3).round(1).tolist())
    padded = TemporalDisaggregator("fernandez", padding=PaddingSettings(before=1, after=1))
    result = padded.fit(annual, high_freq="M")
    print("Artificial targets:", result.preprocessing.artificial_targets.round(1).to_dict())


def example_constraints_rates_and_systems() -> None:
    """Mixed-frequency constraints, a rate, backtesting and reconciliation."""
    rng = np.random.default_rng(SEED)
    months = pd.period_range("2012-01", periods=144, freq="M")
    indicator = pd.Series(100 + np.cumsum(rng.normal(0.2, 1, 144)), index=months, name="x")
    truth = 2 * indicator + np.cumsum(rng.normal(0, 1, 144))
    annual = temporal_aggregate(truth, "Y")

    constraints = (aggregate_constraints(temporal_aggregate(truth, "Q").iloc[[10, 11]], "M")
                   + known_values(truth.iloc[[60, 61]]))
    result = TemporalDisaggregator().fit(annual, indicator, constraints=constraints)
    honoured = np.allclose(result.values.iloc[[60, 61]], truth.iloc[[60, 61]])
    print("Known months honoured:", honoured)

    print(backtest(annual, {
        "chow-lin": Candidate(TemporalDisaggregator(), indicator),
        "fernandez": Candidate(TemporalDisaggregator("fernandez"), indicator),
    }, holdout=4).summary().round(3))

    regions = {f"region_{i}": truth * (0.2 + 0.1 * i) + rng.normal(0, 2, 144) for i in range(3)}
    national = sum(regions.values())
    fits = {name: TemporalDisaggregator().fit(temporal_aggregate(series, "Y"),
                                              (series + rng.normal(0, 4, 144)).rename("x"))
            for name, series in regions.items()}
    system = CrossSectionalReconciler().reconcile_results(fits, national)
    print("Regions add up to the national series:", system.contemporaneous_error < 1e-8)


def example_splicing() -> None:
    """Extend a new-methodology series backwards with a legacy series."""
    months = pd.period_range("2001-01", periods=36, freq="M")
    legacy = pd.Series(np.linspace(1000, 1350, 36), index=months)
    current = (legacy * 1.07).where(months >= pd.Period("2003-01", freq="M"))
    result = SeriesSplicer("interpolated").splice(legacy, current)
    print(result.link, result.values.head(3).round(1).to_dict())


if __name__ == "__main__":
    example_synthetic_two_indicators()
    example_configuration_and_automation()
    example_legacy_long_format()
    example_without_indicator_and_padding()
    example_constraints_rates_and_systems()
    example_splicing()
