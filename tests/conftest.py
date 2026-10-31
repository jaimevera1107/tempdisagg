"""Shared fixtures and data generators for the test-suite."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

#: Seed used by every synthetic data generator (reproducibility).
SEED: int = 20261030


def make_quarterly_problem(
    n_years: int = 30,
    n_indicators: int = 2,
    extra_quarters: int = 0,
    seed: int = SEED,
) -> tuple[pd.Series, pd.DataFrame, pd.Series]:
    """Return (annual target, quarterly indicators, true quarterly series)."""
    rng = np.random.default_rng(seed)
    n_quarters = 4 * n_years + extra_quarters
    index = pd.period_range("1990Q1", periods=n_quarters, freq="Q")
    indicators = pd.DataFrame(
        {
            f"x{i + 1}": 100.0 + np.cumsum(rng.normal(0.5, 1.0, n_quarters))
            for i in range(n_indicators)
        },
        index=index,
    )
    betas = np.linspace(1.5, 0.5, n_indicators)
    noise = np.zeros(n_quarters)
    for t in range(1, n_quarters):
        noise[t] = 0.8 * noise[t - 1] + rng.normal(0.0, 0.5)
    truth = pd.Series(10.0 + indicators.to_numpy() @ betas + noise, index=index)
    annual = truth.iloc[: 4 * n_years].groupby(truth.index[: 4 * n_years].asfreq("Y")).sum()
    annual.index = pd.PeriodIndex(annual.index, freq="Y")
    return annual, indicators, truth


def quiet_fit(model, *args, **kwargs):
    """Fit while silencing expected user warnings (experimental methods)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return model.fit(*args, **kwargs)
