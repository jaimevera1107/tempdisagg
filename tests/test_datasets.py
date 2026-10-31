"""The bundled datasets load correctly and equal the data used for the R parity."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tempdisagg import TemporalDisaggregator
from tempdisagg.datasets import load_denton1971, load_swisspharma

REFERENCE = Path(__file__).parent / "reference"


def test_swisspharma_shapes_and_indexes() -> None:
    sales, trade = load_swisspharma()
    assert isinstance(sales.index, pd.PeriodIndex) and sales.index.freqstr.startswith("Y")
    assert (sales.index[0].year, sales.index[-1].year, len(sales)) == (1975, 2010, 36)
    assert list(trade.columns) == ["exports", "imports"]
    assert str(trade.index[0]) == "1972Q1" and str(trade.index[-1]) == "2011Q2"
    assert not sales.isna().any() and not trade.isna().any().any()


def test_swisspharma_matches_the_r_export() -> None:
    sales, trade = load_swisspharma()
    annual = pd.read_csv(REFERENCE / "swisspharma_annual.csv")
    quarterly = pd.read_csv(REFERENCE / "swisspharma_quarterly.csv")
    np.testing.assert_array_equal(sales.to_numpy(), annual["sales_a"].to_numpy())
    np.testing.assert_array_equal(trade["exports"].to_numpy(), quarterly["exports_q"].to_numpy())
    np.testing.assert_array_equal(trade["imports"].to_numpy(), quarterly["imports_q"].to_numpy())


def test_denton1971_example() -> None:
    totals, indicator = load_denton1971()
    assert totals.tolist() == [500.0, 400.0, 300.0, 400.0, 500.0]
    assert indicator.tolist()[:4] == [50.0, 100.0, 150.0, 100.0] and len(indicator) == 20
    result = TemporalDisaggregator("denton-cholette").fit_arrays(totals, indicator, 4)
    assert result.aggregation_error() < 1e-9
