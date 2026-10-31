"""Tests for input alignment (no implicit imputation, calendar awareness)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tempdisagg import DataAligner, InputValidationError
from tests.conftest import make_quarterly_problem


def test_series_alignment_shapes() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=10, extra_quarters=3)
    data = DataAligner("sum").from_series(annual, indicators)
    assert data.aggregation_matrix.shape == (10, 43)
    assert data.indicators.shape == (43, 2)
    # Extrapolated quarters belong to no group.
    assert not data.aggregation_matrix[:, -3:].any()


def test_monthly_to_daily_uses_the_calendar() -> None:
    days = pd.period_range("2024-01-01", "2024-03-31", freq="D")
    monthly = pd.Series([31.0, 29.0, 31.0], index=pd.period_range("2024-01", periods=3, freq="M"))
    data = DataAligner("sum").from_series(monthly, pd.Series(1.0, index=days))
    np.testing.assert_array_equal(data.aggregation_matrix.sum(axis=1), [31, 29, 31])


def test_strict_mode_rejects_incomplete_constrained_period() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=10)
    with pytest.raises(InputValidationError, match="incomplete"):
        DataAligner("sum", completion=False).from_series(annual, indicators.iloc[1:])


def test_strict_mode_rejects_missing_indicator_values() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=10)
    indicators.iloc[5, 0] = np.nan
    with pytest.raises(InputValidationError, match="missing"):
        DataAligner("sum", completion=False).from_series(annual, indicators)


def test_missing_target_becomes_unconstrained() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=10)
    annual.iloc[4] = np.nan
    with pytest.warns(UserWarning, match="no target value"):
        data = DataAligner("sum").from_series(annual, indicators)
    assert data.n_low == 9
    assert not data.aggregation_matrix[:, 16:20].any()


def test_strict_mode_rejects_indicator_gaps() -> None:
    annual, indicators, _ = make_quarterly_problem(n_years=10)
    with pytest.raises(InputValidationError, match="incomplete"):
        DataAligner("sum", completion=False).from_series(
            annual, indicators.drop(indicators.index[7]))


def test_long_format_completes_incomplete_constrained_group() -> None:
    frame = pd.DataFrame(
        {
            "Index": [2000] * 4 + [2001] * 4 + [2002] * 4 + [2003] * 2,
            "Grain": [1, 2, 3, 4] * 3 + [1, 2],
            "y": [10.0] * 4 + [12.0] * 4 + [11.0] * 4 + [13.0] * 2,
            "X": np.arange(14, dtype=float) + 1,
        }
    )
    with pytest.raises(InputValidationError, match="incomplete"):
        DataAligner("sum", completion=False).from_long_format(frame, x_cols=["X"])
    with pytest.warns(UserWarning):
        data = DataAligner("sum").from_long_format(frame, x_cols=["X"])
    assert data.n_low == 4 and data.n_high == 16 and data.artificial_high.sum() == 2


def test_long_format_rejects_inconsistent_targets() -> None:
    frame = pd.DataFrame(
        {
            "Index": [2000] * 2 + [2001] * 2 + [2002] * 2,
            "Grain": [1, 2] * 3,
            "y": [1.0, 2.0, 3.0, 3.0, 4.0, 4.0],
            "X": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )
    with pytest.raises(InputValidationError, match="inconsistent"):
        DataAligner("sum").from_long_format(frame, x_cols=["X"])


def test_arrays_require_ratio_of_at_least_two() -> None:
    with pytest.raises(InputValidationError):
        DataAligner("sum").from_arrays([1.0, 2.0, 3.0], np.ones(3), frequency_ratio=1)
