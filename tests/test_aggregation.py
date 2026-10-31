"""Tests for the aggregation matrix and temporal aggregation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tempdisagg import (
    Conversion,
    ExactFitWarning,
    InputValidationError,
    TemporalDisaggregator,
    temporal_aggregate,
)
from tempdisagg.core.aggregation import UNCONSTRAINED_GROUP, build_aggregation_matrix


@pytest.mark.parametrize(
    "conversion, expected_row",
    [
        ("sum", [1, 1, 1]),
        ("average", [1 / 3, 1 / 3, 1 / 3]),
        ("first", [1, 0, 0]),
        ("last", [0, 0, 1]),
    ],
)
def test_matrix_rows_follow_conversion(conversion: str, expected_row: list[float]) -> None:
    codes = np.array([0, 0, 0, 1, 1, 1])
    matrix = build_aggregation_matrix(codes, 2, conversion)
    np.testing.assert_allclose(matrix[0, :3], expected_row)
    np.testing.assert_allclose(matrix[1, 3:], expected_row)
    assert np.all(matrix[0, 3:] == 0)


def test_unconstrained_columns_are_zero() -> None:
    codes = np.array([UNCONSTRAINED_GROUP, 0, 0, 1, 1, UNCONSTRAINED_GROUP])
    matrix = build_aggregation_matrix(codes, 2, Conversion.SUM)
    assert not matrix[:, 0].any() and not matrix[:, -1].any()


def test_unequal_group_sizes() -> None:
    matrix = build_aggregation_matrix(np.array([0, 0, 1, 1, 1]), 2, "average")
    np.testing.assert_allclose(matrix.sum(axis=1), [1.0, 1.0])
    np.testing.assert_allclose(matrix[1, 2:], [1 / 3] * 3)


def test_empty_group_is_rejected() -> None:
    with pytest.raises(InputValidationError):
        build_aggregation_matrix(np.array([0, 0, 2, 2]), 3, "sum")


def test_unknown_conversion_is_rejected() -> None:
    with pytest.raises(InputValidationError):
        Conversion.parse("median")


@pytest.mark.parametrize("conversion", ["sum", "average", "first", "last"])
def test_temporal_aggregate_inverts_disaggregation(conversion: str) -> None:
    index = pd.period_range("2000Q1", periods=40, freq="Q")
    indicator = pd.Series(np.linspace(10, 50, 40) + np.sin(np.arange(40)), index=index)
    annual = temporal_aggregate(indicator * 1.3 + 2.0, "Y", conversion)
    with pytest.warns(ExactFitWarning):  # noise-free data: an exact fit is expected
        result = TemporalDisaggregator("chow-lin-maxlog", conversion=conversion).fit(
            annual, indicator)
    recovered = temporal_aggregate(result.values, "Y", conversion)
    np.testing.assert_allclose(recovered.to_numpy(), annual.to_numpy(), rtol=1e-9)


def test_temporal_aggregate_drops_incomplete_periods() -> None:
    index = pd.period_range("2000Q1", periods=10, freq="Q")
    annual = temporal_aggregate(pd.Series(1.0, index=index), "Y")
    assert list(annual.index.year) == [2000, 2001]
