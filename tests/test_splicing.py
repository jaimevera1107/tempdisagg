"""Tests for series splicing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tempdisagg import InputValidationError, SeriesSplicer


def _series() -> tuple[pd.Series, pd.Series]:
    index = pd.period_range("2001-01", periods=24, freq="M")
    legacy = pd.Series(np.linspace(100, 146, 24), index=index)
    current = legacy * 1.1
    current.iloc[:12] = np.nan
    return legacy, current


def test_ratio_recovers_constant_factor() -> None:
    legacy, current = _series()
    result = SeriesSplicer("ratio").splice(legacy, current)
    np.testing.assert_allclose(result.values, legacy * 1.1)
    assert result.link["ratio"] == pytest.approx(1.1)
    assert result.filled.sum() == 12


def test_level_recovers_constant_shift() -> None:
    legacy, _ = _series()
    current = legacy + 5.0
    current.iloc[:10] = np.nan
    result = SeriesSplicer("level").splice(legacy, current)
    np.testing.assert_allclose(result.values, legacy + 5.0)


def test_affine_recovers_linear_link() -> None:
    legacy, _ = _series()
    current = 3.0 + 0.9 * legacy
    current.iloc[:8] = np.nan
    result = SeriesSplicer("affine").splice(legacy, current)
    np.testing.assert_allclose(result.values, 3.0 + 0.9 * legacy)


def test_interpolated_goes_from_anchor_to_link() -> None:
    legacy, current = _series()
    result = SeriesSplicer("interpolated").splice(legacy, current)
    assert result.values.iloc[0] == pytest.approx(legacy.iloc[0])          # anchor: unchanged
    assert result.values.iloc[12] == pytest.approx(current.iloc[12])       # link: current
    factors = (result.values / legacy).iloc[:13].to_numpy()
    assert np.all(np.diff(factors) > 0)                                     # gradual


def test_current_values_are_never_modified() -> None:
    legacy, current = _series()
    for method in ("level", "ratio", "affine", "interpolated"):
        result = SeriesSplicer(method).splice(legacy, current)
        observed = current.notna()
        np.testing.assert_array_equal(result.values[observed], current[observed])


def test_no_overlap_raises() -> None:
    legacy, current = _series()
    with pytest.raises(InputValidationError, match="overlap"):
        SeriesSplicer().splice(legacy.iloc[:6], current.iloc[12:])
