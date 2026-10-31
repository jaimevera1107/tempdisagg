"""Tests for the deprecated 0.2.x wrappers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tempdisagg import InputValidationError, Retropolarizer, TempDisaggModel


def _long_frame() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    years = np.repeat(np.arange(2000, 2015), 12)
    x = 50 + np.cumsum(rng.normal(0.3, 1.0, years.size))
    y = pd.Series(x * 2.0).groupby(years).transform("sum").to_numpy()
    return pd.DataFrame({"Index": years, "Grain": np.tile(np.arange(1, 13), 15), "y": y, "X": x})


def test_temp_disagg_model_still_works() -> None:
    frame = _long_frame()
    with pytest.warns(DeprecationWarning):
        model = TempDisaggModel(method="chow-lin-opt", conversion="sum").fit(frame)
    predictions = model.predict()
    assert predictions.shape == (len(frame), 1)
    assert model.result_.aggregation_error() < 1e-6


def test_retropolarizer_proportion_maps_to_ratio() -> None:
    new_values = [np.nan] * 5 + list(np.arange(6.0, 11.0) * 2)
    frame = pd.DataFrame({"old": np.arange(1.0, 11.0), "new": new_values})
    with pytest.warns(DeprecationWarning):
        completed = Retropolarizer(frame, new_col="new", old_col="old").retropolarize("proportion")
    np.testing.assert_allclose(completed.to_numpy(), np.arange(1.0, 11.0) * 2)


def test_removed_retropolarizer_methods_raise() -> None:
    frame = pd.DataFrame({"old": [1.0, 2.0, 3.0], "new": [np.nan, 4.0, 6.0]})
    with pytest.warns(DeprecationWarning):
        retro = Retropolarizer(frame, new_col="new", old_col="old")
    with pytest.raises(InputValidationError, match="removed"):
        retro.retropolarize("mlp_regression")
