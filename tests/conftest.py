# tests/conftest.py

import pytest
import numpy as np
import pandas as pd

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Index": [2020]*4 + [2021]*4,
        "Grain": [1, 2, 3, 4]*2,
        "y": [100] + [None]*3 + [120] + [None]*3,
        "X": np.linspace(1, 8, 8)
    })
