from tempdisagg.core.numeric_utils import NumericUtils
import numpy as np

def test_power_matrix():
    P = NumericUtils.power_matrix(3)
    assert P.shape == (3, 3)
    assert P[0, 2] == 2
