from tempdisagg.core.rho_optimizer import RhoOptimizer
import numpy as np

def test_optimize_rho_runs():
    C = np.array([[1, 1, 0, 0], [0, 0, 1, 1]])
    X = np.arange(4).reshape(-1, 1)
    y_l = np.array([[3], [7]])

    opt = RhoOptimizer(verbose=False)
    result = opt.optimize(y_l, X, C, method="maxlog")
    assert "rho" in result
    assert -0.9 <= result["rho"] <= 0.99
