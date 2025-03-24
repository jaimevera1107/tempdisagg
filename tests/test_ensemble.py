# tests/test_ensemble.py

from tempdisagg import TempDisaggModel

def test_ensemble_prediction(sample_df):
    model = TempDisaggModel(conversion="sum")
    model.ensemble(sample_df, methods=["ols", "chow-lin", "denton"])
    assert model.y_hat.shape[0] == 8
    assert hasattr(model, "weights_")
    assert abs(sum(model.weights_)-1) < 1e-6
