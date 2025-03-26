import numpy as np
import pandas as pd
import statsmodels.api as sm
import pytest
from tempdisagg import TempDisaggModel

@pytest.fixture(scope="module")
def disagg_data():
    macro = sm.datasets.macrodata.load_pandas().data
    macro["Index"] = macro["year"].astype(int)
    macro["Grain"] = macro["quarter"].astype(int)
    macro["X"] = macro["realcons"]

    col_name = "realgdp"
    gdp_annual = macro.groupby("Index")[col_name].mean().reset_index()
    gdp_annual.columns = ["Index", "y"]

    df = macro.merge(gdp_annual, on="Index", how="left")[["Index", "Grain", "y", "X"]].copy()
    return df

# ---- Individual Model Tests ----

def test_individual_predict_truncated(disagg_data):
    model = TempDisaggModel(method="chow-lin-opt", conversion="average")
    model.fit(disagg_data)
    y_hat = model.predict(full=False)
    assert model._df.shape[0] > disagg_data.shape[0]
    assert y_hat.shape[0] < model._df.shape[0]
    assert y_hat.shape[1] == 1

def test_individual_predict_padded(disagg_data):
    model = TempDisaggModel(method="chow-lin-opt", conversion="average")
    model.fit(disagg_data)
    y_hat = model.predict(full=True)
    assert y_hat.shape == (model._df.shape[0], 1)

def test_individual_adjust_padded(disagg_data):
    model = TempDisaggModel(method="chow-lin-opt", conversion="average")
    model.fit(disagg_data)
    adjusted = model.adjust_output(full=True)
    assert adjusted.shape == (model._df.shape[0], 1)

def test_individual_adjust_truncated(disagg_data):
    model = TempDisaggModel(method="chow-lin-opt", conversion="average")
    model.fit(disagg_data)
    adjusted = model.adjust_output(full=False)
    assert adjusted.shape[0] < model._df.shape[0]
    assert adjusted.shape[1] == 1

# ---- Ensemble Model Tests ----

def test_ensemble_predict_padded(disagg_data):
    model = TempDisaggModel(method="ensemble", conversion="average")
    model.fit(disagg_data, methods=["chow-lin", "litterman"])
    y_hat = model.predict(full=True)
    assert y_hat.shape == (model._df.shape[0], 1)

def test_ensemble_predict_truncated(disagg_data):
    model = TempDisaggModel(method="ensemble", conversion="average")
    model.fit(disagg_data, methods=["chow-lin", "litterman"])
    y_hat = model.predict(full=False)
    assert y_hat.shape[0] < model._df.shape[0]
    assert y_hat.shape[1] == 1

def test_ensemble_adjust_padded(disagg_data):
    model = TempDisaggModel(method="ensemble", conversion="average")
    model.fit(disagg_data, methods=["chow-lin", "litterman"])
    adjusted = model.adjust_output(full=True)
    assert adjusted.shape == (model._df.shape[0], 1)

def test_ensemble_adjust_truncated(disagg_data):
    model = TempDisaggModel(method="ensemble", conversion="average")
    model.fit(disagg_data, methods=["chow-lin", "litterman"])
    adjusted = model.adjust_output(full=False)
    assert adjusted.shape[0] < model._df.shape[0]
    assert adjusted.shape[1] == 1

def test_ensemble_adjust_truncated_no_methods(disagg_data):
    model = TempDisaggModel(method="ensemble", conversion="average")
    model.fit(disagg_data)
    adjusted = model.adjust_output(full=False)
    model.summary()
    model.plot()
    assert adjusted.shape[0] < model._df.shape[0]
    assert adjusted.shape[1] == 1
