import pytest
import statsmodels.api as sm
import matplotlib.pyplot as plt
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

def test_summary_individual(disagg_data):
    model = TempDisaggModel(method="chow-lin-opt", conversion="average")
    model.fit(disagg_data)
    model.summary(metric="mae")

def test_summary_ensemble(disagg_data):
    model = TempDisaggModel(method="ensemble", conversion="average")
    model.fit(disagg_data, methods=["chow-lin", "litterman"])
    model.summary(metric="rmse")

def test_plot_individual(disagg_data):
    model = TempDisaggModel(method="chow-lin", conversion="average")
    model.fit(disagg_data)
    model.plot()

def test_plot_ensemble(disagg_data):
    model = TempDisaggModel(method="ensemble", conversion="average")
    model.fit(disagg_data, methods=["chow-lin", "litterman"])
    model.plot()
