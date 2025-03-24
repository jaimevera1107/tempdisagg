
import pytest
import pandas as pd
import numpy as np
import statsmodels.api as sm
from tempdisagg import TempDisaggModel

def test_fit_predict(sample_df):
    model = TempDisaggModel(method="ols", conversion="sum")
    model.fit(sample_df)
    y_hat = model.predict()
    assert y_hat.shape[0] == 8
    assert (y_hat >= 0).all()



def test_padding_detected_forward_completion():
    data = sm.datasets.macrodata.load_pandas().data
    data['year'] = data['year'].astype(int)
    data['quarter'] = data['quarter'].astype(int)
    data['date'] = pd.period_range(start="1959Q1", periods=len(data), freq='Q').to_timestamp()
    data['X'] = data['realcons']
    data['Index'] = data['year']
    data['Grain'] = data['quarter']
    annual = data.groupby('Index')['realgdp'].mean().reset_index()
    annual.columns = ['Index', 'y']
    df = data.merge(annual, on='Index', how='left')[['Index', 'Grain', 'y', 'X']].copy()
    # Forzar padding hacia adelante quitando el último Q4 conocido (2009Q4)
    df = df[~((df["Index"] == 2009) & (df["Grain"] == 4))]


    model = TempDisaggModel(conversion="average", verbose=False)
    model.fit(df)

    # Verificar que se haya detectado padding hacia adelante
    assert model.df_.shape[0] > df.shape[0]
    assert model.y_hat.shape[0] == model.df_.shape[0]
    assert model.n_pad_after > 0


    print({
    "n_pad_before": model.n_pad_before,
    "n_pad_after": model.n_pad_after,
    "original_length": df.shape[0],
    "completed_length": model.df_.shape[0],
    "y_hat_shape": model.y_hat.shape
})
