
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
    # Cargar y preparar datos simulando corte en el último trimestre
    data = sm.datasets.macrodata.load_pandas().data
    data['year'] = data['year'].astype(int)
    data['quarter'] = data['quarter'].astype(int)
    data_annual = data.groupby('year')['realgdp'].mean().reset_index()
    data_annual.columns = ['Index', 'y']
    data = data.merge(data_annual, left_on='year', right_on='Index', how='left')
    data.rename(columns={'realcons': 'X', 'quarter': 'Grain'}, inplace=True)
    df = data[['Index', 'Grain', 'y', 'X']].head(40)  # Cortar antes del cierre del año

    # Ajustar modelo con verbose para seguimiento
    model = TempDisaggModel(conversion="average", verbose=False)
    model.fit(df)

    # Verificar que se haya detectado padding hacia adelante
    assert model.n_pad_after > 0, "Expected padding after but got 0"
    assert model.df_.shape[0] > df.shape[0], "Completed DataFrame should be longer than original"
    assert model.y_hat.shape[0] == model.df_.shape[0], "Prediction length should match completed DataFrame"

    return {
        "n_pad_before": model.n_pad_before,
        "n_pad_after": model.n_pad_after,
        "original_length": df.shape[0],
        "completed_length": model.df_.shape[0],
        "y_hat_shape": model.y_hat.shape
    }

# Ejecutar manualmente el test y mostrar resultado
test_result = test_padding_detected_forward_completion()
test_result
