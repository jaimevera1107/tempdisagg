from tempdisagg import TempDisaggModel

def test_fit_predict(sample_df):
    model = TempDisaggModel(method="ols", conversion="sum")
    model.fit(sample_df)
    y_hat = model.predict()
    assert y_hat.shape[0] == 8
    assert (y_hat >= 0).all()
