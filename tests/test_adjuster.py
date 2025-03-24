from tempdisagg import TempDisaggModel

def test_adjust_output(sample_df):
    model = TempDisaggModel(method="ols", conversion="average")
    model.fit(sample_df)
    adjusted_df = model.adjust_output(sample_df)
    assert "y_hat" in adjusted_df.columns
    assert (adjusted_df["y_hat"] >= 0).all()
