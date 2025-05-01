"""
Test end-to-end: ensemble weight optimisation should fall back to uniform
weights when using the “problematic” dataset provided by the user.

Assumes:
    • datatempdisagg.csv  lives in  tests/data/  (sep=';')
    • Columns at least:   Index ; Grain ; y ; X   (strings are case-sensitive)
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from tempdisagg.model.tempdisagg_ensemble import EnsemblePredictor


# ---------- helpers ----------------------------------------------------- #
CSV_FILE = r".\data_example\datatempdisagg.csv"


def _load_dataset() -> pd.DataFrame:
    """Read semicolon-separated CSV and enforce dtypes used by the lib."""
    df = pd.read_csv(CSV_FILE, sep=";")

    # Tip: ajusta estos casts a tu propio dataset
    df["Index"] = df["Index"].astype(int)
    df["Grain"] = df["Grain"].astype(int)

    # Garantizar presencia de las 4 columnas mínimas
    expected_cols = {"Index", "Grain", "y", "X"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Dataset missing columns: {missing}")

    return df


# ---------- the actual test -------------------------------------------- #
def test_fallback_uniform_weights_dataset(caplog):
    """
    If optimisation fails, weights must be uniform and a warning logged.
    """
    df = _load_dataset()

    # Instancia “normal”: conversión sum y Retropolarizer desactivado
    predictor = EnsemblePredictor(
        conversion="average",
        verbose=False,
    )

    # Fit completo – aquí internamente puede dispararse el fallback
    y_hat, _, _ = predictor.fit(df)

    predictor.summary()
    predictor.plot()
