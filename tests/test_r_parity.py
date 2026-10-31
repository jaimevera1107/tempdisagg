"""Numerical parity with the R package tempdisagg.

Two layers:

1. **Published values** (always defined): coefficients, standard errors,
   t-values, adjusted R² and residual quantiles printed in the official
   vignette of the R package (``vignette("intro", "tempdisagg")``, also in
   Sax and Steiner, 2013, R Journal 5(2)). They need the swisspharma data.
2. **Full parity**: predictions and statistics for 29 cases written by
   ``tests/reference/generate_r_reference.R``.

Both layers need files produced by the R script. Without them the tests are
skipped, unless the environment variable ``TEMPDISAGG_REQUIRE_R_REFERENCE=1``
is set (as in the CI workflow), in which case missing files fail the run.
The Denton (1971) cases use data fully specified in R's ``?td`` help page.

Tolerances (validated with R 4.5.3, tempdisagg 1.2.0)
-----------------------------------------------------
* Fixed ``rho`` and methods without ``rho``: relative 1e-8 (observed
  agreement 1e-13 to 3e-9).
* Estimated ``rho``: R's ``optimize()`` stops at a tolerance of about 1e-4,
  so the two optima differ slightly. The tests require (i) ``|rho_py - rho_R|
  <= 1e-4``, (ii) that the Python optimum is at least as good as R's on the
  estimation criterion, and (iii) agreement to 1e-8 after re-fitting with
  ``rho`` fixed at R's value, which isolates the formulas from the optimiser.
* R-squared values near zero are compared in absolute terms (1e-8).
* ``denton_prop_h2``: R's solution violates its own annual constraint by
  about 2e-5 (ill-conditioned dense solve), ours by 1e-13; compared at 1e-6.
* ``lit_minrss``: the slope (about 4e-4, R-squared 3e-4) is so sensitive to
  rho that the 15 significant digits with which R writes rho limit the
  re-fit agreement to about 1e-8; compared at 1e-7.
"""

from __future__ import annotations

import dataclasses
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tempdisagg import RhoOptimizerSettings, TemporalDisaggregator
from tempdisagg.core.data import DataAligner
from tests.reference.r_cases import CASES, DENTON_1971_CASES

REFERENCE_DIR: Path = Path(__file__).parent / "reference"
REQUIRE_REFERENCE: bool = os.environ.get("TEMPDISAGG_REQUIRE_R_REFERENCE") == "1"
#: Tolerances (see module docstring).
STRICT_RTOL: float = 1e-8
ESTIMATED_RHO_RTOL: float = 5e-4
RHO_ATOL: float = 1e-4
R2_ATOL: float = 1e-8
CASE_RTOL: dict[str, float] = {"denton_prop_h2": 1e-6, "lit_minrss": 1e-7}
#: Statistics compared in absolute terms (they can be close to zero).
ABSOLUTE_STATISTICS: frozenset[str] = frozenset({"r.squared", "adj.r.squared"})

#: Statistics compared with R (R name -> attribute of RegressionStatistics).
COMPARED_STATISTICS: dict[str, str] = {
    "s_2": "sigma2_ml", "s_2_gls": "sigma2_gls", "rss": "rss", "tss": "tss",
    "r.squared": "r_squared", "adj.r.squared": "adj_r_squared",
    "logl": "log_likelihood", "aic": "aic", "bic": "bic",
}

def _require(path: Path) -> None:
    if not path.exists():
        message = f"R reference file {path.name} missing (run generate_r_reference.R)."
        if REQUIRE_REFERENCE:
            pytest.fail(message)
        pytest.skip(message)


def _quarter_index(years: pd.Series, quarters: pd.Series) -> pd.PeriodIndex:
    return pd.PeriodIndex(
        [pd.Period(year=int(y), quarter=int(q), freq="Q") for y, q in zip(years, quarters)]
    )


def _swisspharma() -> tuple[dict[str, pd.Series], pd.DataFrame]:
    _require(REFERENCE_DIR / "swisspharma_annual.csv")
    annual = pd.read_csv(REFERENCE_DIR / "swisspharma_annual.csv")
    quarterly = pd.read_csv(REFERENCE_DIR / "swisspharma_quarterly.csv")
    indicators = quarterly[["exports_q", "imports_q"]].set_index(
        _quarter_index(quarterly["year"], quarterly["quarter"]))
    sales = pd.Series(annual["sales_a"].to_numpy(),
                      index=pd.PeriodIndex(annual["year"].astype(str), freq="Y"))
    return {"sales": sales}, indicators


def _fit(target: pd.Series, indicators: pd.DataFrame, columns: list[str], preset: str,
         overrides: dict, conversion: str):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalDisaggregator(preset, conversion, **overrides)
        if columns:
            return model.fit(target, indicators[columns])
        return model.fit(target, high_freq="Q")


def _meta(case_id: str) -> pd.Series:
    meta = pd.read_csv(REFERENCE_DIR / "r_meta.csv")
    return meta[meta["id"] == case_id].set_index("key")["value"]


# ----------------------------------------------------------------------
# Layer 1: values published in the R vignette
# ----------------------------------------------------------------------
def test_vignette_m3_chow_lin_one_indicator() -> None:
    """``td(sales.a ~ exports.q)``: rho truncated to 0, 36 -> 158 observations."""
    targets, indicators = _swisspharma()
    result = _fit(targets["sales"], indicators, ["exports_q"], "chow-lin-maxlog", {}, "sum")
    assert (result.data.n_low, result.data.n_high) == (36, 158)
    reg = result.regression
    np.testing.assert_allclose(reg.coefficients, [12.41, 1.339e-02], rtol=5e-4)
    np.testing.assert_allclose(reg.standard_errors, [1.493, 1.672e-04], rtol=5e-4)
    np.testing.assert_allclose(reg.t_values, [8.311, 80.111], rtol=5e-4)
    assert reg.adj_r_squared == pytest.approx(0.9946, abs=5e-5)
    assert result.rho.value == 0.0 and result.rho.truncated
    quantiles = np.quantile(result.residuals, [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(quantiles, [-77.892, -7.711, -4.628, 9.647, 36.448], atol=1e-3)


def test_vignette_m3_without_truncation() -> None:
    """``td(sales.a ~ exports.q, truncated.rho = -1)``."""
    targets, indicators = _swisspharma()
    result = _fit(targets["sales"], indicators, ["exports_q"], "chow-lin-maxlog",
                  {"truncated_rho": -1}, "sum")
    np.testing.assert_allclose(result.coefficients, [12.31579, 0.01341], rtol=5e-4)


def test_vignette_m4_two_indicators() -> None:
    """``td(sales.a ~ exports.q + imports.q)``."""
    targets, indicators = _swisspharma()
    result = _fit(targets["sales"], indicators, ["exports_q", "imports_q"],
                  "chow-lin-maxlog", {}, "sum")
    reg = result.regression
    np.testing.assert_allclose(reg.coefficients, [11.685855, 0.011258, 0.003934], rtol=5e-4)
    np.testing.assert_allclose(reg.standard_errors, [1.493077, 0.001158, 0.002113], rtol=5e-4)
    np.testing.assert_allclose(reg.t_values, [7.827, 9.723, 1.862], rtol=5e-4)
    assert reg.adj_r_squared == pytest.approx(0.9949, abs=5e-5)
    assert result.rho.truncated
    quantiles = np.quantile(result.residuals, [0, 0.25, 0.5, 0.75, 1])
    np.testing.assert_allclose(quantiles, [-61.648, -7.139, -2.825, 5.508, 53.373], atol=1e-3)


# ----------------------------------------------------------------------
# Layer 2: full parity with generated references
# ----------------------------------------------------------------------
@pytest.mark.parametrize("case_id", sorted(CASES))
def test_matches_r(case_id: str) -> None:
    prediction_file = REFERENCE_DIR / f"r_{case_id}.csv"
    _require(prediction_file)
    targets, indicators = _swisspharma()
    target, columns, preset, overrides, conversion = CASES[case_id]
    result = _fit(targets[target], indicators, columns, preset, overrides, conversion)
    meta = _meta(case_id)
    rtol = CASE_RTOL.get(case_id, STRICT_RTOL)

    if result.rho is not None and result.rho.estimated:
        r_rho = float(meta["rho"])
        assert result.rho.value == pytest.approx(r_rho, abs=RHO_ATOL)
        assert result.rho.truncated == bool(meta.get("truncated", 0))
        _compare(result, case_id, meta, ESTIMATED_RHO_RTOL)
        if not result.rho.truncated:
            ours = result.rho.objective
            theirs = _criterion_at(result, columns, targets[target], indicators, r_rho)
            assert ours <= theirs + 1e-9 * max(1.0, abs(theirs))
        refit = _fit_at(result, columns, targets[target], indicators, r_rho)
        _compare(refit, case_id, meta, rtol)
    else:
        _compare(result, case_id, meta, rtol)


@pytest.mark.parametrize("case_id", sorted(DENTON_1971_CASES))
def test_denton_1971_example_matches_r(case_id: str) -> None:
    prediction_file = REFERENCE_DIR / f"r_{case_id}.csv"
    _require(prediction_file)
    quarterly = np.tile([50.0, 100.0, 150.0, 100.0], 5)
    data = DataAligner("sum").from_arrays([500.0, 400.0, 300.0, 400.0, 500.0], quarterly, 4)
    result = TemporalDisaggregator("denton", **DENTON_1971_CASES[case_id]).fit_data(data)
    expected = pd.read_csv(prediction_file)["value"].to_numpy()
    np.testing.assert_allclose(result.values.to_numpy(), expected, rtol=STRICT_RTOL)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _fit_at(result, columns: list[str], target: pd.Series, indicators: pd.DataFrame,
            rho: float):
    """Re-fit the same specification with rho fixed at ``rho``."""
    spec = dataclasses.replace(result.method, rho_method="fixed", fixed_rho=rho,
                               truncated_rho=0.0)
    model = TemporalDisaggregator(spec, result.data.conversion)
    return model.fit(target, indicators[columns]) if columns else model.fit(target, high_freq="Q")


def _criterion_at(result, columns: list[str], target: pd.Series, indicators: pd.DataFrame,
                  rho: float) -> float:
    """Estimation criterion of the same specification evaluated at ``rho``."""
    window = RhoOptimizerSettings(bounds=(rho - 1e-9, rho + 1e-9), grid_size=3)
    spec = dataclasses.replace(result.method, truncated_rho=-1.0, optimizer=window)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalDisaggregator(spec, result.data.conversion)
        fitted = model.fit(target, indicators[columns])
    return float(fitted.rho.objective)


def _compare(result, case_id: str, meta: pd.Series, rtol: float) -> None:
    expected = pd.read_csv(REFERENCE_DIR / f"r_{case_id}.csv")
    expected_index = _quarter_index(expected["year"], expected["quarter"])
    np.testing.assert_allclose(result.values.reindex(expected_index).to_numpy(),
                               expected["value"].to_numpy(), rtol=rtol, err_msg="prediction")
    if result.regression is None:
        return
    reg = result.regression
    r_coefficients = meta[meta.index.str.startswith("coef_")].to_numpy()
    r_errors = meta[meta.index.str.startswith("se_")].to_numpy()
    np.testing.assert_allclose(reg.coefficients.to_numpy(), r_coefficients, rtol=rtol,
                               err_msg="coefficients")
    np.testing.assert_allclose(reg.standard_errors.to_numpy(), r_errors, rtol=rtol,
                               err_msg="standard errors")
    for r_name, attribute in COMPARED_STATISTICS.items():
        value = meta.get(r_name, np.nan)
        if np.isnan(value):
            continue
        if r_name in ABSOLUTE_STATISTICS:
            # Absolute scale; loosened together with rtol before the rho re-fit.
            atol = R2_ATOL if rtol <= CASE_RTOL.get(case_id, STRICT_RTOL) * 100 else rtol
            assert getattr(reg, attribute) == pytest.approx(value, abs=atol), r_name
        else:
            assert getattr(reg, attribute) == pytest.approx(value, rel=rtol), r_name
