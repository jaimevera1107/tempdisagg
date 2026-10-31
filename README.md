<h1 align="center">tempdisagg</h1>

<p align="center">
  <b>Temporal disaggregation of time series in Python</b><br>
  From annual to quarterly, monthly or daily — consistent with the totals, validated against R.
</p>

<p align="center">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-0b6e99">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-0b6e99">
  <img alt="R parity" src="https://img.shields.io/badge/R%20parity-tempdisagg%201.2.0-d9822b">
  <img alt="Status" src="https://img.shields.io/badge/status-beta%20(0.6.2)-7b8794">
</p>

<p align="center">
  <img src="docs/images/hero.png" alt="Quarterly sales estimated from annual totals" width="88%">
</p>

Official statistics often exist only once a year while analysis needs quarters or months.
`tempdisagg` estimates the high-frequency path from related indicators, and every result adds up
exactly to the published figures. It implements the methods of the R package
[`tempdisagg`](https://cran.r-project.org/package=tempdisagg) (Sax and Steiner, 2013) with the same
names and defaults, reproduces its results to machine precision, and adds what production work
usually needs: extra constraints, reconciliation of several series, backtesting and full
diagnostics.

## Installation

```bash
pip install tempdisagg            # numpy, pandas, scipy
pip install "tempdisagg[plot]"    # adds matplotlib for plotting
```

## Quick start

```python
import tempdisagg as td
from tempdisagg.datasets import load_swisspharma

sales, trade = load_swisspharma()          # annual sales, quarterly exports and imports

result = td.TemporalDisaggregator("chow-lin-maxlog").fit(sales, trade[["exports"]])
result.values                              # quarterly series
print(result.summary())                    # coefficients, rho, R², log-likelihood
result.prediction_interval(0.95)           # interval for each quarter

td.disaggregate(sales, method="denton-cholette", high_freq="Q")   # no indicator needed
```

## Methods

| Method | Idea | Presets |
|---|---|---|
| Chow-Lin | regression on the indicators, AR(1) residuals | `chow-lin-maxlog` (default), `chow-lin-minrss-ecotrim`, `chow-lin-minrss-quilis`, `chow-lin-fixed`, `fast` |
| Litterman | random-walk residuals with AR(1) changes | `litterman-maxlog`, `litterman-minrss`, `litterman-fixed` |
| Fernández | random-walk residuals | `fernandez` |
| Dynamic Chow-Lin | the target depends on its own lag (experimental) | `dynamic-maxlog`, `dynamic-minrss`, `dynamic-fixed` |
| OLS | regression without residual correlation | `ols` |
| Denton | adjusts one indicator to the totals, additive or proportional | `denton-cholette`, `denton` |
| Uniform | splits each total evenly | `uniform` |

Every method is configurable — intercept, rho criterion and bounds, truncation, Denton criterion and
order — and `describe()` lists each setting with its R equivalent:

```python
td.TemporalDisaggregator("fernandez", include_intercept=False)
td.TemporalDisaggregator("chow-lin-maxlog", truncated_rho=-1)
td.TemporalDisaggregator("denton-cholette", criterion="additive", h=2)
```

## Beyond a single series

| Need | Tool |
|---|---|
| Some quarters or months are known from another source | `constraints=td.known_values(...)` / `td.aggregate_constraints(...)` |
| Series that must add up to a total every period | `td.CrossSectionalReconciler` |
| Rates and shares (not additive) | `td.disaggregate_ratio`, or `aggregation_weights=` |
| Choosing between models and indicators | `td.backtest` |
| Structural breaks | `td.level_shift`, `td.impulse` as extra indicators |
| Incomplete years, padding of the ends | `td.CompletionSettings`, `td.PaddingSettings` |
| Non-negative output | `td.NonNegativeReconciler` |
| Methodological breaks between two versions of a series | `td.SeriesSplicer` |

Contradictory inputs are rejected with the values that clash; nothing is imputed silently; every
artificial value and every warning is recorded in the result.

## Validated against R

All 29 reference cases — every method, the four aggregation rules, with and without intercept,
Denton orders 0–2 and the Denton (1971) example — were run in R 4.5.3 with tempdisagg 1.2.0.
Predictions, coefficients, standard errors and every reported statistic agree to between 1e-13
and 3e-9. When rho is estimated, R's optimiser stops near 1e-4; the Python optimum is never worse
on the estimation criterion, and refitting at R's rho restores agreement to 1e-13.

<p align="center">
  <img src="docs/images/r_parity_series.png" alt="Python vs R estimates" width="92%">
</p>

<p align="center">
  <img src="docs/images/r_parity_summary.png" alt="Agreement with R per case" width="62%">
</p>

The reference files are committed in `tests/reference/`, so the comparison runs with every
`pytest`. To regenerate them: `Rscript tests/reference/generate_r_reference.R`.

## Examples and data

[`examples/tempdisagg_examples.ipynb`](examples/tempdisagg_examples.ipynb) goes through every method
with real data: Chow-Lin and its variants, Denton with and without an indicator, a test against
known quarters, backtesting, known values as constraints, reconciliation of exports and imports,
rates, incomplete years and padding.

The data ship with the package:

```python
from tempdisagg.datasets import load_swisspharma, load_denton1971
```

`load_swisspharma()` returns the Swiss chemical and pharmaceutical industry data distributed with
the R package (annual sales 1975–2010; quarterly exports and imports 1972–2011; sources: Swiss
Federal Statistical Office and Federal Customs Administration). `load_denton1971()` is the
textbook example of Denton (1971).

## Documentation

| | |
|---|---|
| [Methodology](docs/methodology.md) | formulas and derivations of every method |
| [Statistics](docs/statistics.md) | every number in a result, and its R equivalent |
| [Configuration](docs/configuration.md) | settings of each method |
| [Architecture](docs/architecture.md) | how the code is organised |
| [Testing](docs/testing.md) | what each test guarantees |
| [Migration](MIGRATION.md) | changes from 0.2.x |

## Citation

```bibtex
@misc{verajaramillo2025tempdisagg,
  title         = {tempdisagg: A Python Library for Temporal Disaggregation of Time Series},
  author        = {Jaime A. Vera-Jaramillo},
  year          = {2025},
  eprint        = {2503.22054},
  archivePrefix = {arXiv},
  primaryClass  = {econ.EM}
}
```

Please also cite Sax and Steiner (2013), *Temporal Disaggregation of Time Series*,
The R Journal 5(2), 80–88, for the methods and the reference implementation.

## License

MIT. The swisspharma data come from the R package tempdisagg and Swiss official statistics.
