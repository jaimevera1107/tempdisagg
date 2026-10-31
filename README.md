<!--
  tempdisagg — project README
  Designed for GitHub's Markdown renderer. Detailed methodology remains in /docs.
  Visual asset: docs/images/readme_banner.svg
-->

<div align="center">

# tempdisagg

<p align="center">
  <a href="./docs/images/Tempdisagg Temporal Series Logo.png">
    <img
      src="./docs/images/Tempdisagg Temporal Series Logo.png"
      alt="tempdisagg logo"
      width="320"
    />
  </a>
</p>

<p align="center">
  <sub><sup><em>Logo generated with ChatGPT (OpenAI).</em></sup></sub>
</p>

**Temporal disaggregation of time series, in Python.**

Turn annual or quarterly observations into a finer time series, while preserving the published benchmarks.

[![PyPI](https://img.shields.io/pypi/v/tempdisagg?style=flat-square&label=PyPI&labelColor=15243B&color=29A88F)](https://pypi.org/project/tempdisagg/)
![Python](https://img.shields.io/badge/Python-3.10%2B-15243B?style=flat-square&labelColor=15243B&color=44648E)
[![License](https://img.shields.io/badge/License-MIT-15243B?style=flat-square&labelColor=15243B&color=44648E)](LICENSE)
[![R reference](https://img.shields.io/badge/R%20reference-validated-15243B?style=flat-square&labelColor=15243B&color=29A88F)](#validation-against-r)

[**Get started**](#get-started) · [**Explore methods**](#choose-a-method) · [**Use your data**](#bring-your-own-data) · [**Documentation**](#documentation)

</div>




## Get started

**Install**

```bash
pip install tempdisagg
```

Need charts? Install the optional plotting dependencies with `pip install "tempdisagg[plot]"`.

**Your first estimate**

```python
import tempdisagg as td
from tempdisagg.datasets import load_swisspharma

sales, trade = load_swisspharma()  # annual target, quarterly indicators

result = td.TemporalDisaggregator("chow-lin-maxlog").fit(
    sales, trade[["exports"]]
)

quarterly = result.values
print(result.summary())
```

You now have quarterly estimates and a fitted result with model statistics. You can also call `result.prediction_interval(0.95)`.

**No indicator available?**

```python
quarterly = td.disaggregate(sales, method="denton-cholette", high_freq="Q")
```

<p align="center">
  <img src="docs/images/hero.png" alt="Example of annual sales disaggregated to quarterly estimates" width="92%">
</p>

## What you can build

| **Econometric disaggregation** | **Benchmark-aware workflows** |
|:---|:---|
| Chow–Lin, Litterman, Fernández, OLS, Denton and more. | Known values, aggregation constraints and reconciliation across series. |
| **Model evaluation** | **Real-world data handling** |
| Backtesting, diagnostics, coefficients and uncertainty estimates. | Incomplete periods, padding, non-negative reconciliation and series splicing. |

<details>
<summary><b>See the full feature map</b></summary>

<br>

| Need | API |
|---|---|
| Preserve known subperiods or other benchmark constraints | `td.known_values(...)`, `td.aggregate_constraints(...)` |
| Reconcile several related series | `td.CrossSectionalReconciler` |
| Disaggregate rates and shares | `td.disaggregate_ratio`, `aggregation_weights=` |
| Compare methods and indicator sets | `td.backtest` |
| Represent level shifts or isolated shocks | `td.level_shift`, `td.impulse` |
| Handle incomplete periods or extrapolated ends | `td.CompletionSettings`, `td.PaddingSettings` |
| Enforce non-negative results | `td.NonNegativeReconciler` |
| Join series across methodological breaks | `td.SeriesSplicer` |

Conflicting constraints are rejected; preprocessing choices and warnings are exposed rather than hidden.

</details>

## Choose a method

Different models make different assumptions about how the high-frequency path behaves.

| Family | When to consider it | Method |
|---|---|---|
| **Chow–Lin** | Regression on indicators with AR(1) residuals | `chow-lin-maxlog` **(default)** |
| **Litterman** | Regression with a more persistent error process | `litterman-maxlog` |
| **Fernández** | Regression with random-walk residuals | `fernandez` |
| **Denton / Denton–Cholette** | Match benchmarks while controlling movements | `denton`, `denton-cholette` |
| **OLS** | Regression without serially correlated residuals | `ols` |
| **Uniform** | Even distribution without an indicator | `uniform` |
| **Dynamic Chow–Lin** | Include a lagged target term; experimental | `dynamic-maxlog` |

<details>
<summary><b>Show all method presets and configuration examples</b></summary>

<br>

| Family | Available presets |
|---|---|
| Chow–Lin | `chow-lin-maxlog`, `chow-lin-minrss-ecotrim`, `chow-lin-minrss-quilis`, `chow-lin-fixed`, `fast` |
| Litterman | `litterman-maxlog`, `litterman-minrss`, `litterman-fixed` |
| Fernández | `fernandez` |
| Dynamic Chow–Lin | `dynamic-maxlog`, `dynamic-minrss`, `dynamic-fixed` |
| OLS | `ols` |
| Denton | `denton`, `denton-cholette` |
| Uniform | `uniform` |

```python
td.TemporalDisaggregator("fernandez", include_intercept=False)
td.TemporalDisaggregator("chow-lin-maxlog", truncated_rho=-1)
td.TemporalDisaggregator("denton-cholette", criterion="additive", h=2)
```

See [configuration](docs/configuration.md) for intercepts, rho bounds, Denton criteria and more.

</details>

## Validation against R

The Python implementation is checked against the R [`tempdisagg`](https://cran.r-project.org/package=tempdisagg) reference package (Sax & Steiner, 2013).

| **29** reference cases | **4** aggregation rules | **R 1.2.0** reference package |
|:---:|:---:|:---:|
| Multiple methods, options and edge cases | Sum, average, first and last | Reproducible reference fixtures |

<p align="center">
  <img src="docs/images/r_parity_series.png" alt="Comparison of Python and R disaggregated time series" width="90%">
</p>

<details>
<summary><b>See numerical agreement and reproducibility details</b></summary>

<br>

Reported predictions, coefficients, standard errors and statistics agree to approximately `1e-13`–`3e-9` for the comparable reference settings. Optimized rho values may differ slightly because the numerical optimizers stop at different points; refitting at R's rho recovers approximately `1e-13` agreement.

<p align="center">
  <img src="docs/images/r_parity_summary.png" alt="Summary of numerical differences between Python and R" width="86%">
</p>

The reference data are committed under `tests/reference/` and checked with `pytest`. To regenerate:

```bash
Rscript tests/reference/generate_r_reference.R
```

</details>

## Bring your own data

You provide two ingredients: **`y_low`**, the observed low-frequency series, and **`indicators`**, optional high-frequency signals. The output is an estimated series at the finer frequency.

| Input | Example | Role |
|---|---|---|
| `y_low` | Annual sales: `200` in 2016 | Benchmark that must be reproduced |
| `indicators` | Quarterly exports or activity | Guides how the `200` is allocated |
| `result.values` | Four quarterly estimates | They add up to `200` with `conversion="sum"` |

**Recommended:** a `pandas.Series` with annual `PeriodIndex` for `y_low`, and a quarterly `Series` or `DataFrame` with `PeriodIndex` for indicators.

<details>
<summary><b>Example A — Pandas with calendar dates</b> · recommended</summary>

<br>

```python
import numpy as np
import pandas as pd
import tempdisagg as td

years = pd.period_range("2016", periods=8, freq="Y")
quarters = pd.period_range("2016Q1", periods=32, freq="Q")

y_low = pd.Series(
    [200, 230, 260, 280, 315, 345, 380, 415],
    index=years,
)
indicators = pd.DataFrame(
    {"activity": np.linspace(30, 110, 32)},
    index=quarters,
)

result = td.TemporalDisaggregator("chow-lin-maxlog").fit(y_low, indicators)
estimates = result.values

annual_check = estimates.groupby(estimates.index.asfreq("Y")).sum()
assert np.allclose(annual_check.to_numpy(), y_low.to_numpy())
```

Use multiple columns for multiple indicators. `DatetimeIndex` also works when pandas can infer a frequency.

</details>

<details>
<summary><b>Example B — NumPy arrays</b> · fixed frequency ratio</summary>

<br>

If your observations are already ordered chronologically:

```python
result = td.TemporalDisaggregator("chow-lin-maxlog").fit_arrays(
    y_low.to_numpy(),
    indicators.to_numpy(),
    frequency_ratio=4,
)
```

`frequency_ratio=4` means four quarterly values per annual observation. For calendar-variable conversions such as month to day, prefer pandas indexes.

</details>

<details>
<summary><b>Example C — Long DataFrame</b> · legacy-compatible format</summary>

<br>

Each row is a high-frequency period; the annual target is repeated across its quarters.

| Index | Grain | y | activity |
|---|---:|---:|---:|
| 2016 | 1 | 200 | 30.0 |
| 2016 | 2 | 200 | 32.6 |
| 2016 | 3 | 200 | 35.2 |
| 2016 | 4 | 200 | 37.7 |

```python
frame = pd.DataFrame({
    "Index": np.repeat(years.year, 4),
    "Grain": np.tile([1, 2, 3, 4], len(years)),
    "y": np.repeat(y_low.to_numpy(), 4),
    "activity": indicators["activity"].to_numpy(),
})

result = td.TemporalDisaggregator("chow-lin-maxlog").fit_long_format(
    frame, x_cols=["activity"]
)
```

The `y` field represents the **annual** target, not the quarterly values. Inconsistent targets within one annual group are rejected.

</details>

<details>
<summary><b>Data rules and common pitfalls</b></summary>

<br>

| Topic | What matters |
|---|---|
| Aggregation | `conversion="sum"` is the default. Also supports `"average"`, `"first"`, `"last"`. |
| Dates | Time indexes must be unique, sorted and have an identifiable frequency. |
| Values | Targets and indicators must be numeric; non-finite or duplicated entries are validated. |
| Multicollinearity | Regression methods require an identifiable aggregated indicator design. |
| Missing target | `NaN` in `y_low` means unconstrained, not zero. |
| Missing indicators | With `completion=True` (default), documented completion rules may fill missing values and record them in `result.preprocessing`. Use `completion=False` to reject them. |
| Unconstrained extensions | Indicators beyond the last observed target can generate estimates, but these future periods have no observed benchmark. |
| Output types | `td.disaggregate(...)` returns a `pd.Series`; `TemporalDisaggregator(...).fit(...)` returns a result with diagnostics. |

Read the [configuration guide](docs/configuration.md) for detailed input and model options.

</details>

## Documentation

| Learn | Go to |
|---|---|
| **End-to-end examples** | [Example notebook](examples/tempdisagg_examples.ipynb) |
| **Statistical methods** | [Methodology and equations](docs/methodology.md) |
| **Model outputs** | [Statistics and diagnostics](docs/statistics.md) |
| **API settings** | [Configuration](docs/configuration.md) |
| **Under the hood** | [Architecture](docs/architecture.md) · [Testing](docs/testing.md) |
| **Upgrading** | [Migration guide](MIGRATION.md) · [Changelog](CHANGELOG.md) |

Bundled datasets include `load_swisspharma()` (Swiss industry annual sales and quarterly foreign trade indicators) and `load_denton1971()` (Denton's textbook example).

<details>
<summary><b>Citation</b></summary>

<br>

If this package supports your research, you can cite:

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

For the underlying methods and R reference implementation, also cite Sax and Steiner (2013), *Temporal Disaggregation of Time Series*, **The R Journal**, 5(2), 80–88.

</details>

---

<div align="center">

**Open source under the [MIT License](LICENSE).**

Built and maintained by [Jaime Vera-Jaramillo](https://github.com/jaimevera1107) · [Report an issue](https://github.com/jaimevera1107/tempdisagg/issues)

<sub>The `swisspharma` dataset originates from the R package and Swiss official statistics.</sub>

</div>
