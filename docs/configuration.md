# Configuring each method

Every method is a frozen dataclass in `tempdisagg.methods`. Its fields are its
settings; there is no hidden option. Three equivalent ways to configure it:

```python
from tempdisagg import TemporalDisaggregator, ChowLin, make_method

TemporalDisaggregator("chow-lin-maxlog", include_intercept=False)    # preset + overrides
TemporalDisaggregator(ChowLin(include_intercept=False))               # specification object
TemporalDisaggregator(make_method("chow-lin-maxlog", include_intercept=False))
```

`model.describe()` (or `spec.describe()`) lists every setting with its value,
meaning and R equivalent. The result keeps the exact specification used in
`result.method`.

A setting the method does not have raises `InputValidationError` with the list
of valid ones. A setting that exists but has no effect with the other choices
raises `IneffectiveParameterWarning`.

## Regression-based methods

| Setting | `ChowLin` | `Litterman` | `Fernandez` | `DynamicChowLin` | `OLS` | R |
|---|---|---|---|---|---|---|
| `include_intercept` | True | True | True | True | True | `y ~ x` / `y ~ 0 + x` |
| `rho_method` | `"maxlog"` | `"maxlog"` | — (ρ = 0) | `"maxlog"` | — | method suffix |
| `fixed_rho` | 0.5 | 0.5 | — | 0.5 | — | `fixed.rho` |
| `truncated_rho` | 0.0 | 0.0 | — | 0.0 | — | `truncated.rho` |
| `covariance_form` | `"correlation"` | — | — | — | — | `minrss-ecotrim` / `-quilis` |
| `include_truncation_remainder` | — | — | — | True | — | — |
| `optimizer` | `RhoOptimizerSettings()` | same | — | same | — | — |

`RhoOptimizerSettings(bounds=(-0.999, 0.999), grid_size=199, tolerance=1e-8)`
controls the numerical search of `ρ`.

**Does method X include an intercept?** Every regression method includes one
by default, as in R's formula `y ~ x`. Set `include_intercept=False` for
`y ~ 0 + x`. Denton and uniform estimate no regression, so they have no
intercept setting (asking for one is an error that says so).

## Benchmarking methods

| Setting | `Denton` | `Uniform` | R |
|---|---|---|---|
| `variant` | `"cholette"` | — | `denton-cholette` / `denton` |
| `criterion` | `"proportional"` | — | `criterion` |
| `h` | 1 | — | `h` |

## Presets (R names)

| Preset | Specification |
|---|---|
| `chow-lin-maxlog` | `ChowLin(rho_method="maxlog")` |
| `chow-lin-minrss-ecotrim` | `ChowLin(rho_method="minrss", covariance_form="correlation")` |
| `chow-lin-minrss-quilis` | `ChowLin(rho_method="minrss", covariance_form="covariance")` |
| `chow-lin-fixed` | `ChowLin(rho_method="fixed")` |
| `fast` | `ChowLin(rho_method="fixed", fixed_rho=0.99999)` |
| `litterman-maxlog` / `-minrss` / `-fixed` | `Litterman(rho_method=...)` |
| `fernandez` | `Fernandez()` |
| `dynamic-maxlog` / `-minrss` / `-fixed` | `DynamicChowLin(rho_method=...)` |
| `ols` | `OLS()` |
| `denton-cholette` | `Denton(variant="cholette")` |
| `denton` | `Denton(variant="original")` |
| `uniform` | `Uniform()` |

0.2.x names (`chow-lin-opt`, `litterman-opt`, ...) still resolve, with a
`DeprecationWarning`.

## Methods without indicators

| Method | What it does without an indicator |
|---|---|
| `uniform` | splits each total evenly (no extrapolation) |
| `denton-cholette` | smoothest series meeting the totals (Boot–Feibes–Lisman; `h=1` changes, `h=2` curvature) |
| `denton` | same with the original initial condition |
| `chow-lin-*`, `litterman-*`, `fernandez` | regression on a constant (R: `y ~ 1`); needs `include_intercept=True` |

No indicator column has to be given:

```python
disaggregate(y_annual, method="denton-cholette", high_freq="M")
TemporalDisaggregator("uniform").fit_long_format(df)          # x_cols omitted
TempDisaggModel(method="denton-cholette", X_col=None)         # legacy wrapper
```

## Completion and padding

```python
from tempdisagg import CompletionSettings, PaddingSettings

TemporalDisaggregator("chow-lin-maxlog")                                  # completion on
TemporalDisaggregator("chow-lin-maxlog", completion=False)                # strict mode
TemporalDisaggregator("chow-lin-maxlog",
                      completion=CompletionSettings(edge_method="constant"))
TemporalDisaggregator("chow-lin-maxlog", padding=1)                       # 1 period each side
TemporalDisaggregator("chow-lin-maxlog", padding=(0, 2))                  # 2 periods after only
TemporalDisaggregator("chow-lin-maxlog",
                      padding=PaddingSettings(before=1, after=1, target_method="repeat",
                                              indicator_method="linear", fit_window=4))
```

| `CompletionSettings` | default | meaning |
|---|---|---|
| `enabled` | True | complete the grid and fill indicators; False raises errors |
| `edge_method` | `"linear"` | `"linear"` or `"constant"` extension at the edges |
| `fit_window` | 3 | low-frequency periods used for the linear extension |
| `partial_periods` | `"complete"` | `"complete"`, `"partial"` (constraint on observed sub-periods only) or `"error"` |

| `PaddingSettings` | default | meaning |
|---|---|---|
| `before`, `after` | 0, 0 | low-frequency periods added around the observed targets |
| `indicator_method` | `"linear"` | extension of indicators into padded periods |
| `target_method` | `"linear"` | `"linear"`, `"repeat"` (0.2.x behaviour) or `"none"` |
| `fit_window` | 3 | low-frequency periods used for the linear extensions |

Padded and completed periods are removed from `result.values`
(`result.full_values` keeps them) and listed in `result.preprocessing`.

## Additional information and rates

```python
from tempdisagg import aggregate_constraints, known_values, LinearConstraint

constraints = (aggregate_constraints(quarterly_known, high_freq="M")   # some quarters
               + known_values(months_from_records))                    # some months
TemporalDisaggregator("chow-lin-maxlog").fit(y_annual, indicators, constraints=constraints)

# Annual rate = ratio of annual totals -> weighted average with the denominator
TemporalDisaggregator("chow-lin-maxlog", "average").fit(
    rate_annual, indicators, aggregation_weights=labour_force_monthly)

# Both totals known -> disaggregate numerator and denominator
disaggregate_ratio(informal_annual, employed_annual,
                   TemporalDisaggregator(), TemporalDisaggregator(),
                   informal_indicator, employed_indicator)

# Structural break
X = pd.concat([indicator, level_shift(indicator.index, "2020-04")], axis=1)
```

## Choosing a method and reconciling systems

```python
from tempdisagg import backtest, Candidate, CrossSectionalReconciler

backtest(y_annual, {"chow-lin": Candidate(TemporalDisaggregator(), X),
                    "fernandez": Candidate(TemporalDisaggregator("fernandez"), X)},
         holdout=4).summary()

CrossSectionalReconciler("proportional", h=1).reconcile_results(
    {"Antioquia": res_ant, "Bogota": res_bog, ...}, national_monthly)
```

## Two levels of use

```python
from tempdisagg import disaggregate, TemporalDisaggregator

# Automation: only the series.
series = disaggregate(y_annual, indicators, method="fernandez", include_intercept=False)

# Analysis: the full estimation record.
result = TemporalDisaggregator("fernandez", include_intercept=False).fit(y_annual, indicators)
result.regression.table()          # coefficients with inference
result.prediction_interval(0.9)    # pointwise intervals
result.diagnostics.warnings        # everything that was flagged
```
