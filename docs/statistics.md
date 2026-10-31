# Statistics reported by a fit

All quantities live in immutable objects attached to `DisaggregationResult`.
Symbols follow `docs/methodology.md`; `ẽ = L⁻¹ û_l` are the whitened residuals
(`V = L L'`).

## `result.regression` — `RegressionStatistics` (GLS methods)

| Attribute | Formula | R (`td` object) | Notes |
|---|---|---|---|
| `coefficients` | `b̂` | `coefficients` | |
| `covariance` | `σ̂²_gls (X_l' V⁻¹ X_l)⁻¹` | — | |
| `standard_errors` | `√diag(covariance)` | `se` | |
| `t_values` | `b̂ / se` | summary | |
| `p_values` | `2 P(T_{n_l−k} > |t|)` | summary | Student t |
| `confidence_intervals(level)` | `b̂ ± t_{n_l−k} se` | — | |
| `rss` | `ẽ'ẽ = û_l' V⁻¹ û_l` | `rss` | weighted |
| `tss` | `‖ỹ − P_ι̃ ỹ‖²`, always centred on `ι̃ = L⁻¹ C 1` (with or without intercept) | `tss` | weighted |
| `sigma2_ml` | `rss / n_l` | `s_2` | ML estimator |
| `sigma2_gls` | `rss / (n_l − k)` | `s_2_gls` | unbiased; used for inference |
| `r_squared` | `1 − rss/tss` | `r.squared` | can be negative without an intercept |
| `adj_r_squared` | `1 − (1−R²)(n_l−1)/(n_l−k)` | `adj.r.squared` | |
| `log_likelihood` | concentrated Gaussian, see methodology §2.5 | `logl` | `nan` on exact fit |
| `aic`, `bic` | `log σ̂²_ml + 2k/n_l`, `log σ̂²_ml + k log(n_l)/n_l` | `aic`, `bic` | depend on the scale of `Σ`: compare only models with the same residual structure |
| `aic_loglik`, `bic_loglik` | `−2ℓ + 2p`, `−2ℓ + p log n_l`, `p = k + 1 (+1 if ρ estimated)` | — | include `log|V|`: comparable across residual structures |

**Scale of `σ²`, RSS and TSS.** These depend on how `Σ` is normalised. For
Chow-Lin methods the library reports them with the stationary covariance
`ρ^|i−j| / (1 − ρ²)`, as R does, whatever form was used to estimate `ρ`
(Ecotrim's minimum-RSS criterion uses the correlation form). Estimates,
standard errors and the log-likelihood do not depend on the normalisation.

All conventions in this table were verified against R 4.5.3 / tempdisagg
1.2.0 (`tests/test_r_parity.py`).
| `whitened_residuals` | `ẽ` | — | i.i.d. under a correct model |
| `residual_autocorrelation` | lag-1 autocorrelation of `ẽ` | — | ≈ 0 if the residual model is adequate |
| `n_artificial_obs` | padding targets included in `n_obs` | — | > 0 means inference counts artificial data |

`n_obs` counts every row of the constraint system: observed targets, padding
targets and additional constraints (known values and intermediate totals are
genuine exact observations of linear functionals of `y`). `result.residuals`
covers observed targets only; `result.data.row_kind` gives the origin of each
row.



## `result.rho` — `RhoEstimate`

| Attribute | Meaning |
|---|---|
| `value` | `ρ` used (`nan` if not identified) |
| `method` | `maxlog`, `minrss` or `fixed` |
| `estimated`, `truncated`, `identified` | flags |
| `unconstrained_value` | optimum before truncation |
| `at_bound` | optimum on the search boundary |
| `objective` | `−ℓ(ρ̂)` or `rss(ρ̂)` |
| `profile` | criterion on the grid (`pd.Series`, index `ρ`) |
| `local_minima` | grid locations of local optima |
| `n_evaluations` | criterion evaluations |

## `result.prediction_se` and `prediction_interval(level)`

`√diag Var(ŷ − y)` from methodology §2.7, and `ŷ ± t_{n_l−k} · se`. Conditional
on `ρ` and on Gaussian residuals.

## `result.benchmark` — `BenchmarkStatistics` (Denton)

`variant`, `criterion`, `h`, `penalty` (minimised criterion) and
`indicator_discrepancy = y_l − C x` (what the benchmark distributed).

## `result.diagnostics` — `NumericalDiagnostics`

| Attribute | Meaning |
|---|---|
| `aggregation_error` | `max |C ŷ − y_l|` |
| `condition_number` | of `V` (GLS) or the KKT system (Denton, 1-norm estimate) |
| `design_condition_number` | of the column-scaled whitened design (Belsley–Kuh–Welsch). Values above ~30 indicate collinearity that inflates standard errors; above 1e6 the library warns because ~6 digits are lost |
| `exact_fit` | zero low-frequency residuals |
| `warnings` | every warning raised during the fit, as text |

## `result.preprocessing` — `PreprocessingRecord`

| Attribute | Meaning |
|---|---|
| `completion`, `padding` | settings used |
| `added_periods` | high-frequency periods not present in the input (removed from `values`) |
| `filled_indicators` | mask of filled indicator values on returned periods |
| `n_filled_values` | number of such values |
| `artificial_targets` | padding targets (label → value) |

`result.full_values` contains the estimate on the whole grid, including
artificial periods; `result.to_frame()` flags filled indicator values.

## Evaluation objects

`BacktestResult`: `actual`, `predictions`, `errors`, `failures`, `warnings`
and `summary()` (origins, bias, MAE, RMSE, MAPE per candidate, sorted by RMSE).

`CrossSectionalResult`: `values`, `preliminary`, `adjustments`,
`temporal_error`, `contemporaneous_error`, `redundant_rows`.

`RatioResult`: `numerator`, `denominator`, `ratio`, `low_frequency_error`.

## Warnings

| Warning | When | What to do |
|---|---|---|
| `RhoBoundaryWarning` | `ρ̂` on the search boundary | Often a sign of a misspecified residual model (e.g. random-walk residuals fitted with Chow-Lin): try Litterman/Fernández |
| `RhoNotIdentifiedWarning` | flat criterion | Fix `ρ` or simplify the model |
| `MultipleOptimaWarning` | competing optima | Inspect `result.rho.profile` |
| `ExactFitWarning` | `û_l = 0` | Statistics are degenerate; the estimate is `X b̂` |
| `IllConditionedWarning` | near-collinear regressors or near-singular `V`/KKT | Drop or combine indicators |
| `IndicatorSignWarning` | proportional Denton with a sign-changing indicator | Use `criterion="additive"` |
| `IneffectiveParameterWarning` | a setting ignored by the configuration | Remove it |
| `ExperimentalMethodWarning` | dynamic methods | — |
| `ArtificialDataWarning` | completion or padding created values; regression statistics include padding targets | Review `result.preprocessing` |
| `RedundantConstraintWarning` | a consistent constraint implied by others was dropped | — |
| `MissingTargetWarning` | interior low-frequency periods without target | They are estimated, not constrained |
