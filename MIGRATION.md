# Migrating from tempdisagg 0.2.x

Versions 0.3 and later are a rewrite (0.3.0 was the rewrite; 0.4.0 added
per-method configuration and the full statistical record). Results change wherever 0.2.x was methodologically
incorrect; the list below documents each defect and how it was verified.

## Defects fixed

| # | Component (0.2.13) | Defect | Evidence | 0.3+ |
|---|---|---|---|---|
| 1 | `RhoOptimizer` (`maxlog`) | Log-likelihood did not concentrate out σ², so ρ̂ depended on the **units** of the data | Same data ×1000: ρ̂ went from 0.54 to 0.99 | Concentrated likelihood; ρ̂ invariant (tested) |
| 2 | `RhoOptimizer` | Standard errors ignored σ² (`sqrt(diag((X'V⁻¹X)⁻¹))`); p-values from the normal distribution | Code review | `σ̂²(X'V⁻¹X)⁻¹`, Student-t p-values |
| 3 | `litterman_estimation` | Used `H = I − ρL` only: an AR(1) process, not Litterman's random walk with AR(1) innovations | Code review | `Σ = (Δ'H'HΔ)⁻¹`; Fernández ≡ Litterman(ρ=0) tested |
| 4 | `litterman_opt_estimation` | ρ estimated under the **Chow-Lin** covariance, then plugged into Litterman | Code review | ρ estimated under the Litterman covariance |
| 5 | `denton_estimation` | Estimated a GLS β with `Σ = (D'D)⁻¹`: for h = 1 it was identical to `fernandez` | Max. difference 2e-13 | True benchmarking of the indicator (additive/proportional, order h) |
| 6 | `denton_cholette_estimation` | Added `W = I` to the penalty (issue #4) and ignored the indicator `X` | Issue #4: ‖Dy‖² 35.81 vs optimum 29.45 | Exact KKT solution, uses the indicator |
| 7 | `chow-lin-ecotrim`, `chow-lin-quilis` | Fixed ρ = 0.75 / 0.15. In R these are two **minimum-RSS estimators** of ρ | R documentation | `chow-lin-minrss-ecotrim` / `-quilis` estimate ρ |
| 8 | `fast` | Litterman with ρ = 0.9 (README said "approximation of Denton-Cholette") | Code vs README | Chow-Lin with ρ = 0.99999, as in R |
| 9 | `ols_estimation` | Did not distribute residuals: violated `C y = y_l` | Code review | Residuals spread uniformly; constraint holds |
| 10 | `preprocess_inputs` | `X.reshape(-1, 1)`: a multi-column X was silently flattened | Code review | Any number of indicators |
| 11 | All estimators | `pinv` everywhere; exceptions swallowed and replaced by a **silent fallback** method | Code review | Cholesky/QR/KKT solves; explicit typed errors, no fallback |
| 12 | `TimeSeriesCompleter` | Missing **targets** were interpolated / forward-filled (fabricated annual values); incomplete edge years were padded | Code review; R comparison in the paper | No imputation: unknown targets are unconstrained, extra indicator periods are extrapolated |
| 13 | `ModelFitter` + `Retropolarizer` | `use_retropolarizer=True` crashed (`verbose` argument not accepted), and the error was masked | Reproduced: `TypeError` | Removed; splicing is a separate tool |
| 14 | `Retropolarizer` | Interior gaps linearly interpolated **before** applying the chosen method | Code review | `SeriesSplicer` never alters observed values |
| 15 | `EnsemblePrediction` | Weights minimise ‖C Σwᵢŷᵢ − y_l‖², which is zero for any combination of constrained methods: weights not identified | Algebra | Removed |
| 16 | `PostEstimation` (`average`) | SLSQP failure fell back to zeroing negatives, breaking the constraint | Code review | Exact projection (`NonNegativeReconciler`) |
| 17 | `ConversionMatrixBuilder` / regressions | No intercept option (R default is `y ~ x`) | R documentation | `include_intercept=True` by default |
| 18 | `data_example/datatempdisagg.csv` (now `examples/data/legacy_long_format.csv`) | Targets change within a year; 0.2.x silently took the first value | Validation since 0.3 | Rejected with an explicit error |

## API mapping

| 0.2.x | 0.4 |
|---|---|
| `TempDisaggModel(method, conversion).fit(df)` | `TemporalDisaggregator(method, conversion).fit_long_format(df)` or `.fit(y_low, indicators)`; `disaggregate(...)` for the series only |
| `model.predict()` | `result.values` |
| `model.adjust_output()` | `NonNegativeReconciler().reconcile(result)` |
| `model.summary()` | `print(result.summary())`; statistics in `result.regression`, `result.rho`, `result.diagnostics` |
| `"chow-lin"` | `"chow-lin-fixed"` |
| `"chow-lin-opt"` | `"chow-lin-maxlog"` |
| `"chow-lin-ecotrim"` / `"chow-lin-quilis"` | `"chow-lin-minrss-ecotrim"` / `"chow-lin-minrss-quilis"` |
| `"litterman"` / `"litterman-opt"` | `"litterman-fixed"` / `"litterman-minrss"` (or `"litterman-maxlog"`) |
| `"ensemble"` | removed |
| `Retropolarizer(df, new_col, old_col).retropolarize("proportion")` | `SeriesSplicer("ratio").splice(df[old_col], df[new_col]).values` |
| `retro_method="linear_regression"` | `SeriesSplicer("affine")` |
| polynomial / exponential smoothing / MLP | removed |
| `interpolation_method` | `completion=CompletionSettings(edge_method=...)` |
| manual padding with `y = NaN` | `padding=PaddingSettings(before, after, target_method="repeat")` |
| `fallback_method`, `rho_min`, `rho_max` | removed (`truncated_rho`, `RhoOptimizerSettings(bounds=...)` instead) |

## Behavioural changes to expect

* **Intercept.** Regression methods include a constant by default, as in R.
  Use `include_intercept=False` to reproduce the 0.2.x specification (the
  legacy `TempDisaggModel` wrapper does this automatically).
* **Negative ρ.** Estimated ρ below 0 is truncated to 0 by default
  (`truncated_rho=0`, as in R); 0.2.x allowed down to −0.9.
* **Incomplete years and gaps.** Completed by default (as the 0.2.x
  `TimeSeriesCompleter` did for the indicator), but targets are never
  interpolated or forward-filled: a year without target is estimated, not
  constrained. All artificial values are listed in `result.preprocessing`.
* **Padding.** In 0.2.x, padding rows with `y = NaN` worked because the
  completer forward/backward-filled the target into them. Since 0.5 this is
  explicit: `padding=PaddingSettings(before, after, target_method="repeat")`
  reproduces it; `"linear"` (default) continues the trend instead.
* **Errors are raised, not hidden.** Collinear indicators raise
  `SingularDesignError`; numerical failures raise `NumericalError`.
* **Dependencies.** scikit-learn and matplotlib are no longer required
  (`pip install "tempdisagg[plot]"` for plotting).
