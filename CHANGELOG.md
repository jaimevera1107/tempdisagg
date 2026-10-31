# Changelog

Versioning: the library stays below 1.0 until it is complete and validated
against R. Every major change increases the minor version by 0.1.

## 0.6.3 — 2026-10-31

### Changed
- Synchronize the PyPI distribution with the latest validated GitHub source.
- Incorporate code quality, linting and documentation refinements.
- Preserve the statistical methods and public API introduced in v0.6.2.

## 0.6.2 — 2026-10-31

Documentation release.

### Added
- `tempdisagg.datasets`: `load_swisspharma()` (real data used for the R parity)
  and `load_denton1971()`, installed with the package.
- `examples/tempdisagg_examples.ipynb`: every method and tool with real data,
  executed, with figures.
- `docs/images/`: comparison with R (`r_parity_summary.png`,
  `r_parity_series.png`, `r_parity_statistics.png`) and `hero.png`, built by
  `docs/make_figures.py`.
- New README.
- `tests/test_datasets.py`; the parity case table moved to
  `tests/reference/r_cases.py` (shared with the figure script).

### Changed
- `data_example/datatempdisagg.csv` moved to `examples/data/legacy_long_format.csv`.

## 0.6.1 — 2026-10-31

R parity executed against R 4.5.3 / tempdisagg 1.2.0 (29 cases + vignette).
Predictions, coefficients, standard errors, rho and the log-likelihood already
agreed; three reporting conventions were aligned with R:

### Changed
- **Scale of sigma^2, RSS and TSS for Chow-Lin** now follows the stationary
  covariance `rho^|i-j| / (1 - rho^2)` (R), independently of the form used to
  estimate rho. Estimates, standard errors and likelihood are unchanged.
- **TSS** is always centred on the aggregated constant `L^{-1} C 1`, also
  without intercept and in the dynamic model; **adjusted R-squared** always
  uses `(n - 1) / (n - k)`.
- **`aic` / `bic`** follow R: `log(s2_ml) + 2k/n` and `log(s2_ml) + k log(n)/n`.
  The likelihood-based criteria are kept as `aic_loglik` / `bic_loglik`
  (comparable across residual structures).

### Added
- R reference files committed in `tests/reference/`, so the parity tests run
  in every `pytest` call.
- Stricter parity tests: 1e-8 for fixed/absent rho; for estimated rho, the
  Python optimum must be at least as good as R's on the estimation criterion
  and agree to 1e-8 after re-fitting at R's rho.
- `generate_r_reference.R` is robust: writes next to itself, logs versions
  (`r_session.txt`), continues past failing cases (`r_errors.csv`).

## 0.6.0 — 2026-10-30

### Added
- **Partial periods**: `CompletionSettings(partial_periods="complete" | "partial" |
  "error")` states what the target of an incompletely observed period covers.
- **Additional exact constraints** (`LinearConstraint`, `known_values`,
  `aggregate_constraints`): known high-frequency values, intermediate-frequency
  totals or any linear relation, used directly by every estimator (one-step).
  The stacked system is checked: contradictions raise
  `InfeasibleConstraintError`, redundancies are dropped with
  `RedundantConstraintWarning`. Supported for pandas, NumPy and long-format
  inputs.
- **Aggregation weights** (`aggregation_weights=`, `weights_col=`): weighted
  sums/averages, e.g. rates weighted by their denominator.
- **`disaggregate_ratio`**: ratios of totals through numerator and denominator.
- **`level_shift`, `impulse`**: dummies for structural breaks and events.
- **`backtest`** (`Candidate`, `BacktestResult`): rolling one-step-ahead
  holdout of the low-frequency aggregate to compare models and indicators.
- **`CrossSectionalReconciler`**: two-step reconciliation of several series
  with a high-frequency total (Di Fonzo and Marini, 2011), additive or
  proportional, h = 0 or 1, with consistency checks.
- **General non-negative projection** for overlapping constraints (Dykstra
  iterations with an exact, KKT-certified polish).
- Tests: `test_constraints.py`, `test_evaluation.py`, partial-period tests; the
  randomised contract also draws known-value constraints.

### Fixed
- The example-data test no longer depends on the working directory.

### Not yet done
- The R parity suite has still not been executed against R (no R available in
  the development environment); run `tests/reference/generate_r_reference.R`.

## 0.5.0 — 2026-10-30

### Added
- **Completion of the time grid (default).** Missing sub-periods of
  constrained periods, interior gaps and missing indicator values are filled
  (linear interpolation inside, linear or constant extension at the edges) by
  `GridBuilder`, identically for pandas, NumPy and long-format inputs.
  Targets are never invented. Artificial values are recorded in
  `result.preprocessing`, flagged in `to_frame()`, reported with
  `ArtificialDataWarning` and removed from `result.values`
  (`result.full_values` keeps them). `completion=False` gives strict mode.
- **Padding** (`padding=` int, `(before, after)` or `PaddingSettings`):
  artificial low-frequency periods around the observed targets, with linear or
  constant indicator extension and `"linear"`, `"repeat"` or `"none"` targets.
  Regression statistics report `n_artificial_obs`.
- **Methods without indicators**: no indicator column is required by
  `fit`, `fit_arrays`, `fit_long_format` (`x_cols` optional) or the legacy
  wrapper (`X_col=None`) for uniform, Denton and regressions on a constant.
  `uniform` warns if indicators are passed.
- Tests: `test_preprocessing.py`; the randomised contract now also draws
  missing indicator values and padding.

### Changed
- `fit_long_format(..., x_cols=None)`; `allow_unequal_groups` removed
  (incomplete groups are completed; use period-indexed input for true
  unequal sizes such as days per month).

## 0.4.0 — 2026-10-30

### Added
- **Per-method configuration.** Every method is a frozen specification class
  (`ChowLin`, `Litterman`, `Fernandez`, `DynamicChowLin`, `OLS`, `Denton`,
  `Uniform`) whose fields are its settings: intercept, ρ criterion, fixed ρ,
  truncation, covariance form, truncation remainder, Denton variant /
  criterion / h, and the numerical ρ search (`RhoOptimizerSettings`). R names
  are presets; any field can be overridden (`TemporalDisaggregator("fernandez",
  include_intercept=False)`). Unknown settings raise an error listing the valid
  ones; ineffective settings warn. `describe()` lists every setting with its R
  equivalent.
- **Two-level API.** `disaggregate()` returns only the series (automation);
  `TemporalDisaggregator.fit()` returns the full record.
- **Statistical record.** `RegressionStatistics` (coefficients, covariance,
  t/p, confidence intervals, σ² ML and GLS, RSS, TSS, R², adjusted R²,
  log-likelihood, AIC, BIC, whitened residuals and their autocorrelation),
  `RhoEstimate` (profile, boundary, identification, local optima, untruncated
  value), `BenchmarkStatistics`, `NumericalDiagnostics` (aggregation error,
  condition numbers, warnings).
- **Prediction standard errors and intervals** for regression methods
  (Goldberger BLUP variance, conditional on ρ), validated by Monte Carlo coverage.
- **Typed statistical warnings**: boundary ρ, unidentified ρ, competing optima,
  exact fit, collinearity / ill-conditioning, sign changes in proportional
  Denton, ineffective parameters, experimental methods.
- **Scalability.** Covariances handled through structured factors and
  recursive filters (O(n·n_l²) per evaluation); Denton solved by sparse LU.
  Ten years of daily data from monthly totals take seconds.
- **Tests**: dense textbook reference implementation and cross-checks,
  Monte Carlo validity tests, robustness tests for degenerate and malformed
  data, a 300-case randomised contract test, R parity against the published
  vignette values and 29 generated R cases; CI workflows for Python and R.
- **Documentation**: `docs/methodology.md`, `docs/statistics.md`,
  `docs/configuration.md`, `docs/architecture.md`, `docs/testing.md`.

### Fixed
- Rank and collinearity checks are now scale-invariant (an intercept next to
  an indicator measured in billions was wrongly flagged as collinear).
- Denton-Cholette identification is checked explicitly (rank condition) instead
  of relying on the factorisation failing.

### Changed
- `result.rho` is now a `RhoEstimate` object (`result.rho.value`).
- Regression statistics moved to `result.regression`.
- `rho_bounds` moved to `RhoOptimizerSettings(bounds=...)`.
- Denton variant values are `"cholette"` / `"original"`.

## 0.3.0 — 2026-10-30

Complete rewrite of 0.2.13 (see MIGRATION.md for the 18 corrected defects):
R-compatible methods and names, multiple indicators, extrapolation,
calendar-aware alignment, exact Denton (issue #4), unit-invariant ρ, no silent
imputation or fallback, `SeriesSplicer`, `NonNegativeReconciler`,
`temporal_aggregate`, removal of the ensemble. 0.2.x interfaces kept as
deprecated wrappers, to be removed in 1.0.
