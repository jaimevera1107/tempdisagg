# Architecture

```
tempdisagg/
├── core/
│   ├── aggregation.py      Conversion rules, aggregation matrix C, temporal_aggregate (R: ta)
│   ├── data.py             DataAligner: pandas / NumPy / long format -> DisaggregationData
│   ├── constraints.py      LinearConstraint, known_values, aggregate_constraints, row checks
│   └── linalg.py           difference operators, Cholesky helpers (no pseudo-inverses)
├── covariance/
│   ├── factors.py          LowerTriangularFactor: Σ = T T' applied by recursive filters
│   └── structures.py       white noise, AR(1), random walk + AR(1), dynamic
├── estimation/
│   ├── gls.py              GLS on the low-frequency equation and its statistics
│   ├── rho.py              grid + Brent search of ρ with diagnostics
│   ├── regression.py       RegressionDisaggregator: the single engine for all GLS methods
│   ├── diagnostics.py      DiagnosticLog: warnings collected during a fit
│   └── base.py             MethodOutput and the Disaggregator protocol
├── preprocessing/
│   ├── settings.py         CompletionSettings, PaddingSettings
│   ├── calendar.py         Period / Ratio / Grain calendars (how periods nest)
│   ├── extension.py        interpolation and edge extension of series
│   └── grid.py             GridBuilder: complete grid, padding, constraints
├── benchmarking/
│   └── denton.py           Denton / Denton-Cholette (sparse KKT), uniform
├── methods.py              method specifications (configuration) and R presets
├── model.py                TemporalDisaggregator, disaggregate(), result assembly
├── results.py              DisaggregationResult and its statistics objects
├── reconciliation/
│   ├── non_negative.py     exact non-negative projection (separable and general)
│   └── cross_section.py    two-step cross-sectional reconciliation (Di Fonzo-Marini)
├── evaluation/
│   └── backtest.py         rolling one-step-ahead backtest
├── ratios.py               ratio of totals via numerator and denominator
├── indicators.py           level_shift and impulse dummies
├── splicing/               level / ratio / affine / interpolated splicing
├── plotting.py             optional matplotlib helpers
└── legacy.py               deprecated 0.2.x wrappers (removed in 1.0)
```

## Data flow

```
inputs ──DataAligner──> DisaggregationData ──Disaggregator.disaggregate──> MethodOutput
                                                                              │
                         DisaggregationResult <──_assemble_result (model.py)──┘
```

1. `DataAligner` validates the input format and translates its labels into a
   calendar; `GridBuilder` completes the grid, applies padding and builds `C`.
   Targets are never imputed; artificial values are recorded.
2. `make_method` resolves a preset or a specification and its overrides.
3. `spec.build()` creates the estimator (`RegressionDisaggregator`,
   `DentonBenchmarker` or `UniformDistributor`).
4. The estimator returns a `MethodOutput` (arrays + `DiagnosticLog`).
5. `model.py` emits the warnings and assembles immutable result objects.

## Design rules

* **One engine per family.** All GLS methods differ only by a residual
  structure and a ρ rule; there is no per-method copy of the GLS algebra.
* **No silent behaviour.** No imputation of inputs, no fallback method, no
  pseudo-inverse. Failures raise a typed `TempDisaggError`; statistical issues
  raise a typed `TempDisaggWarning` that is also stored in the result.
* **Composition over inheritance.** Specifications compose a structure and
  settings; results compose statistics objects.
* **Immutability.** Specifications and results are frozen dataclasses.
* **Verifiability.** Every optimised computation has a dense textbook
  counterpart in `tests/reference_implementation.py`.

## Adding a method

1. If it is a GLS method, write a `ResidualStructure` (a `factor(rho, n)` that
   returns a `LowerTriangularFactor`) in `covariance/structures.py`.
2. Add a specification dataclass in `methods.py` with documented fields and a
   `build()` method; register an R-named preset in `PRESETS`.
3. Add a dense formula to `tests/reference_implementation.py`, a cross-check in
   `tests/test_reference_engine.py`, and an R case in
   `tests/reference/generate_r_reference.R` / `tests/test_r_parity.py`.
