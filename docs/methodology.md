# Methodology

This document derives every estimator implemented in `tempdisagg`, with the
notation used in the code. References are listed at the end.

## 1. Notation and the aggregation constraint

| Symbol | Meaning | Code |
|---|---|---|
| `n_l` | observed low-frequency values | `data.n_low` |
| `n` | high-frequency periods (including extrapolated ones) | `data.n_high` |
| `y_l` | low-frequency target, `n_l × 1` | `data.y_low` |
| `y` | unknown high-frequency series, `n × 1` | `result.values` |
| `X` | high-frequency regressors (constant and/or indicators), `n × k` | `_design()` |
| `C` | aggregation matrix, `n_l × n` | `data.aggregation_matrix` |

Every method returns an estimate satisfying **`C ŷ = y_l` exactly**. Row `g`
of `C` holds the weights of low-frequency period `g`:

| conversion | weights within the period |
|---|---|
| `sum` | 1, 1, …, 1 |
| `average` | 1/m, …, 1/m (m = number of sub-periods, may vary: days per month) |
| `first` | 1, 0, …, 0 |
| `last` | 0, …, 0, 1 |

High-frequency periods outside every observed low-frequency period have a zero
column in `C`; they are **extrapolated** (or retropolated) by the same model
(Wei, 1994). A low-frequency period with a missing target simply has no row, so
its sub-periods are estimated without a constraint. Nothing is imputed.

## 1b. Preprocessing: completion and padding

**Completion (default).** The residual models and the Denton difference
operators assume a regular grid of consecutive periods, and the constraint of
a low-frequency period needs all its sub-periods. `GridBuilder`
(`preprocessing/grid.py`) therefore builds the grid from the first to the last
observed or required high-frequency period, including

* missing sub-periods of every period with a target (the target is assumed to
  cover the whole period, as in 0.2.x);
* interior gaps.

Missing indicator values are filled by linear interpolation inside the sample
and by `edge_method` (`"linear"`: OLS trend over the last `fit_window`
low-frequency periods; `"constant"`) at the edges. A low-frequency period
without a target is **never** given an invented target: it stays unconstrained
and the model estimates it. Partial edge periods without a target are not
padded out. All artificial values are recorded (`result.preprocessing`) and
removed from `result.values`. `completion=False` turns all of this into errors.

**Partial periods.** A target whose period is not fully observed (e.g. the
first year starts in April) is ambiguous: the value may cover the whole year
or only the observed months. `CompletionSettings(partial_periods=...)` makes
the choice explicit:

| policy | constraint | grid |
|---|---|---|
| `"complete"` (default) | whole period (missing sub-periods added, indicator extended) | completed |
| `"partial"` | observed sub-periods only, with the conversion rule applied to them | interior gaps added, left unconstrained |
| `"error"` | — | error |

For `first`/`last` conversions a partial constraint needs the first/last
sub-period itself to be observed.

**Padding (optional).** `PaddingSettings(before, after, ...)` adds
low-frequency periods before the first and after the last observed target,
with indicators extended (`indicator_method`) and artificial targets
(`target_method`: `"linear"` trend of the last `fit_window` targets,
`"repeat"` the edge target — the behaviour of 0.2.x, whose completer
forward/backward-filled the target — or `"none"`).

Why artificial targets are needed: for Chow-Lin (stationary AR(1)),
Litterman/Fernández with an intercept, and Denton-Cholette, adding
*unconstrained* periods outside the sample leaves the in-sample estimate
unchanged (the in-sample block of `Σ` and the constrained optimum do not
depend on free periods; for random-walk models the shifted initial condition
is a constant absorbed by the intercept). `tests/test_preprocessing.py`
verifies this numerically. Only the original Denton reacts, because its
initial condition moves. Padding therefore acts through the artificial
constraints, which anchor the ends of the sample.

Padding is a heuristic, not an estimator: the artificial targets are an
assumption about the continuation of the series, and regression statistics
computed with them count artificial observations as data
(`regression.n_artificial_obs`, `ArtificialDataWarning`).

## 1c. Additional constraints and aggregation weights

**Additional exact constraints.** Information beyond the target — known
high-frequency values, totals at an intermediate frequency, any linear relation
`a' y = b` — is appended as rows of `C`. The estimators then use it directly:

* GLS methods treat each row as one more exact observation of `C y`; the BLUE
  formula, the likelihood and the prediction variance are unchanged in form
  (`tests/test_constraints.py` checks the result against the textbook formula
  with the stacked `C`);
* Denton methods add the rows to the KKT system.

This one-step approach keeps the estimators optimal; estimating first and
reconciling afterwards does not. Before estimation the stacked system is
checked row by row with modified Gram-Schmidt (targets first): a row that is a
linear combination of earlier rows must carry the implied value — otherwise
the system is contradictory (`InfeasibleConstraintError`, with the conflicting
values); if it does, it is redundant and dropped (`RedundantConstraintWarning`),
because redundant rows make `V = CΣC'` singular. Constraints must refer to
periods of the output grid: they never extend it.

**Aggregation weights.** With weights `w_t`, a `sum` target is `Σ w_t y_t` and an
`average` target is `Σ w_t y_t / Σ w_t`. The main use is rates: if the annual
rate is the ratio of annual totals, `r_year = Σ N_t r_t / Σ N_t` with the
denominator `N_t` (e.g. labour force), i.e. a weighted average.

**Ratios through their components.** When both annual totals are known,
`disaggregate_ratio` disaggregates numerator and denominator (each with its own
model) and divides; with `sum` conversion, `Σ num / Σ den` reproduces the
official annual ratio exactly.

**Structural breaks.** A step dummy (`level_shift`) or impulse dummy
(`impulse`) added to the indicators lets regression methods represent a level
change; its coefficient is estimated from the low-frequency data, so the break
must be visible there (low-frequency observations on both sides).

## 2. Regression-based methods (Chow-Lin family)

### 2.1 Model

    y = X b + u,        Var(u) = σ² Σ(ρ)
    y_l = C y = X_l b + u_l,   X_l = C X,   Var(u_l) = σ² V,   V = C Σ C'

### 2.2 Estimator (Chow and Lin, 1971)

GLS on the low-frequency equation and BLUE distribution of the residuals:

    b̂ = (X_l' V⁻¹ X_l)⁻¹ X_l' V⁻¹ y_l
    û_l = y_l − X_l b̂
    ŷ = X b̂ + Σ C' V⁻¹ û_l

Sax and Steiner (2013) write `ŷ = p + D û_l` with the *preliminary series*
`p = X b̂` and the *distribution matrix* `D = Σ C' V⁻¹`. Since `C D = I`,
`C ŷ = C p + û_l = y_l`.

### 2.3 Residual structures

| Method | Process for `u_t` | `Σ` | Factor `T` (`Σ = T T'`) |
|---|---|---|---|
| `ols` | `e_t` | `I` | `I` |
| `chow-lin-*` | `ρ u_{t−1} + e_t`, stationary start | `ρ^|i−j| / (1−ρ²)` (covariance) or `ρ^|i−j|` (correlation) | `(I − ρL)⁻¹ S` |
| `litterman-*` | `(1−L)(1−ρL) u_t = e_t`, zero start | `(Δ' H' H Δ)⁻¹` | `(I − L)⁻¹ (I − ρL)⁻¹` |
| `fernandez` | `u_{t−1} + e_t` (Litterman, ρ = 0) | `(Δ' Δ)⁻¹` | `(I − L)⁻¹` |
| `dynamic-*` | see 2.6 | `H⁻¹ H⁻ᵀ` | `(I − ρL)⁻¹` |

with `H = I − ρL`, `Δ = I − L` and `S = diag(1/√(1−ρ²), 1, …, 1)` for the
covariance form (`S = diag(1, √(1−ρ²), …)` for the correlation form).

**Implementation.** Products with `T` and `T'` are first-order recursive
filters (`scipy.signal.lfilter`, run backwards for `T'`). The code computes
`W = C T` (`n_l × n`), so `V = W W'`, and the distribution term is
`Σ C' V⁻¹ û_l = T (W' V⁻¹ û_l)`. One likelihood evaluation costs
`O(n n_l²)` instead of `O(n² n_l)`, which makes monthly → daily problems fast.
`tests/reference_implementation.py` builds `Σ` densely and the test-suite checks
that both routes agree to ~1e-8.

### 2.4 Whitening and inference

With `V = L L'` (Cholesky), `ỹ = L⁻¹ y_l`, `X̃ = L⁻¹ X_l`, the problem becomes
OLS with `Var(e) = σ² I`. It is solved by QR after scaling each column of `X̃`
to unit length (so rank and conditioning do not depend on units).
See `docs/statistics.md` for every reported statistic.

### 2.5 Estimating ρ

* **maxlog** — maximise the concentrated Gaussian log-likelihood

      ℓ(ρ) = −n_l/2 [log(2π) + 1 + log(rss(ρ)/n_l)] − ½ log|V(ρ)|,
      rss(ρ) = û_l' V(ρ)⁻¹ û_l.

  `σ²` is concentrated out, so `ℓ` is invariant to rescaling `y_l` or `V`:
  the estimate does not depend on units, and the correlation and covariance
  forms of the AR(1) give identical results.

* **minrss** — minimise `rss(ρ)`. Because `V` is not normalised, the result
  depends on the scaling of `Σ(ρ)`: the correlation form gives
  `chow-lin-minrss-ecotrim`, the covariance form `chow-lin-minrss-quilis`
  (Quilis multiplies the correlation matrix by `1/(1−ρ²)`).

* **fixed** — `ρ` supplied by the user (`fixed_rho`).

**Search.** The criterion is evaluated on a grid over `(−0.999, 0.999)`
(199 points by default), then refined by a bounded Brent search between the
neighbours of the best grid point. This avoids stopping at a local optimum.
The full profile is returned in `result.rho.profile`.

**Truncation.** An estimate below `truncated_rho` (default 0) is replaced by
it, as in R (`truncated.rho`); `-1` disables truncation. The untruncated value
is kept in `result.rho.unconstrained_value`.

**Identification checks.** Collinearity and exact fit do not depend on `ρ`
(`V` is non-singular), so they are checked once before the search:

* collinear regressors → `SingularDesignError`;
* exact fit (`û_l = 0`) → `ŷ = X b̂` for every `ρ`; `ρ` is reported as not
  identified (`nan`) with an `ExactFitWarning`.

After the search the library reports a flat criterion
(`RhoNotIdentifiedWarning`), an optimum on the boundary
(`RhoBoundaryWarning`) and competing local optima (`MultipleOptimaWarning`):
admissible optima within `χ²₁(0.95)/2 ≈ 1.92` log-likelihood units of the global
one (5 % in RSS for `minrss`), i.e. not distinguishable by a likelihood-ratio
test.

### 2.6 Dynamic Chow-Lin (Santos Silva and Cardoso, 2001) — experimental

    y_t = ρ y_{t−1} + x_t' b + e_t,   e_t white noise.

Solving the recursion: `y = H⁻¹ X b + H⁻¹ e₁ (ρ y₀) + H⁻¹ e`. Hence the
regressors are filtered by `H⁻¹`, the vector `ρ^t` (the *truncation remainder*,
coefficient `y₀`) is appended when `ρ ≠ 0`, and `Σ = H⁻¹ H⁻ᵀ`. The method is
marked experimental, as in R, until full parity is established.

### 2.7 Prediction variance

The BLUP error covariance (Goldberger, 1962), conditional on `ρ`:

    Var(ŷ − y) = σ² (Σ − G V⁻¹ G') + Q Var(b̂) Q',
    G = Σ C',   Q = X − G V⁻¹ X_l,   Var(b̂) = σ² (X_l' V⁻¹ X_l)⁻¹.

Only the diagonal is computed (`O(n n_l²)`). Properties verified by the tests:
`C Var(ŷ − y) C' = 0` (aggregates are known exactly), zero variance for the
constrained sub-period under `first`/`last`, growth with the extrapolation
horizon for random-walk models, and nominal coverage of 95 % intervals in Monte
Carlo simulations when `ρ` is known. Intervals ignore the sampling uncertainty
of an estimated `ρ`, so they are conditional.

## 3. Denton benchmarking

Given one indicator `x` (a constant if none), find `y` closest to `x` in
movement subject to `C y = y_l`:

    additive:      min ‖D (y − x)‖²
    proportional:  min ‖D diag(1/x) (y − x)‖²

`D` is the `h`-th difference operator (`h = 0` levels, 1 changes, 2
curvature). The **original Denton** (1971) keeps the first `h` rows of `D`,
which embed the initial condition `y₀ = x₀` and distort the start of the
series. **Denton-Cholette** (Dagum and Cholette, 2006) drops them.

With `A = D'D`, the optimum solves the KKT system

    [ A   C' ] [ y ]   [ A x ]
    [ C   0  ] [ λ ] = [ y_l ].

No regulariser is added (issue #4 of 0.2.13 added `I` to `A`, which changed the
objective). For Denton-Cholette, `A` is singular on the polynomials of degree
`< h` (times `x` if proportional); the system is non-singular iff `C N` has full
column rank, where `N` spans that null space. The code checks this rank
condition explicitly and raises `NumericalError` with an explanation otherwise.
The system is sparse and solved by sparse LU; its condition number is estimated
with Higham's 1-norm estimator.

**Without an indicator** Denton uses a constant `x = 1`; proportional and
additive criteria then coincide, and Denton-Cholette with `h = 1` (`h = 2`)
is the minimum first (second) difference interpolation of Boot, Feibes and
Lisman (1967): the smoothest series compatible with the totals.

**Uniform** distributes `y_l` evenly: `ŷ = C'(CC')⁻¹ y_l`, defined for `sum`
and `average` only and never extrapolated.

## 4. Non-negative reconciliation

*Separable case* — see below. *General case* (overlapping constraint rows,
such as quarters inside years or known values): Dykstra's alternating
projections between `{C y = b}` and the orthant converge to the exact
projection onto their intersection; the iterate is then polished on its active
set in closed form and accepted only if the KKT conditions hold
(`y_F ≥ 0`, multipliers of the zero set `≥ 0`). Non-convergence means the
constraints admit no non-negative solution (`InfeasibleConstraintError`).

### 4.1 Separable case

    min ‖y − ŷ‖²   s.t.   C y = y_l,   y ≥ 0.

Each period belongs to at most one group, so the problem separates. For a
group with weights `c_i > 0`, the KKT conditions give
`y_i = max(0, ŷ_i + λ c_i)`, and `Σ c_i y_i(λ)` is non-decreasing in `λ`; `λ` is
bracketed by bisection and computed in closed form on the identified active set.
Periods outside every group are clipped at zero. A negative `y_l` makes the
problem infeasible (`InfeasibleConstraintError`).

## 4b. Cross-sectional reconciliation

`S` series disaggregated separately satisfy `C_s y_s = a_s` but not, in
general, a high-frequency total `Σ_s y_s = z`. Following the two-step approach
of Di Fonzo and Marini (2011), the reconciled system solves

    min Σ_s ‖D_h W_s (y_s − p_s)‖²   s.t.   C_s y_s = a_s (all s),   Σ_s y_s = z,

with `p_s` the step-one estimates, `W_s = I` (additive) or `diag(1/p_s)`
(proportional) and `D_h = I` (`h = 0`, minimal adjustments) or the first
difference without initial condition (`h = 1`, movement preservation). The
constraint sets overlap: in every period where all series have targets,
`Σ_s a_s(g)` must equal `C_g z`. A mismatch raises `InfeasibleConstraintError`
listing the periods; otherwise one temporal row per such period is redundant
and removed. The sparse KKT system is solved by LU.

## 4c. Backtesting

Constrained methods reproduce the targets exactly, so in-sample fit cannot rank
them. `backtest` holds out each of the last `holdout` observed periods in turn,
fits with the earlier targets only, aggregates the extrapolated estimate over
the held-out period and compares it with the actual target (rolling origin,
one step ahead). It measures extrapolation of the aggregate, not the shape
within periods, which is unobservable without a high-frequency benchmark.

## 5. Splicing

Legacy series `o_t`, current series `c_t`, overlap `O`:

| method | link | spliced value where `c_t` is missing |
|---|---|---|
| level | `d = mean_O(c − o)` | `o_t + d` |
| ratio | `k = Σ_O c / Σ_O o` | `k o_t` |
| affine | OLS of `c` on `o` over `O` | `a + b o_t` |
| interpolated | ratio `r` (or shift) at the first overlap period `T`, neutral at anchor `t₀` | `o_t [1 + (r − 1)(t − t₀)/(T − t₀)]` |

The interpolated method (de la Fuente, 2014) assumes the methodological
discrepancy built up gradually since `t₀`, where the legacy series is taken as
correct. Observed values of the current series are never modified.

## References

* Boot, J. C. G., Feibes, W. and Lisman, J. H. C. (1967). Further methods of derivation of quarterly figures from annual data. *Applied Statistics* 16(1), 65–75.
* Chow, G. C. and Lin, A. (1971). Best linear unbiased interpolation, distribution, and extrapolation of time series by related series. *Review of Economics and Statistics* 53(4), 372–375.
* Dagum, E. B. and Cholette, P. A. (2006). *Benchmarking, Temporal Distribution, and Reconciliation Methods for Time Series*. Springer.
* de la Fuente Moreno, A. (2014). A "mixed" splicing procedure for economic time series. *Estadística Española* 56(183), 107–121.
* Denton, F. T. (1971). Adjustment of monthly or quarterly series to annual totals. *JASA* 66(333), 99–102.
* Di Fonzo, T. and Marini, M. (2011). Simultaneous and two-step reconciliation of systems of time series: methodological and practical issues. *JRSS C* 60(2), 143–164.
* Fernández, R. B. (1981). A methodological note on the estimation of time series. *Review of Economics and Statistics* 63(3), 471–476.
* Goldberger, A. S. (1962). Best linear unbiased prediction in the generalized linear regression model. *JASA* 57(298), 369–375.
* Litterman, R. B. (1983). A random walk, Markov model for the distribution of time series. *JBES* 1(2), 169–173.
* Quilis, E. M. (2009). Temporal disaggregation using Matlab (technical note).
* Santos Silva, J. M. C. and Cardoso, F. N. (2001). The Chow-Lin method using dynamic models. *Economic Modelling* 18, 269–280.
* Sax, C. and Steiner, P. (2013). Temporal disaggregation of time series. *The R Journal* 5(2), 80–88.
* Wei, W. W. S. (1994). *Time Series Analysis*. Addison-Wesley.
