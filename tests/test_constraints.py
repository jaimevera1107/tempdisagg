"""Additional linear constraints and aggregation weights."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from tempdisagg import (
    InfeasibleConstraintError,
    InputValidationError,
    LinearConstraint,
    NonNegativeReconciler,
    RedundantConstraintWarning,
    TemporalDisaggregator,
    aggregate_constraints,
    known_values,
    temporal_aggregate,
)
from tests import reference_implementation as ref

SEED: int = 11


def _monthly_problem(n_years: int = 6):
    rng = np.random.default_rng(SEED)
    months = pd.period_range("2015-01", periods=12 * n_years, freq="M")
    x = pd.Series(100 + np.cumsum(rng.normal(0.3, 1, len(months))), index=months, name="x")
    truth = 2 * x + np.cumsum(rng.normal(0, 1, len(months)))
    return temporal_aggregate(truth, "Y"), x, truth


def _quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


@pytest.mark.parametrize("method", ["chow-lin-maxlog", "litterman-maxlog", "fernandez", "ols",
                                    "denton-cholette", "denton"])
def test_mixed_frequency_constraints_hold_exactly(method: str) -> None:
    annual, x, truth = _monthly_problem()
    quarters = temporal_aggregate(truth, "Q").iloc[[2, 7, 13]]
    months = truth.iloc[[20, 21, 50]]
    constraints = aggregate_constraints(quarters, "M") + known_values(months)
    result = _quiet(TemporalDisaggregator(method).fit, annual, x, constraints=constraints)
    np.testing.assert_allclose(result.values.loc[months.index], months, rtol=1e-9)
    by_quarter = temporal_aggregate(result.values, "Q").loc[quarters.index]
    np.testing.assert_allclose(by_quarter, quarters, rtol=1e-9)
    assert result.aggregation_error() < 1e-8 * float(annual.max())


def test_constraints_enter_gls_as_exact_observations() -> None:
    """One-step estimator equals the textbook BLUE with the stacked constraint matrix."""
    annual, x, truth = _monthly_problem(4)
    constraints = known_values(truth.iloc[[5, 30]])
    result = TemporalDisaggregator("chow-lin-fixed", fixed_rho=0.6).fit(
        annual, x, constraints=constraints)
    data = result.data
    design = np.column_stack([np.ones(len(x)), x.to_numpy()])
    expected = ref.gls_disaggregation(data.y_low, design, data.aggregation_matrix,
                                      ref.sigma_chow_lin(0.6, len(x)))
    np.testing.assert_allclose(result.values, expected["y_hat"], rtol=1e-9)
    assert result.regression.n_obs == len(annual) + 2


def test_known_values_improve_accuracy_near_them() -> None:
    annual, x, truth = _monthly_problem()
    plain = TemporalDisaggregator().fit(annual, x)
    informed = TemporalDisaggregator().fit(annual, x, constraints=known_values(truth.iloc[30:33]))
    window = slice(28, 36)
    error_plain = np.abs(plain.values.iloc[window] - truth.iloc[window]).sum()
    error_informed = np.abs(informed.values.iloc[window] - truth.iloc[window]).sum()
    assert error_informed < error_plain


def test_contradictory_constraints_are_rejected() -> None:
    annual, x, truth = _monthly_problem()
    quarters = temporal_aggregate(truth, "Q").iloc[:4] * 1.02  # do not add up to the year
    with pytest.raises(InfeasibleConstraintError, match="contradicts"):
        TemporalDisaggregator().fit(annual, x, constraints=aggregate_constraints(quarters, "M"))


def test_redundant_constraints_are_dropped_with_warning() -> None:
    annual, x, truth = _monthly_problem()
    quarters = temporal_aggregate(truth, "Q").iloc[:4]  # consistent: sum to the first year
    with pytest.warns(RedundantConstraintWarning, match="1 redundant"):
        result = TemporalDisaggregator().fit(annual, x,
                                             constraints=aggregate_constraints(quarters, "M"))
    assert result.aggregation_error() < 1e-8 * float(annual.max())


def test_constraint_outside_the_output_is_rejected() -> None:
    annual, x, _ = _monthly_problem()
    outside = LinearConstraint(pd.Series([1.0], index=[pd.Period("1990-01", "M")]), 5.0)
    with pytest.raises(InputValidationError):
        _quiet(TemporalDisaggregator().fit, annual, x, constraints=[outside])


def test_invalid_constraints() -> None:
    with pytest.raises(InputValidationError):
        LinearConstraint(pd.Series(dtype=float), 1.0)
    with pytest.raises(InputValidationError):
        LinearConstraint(pd.Series([0.0, 0.0], index=[1, 2]), 1.0)
    with pytest.raises(InputValidationError):
        LinearConstraint(pd.Series([1.0]), np.inf)


def test_uniform_rejects_additional_constraints() -> None:
    annual, x, truth = _monthly_problem()
    with pytest.raises(InputValidationError, match="denton-cholette"):
        TemporalDisaggregator("uniform").fit(annual, high_freq="M",
                                             constraints=known_values(truth.iloc[[3]]))


def test_constraints_with_array_and_long_format_labels() -> None:
    rng = np.random.default_rng(1)
    x = 50 + np.cumsum(rng.normal(0, 1, 40))
    y_low = x.reshape(10, 4).sum(axis=1) + rng.normal(0, 1, 10)
    fixed = LinearConstraint(pd.Series([1.0], index=[7]), 55.0)
    result = TemporalDisaggregator().fit_arrays(y_low, x, 4, constraints=[fixed])
    assert result.values.iloc[7] == pytest.approx(55.0)
    frame = pd.DataFrame({"Index": np.repeat(np.arange(2000, 2010), 4),
                          "Grain": np.tile(np.arange(1, 5), 10),
                          "y": np.repeat(y_low, 4), "X": x})
    fixed_long = LinearConstraint(pd.Series([1.0], index=[(2001, 4)]), 55.0)
    result_long = TemporalDisaggregator().fit_long_format(frame, x_cols=["X"],
                                                          constraints=[fixed_long])
    assert result_long.values.loc[(2001, 4)] == pytest.approx(55.0)


def test_non_negative_reconciliation_with_overlapping_constraints() -> None:
    annual, x, truth = _monthly_problem()
    constraints = aggregate_constraints(temporal_aggregate(truth, "Q").iloc[[1, 5]], "M")
    result = TemporalDisaggregator().fit(annual, x - 150, constraints=constraints)
    data = result.data
    perturbed = result.full_values.to_numpy() - 260.0 * np.sin(np.arange(len(x)))
    projected = NonNegativeReconciler().project(perturbed, data.aggregation_matrix, data.y_low)
    assert projected.min() >= 0.0
    np.testing.assert_allclose(data.aggregation_matrix @ projected, data.y_low, rtol=1e-8)


def test_general_projection_matches_a_reference_solver() -> None:
    from scipy.optimize import minimize

    rng = np.random.default_rng(4)
    matrix = np.vstack([np.ones(12), np.r_[np.ones(3), np.zeros(9)], np.eye(12)[7]])
    targets = np.array([20.0, 4.0, 2.0])
    y_hat = rng.normal(1, 3, 12)
    projected = NonNegativeReconciler().project(y_hat, matrix, targets)
    reference = minimize(lambda v: np.sum((v - y_hat) ** 2), np.ones(12), method="SLSQP",
                         constraints=[{"type": "eq", "fun": lambda v: matrix @ v - targets}],
                         bounds=[(0, None)] * 12, options={"ftol": 1e-14, "maxiter": 2000})
    np.testing.assert_allclose(projected, reference.x, atol=1e-6)


def test_infeasible_general_projection() -> None:
    matrix = np.vstack([np.ones(4), np.r_[1.0, 1.0, 0.0, 0.0]])
    with pytest.raises(InfeasibleConstraintError):
        NonNegativeReconciler().project(np.array([-1.0, 2.0, 1.0, 1.0]), matrix,
                                        np.array([3.0, 5.0]))


# ----------------------------------------------------------------------
# Aggregation weights (rates)
# ----------------------------------------------------------------------
def test_weighted_average_reproduces_rate_definition() -> None:
    rng = np.random.default_rng(2)
    months = pd.period_range("2012-01", periods=96, freq="M")
    labour = pd.Series(1000 + np.cumsum(rng.normal(1, 3, 96)), index=months)
    rate = pd.Series(0.1 + 0.01 * np.sin(np.arange(96) / 5), index=months)
    annual_rate = ((rate * labour).groupby(months.asfreq("Y")).sum()
                   / labour.groupby(months.asfreq("Y")).sum())
    annual_rate.index = pd.PeriodIndex(annual_rate.index, freq="Y")
    result = TemporalDisaggregator("chow-lin-maxlog", "average").fit(
        annual_rate, (rate + rng.normal(0, 0.002, 96)).rename("r"), aggregation_weights=labour)
    implied = ((result.values * labour).groupby(months.asfreq("Y")).sum()
               / labour.groupby(months.asfreq("Y")).sum())
    np.testing.assert_allclose(implied, annual_rate, rtol=1e-10)


def test_invalid_weights() -> None:
    annual, x, _ = _monthly_problem()
    with pytest.raises(InputValidationError, match="positive"):
        TemporalDisaggregator(conversion="average").fit(annual / 12, x,
                                                        aggregation_weights=x * 0 - 1)
    with pytest.raises(InputValidationError, match="'sum' and 'average'"):
        TemporalDisaggregator(conversion="first").fit(annual, x, aggregation_weights=x)
