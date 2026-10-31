"""Tests for the exact non-negative reconciliation."""

from __future__ import annotations

import numpy as np
import pytest

from tempdisagg import InfeasibleConstraintError, NonNegativeReconciler
from tempdisagg.core.aggregation import build_aggregation_matrix


def test_closed_form_example() -> None:
    matrix = build_aggregation_matrix(np.array([0, 0, 0]), 1, "sum")
    y_hat = np.array([-1.0, 2.0, 3.0])
    projected = NonNegativeReconciler().project(y_hat, matrix, np.array([4.0]))
    np.testing.assert_allclose(projected, [0.0, 1.5, 2.5])


@pytest.mark.parametrize("conversion", ["sum", "average", "first", "last"])
def test_constraint_and_sign_hold(conversion: str) -> None:
    rng = np.random.default_rng(3)
    codes = np.repeat(np.arange(6), 4)
    matrix = build_aggregation_matrix(codes, 6, conversion)
    y_hat = rng.normal(1.0, 2.0, codes.size)
    y_low = np.abs(matrix @ y_hat) + 0.1
    projected = NonNegativeReconciler().project(y_hat, matrix, y_low)
    assert np.all(projected >= 0)
    np.testing.assert_allclose(matrix @ projected, y_low, rtol=1e-10, atol=1e-12)


def test_projection_is_optimal_against_random_feasible_points() -> None:
    rng = np.random.default_rng(7)
    matrix = build_aggregation_matrix(np.zeros(5, dtype=int), 1, "sum")
    y_hat = np.array([-2.0, 0.5, 3.0, -0.5, 1.0])
    projected = NonNegativeReconciler().project(y_hat, matrix, np.array([3.0]))
    best = np.sum((projected - y_hat) ** 2)
    for _ in range(500):
        candidate = rng.dirichlet(np.ones(5)) * 3.0  # feasible: >= 0, sums to 3
        assert np.sum((candidate - y_hat) ** 2) >= best - 1e-12


def test_negative_target_is_infeasible() -> None:
    matrix = build_aggregation_matrix(np.zeros(3, dtype=int), 1, "sum")
    with pytest.raises(InfeasibleConstraintError):
        NonNegativeReconciler().project(np.array([-1.0, -1.0, -1.0]), matrix, np.array([-3.0]))
