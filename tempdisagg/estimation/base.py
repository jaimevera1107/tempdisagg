"""Common output type and interface shared by every disaggregation method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from tempdisagg.core.data import DisaggregationData
from tempdisagg.estimation.diagnostics import DiagnosticLog
from tempdisagg.estimation.gls import GLSFit
from tempdisagg.estimation.rho import RhoMethod, RhoSearchResult


@dataclass(frozen=True)
class RhoOutcome:
    """How ``rho`` was determined.

    Attributes
    ----------
    value : float
        ``rho`` used for the estimate (``nan`` if not identified).
    method : RhoMethod
        Criterion used.
    truncated : bool
        Whether the estimate was replaced by the truncation bound.
    unconstrained_value : float or None
        Estimate before truncation (``None`` for fixed ``rho``).
    search : RhoSearchResult or None
        Search diagnostics when ``rho`` was estimated.
    identified : bool
        False when ``rho`` cannot be determined from the data.
    """

    value: float
    method: RhoMethod
    truncated: bool = False
    unconstrained_value: float | None = None
    search: RhoSearchResult | None = None
    identified: bool = True

    @property
    def estimated(self) -> bool:
        """Whether ``rho`` was estimated from the data."""
        return self.method is not RhoMethod.FIXED


@dataclass(frozen=True)
class DentonOutcome:
    """Statistics of a Denton benchmark.

    Attributes
    ----------
    penalty : float
        Value of ``||D (y - x)||^2`` (additive) or
        ``||D diag(1/x)(y - x)||^2`` (proportional) at the solution.
    condition_number : float
        Condition number of the KKT system.
    """

    penalty: float
    condition_number: float


@dataclass(frozen=True)
class MethodOutput:
    """Raw output of one disaggregation method.

    Attributes
    ----------
    values : np.ndarray
        High-frequency estimate, shape ``(n_high,)``.
    fitted_low : np.ndarray
        Low-frequency values before residual distribution (``C X b`` for
        regressions, ``C x`` for Denton, ``y_l`` for uniform).
    diagnostics : DiagnosticLog
        Warnings raised during estimation.
    coefficient_names : tuple of str
        Regression coefficient names (empty otherwise).
    gls : GLSFit or None
        Regression statistics.
    rho : RhoOutcome or None
        ``rho`` details for structures that use it.
    prediction_variance : np.ndarray or None
        ``Var(y_hat - y)`` per high-frequency period, conditional on ``rho``.
    denton : DentonOutcome or None
        Benchmark statistics.
    """

    values: np.ndarray
    fitted_low: np.ndarray
    diagnostics: DiagnosticLog
    coefficient_names: tuple[str, ...] = ()
    gls: GLSFit | None = None
    rho: RhoOutcome | None = None
    prediction_variance: np.ndarray | None = None
    denton: DentonOutcome | None = None


class Disaggregator(Protocol):
    """Interface implemented by every method family."""

    def disaggregate(self, data: DisaggregationData) -> MethodOutput:
        """Estimate the high-frequency series for ``data``."""
