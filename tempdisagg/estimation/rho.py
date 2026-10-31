"""Estimation of the autoregressive parameter ``rho``.

Criteria (Sax and Steiner, 2013)
--------------------------------
``maxlog``  maximise the concentrated log-likelihood of the low-frequency
            equation (see :mod:`tempdisagg.estimation.gls`).
``minrss``  minimise the weighted residual sum of squares
            ``u_l' V(rho)^{-1} u_l``. Because ``V`` is not normalised, the
            result depends on how ``Sigma(rho)`` is scaled (Ecotrim vs Quilis).
``fixed``   ``rho`` is supplied by the user.

Search strategy
---------------
The criterion can be multimodal or flat, so a single bounded Brent search can
stop at a local optimum. :class:`RhoEstimator` evaluates a grid over the
open interval, refines around the best grid point with a bounded scalar
search, and reports diagnostics: the full profile, whether the optimum lies on
the boundary, whether the criterion is flat (``rho`` not identified) and how
many local optima the grid reveals.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import minimize_scalar

from tempdisagg.exceptions import InputValidationError, NumericalError, TempDisaggError

logger = logging.getLogger(__name__)

#: Relative range below which the criterion profile is considered flat.
FLATNESS_TOLERANCE: float = 1e-10


class RhoMethod(str, Enum):
    """How ``rho`` is obtained."""

    MAXLOG = "maxlog"
    MINRSS = "minrss"
    FIXED = "fixed"


@dataclass(frozen=True)
class RhoOptimizerSettings:
    """Numerical settings of the ``rho`` search.

    Parameters
    ----------
    bounds : tuple of float, default (-0.999, 0.999)
        Open interval searched (as in the R package).
    grid_size : int, default 199
        Grid points evaluated before refinement (>= 3).
    tolerance : float, default 1e-8
        Absolute tolerance of the refinement.
    """

    bounds: tuple[float, float] = (-0.999, 0.999)
    grid_size: int = 199
    tolerance: float = 1e-8

    def __post_init__(self) -> None:
        lower, upper = self.bounds
        if not -1.0 < lower < upper < 1.0:
            raise InputValidationError("rho bounds must satisfy -1 < lower < upper < 1.")
        if int(self.grid_size) < 3:
            raise InputValidationError("grid_size must be at least 3.")
        if not 0.0 < self.tolerance < 1e-2:
            raise InputValidationError("tolerance must lie in (0, 0.01).")


@dataclass(frozen=True)
class RhoSearchResult:
    """Outcome of a ``rho`` search.

    Attributes
    ----------
    rho : float
        Minimiser of the criterion.
    objective : float
        Criterion at ``rho`` (negative log-likelihood or RSS).
    grid, profile : np.ndarray
        Grid points and criterion values (``inf`` where evaluation failed).
    n_evaluations : int
        Total criterion evaluations.
    refined : bool
        Whether the local refinement improved on the grid.
    at_bound : bool
        Whether ``rho`` lies on the boundary of the search interval.
    identified : bool
        False when the criterion is numerically flat.
    local_minima : tuple of float
        Grid locations of local minima (more than one signals multimodality).
    """

    rho: float
    objective: float
    grid: np.ndarray
    profile: np.ndarray
    n_evaluations: int
    refined: bool
    at_bound: bool
    identified: bool
    local_minima: tuple[float, ...]


class RhoEstimator:
    """Global-then-local scalar search for ``rho``."""

    def __init__(self, settings: RhoOptimizerSettings | None = None) -> None:
        self.settings: RhoOptimizerSettings = settings or RhoOptimizerSettings()

    def estimate(self, loss: Callable[[float], float]) -> RhoSearchResult:
        """Minimise ``loss`` over the configured interval.

        ``loss`` may raise library errors (or return ``nan``) for invalid
        values of ``rho``; such points count as infinitely bad.

        Raises
        ------
        NumericalError
            If ``loss`` cannot be evaluated anywhere on the grid.
        """
        counter = [0]
        safe_loss = self._safe(loss, counter)
        lower, upper = self.settings.bounds
        grid = np.linspace(lower, upper, int(self.settings.grid_size))
        profile = np.array([safe_loss(rho) for rho in grid])
        finite = np.isfinite(profile)
        if not finite.any():
            raise NumericalError("The rho criterion could not be evaluated on the grid.")

        best = int(np.argmin(np.where(finite, profile, np.inf)))
        rho, objective, refined = float(grid[best]), float(profile[best]), False
        bracket = (grid[max(best - 1, 0)], grid[min(best + 1, grid.size - 1)])
        search = minimize_scalar(
            safe_loss, bounds=bracket, method="bounded",
            options={"xatol": self.settings.tolerance},
        )
        if search.success and np.isfinite(search.fun) and search.fun < objective:
            rho, objective, refined = float(search.x), float(search.fun), True

        edge = max(10.0 * self.settings.tolerance, 1e-7)
        at_bound = abs(rho - lower) <= edge or abs(rho - upper) <= edge
        values = profile[finite]
        spread = float(values.max() - values.min())
        identified = spread > FLATNESS_TOLERANCE * (1.0 + abs(float(values.min())))
        return RhoSearchResult(
            rho=rho,
            objective=objective,
            grid=grid,
            profile=profile,
            n_evaluations=counter[0],
            refined=refined,
            at_bound=bool(at_bound),
            identified=bool(identified),
            local_minima=self._local_minima(grid, profile),
        )

    @staticmethod
    def _local_minima(grid: np.ndarray, profile: np.ndarray) -> tuple[float, ...]:
        """Grid points lower than both neighbours (endpoints: one neighbour)."""
        padded = np.concatenate([[np.inf], profile, [np.inf]])
        is_min = (padded[1:-1] < padded[:-2]) & (padded[1:-1] < padded[2:])
        is_min &= np.isfinite(profile)
        return tuple(float(rho) for rho in grid[is_min])

    @staticmethod
    def _safe(loss: Callable[[float], float], counter: list[int]) -> Callable[[float], float]:
        """Wrap ``loss`` so numerical failures return ``inf``."""

        def wrapped(rho: float) -> float:
            counter[0] += 1
            try:
                value = float(loss(float(rho)))
            except (TempDisaggError, ValueError, ArithmeticError, np.linalg.LinAlgError) as exc:
                logger.debug("rho=%.6f rejected: %s", rho, exc)
                return np.inf
            return value if np.isfinite(value) else np.inf

        return wrapped
