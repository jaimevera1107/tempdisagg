"""Filling and extending one-dimensional series.

These helpers create the artificial values used by completion and padding.
They work on positions (0, 1, 2, ...) of a regular grid, so the result does
not depend on how periods are labelled.
"""

from __future__ import annotations

import numpy as np

from tempdisagg.exceptions import InputValidationError
from tempdisagg.preprocessing.settings import EdgeMethod


def extend(
    positions: np.ndarray,
    values: np.ndarray,
    targets: np.ndarray,
    method: EdgeMethod,
    window: int,
) -> np.ndarray:
    """Extend known ``(positions, values)`` to ``targets`` beyond one edge.

    Parameters
    ----------
    positions, values : np.ndarray
        Known points, sorted by position.
    targets : np.ndarray
        Positions to fill; all on the same side of the known points.
    method : EdgeMethod
        ``linear``: OLS line through the ``window`` known points closest to
        the targets. ``constant``: value of the closest known point.
    window : int
        Number of known points used by the linear fit (>= 2).
    """
    if positions.size == 0:
        raise InputValidationError("Cannot extend a series without known values.")
    after = targets.min() > positions.max()
    if method is EdgeMethod.CONSTANT or positions.size < 2:
        return np.full(targets.shape, values[-1] if after else values[0], dtype=float)
    tail = slice(-window, None) if after else slice(0, window)
    slope, intercept = np.polyfit(positions[tail].astype(float), values[tail], deg=1)
    return intercept + slope * targets.astype(float)


def fill_missing(values: np.ndarray, method: EdgeMethod, window: int) -> np.ndarray:
    """Fill NaN entries: linear interpolation inside, :func:`extend` at edges."""
    values = np.asarray(values, dtype=float)
    missing = np.isnan(values)
    if not missing.any():
        return values.copy()
    positions = np.arange(values.size)
    known = positions[~missing]
    if known.size == 0:
        raise InputValidationError("An indicator has no observed values.")
    filled = values.copy()
    inside = missing & (positions > known[0]) & (positions < known[-1])
    filled[inside] = np.interp(positions[inside], known, values[~missing])
    for edge in (positions[missing & (positions < known[0])],
                 positions[missing & (positions > known[-1])]):
        if edge.size:
            filled[edge] = extend(known, values[~missing], edge, method, window)
    return filled
