"""Conformity between a disaggregated series and its indicators.

Every constrained method reproduces the low-frequency data, so the totals
cannot tell methods apart. What differs is how the high-frequency estimate
moves relative to the indicator. Quilis (2018, Table 4) compares methods
with four statistics, computed here for each indicator ``x`` and the
estimate ``y`` over the returned periods:

``corr_levels``        correlation of ``y`` and ``x`` in levels
``corr_changes``       correlation of their period-on-period changes
``sd_changes``         standard deviation of the changes of ``y``
``volatility_ratio``   ``sd(changes of y) / sd(changes of x)``

Changes are growth rates in percent when both series are strictly positive
and first differences otherwise (``change`` column). A method whose
``corr_changes`` is clearly lower, or whose volatility ratio is clearly
higher, than the others is adding movement that the indicator does not have
(Quilis finds this for the dynamic Chow-Lin model). Indicator values filled
by completion are included; check ``result.preprocessing`` if there are many.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from tempdisagg.exceptions import InputValidationError
from tempdisagg.results import DisaggregationResult

#: Minimum number of periods needed for meaningful correlations.
MIN_PERIODS: int = 4


def _changes(values: np.ndarray, growth: bool) -> np.ndarray:
    if growth:
        return 100.0 * (values[1:] / values[:-1] - 1.0)
    return np.diff(values)


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0.0 or np.std(b) == 0.0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def conformity(
    result: DisaggregationResult,
    indicators: pd.Series | pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Conformity statistics between the estimate and each indicator.

    Parameters
    ----------
    result : DisaggregationResult
        A fitted disaggregation.
    indicators : pd.Series or pd.DataFrame, optional
        Series to compare with, indexed like ``result.values``. Default: the
        indicators used in the fit (constant columns, such as dummies with no
        variation, are skipped). Pass one explicitly for methods fitted
        without indicators.

    Returns
    -------
    pd.DataFrame
        One row per indicator with ``change``, ``corr_levels``,
        ``corr_changes``, ``sd_changes``, ``sd_changes_indicator`` and
        ``volatility_ratio``.
    """
    estimate = result.values
    if indicators is None:
        real = ~result.data.artificial_high
        frame = pd.DataFrame(result.data.indicators[real], index=result.data.high_index[real],
                             columns=list(result.data.indicator_names))
    elif isinstance(indicators, pd.Series):
        frame = indicators.to_frame(name=str(indicators.name or "indicator"))
    elif isinstance(indicators, pd.DataFrame):
        frame = indicators
    else:
        raise InputValidationError("indicators must be a pandas Series or DataFrame.")
    frame = frame.reindex(estimate.index)
    frame = frame.loc[:, frame.notna().all() & (frame.std() > 0)]
    if frame.shape[1] == 0:
        raise InputValidationError(
            "No varying indicator to compare with: pass `indicators` explicitly.")
    if len(estimate) < MIN_PERIODS:
        raise InputValidationError(f"At least {MIN_PERIODS} periods are required.")

    y = estimate.to_numpy(dtype=float)
    rows = {}
    for name in frame.columns:
        x = frame[name].to_numpy(dtype=float)
        growth = bool(np.all(y > 0) and np.all(x > 0))
        dy, dx = _changes(y, growth), _changes(x, growth)
        sd_y, sd_x = float(np.std(dy, ddof=1)), float(np.std(dx, ddof=1))
        rows[name] = {
            "change": "growth %" if growth else "difference",
            "corr_levels": _correlation(y, x),
            "corr_changes": _correlation(dy, dx),
            "sd_changes": sd_y,
            "sd_changes_indicator": sd_x,
            "volatility_ratio": sd_y / sd_x if sd_x > 0 else float("nan"),
        }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index.name = "indicator"
    return table
