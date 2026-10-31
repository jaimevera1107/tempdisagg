"""Deterministic indicators for structural breaks and one-off events.

Because regression methods accept several indicators, a structural break can
be represented by adding a dummy regressor next to the economic indicators:

* :func:`level_shift` — 0 before ``start``, 1 from ``start`` on: a permanent
  change in level (e.g. a methodological change or the 2020 shock);
* :func:`impulse` — 1 in the given periods, 0 elsewhere: a transitory event.

Identification. The coefficient of a dummy is estimated from the
low-frequency data, so the event must be visible there: a level shift needs
low-frequency observations on both sides of ``start``, and a dummy that is
constant over the sample is collinear with the intercept (the library then
raises :class:`~tempdisagg.exceptions.SingularDesignError`). With few
low-frequency observations every extra regressor costs a degree of freedom.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from tempdisagg.exceptions import InputValidationError


def level_shift(index: pd.Index, start: object, name: str | None = None) -> pd.Series:
    """Return a step dummy: 0 before ``start``, 1 from ``start`` (inclusive).

    Parameters
    ----------
    index : pd.Index
        Ordered high-frequency index (typically the indicators' PeriodIndex).
    start : label
        First period of the new level.
    name : str, optional
        Series name (default ``"shift_<start>"``).
    """
    if not index.is_monotonic_increasing:
        raise InputValidationError("index must be sorted.")
    if isinstance(index, pd.PeriodIndex) and not isinstance(start, pd.Period):
        start = pd.Period(start, freq=index.freq)
    values = (index >= start).astype(float)
    if values.min() == values.max():
        raise InputValidationError(f"{start} does not split the index; the dummy is constant.")
    return pd.Series(values, index=index, name=name or f"shift_{start}")


def impulse(index: pd.Index, periods: Iterable[object], name: str | None = None) -> pd.Series:
    """Return a dummy equal to 1 in ``periods`` and 0 elsewhere."""
    labels = list(periods)
    if isinstance(index, pd.PeriodIndex):
        labels = [p if isinstance(p, pd.Period) else pd.Period(p, freq=index.freq) for p in labels]
    missing = [label for label in labels if label not in index]
    if missing:
        raise InputValidationError(f"Periods not in the index: {missing}.")
    values = np.asarray(index.isin(labels), dtype=float)
    return pd.Series(values, index=index, name=name or "impulse")
