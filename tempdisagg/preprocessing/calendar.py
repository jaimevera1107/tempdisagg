"""Calendars: how high-frequency periods nest inside low-frequency periods.

Each input format has its own labels, but completion and padding only need
four operations, which every calendar implements:

* ``subperiods(low)``      ordered high-frequency labels inside ``low``;
* ``parents(high_index)``  low-frequency label of each high-frequency label;
* ``shift(low, k)``        the low-frequency label ``k`` periods away;
* ``low_range(a, b)``      every low-frequency label from ``a`` to ``b``.

=================  ==========================  ===============================
Calendar           Labels                      Used by
=================  ==========================  ===============================
PeriodCalendar     pandas Periods              ``fit`` (calendar aware)
RatioCalendar      integers (positions)        ``fit_arrays``
GrainCalendar      (index, grain) pairs        ``fit_long_format``
=================  ==========================  ===============================
"""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
import pandas as pd

from tempdisagg.exceptions import InputValidationError


class Calendar(Protocol):
    """Operations needed to build a complete time grid."""

    def subperiods(self, low: object) -> pd.Index:
        """High-frequency labels inside ``low`` (chronological)."""

    def parents(self, high: pd.Index) -> pd.Index:
        """Low-frequency label of each high-frequency label."""

    def shift(self, low: object, steps: int) -> object:
        """Low-frequency label ``steps`` periods after ``low``."""

    def low_range(self, first: object, last: object) -> pd.Index:
        """Every low-frequency label from ``first`` to ``last``."""

    def high_index(self, labels: Sequence[object]) -> pd.Index:
        """Build an index of high-frequency labels."""


class PeriodCalendar:
    """Calendar of pandas periods (e.g. years containing months or days)."""

    def __init__(self, low_freq: object, high_freq: object) -> None:
        self.low_freq = low_freq
        self.high_freq = high_freq

    def subperiods(self, low: pd.Period) -> pd.Index:
        index = pd.period_range(low.asfreq(self.high_freq, how="start"),
                                low.asfreq(self.high_freq, how="end"), freq=self.high_freq)
        if len(index) < 2:
            raise InputValidationError(
                "The indicator frequency must be higher than the target frequency."
            )
        return index

    def parents(self, high: pd.Index) -> pd.Index:
        return pd.PeriodIndex(high, freq=self.high_freq).asfreq(self.low_freq)

    def shift(self, low: pd.Period, steps: int) -> pd.Period:
        return low + steps

    def low_range(self, first: pd.Period, last: pd.Period) -> pd.Index:
        return pd.period_range(first, last, freq=self.low_freq)

    def high_index(self, labels: Sequence[object]) -> pd.Index:
        return pd.PeriodIndex(list(labels), freq=self.high_freq)


class RatioCalendar:
    """Integer positions with a fixed number of sub-periods per period."""

    def __init__(self, ratio: int) -> None:
        if isinstance(ratio, bool) or not isinstance(ratio, (int, np.integer)) or ratio < 2:
            raise InputValidationError("frequency_ratio must be an integer >= 2.")
        self.ratio = int(ratio)

    def subperiods(self, low: int) -> pd.Index:
        return pd.Index(range(int(low) * self.ratio, (int(low) + 1) * self.ratio))

    def parents(self, high: pd.Index) -> pd.Index:
        return pd.Index(np.asarray(high, dtype=int) // self.ratio)

    def shift(self, low: int, steps: int) -> int:
        return int(low) + steps

    def low_range(self, first: int, last: int) -> pd.Index:
        return pd.Index(range(int(first), int(last) + 1))

    def high_index(self, labels: Sequence[object]) -> pd.Index:
        return pd.Index(list(labels), dtype=int)


class GrainCalendar:
    """``(index, grain)`` pairs of the 0.2.x long format.

    Every low-frequency label has the same grains (the sorted union of those
    observed). Shifting and gap filling require integer low labels (years).
    """

    def __init__(self, grains: Sequence[object], low_labels: pd.Index) -> None:
        self.grains = tuple(sorted(set(grains)))
        if len(self.grains) < 2:
            raise InputValidationError("At least two distinct grains are required.")
        self.numeric = pd.api.types.is_integer_dtype(low_labels)
        self._labels = pd.Index(sorted(set(low_labels)))

    def subperiods(self, low: object) -> pd.Index:
        return pd.MultiIndex.from_tuples([(low, grain) for grain in self.grains])

    def parents(self, high: pd.Index) -> pd.Index:
        return pd.Index(high.get_level_values(0))

    def shift(self, low: object, steps: int) -> object:
        if not self.numeric:
            raise InputValidationError("Padding requires integer low-frequency labels.")
        return int(low) + steps

    def low_range(self, first: object, last: object) -> pd.Index:
        if not self.numeric:
            return self._labels[(self._labels >= first) & (self._labels <= last)]
        return pd.Index(range(int(first), int(last) + 1))

    def high_index(self, labels: Sequence[object]) -> pd.Index:
        return pd.MultiIndex.from_tuples(list(labels))
