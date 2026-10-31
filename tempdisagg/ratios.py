"""Disaggregation of rates and proportions.

A rate is not additive: an annual unemployment rate is not the sum, nor in
general the simple mean, of monthly rates. Two coherent approaches:

1. **Numerator and denominator** (:func:`disaggregate_ratio`). When both
   low-frequency totals are known (e.g. informal workers and employed
   workers), disaggregate each with its own model and indicators and compute
   the ratio. The annual ratio of sums reproduces the official ratio
   exactly, and both components remain available.

2. **Rate weighted by its denominator.** When only the annual rate and a
   high-frequency denominator ``N_t`` are known, the rate satisfies
   ``r_year = sum_t N_t r_t / sum_t N_t``: fit with
   ``conversion="average"`` and ``aggregation_weights=N``::

       TemporalDisaggregator("chow-lin-maxlog", "average").fit(
           rate_annual, indicators, aggregation_weights=labour_force_monthly)

The official definition of the aggregate rate must be checked first: the
formulas above assume it is the ratio of totals over the year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from tempdisagg.core.aggregation import Conversion
from tempdisagg.exceptions import InputValidationError
from tempdisagg.model import TemporalDisaggregator
from tempdisagg.results import DisaggregationResult


@dataclass(frozen=True)
class RatioResult:
    """Disaggregated numerator, denominator and their ratio.

    Attributes
    ----------
    numerator, denominator : DisaggregationResult
    ratio : pd.Series
        ``numerator / denominator`` per high-frequency period.
    low_frequency_error : float
        Largest absolute difference between ``sum(num) / sum(den)`` per
        low-frequency period and ``num_low / den_low`` (should be ~0).
    """

    numerator: DisaggregationResult
    denominator: DisaggregationResult
    ratio: pd.Series
    low_frequency_error: float


def disaggregate_ratio(
    numerator_low: pd.Series,
    denominator_low: pd.Series,
    numerator_model: TemporalDisaggregator,
    denominator_model: TemporalDisaggregator,
    numerator_indicators: pd.Series | pd.DataFrame | None = None,
    denominator_indicators: pd.Series | pd.DataFrame | None = None,
    high_freq: str | None = None,
) -> RatioResult:
    """Disaggregate a ratio of totals through its numerator and denominator.

    Both models must use ``conversion="sum"`` (the ratio of sums is then
    preserved exactly). The denominator estimate must be strictly positive.

    Raises
    ------
    InputValidationError
        On non-sum conversions, misaligned outputs or a non-positive
        denominator estimate.
    """
    for model in (numerator_model, denominator_model):
        if model.conversion is not Conversion.SUM:
            raise InputValidationError("Both models must use conversion='sum'.")
    numerator = numerator_model.fit(numerator_low, numerator_indicators, high_freq)
    denominator = denominator_model.fit(denominator_low, denominator_indicators, high_freq)
    if not numerator.values.index.equals(denominator.values.index):
        raise InputValidationError("Numerator and denominator outputs cover different periods.")
    if np.any(denominator.values.to_numpy() <= 0):
        raise InputValidationError(
            "The denominator estimate is not strictly positive; the ratio is undefined. "
            "Consider NonNegativeReconciler or a different model for the denominator."
        )
    ratio = (numerator.values / denominator.values).rename("ratio")

    index = numerator.values.index
    if isinstance(index, pd.PeriodIndex) and isinstance(numerator_low.index, pd.PeriodIndex):
        low_freq = numerator_low.index.freq
        num_sum = numerator.values.groupby(index.asfreq(low_freq)).sum()
        den_sum = denominator.values.groupby(index.asfreq(low_freq)).sum()
        official = (numerator_low / denominator_low).dropna()
        official.index = pd.PeriodIndex(official.index, freq=low_freq)
        implied = (num_sum / den_sum).reindex(official.index)
        error = float(np.nanmax(np.abs(implied.to_numpy() - official.to_numpy())))
    else:
        error = float("nan")
    return RatioResult(numerator, denominator, ratio, error)
