"""Backward (and gap) splicing of a legacy series onto a current series.

Statistical offices periodically change methodology, leaving a long *legacy*
series and a shorter *current* one. Splicing (retropolation) rescales the
legacy series so that it can extend the current one. Methods:

``level``
    Constant additive link: ``current - legacy`` averaged over the overlap.
``ratio``
    Constant multiplicative link: ``sum(current) / sum(legacy)`` over the
    overlap (ratio of sums, robust to small individual values).
``affine``
    ``current = a + b * legacy`` estimated by OLS on the overlap.
``interpolated``
    De la Fuente (2014): the discrepancy observed at the link period is
    assumed to have built up gradually since an anchor period, where the
    legacy series is taken as correct. The link factor is interpolated
    linearly from neutral at the anchor to its observed value at the link.

Where the current series is observed it is always kept unchanged.

Reference: de la Fuente Moreno, A. (2014). A "mixed" splicing procedure for
economic time series. Estadística Española, 56(183), 107-121.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from tempdisagg.exceptions import InputValidationError


class SplicingMethod(str, Enum):
    """Available splicing methods."""

    LEVEL = "level"
    RATIO = "ratio"
    AFFINE = "affine"
    INTERPOLATED = "interpolated"


class InterpolationKind(str, Enum):
    """Link used by the interpolated method."""

    RATIO = "ratio"
    LEVEL = "level"


@dataclass(frozen=True)
class SplicingResult:
    """Output of :meth:`SeriesSplicer.splice`.

    Attributes
    ----------
    values : pd.Series
        Spliced series.
    method : SplicingMethod
        Method used.
    overlap_index : pd.Index
        Periods where both series were observed and used for the link.
    link : dict of str to float
        Link parameters (e.g. ``{"ratio": 1.03}`` or ``{"a": ..., "b": ...}``).
    filled : pd.Series
        Boolean mask of periods filled from the legacy series.
    """

    values: pd.Series
    method: SplicingMethod
    overlap_index: pd.Index
    link: dict[str, float]
    filled: pd.Series


class SeriesSplicer:
    """Splice a legacy series onto a current series.

    Parameters
    ----------
    method : SplicingMethod or str, default "ratio"
    overlap_periods : int, optional
        Use only the first ``overlap_periods`` overlapping periods to compute
        the link (closest to the methodological break). Default: all.
    interpolation_kind : {"ratio", "level"}, default "ratio"
        Link interpolated by the ``interpolated`` method.
    anchor : label, optional
        Period where the legacy series is assumed correct for the
        ``interpolated`` method. Default: the first legacy observation.
    """

    def __init__(
        self,
        method: SplicingMethod | str = SplicingMethod.RATIO,
        overlap_periods: int | None = None,
        interpolation_kind: InterpolationKind | str = InterpolationKind.RATIO,
        anchor: object | None = None,
    ) -> None:
        try:
            self.method: SplicingMethod = SplicingMethod(method)
            self.interpolation_kind: InterpolationKind = InterpolationKind(interpolation_kind)
        except ValueError as exc:
            raise InputValidationError(str(exc)) from exc
        if overlap_periods is not None and overlap_periods < 1:
            raise InputValidationError("overlap_periods must be a positive integer.")
        self.overlap_periods: int | None = overlap_periods
        self.anchor: object | None = anchor

    def splice(self, legacy: pd.Series, current: pd.Series) -> SplicingResult:
        """Return the current series extended with the rescaled legacy series."""
        legacy_aligned, current_aligned = self._align(legacy, current)
        both = legacy_aligned.notna() & current_aligned.notna()
        overlap = legacy_aligned.index[both]
        if self.overlap_periods is not None:
            overlap = overlap[: self.overlap_periods]
        if len(overlap) == 0:
            raise InputValidationError("The series do not overlap; a link cannot be estimated.")

        legacy_link = legacy_aligned.loc[overlap].to_numpy(dtype=float)
        current_link = current_aligned.loc[overlap].to_numpy(dtype=float)
        to_fill = current_aligned.isna() & legacy_aligned.notna()

        if self.method is SplicingMethod.INTERPOLATED:
            adjusted, link = self._interpolated(legacy_aligned, current_aligned, overlap)
        else:
            adjusted, link = self._constant_link(legacy_aligned, legacy_link, current_link)

        values = current_aligned.where(~to_fill, adjusted)
        values.name = current.name
        return SplicingResult(values, self.method, overlap, link, to_fill.rename("filled"))

    # ------------------------------------------------------------------
    # Methods
    # ------------------------------------------------------------------
    def _constant_link(
        self, legacy: pd.Series, legacy_link: np.ndarray, current_link: np.ndarray
    ) -> tuple[pd.Series, dict[str, float]]:
        if self.method is SplicingMethod.LEVEL:
            shift = float(np.mean(current_link - legacy_link))
            return legacy + shift, {"level": shift}
        if self.method is SplicingMethod.RATIO:
            denominator = float(np.sum(legacy_link))
            if denominator == 0.0:
                raise InputValidationError("The legacy series sums to zero on the overlap.")
            ratio = float(np.sum(current_link)) / denominator
            return legacy * ratio, {"ratio": ratio}
        # Affine
        if len(legacy_link) < 2 or np.ptp(legacy_link) == 0.0:
            raise InputValidationError(
                "The affine method needs at least two overlap periods with "
                "different legacy values."
            )
        slope, intercept = np.polyfit(legacy_link, current_link, deg=1)
        return intercept + slope * legacy, {"a": float(intercept), "b": float(slope)}

    def _interpolated(
        self, legacy: pd.Series, current: pd.Series, overlap: pd.Index
    ) -> tuple[pd.Series, dict[str, float]]:
        link_label = overlap[0]
        legacy_value = float(legacy.loc[link_label])
        observed = legacy.dropna()
        anchor = self.anchor if self.anchor is not None else observed.index[0]
        if anchor not in legacy.index:
            raise InputValidationError(f"Anchor {anchor!r} is not in the legacy index.")

        positions = np.arange(len(legacy), dtype=float)
        anchor_pos = float(legacy.index.get_loc(anchor))
        link_pos = float(legacy.index.get_loc(link_label))
        if anchor_pos >= link_pos:
            raise InputValidationError("The anchor must precede the first overlap period.")

        # Weight goes from 0 at the anchor to 1 at the link and stays at 1
        # afterwards (forward gaps use the full link).
        weight = np.clip((positions - anchor_pos) / (link_pos - anchor_pos), 0.0, 1.0)
        current_value = float(current.loc[link_label])
        if self.interpolation_kind is InterpolationKind.RATIO:
            if legacy_value == 0.0:
                raise InputValidationError("The legacy value at the link period is zero.")
            ratio = current_value / legacy_value
            factor = 1.0 + (ratio - 1.0) * weight
            return legacy * factor, {"ratio_at_link": ratio}
        shift = current_value - legacy_value
        return legacy + shift * weight, {"level_at_link": shift}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _align(self, legacy: pd.Series, current: pd.Series) -> tuple[pd.Series, pd.Series]:
        for label, series in (("legacy", legacy), ("current", current)):
            if not isinstance(series, pd.Series):
                raise InputValidationError(f"{label} must be a pandas Series.")
            if not series.index.is_unique:
                raise InputValidationError(f"{label} index must be unique.")
        index = legacy.index.union(current.index).sort_values()
        legacy_aligned = pd.to_numeric(legacy, errors="raise").reindex(index).astype(float)
        current_aligned = pd.to_numeric(current, errors="raise").reindex(index).astype(float)
        return legacy_aligned, current_aligned
