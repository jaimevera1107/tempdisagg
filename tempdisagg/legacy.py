"""Backward-compatible wrappers for tempdisagg 0.2.x.

These classes keep existing scripts running while emitting
:class:`DeprecationWarning`. Results differ from 0.2.x wherever 0.2.x was
methodologically wrong (see MIGRATION.md). They will be removed in 1.0.
"""

from __future__ import annotations

import dataclasses
import warnings
from typing import Any

import numpy as np
import pandas as pd

from tempdisagg.exceptions import InputValidationError
from tempdisagg.methods import make_method
from tempdisagg.model import TemporalDisaggregator
from tempdisagg.reconciliation.non_negative import NonNegativeReconciler
from tempdisagg.results import DisaggregationResult
from tempdisagg.splicing.series_splicer import SeriesSplicer

#: Default ``rho`` of the fixed-rho methods in 0.2.x.
LEGACY_FIXED_RHO: dict[str, float] = {"chow-lin": 0.5, "litterman": 0.9}
#: 0.2.x Retropolarizer methods still supported, mapped to splicing methods.
LEGACY_SPLICING_METHODS: dict[str, str] = {
    "proportion": "ratio",
    "linear_regression": "affine",
}


def _deprecated(old: str, new: str) -> None:
    warnings.warn(
        f"{old} is deprecated and will be removed in tempdisagg 1.0; use {new}.",
        DeprecationWarning,
        stacklevel=3,
    )


class TempDisaggModel:
    """Deprecated 0.2.x interface; use :class:`TemporalDisaggregator`.

    Differences with 0.2.x: no silent fallback method, targets are never
    interpolated (incomplete groups are completed, unknown targets stay
    unconstrained), no intercept (as in 0.2.x), ``X_col=None`` for methods
    without indicator, and ``method="ensemble"`` is no longer available.
    """

    def __init__(
        self,
        method: str = "chow-lin",
        conversion: str = "sum",
        grain_col: str = "Grain",
        index_col: str = "Index",
        y_col: str = "y",
        X_col: str | None = "X",  # noqa: N803 - legacy argument name
        **ignored: Any,
    ) -> None:
        _deprecated("TempDisaggModel", "TemporalDisaggregator")
        if ignored:
            warnings.warn(
                f"Ignored legacy arguments: {sorted(ignored)} (padding, "
                "interpolation and fallback were removed in 0.3).",
                DeprecationWarning,
                stacklevel=2,
            )
        self.method: str = method
        self.conversion: str = conversion
        self.grain_col: str = grain_col
        self.index_col: str = index_col
        self.y_col: str = y_col
        self.X_col: str = X_col
        self.result_: DisaggregationResult | None = None
        self._adjusted: pd.Series | None = None

    def fit(self, df: pd.DataFrame) -> TempDisaggModel:
        """Fit on the 0.2.x long format."""
        spec = make_method(self.method)
        fields = {f.name for f in dataclasses.fields(spec)}
        overrides: dict[str, Any] = {}
        if "include_intercept" in fields:
            overrides["include_intercept"] = False  # 0.2.x had no intercept
        if self.method in LEGACY_FIXED_RHO:
            overrides["fixed_rho"] = LEGACY_FIXED_RHO[self.method]
        model = TemporalDisaggregator(spec, self.conversion, **overrides)
        if self.X_col is None:
            x_cols: list[str] = []
        else:
            x_cols = [self.X_col] if isinstance(self.X_col, str) else list(self.X_col)
        self.result_ = model.fit_long_format(
            df, self.index_col, self.grain_col, self.y_col, x_cols
        )
        self._adjusted = None
        return self

    def predict(self, full: bool = True) -> np.ndarray:
        """Return the estimate as an ``(n, 1)`` array (0.2.x shape)."""
        return self._require_result().values.to_numpy().reshape(-1, 1)

    def adjust_output(self, full: bool = True) -> np.ndarray:
        """Return the non-negative reconciled estimate as an ``(n, 1)`` array."""
        result = self._require_result()
        self._adjusted = NonNegativeReconciler().reconcile(result)
        return self._adjusted.to_numpy().reshape(-1, 1)

    def summary(self) -> None:
        """Print the summary."""
        print(self._require_result().summary())

    def _require_result(self) -> DisaggregationResult:
        if self.result_ is None:
            raise RuntimeError("Call fit() first.")
        return self.result_


class Retropolarizer:
    """Deprecated 0.2.x interface; use :class:`SeriesSplicer`.

    ``proportion`` maps to ratio splicing and ``linear_regression`` to affine
    splicing. The polynomial, exponential-smoothing and MLP options were
    removed (no methodological basis with few overlapping observations).
    Unlike 0.2.x, interior gaps are no longer linearly interpolated before
    splicing.
    """

    def __init__(self, df: pd.DataFrame, new_col: str, old_col: str, **ignored: Any) -> None:
        _deprecated("Retropolarizer", "SeriesSplicer")
        missing = {new_col, old_col} - set(df.columns)
        if missing:
            raise InputValidationError(f"Missing columns: {sorted(missing)}.")
        self.df: pd.DataFrame = df.copy()
        self.new_col: str = new_col
        self.old_col: str = old_col

    def retropolarize(self, method: str = "proportion") -> pd.Series:
        """Return ``new_col`` completed from ``old_col``."""
        if method not in LEGACY_SPLICING_METHODS:
            raise InputValidationError(
                f"Retropolarizer method '{method}' was removed in 0.3. Supported: "
                f"{sorted(LEGACY_SPLICING_METHODS)}, or use SeriesSplicer directly."
            )
        splicer = SeriesSplicer(method=LEGACY_SPLICING_METHODS[method])
        result = splicer.splice(self.df[self.old_col], self.df[self.new_col])
        return result.values.reindex(self.df.index)
