"""Optional plotting helpers (requires matplotlib)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tempdisagg.results import DisaggregationResult

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure


def plot_result(result: DisaggregationResult, **figure_kwargs: Any) -> "Figure":
    """Plot the high-frequency estimate and the low-frequency residuals.

    The top panel shows the estimate; extrapolated periods (not covered by an
    observed low-frequency value) are drawn dashed. The bottom panel shows
    the low-frequency residuals of the regression or benchmark.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError("plot_result requires matplotlib: pip install tempdisagg[plot]") from exc

    figure, (top, bottom) = plt.subplots(2, 1, sharex=False, **figure_kwargs)
    values = result.values
    positions = np.arange(len(values))
    constrained = result.data.aggregation_matrix.any(axis=0)[~result.data.artificial_high]

    top.plot(positions, values.to_numpy(), color="C0", label="estimate")
    if result.prediction_se is not None:
        band = result.prediction_interval()
        top.fill_between(positions, band["lower"].to_numpy(), band["upper"].to_numpy(),
                         color="C0", alpha=0.2, label="95% interval (given rho)")
    if not constrained.all():
        masked = np.where(constrained, np.nan, values.to_numpy())
        top.plot(positions, masked, color="C1", linestyle="--", label="extrapolated")
    top.set_title(f"Temporal disaggregation ({result.method_name})")
    top.legend()
    _label_axis(top, values.index, positions)

    residuals = result.residuals
    bottom.bar(np.arange(len(residuals)), residuals.to_numpy(), color="C2")
    bottom.axhline(0.0, color="black", linewidth=0.8)
    bottom.set_title("Low-frequency residuals")
    _label_axis(bottom, residuals.index, np.arange(len(residuals)))
    figure.tight_layout()
    return figure


def _label_axis(axis: Any, index: Any, positions: np.ndarray, max_ticks: int = 10) -> None:
    """Place at most ``max_ticks`` readable tick labels."""
    step = max(1, len(positions) // max_ticks)
    ticks = positions[::step]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(index[i]) for i in ticks], rotation=45, ha="right")
