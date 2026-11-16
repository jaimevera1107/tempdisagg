"""Optional plotting helpers (requires matplotlib)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from tempdisagg.results import DisaggregationResult

if TYPE_CHECKING:  # pragma: no cover
    from matplotlib.figure import Figure


def plot_result(result: DisaggregationResult, **figure_kwargs: Any) -> Figure:
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
    constrained = np.asarray(result.data.aggregation_matrix.any(axis=0))[
        ~result.data.artificial_high
    ]

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


def plot_rho_profile(result: DisaggregationResult, ax: Any = None) -> Any:
    """Plot the estimation criterion of rho over the search grid.

    For ``maxlog`` methods the curve is the log-likelihood (higher is
    better); for ``minrss`` the weighted residual sum of squares (lower is
    better). Marked on the curve: the optimum, the value actually used (after
    truncation), the truncation bound and every local optimum. A flat curve
    means rho is poorly identified; several peaks mean the optimum is fragile.
    Quilis (2018) uses this plot, for instance, to see that dropping the
    intercept sharpens the likelihood.

    Parameters
    ----------
    result : DisaggregationResult
        A fit whose rho was estimated (``*-maxlog`` or ``*-minrss``).
    ax : matplotlib Axes, optional
        Axes to draw on; a new figure is created otherwise.

    Returns
    -------
    matplotlib Axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "plot_rho_profile requires matplotlib: pip install tempdisagg[plot]") from exc
    rho = result.rho
    if rho is None or rho.profile is None:
        raise ValueError("The fit has no rho profile: use a method that estimates rho.")
    profile = rho.profile
    loglik = rho.method == "maxlog"
    values = -profile.to_numpy() if loglik else profile.to_numpy()
    finite = np.isfinite(values)
    grid = profile.index.to_numpy()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 3.2))
    ax.plot(grid[finite], values[finite], color="#0b6e99", linewidth=1.6)
    if rho.objective is not None and rho.unconstrained_value is not None:
        best = -rho.objective if loglik else rho.objective
        ax.scatter([rho.unconstrained_value], [best], color="#d9822b", zorder=3,
                   label=f"optimum {rho.unconstrained_value:.3f}")
    for minimum in rho.local_minima:
        if rho.unconstrained_value is None or abs(minimum - rho.unconstrained_value) > 1e-2:
            ax.axvline(minimum, color="#7b8794", linestyle=":", linewidth=0.9)
    if rho.truncated:
        ax.axvline(rho.value, color="#1f2933", linestyle="--", linewidth=0.9,
                   label=f"used (truncated) {rho.value:.3f}")
    ax.set_xlabel("rho")
    ax.set_ylabel("log-likelihood" if loglik else "weighted RSS")
    ax.set_title(f"rho profile ({result.method_name})", loc="left")
    ax.legend(loc="best", fontsize=8.5, frameon=False)
    return ax


def _label_axis(axis: Any, index: Any, positions: np.ndarray, max_ticks: int = 10) -> None:
    """Place at most ``max_ticks`` readable tick labels."""
    step = max(1, len(positions) // max_ticks)
    ticks = positions[::step]
    axis.set_xticks(ticks)
    axis.set_xticklabels([str(index[i]) for i in ticks], rotation=45, ha="right")
