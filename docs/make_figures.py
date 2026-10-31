"""Build the figures in docs/images/ (run from the repository root).

    python docs/make_figures.py

Figures
-------
hero.png                 quarterly sales estimated from annual totals (README)
r_parity_summary.png     largest relative difference with R, per case
r_parity_series.png      Python and R estimates for six methods
r_parity_statistics.png  agreement of each reported statistic

The R values come from tests/reference/, generated with
tests/reference/generate_r_reference.R (R 4.5.3, tempdisagg 1.2.0).
"""

from __future__ import annotations

import dataclasses
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tempdisagg import TemporalDisaggregator  # noqa: E402
from tempdisagg.datasets import load_swisspharma  # noqa: E402
from tests.reference.r_cases import CASES  # noqa: E402

REFERENCE = ROOT / "tests" / "reference"
OUTPUT = ROOT / "docs" / "images"
INK, MUTED, ACCENT, ACCENT_2, GRID = "#1f2933", "#7b8794", "#0b6e99", "#d9822b", "#e4e7eb"
STATISTICS = {"coefficients": None, "std. errors": None, "s_2": "sigma2_ml",
              "s_2_gls": "sigma2_gls", "rss": "rss", "tss": "tss", "r.squared": "r_squared",
              "adj.r.squared": "adj_r_squared", "logl": "log_likelihood", "aic": "aic",
              "bic": "bic"}


def style() -> None:
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
        "axes.spines.right": False, "legend.frameon": False,
    })


# ----------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------
def r_inputs() -> tuple[pd.Series, pd.DataFrame]:
    sales, trade = load_swisspharma()
    return sales, trade.rename(columns={"exports": "exports_q", "imports": "imports_q"})


def fit_case(case_id: str, rho: float | None = None):
    target, columns, preset, overrides, conversion = CASES[case_id]
    sales, trade = r_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TemporalDisaggregator(preset, conversion, **overrides)
        if rho is not None:
            spec = dataclasses.replace(model.method, rho_method="fixed", fixed_rho=rho,
                                       truncated_rho=0.0)
            model = TemporalDisaggregator(spec, conversion)
        return model.fit(sales, trade[columns]) if columns else model.fit(sales, high_freq="Q")


def r_values(case_id: str) -> pd.Series:
    frame = pd.read_csv(REFERENCE / f"r_{case_id}.csv")
    index = pd.PeriodIndex([pd.Period(year=int(y), quarter=int(q), freq="Q")
                            for y, q in zip(frame["year"], frame["quarter"])])
    return pd.Series(frame["value"].to_numpy(), index=index)


def r_meta(case_id: str) -> pd.Series:
    meta = pd.read_csv(REFERENCE / "r_meta.csv")
    return meta[meta["id"] == case_id].set_index("key")["value"]


def relative_differences(result, case_id: str) -> dict[str, float]:
    expected = r_values(case_id)
    diffs = {"prediction": float(np.max(np.abs(
        result.values.reindex(expected.index) / expected - 1)))}
    if result.regression is None:
        return diffs
    meta, reg = r_meta(case_id), result.regression
    k = len(reg.coefficients)
    diffs["coefficients"] = float(np.max(np.abs(
        reg.coefficients.to_numpy() / meta[[f"coef_{i + 1}" for i in range(k)]].to_numpy() - 1)))
    diffs["std. errors"] = float(np.max(np.abs(
        reg.standard_errors.to_numpy() / meta[[f"se_{i + 1}" for i in range(k)]].to_numpy() - 1)))
    for label, attribute in STATISTICS.items():
        if attribute is None or np.isnan(meta.get(label, np.nan)):
            continue
        value, reference = getattr(reg, attribute), meta[label]
        absolute = label in ("r.squared", "adj.r.squared")
        diffs[label] = float(abs(value - reference) if absolute else abs(value / reference - 1))
    return diffs


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def hero() -> None:
    sales, trade = load_swisspharma()
    result = TemporalDisaggregator("chow-lin-maxlog").fit(sales, trade)
    band = result.prediction_interval(0.95)
    x = band.index.to_timestamp()
    annual_mean = (sales / 4).reindex(band.index.asfreq("Y")).to_numpy()

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.step(x, annual_mean, where="post", color=MUTED, linewidth=1.2,
            label="annual total / 4 (observed)")
    ax.fill_between(x, band["lower"], band["upper"], color=ACCENT, alpha=0.18, linewidth=0,
                    label="95% interval")
    ax.plot(x, band["estimate"], color=ACCENT, linewidth=1.8, label="quarterly estimate")
    first, last = sales.index[0].start_time, sales.index[-1].end_time
    shaded = ((x[0], first, "retropolated", "left"),
              (last, x[-1] + pd.Timedelta(days=90), "extrapolated", "right"))
    for start, end, text, align in shaded:
        ax.axvspan(start, end, color=ACCENT_2, alpha=0.12, linewidth=0)
        ax.text(start if align == "left" else end, 285, text, color=ACCENT_2,
                fontsize=8.5, ha=align, va="top")
    ax.set_ylim(0, 300)
    ax.set_title("Quarterly sales from annual totals (Chow-Lin, exports and imports)", loc="left")
    ax.set_ylabel("sales index")
    ax.legend(loc="lower right", fontsize=9)
    fig.savefig(OUTPUT / "hero.png")
    plt.close(fig)


def parity_summary() -> None:
    rows = []
    for case_id in sorted(CASES):
        result = fit_case(case_id)
        worst = max(relative_differences(result, case_id).values())
        refit = None
        if result.rho is not None and result.rho.estimated and not result.rho.truncated:
            refit = max(relative_differences(fit_case(case_id, float(r_meta(case_id)["rho"])),
                                             case_id).values())
        rows.append((case_id, worst, refit))
    frame = pd.DataFrame(rows, columns=["case", "direct", "refit"]).sort_values("direct")
    floor = 1e-16

    fig, ax = plt.subplots(figsize=(8.5, 7.2))
    y = np.arange(len(frame))
    ax.barh(y, np.maximum(frame["direct"], floor), color=ACCENT, height=0.6,
            label="as fitted")
    refit = frame["refit"].notna()
    ax.scatter(np.maximum(frame.loc[refit, "refit"], floor), y[refit.to_numpy()], color=ACCENT_2,
               zorder=3, s=28, label="refit with R's rho (estimated-rho cases)")
    ax.axvline(1e-8, color=INK, linestyle="--", linewidth=0.9)
    ax.text(1.3e-8, len(frame) - 0.6, "test tolerance 1e-8", fontsize=8.5, color=INK)
    ax.set_xscale("log")
    ax.set_xlim(1e-16, 1e-2)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["case"], fontsize=8.5)
    ax.set_xlabel("largest relative difference with R (predictions, coefficients, statistics)")
    ax.set_title(f"Agreement with R tempdisagg 1.2.0 ({len(frame)} cases)", loc="left",
                 pad=26)
    ax.text(0, 1.012, "Bars: model as fitted. With estimated rho, R's optimiser stops near "
            "1e-4; dots: same model re-fitted at R's rho.", transform=ax.transAxes,
            fontsize=8.5, color=MUTED)
    ax.legend(loc="lower right", fontsize=8.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(OUTPUT / "r_parity_summary.png")
    plt.close(fig)


def parity_series() -> None:
    cases = [("cl_maxlog_two", "Chow-Lin (two indicators)"), ("lit_fixed", "Litterman, rho = 0.7"),
             ("fernandez", "Fernandez"), ("dyn_fixed", "Dynamic Chow-Lin, rho = 0.5"),
             ("dc_prop", "Denton-Cholette, proportional"),
             ("dc_noind", "Denton-Cholette, no indicator")]
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.4), sharex=True)
    for ax, (case_id, title) in zip(axes.ravel(), cases):
        expected = r_values(case_id)
        result = fit_case(case_id).values.reindex(expected.index)
        diff = float(np.max(np.abs(result / expected - 1)))  # over the whole output
        sample = expected.index >= pd.Period("1975Q1", freq="Q")
        expected, result = expected[sample], result[sample]
        x = expected.index.to_timestamp()
        ax.plot(x, result, color=ACCENT, linewidth=1.6, label="Python")
        ax.plot(x[::3], expected.to_numpy()[::3], "o", color=ACCENT_2, markersize=3,
                label="R (every 3rd quarter)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.xaxis.set_major_locator(matplotlib.dates.YearLocator(10))
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%Y"))
        ax.text(0.02, 0.9, f"max rel. diff {diff:.0e}", transform=ax.transAxes, fontsize=8.5,
                color=INK)
    axes[0, 0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Quarterly swisspharma sales: Python vs R (shown from 1975; differences "
                 "computed over the full output)", x=0.01, ha="left", fontweight="bold",
                 color=INK)
    fig.tight_layout()
    fig.savefig(OUTPUT / "r_parity_series.png")
    plt.close(fig)


def parity_statistics() -> None:
    collected: dict[str, list[float]] = {}
    for case_id in sorted(CASES):
        result = fit_case(case_id)
        if result.rho is not None and result.rho.estimated and not result.rho.truncated:
            result = fit_case(case_id, float(r_meta(case_id)["rho"]))
        for label, value in relative_differences(result, case_id).items():
            if label != "prediction":
                collected.setdefault(label, []).append(max(value, 1e-16))
    labels = [label for label in STATISTICS if label in collected]
    worst = [max(collected[label]) for label in labels]
    median = [float(np.median(collected[label])) for label in labels]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.bar(labels, worst, color=ACCENT, width=0.6, label="worst case")
    ax.scatter(labels, median, color=ACCENT_2, zorder=3, s=30, label="median over cases")
    ax.axhline(1e-8, color=INK, linestyle="--", linewidth=0.9)
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 1e-4)
    ax.set_ylabel("difference with R")
    ax.set_title("Reported statistics vs R (regression cases, rho fixed at R's value)",
                 loc="left")
    ax.legend(loc="upper right", fontsize=8.5)
    ax.tick_params(axis="x", rotation=30)
    ax.grid(axis="x", visible=False)
    fig.savefig(OUTPUT / "r_parity_statistics.png")
    plt.close(fig)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    style()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hero()
        parity_summary()
        parity_series()
        parity_statistics()
    print("Figures written to", OUTPUT)


if __name__ == "__main__":
    main()
