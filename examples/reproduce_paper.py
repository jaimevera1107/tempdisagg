"""Reproduce every number, table and figure of the tempdisagg paper.

Usage
-----
    git clone https://github.com/jaimevera1107/tempdisagg   # holds the R reference files
    pip install "tempdisagg[plot]==0.6.6" statsmodels
    python tempdisagg/examples/reproduce_paper.py --repo tempdisagg --out paper_output

Writes ``figures/*.pdf`` and ``tables/*.tex`` under ``--out`` and prints a log
of every quantity quoted in the text. Monte Carlo experiments use fixed seeds.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import tempdisagg as td  # noqa: E402
from tempdisagg import TemporalDisaggregator  # noqa: E402
from tempdisagg.core.data import DataAligner  # noqa: E402
from tempdisagg.datasets import load_swisspharma  # noqa: E402
from tempdisagg.plotting import plot_rho_profile  # noqa: E402

INK, MUTED, ACCENT, ACCENT_2, GRID = "#1f2933", "#7b8794", "#0b6e99", "#d9822b", "#e4e7eb"
LOG: dict[str, object] = {}


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def style() -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 9.5, "axes.edgecolor": MUTED,
        "axes.labelcolor": INK, "axes.titlecolor": INK, "xtick.color": MUTED,
        "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def quiet(func, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return func(*args, **kwargs)


def fit(preset: str, y, x=None, conversion: str = "sum", **overrides):
    def run():
        model = TemporalDisaggregator(preset, conversion, **overrides)
        return model.fit(y, x) if x is not None else model.fit(y, high_freq="Q")
    return quiet(run)


def fmt(value: float, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:,.{digits}f}"


def sci(value: float | None) -> str:
    if value is None or np.isnan(value):
        return "--"
    if value == 0:
        return "$0$"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"${mantissa}\\times 10^{{{int(exponent)}}}$"


def write_table(out: Path, name: str, header: list[str], rows: list[list[str]],
                align: str) -> None:
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule",
             " & ".join(header) + " \\\\", "\\midrule"]
    lines += [" & ".join(row) + " \\\\" for row in rows]
    lines += ["\\bottomrule", "\\end{tabular}"]
    (out / "tables" / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def tt(text: str) -> str:
    return "\\texttt{" + text.replace("_", "\\_") + "}"


# ----------------------------------------------------------------------
# 1. Swiss pharmaceutical sales (R vignette data)
# ----------------------------------------------------------------------
def swisspharma(out: Path) -> None:
    sales, trade = load_swisspharma()
    result = fit("chow-lin-maxlog", sales, trade[["exports"]])
    reg = result.regression
    LOG["swiss_n_low"], LOG["swiss_n_high"] = int(len(sales)), int(len(result.values))
    LOG["swiss_rho_unconstrained"] = float(result.rho.unconstrained_value)
    LOG["swiss_aggregation_error"] = float(result.aggregation_error())
    LOG["swiss_r2"] = float(reg.r_squared)
    rows = []
    for name in reg.coefficients.index:
        rows.append([tt(name), f"{reg.coefficients[name]:.6g}", f"{reg.standard_errors[name]:.6g}",
                     f"{reg.t_values[name]:.3f}", sci(float(reg.p_values[name]))])
    write_table(out, "swiss_coefficients", ["", "Estimate", "Std.\\ error", "$t$", "$p$"],
                rows, "lrrrr")

    band = result.prediction_interval(0.95)
    x = band.index.to_timestamp()
    annual = (sales / 4).reindex(band.index.asfreq("Y")).to_numpy()
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    ax.step(x, annual, where="post", color=MUTED, linewidth=1.0,
            label="annual total / 4 (observed)")
    ax.fill_between(x, band["lower"], band["upper"], color=ACCENT, alpha=0.2, linewidth=0,
                    label="95% prediction interval")
    ax.plot(x, band["estimate"], color=ACCENT, linewidth=1.4, label="quarterly estimate")
    first, last = sales.index[0].start_time, sales.index[-1].end_time
    for start, end in ((x[0], first), (last, x[-1] + pd.Timedelta(days=90))):
        ax.axvspan(start, end, color=ACCENT_2, alpha=0.12, linewidth=0)
    ax.text(x[0], 270, "retro-\npolated", color=ACCENT_2, fontsize=7, va="top")
    ax.text(x[-1] + pd.Timedelta(days=90), 8, "extrapolated", color=ACCENT_2,
            fontsize=7, va="bottom", ha="right")
    ax.set_ylim(0, 280)
    ax.set_ylabel("sales index")
    ax.legend(loc="upper left", fontsize=7.5, bbox_to_anchor=(0.12, 1.0))
    fig.savefig(out / "figures" / "swisspharma_chowlin_interval.pdf")
    plt.close(fig)

    # rho profile with and without intercept
    no_constant = fit("chow-lin-maxlog", sales, trade[["exports"]], include_intercept=False)
    LOG["swiss_rho_no_intercept"] = float(no_constant.rho.value)
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.5))
    plot_rho_profile(result, ax=axes[0])
    plot_rho_profile(no_constant, ax=axes[1])
    axes[0].set_title("(a) with intercept", loc="left")
    axes[1].set_title("(b) without intercept", loc="left")
    for ax in axes:
        ax.legend(fontsize=6.5, loc="lower center")
    fig.tight_layout()
    fig.savefig(out / "figures" / "rho_profile_intercept.pdf")
    plt.close(fig)

    # backtest
    candidates = {
        "Chow-Lin, exports": td.Candidate(TemporalDisaggregator("chow-lin-maxlog"),
                                          trade[["exports"]]),
        "Chow-Lin, exports + imports": td.Candidate(TemporalDisaggregator("chow-lin-maxlog"),
                                                    trade),
        "Fern\\'andez, exports": td.Candidate(TemporalDisaggregator("fernandez"),
                                              trade[["exports"]]),
        "Denton-Cholette, exports": td.Candidate(TemporalDisaggregator("denton-cholette"),
                                                 trade[["exports"]]),
    }
    summary = quiet(td.backtest, sales, candidates, holdout=5).summary()
    write_table(out, "swiss_backtest", ["Candidate", "Origins", "Bias", "MAE", "RMSE",
                                        "MAPE (\\%)"],
                [[name, str(int(r["n_origins"])), fmt(r["bias"]), fmt(r["MAE"]),
                  fmt(r["RMSE"]), fmt(r["MAPE_%"])] for name, r in summary.iterrows()],
                "lrrrrr")
    LOG["backtest"] = summary.round(3).to_dict(orient="index")

    # conformity and revisions
    names = ["chow-lin-maxlog", "fernandez", "litterman-maxlog", "denton-cholette",
             "dynamic-maxlog", "proietti-maxlog"]
    fits = {n: fit(n, sales, trade[["exports"]]) for n in names}
    conf = pd.concat({n: td.conformity(r) for n, r in fits.items()}).droplevel(1)
    write_table(out, "swiss_conformity", ["Method", "corr.\\ levels", "corr.\\ growth",
                                          "sd growth (est.)", "sd growth (ind.)",
                                          "Volatility ratio"],
                [[tt(n), fmt(r["corr_levels"], 3), fmt(r["corr_changes"], 3),
                  fmt(r["sd_changes"]), fmt(r["sd_changes_indicator"]),
                  fmt(r["volatility_ratio"], 3)] for n, r in conf.iterrows()], "lrrrrr")
    LOG["conformity"] = conf.drop(columns="change").round(4).to_dict(orient="index")

    rev_candidates = {n: td.Candidate(TemporalDisaggregator(n), trade[["exports"]])
                      for n in names[:5]}
    revisions = quiet(td.revision_analysis, sales, rev_candidates, n_vintages=8).summary()
    write_table(out, "swiss_revisions", ["Method", "MAR level (\\%)", "MAR growth (pp)",
                                         "MAR level, latest", "MAR growth, latest",
                                         "Max.\\ level (\\%)"],
                [[tt(n), fmt(r["MAR_level"]), fmt(r["MAR_change"]), fmt(r["MAR_level_latest"]),
                  fmt(r["MAR_change_latest"]), fmt(r["max_level"])]
                 for n, r in revisions.iterrows()], "lrrrrr")
    LOG["revisions"] = revisions.drop(columns="change_units").round(4).to_dict(orient="index")

    cft = td.common_factor_test(fits["proietti-maxlog"])
    LOG["swiss_common_factor"] = {"F": float(cft.f_statistic), "p": float(cft.p_value),
                                  "df": [int(cft.df_numerator), int(cft.df_denominator)],
                                  "rho": float(cft.rho)}


# ----------------------------------------------------------------------
# 2. Agreement with R
# ----------------------------------------------------------------------
STATISTICS = {"s_2": "sigma2_ml", "s_2_gls": "sigma2_gls", "rss": "rss", "tss": "tss",
              "r.squared": "r_squared", "adj.r.squared": "adj_r_squared",
              "logl": "log_likelihood", "aic": "aic", "bic": "bic"}


def parity(out: Path, repo: Path) -> None:
    sys.path.insert(0, str(repo))
    from tests.reference.r_cases import CASES, DENTON_1971_CASES

    reference = repo / "tests" / "reference"
    meta_all = pd.read_csv(reference / "r_meta.csv")
    sales, trade = load_swisspharma()
    trade = trade.rename(columns={"exports": "exports_q", "imports": "imports_q"})

    def r_values(case_id: str) -> pd.Series:
        frame = pd.read_csv(reference / f"r_{case_id}.csv")
        index = pd.PeriodIndex([pd.Period(year=int(y), quarter=int(q), freq="Q")
                                for y, q in zip(frame["year"], frame["quarter"], strict=True)])
        return pd.Series(frame["value"].to_numpy(), index=index)

    def run(case_id: str, rho: float | None = None):
        _, columns, preset, overrides, conversion = CASES[case_id]
        model = quiet(TemporalDisaggregator, preset, conversion, **overrides)
        if rho is not None:
            spec = dataclasses.replace(model.method, rho_method="fixed", fixed_rho=rho,
                                       truncated_rho=0.0)
            model = quiet(TemporalDisaggregator, spec, conversion)
        return quiet(model.fit, sales, trade[columns]) if columns else \
            quiet(model.fit, sales, high_freq="Q")

    def difference(result, case_id: str) -> float:
        expected = r_values(case_id)
        worst = float(np.max(np.abs(result.values.reindex(expected.index) / expected - 1)))
        meta = meta_all[meta_all["id"] == case_id].set_index("key")["value"]
        reg = result.regression
        if reg is None:
            return worst
        k = len(reg.coefficients)
        worst = max(worst, float(np.max(np.abs(
            reg.coefficients.to_numpy() / meta[[f"coef_{i + 1}" for i in range(k)]]
            .to_numpy(dtype=float) - 1))))
        worst = max(worst, float(np.max(np.abs(
            reg.standard_errors.to_numpy() / meta[[f"se_{i + 1}" for i in range(k)]]
            .to_numpy(dtype=float) - 1))))
        for label, attribute in STATISTICS.items():
            reference_value = float(meta.get(label, np.nan))
            if np.isnan(reference_value):
                continue
            value = float(getattr(reg, attribute))
            absolute = label in ("r.squared", "adj.r.squared")
            worst = max(worst, abs(value - reference_value) if absolute
                        else abs(value / reference_value - 1))
        return worst

    rows, records = [], []
    for case_id in sorted(CASES):
        _, columns, preset, overrides, conversion = CASES[case_id]
        result = run(case_id)
        meta = meta_all[meta_all["id"] == case_id].set_index("key")["value"]
        r_rho = float(meta.get("rho", np.nan))
        direct = difference(result, case_id)
        refit = None
        estimated = result.rho is not None and result.rho.estimated and not result.rho.truncated
        if estimated:
            refit = difference(run(case_id, r_rho), case_id)
        py_rho = float(result.rho.value) if result.rho is not None else np.nan
        settings = ", ".join(f"{k}={v}" for k, v in overrides.items())
        records.append({"case": case_id, "preset": preset, "settings": settings,
                        "conversion": conversion, "rho_py": py_rho, "rho_r": r_rho,
                        "direct": direct, "refit": refit})
    for case_id, settings in sorted(DENTON_1971_CASES.items()):
        quarterly = np.tile([50.0, 100.0, 150.0, 100.0], 5)
        data = DataAligner("sum").from_arrays([500.0, 400.0, 300.0, 400.0, 500.0], quarterly, 4)
        result = TemporalDisaggregator("denton", **settings).fit_data(data)
        expected = pd.read_csv(reference / f"r_{case_id}.csv")["value"].to_numpy()
        records.append({"case": case_id, "preset": "denton",
                        "settings": ", ".join(f"{k}={v}" for k, v in settings.items()),
                        "conversion": "sum", "rho_py": np.nan, "rho_r": np.nan,
                        "direct": float(np.max(np.abs(result.values.to_numpy() / expected - 1))),
                        "refit": None})
    frame = pd.DataFrame(records)
    for r in frame.itertuples():
        settings = r.settings.replace("_", "\\_") if r.settings else ""
        rows.append([tt(r.case), tt(r.preset), settings if settings else "--", r.conversion,
                     fmt(r.rho_py, 4), fmt(r.rho_r, 4), sci(r.direct),
                     sci(r.refit) if r.refit is not None else "--"])
    write_table(out, "r_parity", ["Case", "Preset", "Settings", "Conv.", "$\\hat\\rho$ (Py)",
                                  "$\\hat\\rho$ (R)", "As fitted", "At R's $\\rho$"],
                rows, "lllcrrrr")
    fixed = frame[frame["refit"].isna()]["direct"]
    LOG["parity_n_cases"] = int(len(frame))
    LOG["parity_fixed_max"] = float(fixed.max())
    LOG["parity_fixed_min"] = float(fixed.min())
    LOG["parity_refit_max"] = float(frame["refit"].dropna().max())
    LOG["parity_estimated_direct_max"] = float(frame[frame["refit"].notna()]["direct"].max())
    LOG["parity_records"] = frame.to_dict(orient="records")

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    frame = frame.sort_values("direct")
    y = np.arange(len(frame))
    ax.barh(y, np.maximum(frame["direct"], 1e-16), color=ACCENT, height=0.6, label="as fitted")
    mask = frame["refit"].notna().to_numpy()
    ax.scatter(np.maximum(frame.loc[mask, "refit"].astype(float), 1e-16), y[mask],
               color=ACCENT_2, zorder=3, s=18, label="re-fitted at R's $\\rho$")
    ax.axvline(1e-8, color=INK, linestyle="--", linewidth=0.8)
    ax.set_xscale("log")
    ax.set_xlim(1e-16, 1e-2)
    ax.set_yticks(y)
    ax.set_yticklabels(frame["case"], fontsize=7)
    ax.set_xlabel("largest relative difference with R (predictions, coefficients, statistics)")
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="y", visible=False)
    fig.savefig(out / "figures" / "r_parity_cases.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# 3. Recovery of known high-frequency series
# ----------------------------------------------------------------------
RECOVERY_METHODS = [
    ("chow-lin-maxlog", {}), ("chow-lin-maxlog", {"include_intercept": False}),
    ("fernandez", {}), ("litterman-maxlog", {}), ("dynamic-maxlog", {}),
    ("proietti-maxlog", {}), ("denton-cholette", {}), ("denton", {}),
    ("uniform", {}),
]


def growth(series: pd.Series) -> pd.Series:
    return 100 * series.pct_change()


def recovery_table(truth: pd.Series, y_low: pd.Series, x: pd.DataFrame | None,
                   conversion: str, extrapolate: pd.Series | None = None) -> list[dict]:
    """Errors of each method against ``truth``; ``extrapolate`` holds later true values."""
    rows = []
    for preset, overrides in RECOVERY_METHODS:
        uses_x = preset != "uniform"
        result = fit(preset, y_low, x if uses_x else None, conversion, **overrides)
        estimate = result.values
        inside = estimate.reindex(truth.index).dropna()
        common = truth.reindex(inside.index)
        g_est, g_true = growth(inside).dropna(), growth(common).dropna()
        row = {"method": preset + (" (no intercept)" if overrides else ""),
               "rmse_level": float(np.sqrt(np.mean((inside - common) ** 2))),
               "rmse_growth": float(np.sqrt(np.mean((g_est - g_true) ** 2))),
               "corr_growth": float(np.corrcoef(g_est, g_true)[0, 1]),
               "rho": float(result.rho.value) if result.rho is not None else np.nan}
        if extrapolate is not None and preset != "uniform":
            row["rmse_extrapolation"] = float(np.sqrt(np.mean(
                (estimate.reindex(extrapolate.index) - extrapolate) ** 2)))
        rows.append(row)
    return rows


def recovery(out: Path) -> None:
    # (a) Swiss exports from imports, 1975-2010
    _, trade = load_swisspharma()
    sample = (trade.index.year >= 1975) & (trade.index.year <= 2010)
    exports = trade.loc[sample, "exports"]
    exports_a = td.temporal_aggregate(exports, "Y")
    rows_a = recovery_table(exports, exports_a, trade.loc[sample, ["imports"]], "sum")

    # (b) US real GDP from real consumption, 1959-2008, extrapolating 2009Q1-Q3
    import statsmodels.api as sm
    macro = sm.datasets.macrodata.load_pandas().data
    index = pd.PeriodIndex([pd.Period(year=int(y), quarter=int(q), freq="Q")
                            for y, q in zip(macro["year"], macro["quarter"], strict=True)])
    gdp = pd.Series(macro["realgdp"].to_numpy(), index=index)
    cons = pd.DataFrame({"realcons": macro["realcons"].to_numpy()}, index=index)
    complete = gdp[gdp.index.year <= 2008]
    gdp_a = td.temporal_aggregate(complete, "Y", conversion="average")
    rows_b = recovery_table(complete, gdp_a, cons, "average", extrapolate=gdp.iloc[-3:])

    def render(rows, name, extra):
        header = ["Method", "$\\hat\\rho$", "RMSE level", "RMSE growth (pp)",
                  "corr.\\ growth"] + (["RMSE 2009Q1--Q3"] if extra else [])
        body = []
        for r in rows:
            line = [tt(r["method"]), fmt(r["rho"], 3), fmt(r["rmse_level"], 1),
                    fmt(r["rmse_growth"], 3), fmt(r["corr_growth"], 3)]
            if extra:
                line.append(fmt(r.get("rmse_extrapolation", np.nan), 1))
            body.append(line)
        write_table(out, name, header, body, "lrrrr" + ("r" if extra else ""))

    render(rows_a, "recovery_exports", False)
    render(rows_b, "recovery_gdp", True)
    LOG["recovery_exports"] = rows_a
    LOG["recovery_gdp"] = rows_b
    LOG["gdp_n_low"], LOG["gdp_n_high"] = int(len(gdp_a)), int(len(complete))

    # figure: q/q growth of GDP, truth vs two methods, 1990-2008
    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    window = complete.index.year >= 1995
    g_true = growth(complete)[window]
    x = g_true.index.to_timestamp()
    ax.plot(x, g_true, color=INK, linewidth=1.3, label="true quarterly growth")
    styles = {"chow-lin-maxlog": ACCENT, "denton-cholette": ACCENT_2}
    for preset, colour in styles.items():
        est = fit(preset, gdp_a, cons, "average").values
        ax.plot(x, growth(est).reindex(g_true.index), color=colour, linewidth=1.0,
                linestyle="--" if preset.startswith("denton") else "-", label=preset)
    ax.set_ylabel("q/q growth (%)")
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(fontsize=7.5, loc="lower left", ncol=3)
    fig.savefig(out / "figures" / "gdp_growth_recovery.pdf")
    plt.close(fig)


# ----------------------------------------------------------------------
# 4. Monte Carlo: interval coverage and the common-factor test
# ----------------------------------------------------------------------
def simulate_chow_lin(rng, n_low=30, rho=0.7, beta=(5.0, 1.5)):
    n = 4 * n_low
    x = 20 + np.cumsum(rng.normal(0.2, 1.0, n))
    u = np.empty(n)
    u[0] = rng.normal(0, 1 / np.sqrt(1 - rho ** 2))
    for t in range(1, n):
        u[t] = rho * u[t - 1] + rng.normal()
    y = beta[0] + beta[1] * x + u
    return DataAligner("sum").from_arrays(y.reshape(n_low, 4).sum(axis=1), x, 4), y


def simulate_adl(rng, n_low=40, rho=0.6, b0=1.0, b1=0.4):
    n = 4 * n_low
    x = 20 + np.cumsum(rng.normal(0.2, 1.0, n))
    y = np.empty(n)
    y[0] = 5 + b0 * x[0]
    for t in range(1, n):
        y[t] = rho * y[t - 1] + 2.0 + b0 * x[t] + b1 * x[t - 1] + rng.normal()
    return DataAligner("sum").from_arrays(y.reshape(n_low, 4).sum(axis=1), x, 4)


def monte_carlo(out: Path, replications: int) -> None:
    rows = []
    for label, preset, extra in (("$\\rho$ known (0.7)", "chow-lin-fixed", {"fixed_rho": 0.7}),
                                 ("$\\rho$ estimated (maxlog)", "chow-lin-maxlog", {})):
        rng = np.random.default_rng(2026)
        model = quiet(TemporalDisaggregator, preset, **extra)
        covered, total, beta_hits = 0, 0, 0
        for _ in range(replications):
            data, truth = simulate_chow_lin(rng)
            result = quiet(model.fit_data, data)
            band = result.prediction_interval(0.95)
            covered += int(np.sum((band["lower"] <= truth) & (truth <= band["upper"])))
            total += truth.size
            ci = result.regression.confidence_intervals(0.95).iloc[1]
            beta_hits += int(ci.iloc[0] <= 1.5 <= ci.iloc[1])
        rows.append([label, fmt(100 * covered / total, 1), fmt(100 * beta_hits / replications, 1)])
        LOG[f"coverage_{preset}"] = {"prediction": covered / total,
                                     "beta": beta_hits / replications}
    write_table(out, "mc_coverage", ["Setting", "Prediction intervals (\\%)",
                                     "Slope intervals (\\%)"], rows, "lrr")

    size_rows = []
    for n_low in (20, 40, 80):
        rng = np.random.default_rng(100 + n_low)
        rejections = 0
        for _ in range(replications):
            result = quiet(lambda d: TemporalDisaggregator("proietti-maxlog").fit_data(d),
                           simulate_chow_lin(rng, n_low=n_low)[0])
            rejections += td.common_factor_test(result).p_value < 0.05
        rng = np.random.default_rng(200 + n_low)
        power = 0
        for _ in range(replications):
            result = quiet(lambda d: TemporalDisaggregator("proietti-maxlog").fit_data(d),
                           simulate_adl(rng, n_low=n_low))
            power += td.common_factor_test(result).p_value < 0.05
        size_rows.append([str(n_low), fmt(100 * rejections / replications, 1),
                          fmt(100 * power / replications, 1)])
        LOG[f"common_factor_{n_low}"] = {"size": rejections / replications,
                                         "power": power / replications}
    write_table(out, "mc_common_factor", ["$n_l$", "Size (\\%)", "Power (\\%)"], size_rows,
                "rrr")
    LOG["mc_replications"] = replications


# ----------------------------------------------------------------------
# 5. Computational cost
# ----------------------------------------------------------------------
def timing(out: Path) -> None:
    rng = np.random.default_rng(5)
    rows = []
    for label, n_low, ratio in (("annual $\\to$ quarterly", 40, 4),
                                ("quarterly $\\to$ monthly", 80, 3),
                                ("annual $\\to$ monthly", 40, 12),
                                ("monthly $\\to$ daily (30-day months)", 120, 30)):
        n = n_low * ratio
        x = 100 + np.cumsum(rng.normal(0.1, 1.0, n))
        y = (2 + 0.5 * x + rng.normal(size=n)).reshape(n_low, ratio).sum(axis=1)
        data = DataAligner("sum").from_arrays(y, x, ratio)
        for preset in ("chow-lin-maxlog", "denton-cholette"):
            model = quiet(TemporalDisaggregator, preset)
            times = []
            for _ in range(5):
                start = time.perf_counter()
                quiet(model.fit_data, data)
                times.append(time.perf_counter() - start)
            rows.append([label, str(n_low), str(n), tt(preset), fmt(float(np.median(times)), 3)])
    write_table(out, "timing", ["Problem", "$n_l$", "$n$", "Method", "Median time (s)"],
                rows, "lrrlr")
    LOG["timing_platform"] = f"{platform.processor() or platform.machine()}, " \
                             f"Python {platform.python_version()}"
    LOG["timing"] = rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True,
                        help="clone of github.com/jaimevera1107/tempdisagg (R references)")
    parser.add_argument("--out", type=Path, default=Path("."))
    parser.add_argument("--replications", type=int, default=500)
    args = parser.parse_args()
    for sub in ("figures", "tables"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)
    style()
    LOG["tempdisagg_version"] = td.__version__
    swisspharma(args.out)
    parity(args.out, args.repo.resolve())
    recovery(args.out)
    monte_carlo(args.out, args.replications)
    timing(args.out)
    (args.out / "results.json").write_text(json.dumps(LOG, indent=2, default=float))
    print(json.dumps(LOG, indent=2, default=float))


if __name__ == "__main__":
    main()
