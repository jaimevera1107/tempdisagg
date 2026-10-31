"""Cases compared with the R package (shared by the tests and docs/make_figures.py).

Keep in sync with ``generate_r_reference.R``.
"""

from __future__ import annotations

# id -> (target, indicator columns, preset, overrides, conversion)
CASES: dict[str, tuple[str, list[str], str, dict, str]] = {
    "cl_maxlog_two": ("sales", ["exports_q", "imports_q"], "chow-lin-maxlog", {}, "sum"),
    "cl_maxlog": ("sales", ["exports_q"], "chow-lin-maxlog", {}, "sum"),
    "cl_maxlog_notrunc": ("sales", ["exports_q"], "chow-lin-maxlog", {"truncated_rho": -1}, "sum"),
    "cl_maxlog_noint": ("sales", ["exports_q"], "chow-lin-maxlog",
                        {"include_intercept": False}, "sum"),
    "cl_ecotrim": ("sales", ["exports_q"], "chow-lin-minrss-ecotrim",
                   {"truncated_rho": -1}, "sum"),
    "cl_quilis": ("sales", ["exports_q"], "chow-lin-minrss-quilis", {"truncated_rho": -1}, "sum"),
    "cl_fixed": ("sales", ["exports_q"], "chow-lin-fixed", {"fixed_rho": 0.5}, "sum"),
    "cl_avg": ("sales", ["exports_q"], "chow-lin-maxlog", {}, "average"),
    "cl_first": ("sales", ["exports_q"], "chow-lin-fixed", {"fixed_rho": 0.6}, "first"),
    "cl_last": ("sales", ["exports_q"], "chow-lin-fixed", {"fixed_rho": 0.6}, "last"),
    "lit_maxlog": ("sales", ["exports_q"], "litterman-maxlog", {"truncated_rho": -1}, "sum"),
    "lit_minrss": ("sales", ["exports_q"], "litterman-minrss", {"truncated_rho": -1}, "sum"),
    "lit_fixed": ("sales", ["exports_q"], "litterman-fixed", {"fixed_rho": 0.7}, "sum"),
    "fernandez": ("sales", ["exports_q"], "fernandez", {}, "sum"),
    "fernandez_noint": ("sales", ["exports_q"], "fernandez", {"include_intercept": False}, "sum"),
    "ols": ("sales", ["exports_q"], "ols", {}, "sum"),
    "fast": ("sales", ["exports_q"], "fast", {}, "sum"),
    "dyn_maxlog": ("sales", ["exports_q"], "dynamic-maxlog", {}, "sum"),
    "dyn_fixed": ("sales", ["exports_q"], "dynamic-fixed", {"fixed_rho": 0.5}, "sum"),
    "dc_prop": ("sales", ["exports_q"], "denton-cholette",
                {"criterion": "proportional", "h": 1}, "sum"),
    "dc_add_h0": ("sales", ["exports_q"], "denton-cholette",
                  {"criterion": "additive", "h": 0}, "sum"),
    "dc_add_h2": ("sales", ["exports_q"], "denton-cholette",
                  {"criterion": "additive", "h": 2}, "sum"),
    "denton_add": ("sales", ["exports_q"], "denton", {"criterion": "additive", "h": 1}, "sum"),
    "denton_prop_h2": ("sales", ["exports_q"], "denton",
                       {"criterion": "proportional", "h": 2}, "sum"),
    "dc_noind": ("sales", [], "denton-cholette", {}, "sum"),
    "uniform": ("sales", [], "uniform", {}, "sum"),
}
DENTON_1971_CASES: dict[str, dict] = {
    "denton1971_add_h1": {"variant": "original", "criterion": "additive", "h": 1},
    "denton1971_prop_h1": {"variant": "original", "criterion": "proportional", "h": 1},
    "denton1971_dc_add_h1": {"variant": "cholette", "criterion": "additive", "h": 1},
}
