# Generate reference results with the R package tempdisagg (>= 1.2.0).
#
# Usage (any working directory):
#     Rscript tests/reference/generate_r_reference.R
#
# Files are written NEXT TO THIS SCRIPT (tests/reference/):
#   swisspharma_annual.csv, swisspharma_quarterly.csv   inputs
#   r_<case>.csv                                         high-frequency predictions
#   r_meta.csv                                           rho, truncation, coefficients,
#                                                        standard errors, statistics
#   r_session.txt                                        R and package versions
#   r_errors.csv                                         cases that failed (if any)
# A failing case is logged and skipped; it does not stop the others.
# tests/test_r_parity.py compares the Python implementation with these files.

# ---------------------------------------------------------------------------
# Output directory = directory of this script (independent of the cwd)
# ---------------------------------------------------------------------------
script_arg <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
out_dir <- if (length(script_arg) == 1) {
  dirname(normalizePath(sub("^--file=", "", script_arg)))
} else {
  file.path("tests", "reference")
}
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
cat("Output directory:", out_dir, "\n")

if (!requireNamespace("tempdisagg", quietly = TRUE)) {
  stop("The R package 'tempdisagg' is not installed. Run:\n",
       "  install.packages('tempdisagg', repos = 'https://cloud.r-project.org')")
}
library(tempdisagg)
session <- c(R.version.string, paste("tempdisagg", as.character(packageVersion("tempdisagg"))))
writeLines(session, file.path(out_dir, "r_session.txt"))
cat(session, sep = "\n")

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
data(swisspharma)
year_of <- function(x) as.integer(floor(time(x) + 1e-9))

write.csv(data.frame(year = year_of(sales.a), sales_a = as.numeric(sales.a)),
          file.path(out_dir, "swisspharma_annual.csv"), row.names = FALSE)
quarterly_ts <- na.omit(cbind(exports.q, imports.q))
write.csv(data.frame(year = year_of(quarterly_ts),
                     quarter = as.integer(cycle(quarterly_ts)),
                     exports_q = as.numeric(quarterly_ts[, "exports.q"]),
                     imports_q = as.numeric(quarterly_ts[, "imports.q"])),
          file.path(out_dir, "swisspharma_quarterly.csv"), row.names = FALSE)
exports.q <- quarterly_ts[, "exports.q"]
imports.q <- quarterly_ts[, "imports.q"]

# Denton (1971) example from ?td (data fully specified in the help page).
d.q <- ts(rep(c(50, 100, 150, 100), 5), frequency = 4)
d.a <- ts(c(500, 400, 300, 400, 500))

# ---------------------------------------------------------------------------
# Cases: id, formula, method, extra arguments (keep in sync with test_r_parity.py)
# ---------------------------------------------------------------------------
cases <- list(
  list("cl_maxlog_two", sales.a ~ exports.q + imports.q, "chow-lin-maxlog", list()),
  list("cl_maxlog", sales.a ~ exports.q, "chow-lin-maxlog", list()),
  list("cl_maxlog_notrunc", sales.a ~ exports.q, "chow-lin-maxlog", list(truncated.rho = -1)),
  list("cl_maxlog_noint", sales.a ~ 0 + exports.q, "chow-lin-maxlog", list()),
  list("cl_ecotrim", sales.a ~ exports.q, "chow-lin-minrss-ecotrim", list(truncated.rho = -1)),
  list("cl_quilis", sales.a ~ exports.q, "chow-lin-minrss-quilis", list(truncated.rho = -1)),
  list("cl_fixed", sales.a ~ exports.q, "chow-lin-fixed", list(fixed.rho = 0.5)),
  list("cl_avg", sales.a ~ exports.q, "chow-lin-maxlog", list(conversion = "average")),
  list("cl_first", sales.a ~ exports.q, "chow-lin-fixed", list(conversion = "first", fixed.rho = 0.6)),
  list("cl_last", sales.a ~ exports.q, "chow-lin-fixed", list(conversion = "last", fixed.rho = 0.6)),
  list("lit_maxlog", sales.a ~ exports.q, "litterman-maxlog", list(truncated.rho = -1)),
  list("lit_minrss", sales.a ~ exports.q, "litterman-minrss", list(truncated.rho = -1)),
  list("lit_fixed", sales.a ~ exports.q, "litterman-fixed", list(fixed.rho = 0.7)),
  list("fernandez", sales.a ~ exports.q, "fernandez", list()),
  list("fernandez_noint", sales.a ~ 0 + exports.q, "fernandez", list()),
  list("ols", sales.a ~ exports.q, "ols", list()),
  list("fast", sales.a ~ exports.q, "fast", list()),
  list("dyn_maxlog", sales.a ~ exports.q, "dynamic-maxlog", list()),
  list("dyn_fixed", sales.a ~ exports.q, "dynamic-fixed", list(fixed.rho = 0.5)),
  list("dc_prop", sales.a ~ 0 + exports.q, "denton-cholette", list(criterion = "proportional", h = 1)),
  list("dc_add_h0", sales.a ~ 0 + exports.q, "denton-cholette", list(criterion = "additive", h = 0)),
  list("dc_add_h2", sales.a ~ 0 + exports.q, "denton-cholette", list(criterion = "additive", h = 2)),
  list("denton_add", sales.a ~ 0 + exports.q, "denton", list(criterion = "additive", h = 1)),
  list("denton_prop_h2", sales.a ~ 0 + exports.q, "denton", list(criterion = "proportional", h = 2)),
  list("dc_noind", sales.a ~ 1, "denton-cholette", list(to = "quarterly")),
  list("uniform", sales.a ~ 1, "uniform", list(to = "quarterly")),
  list("denton1971_add_h1", d.a ~ 0 + d.q, "denton", list(criterion = "additive", h = 1)),
  list("denton1971_prop_h1", d.a ~ 0 + d.q, "denton", list(criterion = "proportional", h = 1)),
  list("denton1971_dc_add_h1", d.a ~ 0 + d.q, "denton-cholette", list(criterion = "additive", h = 1))
)

scalar <- function(x) if (is.null(x) || length(x) == 0) NA_real_ else as.numeric(x)[1]

meta <- data.frame(id = character(), key = character(), value = numeric())
errors <- data.frame(id = character(), message = character())
add <- function(meta, id, key, value) rbind(meta, data.frame(id = id, key = key, value = value))

for (case in cases) {
  id <- case[[1]]
  outcome <- tryCatch({
    model <- do.call(td, c(list(formula = case[[2]], method = case[[3]]), case[[4]]))
    pred <- predict(model)
    write.csv(data.frame(position = seq_along(pred), year = year_of(pred),
                         quarter = as.integer(cycle(pred)), value = as.numeric(pred)),
              file.path(out_dir, paste0("r_", id, ".csv")), row.names = FALSE)
    for (key in c("rho", "s_2", "s_2_gls", "tss", "rss", "r.squared",
                  "adj.r.squared", "logl", "aic", "bic")) {
      meta <- add(meta, id, key, scalar(model[[key]]))
    }
    meta <- add(meta, id, "truncated", scalar(isTRUE(model$truncated)))
    if (!is.null(model$coefficients)) {
      for (i in seq_along(model$coefficients)) {
        meta <- add(meta, id, paste0("coef_", i), unname(model$coefficients[i]))
        meta <- add(meta, id, paste0("se_", i), unname(model$se[i]))
      }
    }
    meta
  }, error = function(e) e)
  if (inherits(outcome, "error")) {
    cat(sprintf("  FAILED  %-22s %s\n", id, conditionMessage(outcome)))
    errors <- rbind(errors, data.frame(id = id, message = conditionMessage(outcome)))
  } else {
    meta <- outcome
    cat(sprintf("  ok      %s\n", id))
  }
}

write.csv(meta, file.path(out_dir, "r_meta.csv"), row.names = FALSE)
if (nrow(errors) > 0) {
  write.csv(errors, file.path(out_dir, "r_errors.csv"), row.names = FALSE)
}
n_csv <- length(list.files(out_dir, pattern = "\\.csv$"))
cat(sprintf("\nDone: %d of %d cases ok; %d CSV files in %s\n",
            length(cases) - nrow(errors), length(cases), n_csv, out_dir))
