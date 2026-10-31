"""Real datasets for examples and tests.

``load_swisspharma``
    Swiss chemical and pharmaceutical industry: annual sales index
    (1975-2010) and quarterly exports and imports in millions of CHF
    (1972Q1-2011Q2). Distributed with the R package tempdisagg (Sax and
    Steiner); original sources: Swiss Federal Statistical Office (sales) and
    Swiss Federal Customs Administration (trade). These are the data used to
    validate this library against R.

``load_denton1971``
    The textbook example of Denton (1971), also used in R's ``?td``: a
    quarterly indicator repeating 50, 100, 150, 100 and annual totals
    500, 400, 300, 400, 500.

Both loaders return pandas objects indexed by ``PeriodIndex`` (Denton: plain
positions), ready for :class:`~tempdisagg.TemporalDisaggregator`.
"""

from __future__ import annotations

from importlib import resources

import pandas as pd

_PACKAGE = "tempdisagg.datasets"


def _read(name: str) -> pd.DataFrame:
    with resources.files(_PACKAGE).joinpath("data", name).open("r", encoding="utf-8") as handle:
        return pd.read_csv(handle)


def load_swisspharma() -> tuple[pd.Series, pd.DataFrame]:
    """Return ``(sales_annual, trade_quarterly)``.

    Returns
    -------
    sales_annual : pd.Series
        Annual sales index, yearly PeriodIndex 1975-2010, name ``"sales"``.
    trade_quarterly : pd.DataFrame
        Columns ``exports`` and ``imports`` (millions of CHF), quarterly
        PeriodIndex 1972Q1-2011Q2 (it extends beyond the annual series, so
        the first and last years are retropolated and extrapolated).
    """
    annual = _read("swisspharma_annual.csv")
    sales = pd.Series(annual["sales_a"].to_numpy(),
                      index=pd.PeriodIndex(annual["year"].astype(str), freq="Y").rename(None),
                      name="sales")
    quarterly = _read("swisspharma_quarterly.csv")
    index = pd.PeriodIndex(
        [pd.Period(year=int(y), quarter=int(q), freq="Q")
         for y, q in zip(quarterly["year"], quarterly["quarter"])])
    trade = pd.DataFrame({"exports": quarterly["exports_q"].to_numpy(),
                          "imports": quarterly["imports_q"].to_numpy()}, index=index)
    return sales, trade


def load_denton1971() -> tuple[pd.Series, pd.Series]:
    """Return ``(annual_totals, quarterly_indicator)`` of Denton's example.

    Indexed by positions (years 1-5, quarters 1-20); use
    :meth:`~tempdisagg.TemporalDisaggregator.fit_arrays` with
    ``frequency_ratio=4``.
    """
    annual = _read("denton1971_annual.csv")
    quarterly = _read("denton1971_quarterly.csv")
    return (pd.Series(annual["total"].to_numpy(dtype=float), name="total"),
            pd.Series(quarterly["indicator"].to_numpy(dtype=float), name="indicator"))


__all__ = ["load_denton1971", "load_swisspharma"]
