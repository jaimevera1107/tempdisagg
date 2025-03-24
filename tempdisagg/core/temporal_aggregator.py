import numpy as np
import pandas as pd
import logging
from tempdisagg.utils.logging_utils import VerboseLogger

class TemporalAggregator:
    """
    Aggregates a high-frequency time series into a low-frequency series
    using one of four supported rules: sum, average, first, or last.
    """

    def __init__(self, conversion="sum", freq_ratio=None, grain_col="Grain", index_col="Index", verbose=False):
        """
        Parameters
        ----------
        conversion : str
            Aggregation rule: 'sum', 'average', 'first', or 'last'.
        freq_ratio : int
            Number of high-frequency units per low-frequency unit.
        grain_col : str
            Column name for high-frequency grain.
        index_col : str
            Column name for low-frequency index.
        verbose : bool
            Enable logging messages.
        """
        self.conversion = conversion
        self.freq_ratio = freq_ratio
        self.grain_col = grain_col
        self.index_col = index_col
        self.verbose = verbose
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        valid = ["sum", "average", "first", "last"]
        if self.conversion not in valid:
            raise ValueError(f"Invalid conversion: {self.conversion}. Must be one of {valid}.")

    def aggregate(self, df, value_col):
        """
        Aggregate the high-frequency series to low-frequency.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing high-frequency data.
        value_col : str
            Name of the value column to aggregate.

        Returns
        -------
        pd.DataFrame
            Low-frequency aggregated series.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")
        if value_col not in df.columns:
            raise ValueError(f"Column '{value_col}' not found in input DataFrame.")

        if self.index_col not in df.columns or self.grain_col not in df.columns:
            raise ValueError("Both 'index_col' and 'grain_col' must exist in the DataFrame.")

        df = df.copy()
        df = df.sort_values([self.index_col, self.grain_col])

        if self.verbose:
            self.logger.info(f"Aggregating using rule: {self.conversion}")

        if self.conversion == "sum":
            agg_func = "sum"
        elif self.conversion == "average":
            agg_func = "mean"
        elif self.conversion == "first":
            agg_func = "first"
        elif self.conversion == "last":
            agg_func = "last"
        else:
            raise ValueError(f"Unsupported conversion: {self.conversion}")

        result = df.groupby(self.index_col, as_index=False)[value_col].agg(agg_func)
        result.rename(columns={value_col: f"{value_col}_agg"}, inplace=True)

        return result
