import numpy as np
import pandas as pd
import warnings

from tempdisagg.utils.logging_utils import VerboseLogger
from tempdisagg.preprocessing.timeseries_completer import TimeSeriesCompleter
from tempdisagg.preprocessing.conversion_matrix_builder import ConversionMatrixBuilder

class DisaggInputPreparer:
    """
    Prepares aligned inputs for temporal disaggregation models.

    This class orchestrates:
        - Completion of the time series (via TimeSeriesCompleter).
        - Construction of the conversion matrix (via ConversionMatrixBuilder).
        - Extraction of y_l, X, and C with validated consistency.
    """

    def __init__(self, conversion, grain_col="Grain", index_col="Index",
                 y_col="y", X_col="X", verbose=False, interpolation_method="nearest",
                 use_retropolarizer=False, retro_method="linear_regression", retro_aux_col=None):
        """
        Initialize the DisaggInputPreparer instance.

        Parameters
        ----------
        conversion : str
            Aggregation method: "sum", "average", "first", or "last".
        grain_col : str
            Column name representing high-frequency identifiers.
        index_col : str
            Column name representing low-frequency groups.
        y_col : str
            Column name for low-frequency target variable.
        X_col : str
            Column name for high-frequency predictor variable.
        verbose : bool
            Whether to enable logging messages.
        interpolation_method : str
            Method used to impute missing values during completion.
        use_retropolarizer : bool
            Whether to use Retropolarizer instead of standard interpolation for y_col.
        retro_method : str
            Method used by Retropolarizer (e.g. 'linear_regression').
        retro_aux_col : str or None
            Auxiliary column to use as predictor for retropolarization. If None, uses X_col.
        """
        self.conversion = conversion
        self.grain_col = grain_col
        self.index_col = index_col
        self.y_col = y_col
        self.X_col = X_col
        self.verbose = verbose
        self.interpolation_method = interpolation_method
        self.use_retropolarizer = use_retropolarizer
        self.retro_method = retro_method
        self.retro_aux_col = retro_aux_col
        self.df_full = None
        self.padding_info = {}

        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

    def prepare(self, dataframe):
        """
        Prepare aligned components for disaggregation: y_l, X, and C.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Input DataFrame containing index, grain, y_col, and X_col.

        Returns
        -------
        tuple
            (y_l, X, C):
            - y_l: np.ndarray of shape (n_low, 1) – low-frequency targets.
            - X:   np.ndarray of shape (n_high, 1) – high-frequency predictors.
            - C:   np.ndarray of shape (n_low, n_high) – conversion matrix.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError(f"Expected input as pandas DataFrame, got {type(dataframe)}.")

        required_cols = {self.index_col, self.grain_col, self.y_col, self.X_col}
        missing = required_cols - set(dataframe.columns)
        if missing:
            raise ValueError(f"Missing required columns in DataFrame: {missing}")

        completer = TimeSeriesCompleter(
            df=dataframe,
            index_col=self.index_col,
            grain_col=self.grain_col,
            y_col=self.y_col,
            X_col=self.X_col,
            interpolation_method=self.interpolation_method,
            verbose=self.verbose,
            use_retropolarizer=self.use_retropolarizer,
            retro_method=self.retro_method,
            retro_aux_col=self.retro_aux_col
        )
        completed_df, padding_info = completer.complete_series()
        self.df_full = completed_df
        self.padding_info = padding_info

        builder = ConversionMatrixBuilder(
            conversion=self.conversion,
            grain_col=self.grain_col,
            index_col=self.index_col,
            verbose=self.verbose
        )
        C = builder.build(completed_df)

        try:
            y_l = completed_df.groupby(self.index_col)[self.y_col].first().values.reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Error extracting 'y_l': {e}")

        try:
            X = completed_df[self.X_col].values.reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Error extracting 'X': {e}")

        if C.shape[0] != y_l.shape[0]:
            raise ValueError(f"Row mismatch: C has {C.shape[0]} rows but y_l has {y_l.shape[0]}.")
        if C.shape[1] != X.shape[0]:
            raise ValueError(f"Column mismatch: C has {C.shape[1]} columns but X has {X.shape[0]} rows.")

        if y_l.shape[0] < 3:
            warnings.warn(f"Only {y_l.shape[0]} observations in 'y_l'.", UserWarning)

        if self.verbose:
            self.logger.info("Disaggregation inputs prepared successfully.")
            self.logger.info(f"  → y_l shape: {y_l.shape}")
            self.logger.info(f"  → X shape: {X.shape}")
            self.logger.info(f"  → C shape: {C.shape}")

        return y_l, X, C, completed_df, padding_info
