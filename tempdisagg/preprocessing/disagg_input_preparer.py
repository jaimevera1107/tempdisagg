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
                 y_col="y", X_col="X", verbose=False, interpolation_method="nearest"):
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
        """
        # Store initialization parameters
        self.conversion = conversion
        self.grain_col = grain_col
        self.index_col = index_col
        self.y_col = y_col
        self.X_col = X_col
        self.verbose = verbose
        self.interpolation_method = interpolation_method

        # Create logger with verbosity control
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

        Raises
        ------
        ValueError
            If validation fails or shapes are inconsistent.
        """
        # Validate that input is a DataFrame
        if not isinstance(dataframe, pd.DataFrame):
            raise ValueError(f"Expected input as pandas DataFrame, got {type(dataframe)}.")

        # Check for required columns
        required_cols = {self.index_col, self.grain_col, self.y_col, self.X_col}
        missing = required_cols - set(dataframe.columns)
        if missing:
            raise ValueError(f"Missing required columns in DataFrame: {missing}")

        # Step 1: Complete the time series
        completer = TimeSeriesCompleter(
            df=dataframe,
            index_col=self.index_col,
            grain_col=self.grain_col,
            y_col=self.y_col,
            X_col=self.X_col,
            interpolation_method=self.interpolation_method,
            verbose=self.verbose
        )
        completed_df, padding_info = completer.complete_series()

        # Step 2: Build the conversion matrix
        builder = ConversionMatrixBuilder(
            conversion=self.conversion,
            grain_col=self.grain_col,
            index_col=self.index_col,
            verbose=self.verbose
        )
        C = builder.build(completed_df)

        # Step 3: Extract low-frequency target variable y_l
        try:
            y_l = completed_df.groupby(self.index_col)[self.y_col].first().values.reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Error extracting 'y_l' from column '{self.y_col}': {str(e)}")

        # Step 4: Extract high-frequency predictor variable X
        try:
            X = completed_df[self.X_col].values.reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Error extracting 'X' from column '{self.X_col}': {str(e)}")

        # Step 5: Validate alignment of dimensions
        if C.shape[0] != y_l.shape[0]:
            raise ValueError(
                f"Row mismatch: C has {C.shape[0]} rows but y_l has {y_l.shape[0]}."
            )
        if C.shape[1] != X.shape[0]:
            raise ValueError(
                f"Column mismatch: C has {C.shape[1]} columns but X has {X.shape[0]} rows."
            )

        # Step 6: Warn if too few observations
        if y_l.shape[0] < 3:
            warnings.warn(
                f"Only {y_l.shape[0]} observations in 'y_l'. Model output may be unstable.",
                UserWarning
            )

        # Step 7: Log result shapes
        if self.verbose:
            self.logger.info("Disaggregation inputs prepared successfully.")
            self.logger.info(f"  → y_l shape: {y_l.shape}")
            self.logger.info(f"  → X shape: {X.shape}")
            self.logger.info(f"  → C shape: {C.shape}")

        return y_l, X, C, padding_info