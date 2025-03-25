import numpy as np
import pandas as pd
import warnings

from tempdisagg.utils.logging_utils import VerboseLogger

class InputPreprocessor:
    """
    Validates and extracts input arrays for temporal disaggregation models.

    Given a DataFrame and a conversion matrix, this class extracts:
        - y_l: Low-frequency target (first value per group).
        - X:   High-frequency regressor values.
        - C:   Conversion matrix (already built externally).
    Ensures consistency in shapes and types.
    """

    def __init__(self,index_col="Index", y_col="y", X_col="X", verbose=False):
        """
        Initialize the InputPreprocessor instance.

        Parameters
        ----------
        index_col : str
            Column name representing the low-frequency group (e.g., year).
        y_col : str
            Column name for the low-frequency target variable.
        X_col : str
            Column name for the high-frequency explanatory variable.
        verbose : bool
            Whether to enable logging messages.
        """
        # Store column parameters
        self.index_col = index_col
        self.y_col = y_col
        self.X_col = X_col
        self.verbose = verbose

        # Create logger with verbosity control
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

    def validate_and_format(self, df, C):
        """
        Validate inputs and return aligned matrices (y_l, X, C).

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame containing both low- and high-frequency data.
        C : np.ndarray
            Conversion matrix to map high-frequency to low-frequency values.

        Returns
        -------
        tuple
            Tuple (y_l, X, C) with shapes aligned for model estimation.

        Raises
        ------
        ValueError
            If any validation step fails.
        """
        # Ensure df is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"Expected input 'df' to be a pandas DataFrame. Got {type(df)} instead.")

        # Check required columns are present
        required_columns = {self.index_col, self.y_col, self.X_col}
        missing_columns = required_columns - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns in DataFrame: {missing_columns}")

        # Extract low-frequency target: first observation per group
        try:
            y_l = df.groupby(self.index_col)[self.y_col].first().values.reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Failed to extract 'y_l' from column '{self.y_col}': {str(e)}")

        # Extract high-frequency regressor as column vector
        try:
            X = df[self.X_col].values.reshape(-1, 1)
        except Exception as e:
            raise ValueError(f"Failed to extract 'X' from column '{self.X_col}': {str(e)}")

        # Validate conversion matrix type
        if not isinstance(C, np.ndarray):
            raise ValueError(f"Expected conversion matrix 'C' as numpy.ndarray, got {type(C)}.")

        # Validate shape consistency: rows of C match y_l
        if C.shape[0] != y_l.shape[0]:
            raise ValueError(
                f"Shape mismatch: C has {C.shape[0]} rows, but y_l has {y_l.shape[0]} rows."
            )

        # Validate shape consistency: columns of C match X
        if C.shape[1] != X.shape[0]:
            raise ValueError(
                f"Shape mismatch: C has {C.shape[1]} columns, but X has {X.shape[0]} rows."
            )

        # Warn if there are too few observations
        if y_l.shape[0] < 3:
            warnings.warn(
                f"Only {y_l.shape[0]} observations in 'y_l'. Disaggregation may be unstable.",
                UserWarning
            )

        # Log resulting shapes
        if self.verbose:
            self.logger.info("Input preprocessing successful.")
            self.logger.info(f"  → y_l shape: {y_l.shape}")
            self.logger.info(f"  → X shape: {X.shape}")
            self.logger.info(f"  → C shape: {C.shape}")

        return y_l, X, C