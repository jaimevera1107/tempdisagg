import pandas as pd
import numpy as np
import warnings

from tempdisagg.utils.logging_utils import VerboseLogger


class TimeSeriesCompleter:
    """
    Completes and imputes missing values in a time series DataFrame.
    Ensures all combinations of index and grain exist and performs robust interpolation.

    Parameters
    ----------
    df : pd.DataFrame
        Input time series DataFrame.
    index_col : str
        Name of the column representing the time index (e.g., year).
    grain_col : str
        Name of the column representing the grain/frequency (e.g., month).
    y_col : str
        Name of the target variable to impute.
    X_col : str
        Name of the secondary feature column to impute.
    interpolation_method : str
        Method used for interpolation (default is 'linear').
    verbose : bool
        If True, enables informative logging.
    """

    def __init__(self, df, index_col, grain_col, y_col, X_col,
                 interpolation_method='nearest', verbose=False):
        
        # Store input arguments
        self.df = df.copy()
        self.index_col = index_col
        self.grain_col = grain_col
        self.y_col = y_col
        self.X_col = X_col
        self.interpolation_method = interpolation_method
        self.verbose = verbose

        # Placeholder for output DataFrame
        self.df_full = pd.DataFrame()

        # Use centralized logger factory
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        # Validate inputs at initialization
        self._validate_input()

    def complete_series(self):
        """
        Completes missing combinations and imputes target columns.

        Returns
        -------
        pd.DataFrame
            Completed DataFrame with all combinations and no missing values in target columns.
        """
        try:
            # Generate full set of index-grain combinations
            if self.verbose:
                self.logger.info("Creating full index with all combinations...")
            self._create_full_index()
        except Exception as e:
            # Log and return empty DataFrame in case of error
            self.logger.error(f"Failed to create full index: {e}")
            return self.df_full

        # Loop over each target column to impute
        for col in [self.y_col, self.X_col]:
            try:
                if self.verbose:
                    self.logger.info(f"Imputing values in column: '{col}'...")
                self._impute_column(col)
            except Exception as e:
                # Log and return partially completed DataFrame if imputation fails
                self.logger.error(f"Imputation failed for '{col}': {e}")
                return self.df_full

        # Validate that no NaNs remain
        self._validate_output_no_nans()
        
        original_keys = list(zip(self.df[self.index_col], self.df[self.grain_col]))
        completed_keys = list(zip(self.df_full[self.index_col], self.df_full[self.grain_col]))
        match_positions = [completed_keys.index(key) for key in original_keys if key in completed_keys]

        if not match_positions:
            raise ValueError("Original keys not found in completed DataFrame — check index/grain consistency.")

        start = min(match_positions)
        end = max(match_positions)

        n_pad_before = start
        n_pad_after = len(completed_keys) - end - 1

        padding_info = {
            "original_length": len(original_keys),
            "completed_length": len(completed_keys),
            "n_pad_before": n_pad_before,
            "n_pad_after": n_pad_after
        }

        return self.df_full, padding_info

    def _validate_input(self):
        """
        Validates the input DataFrame for required structure and integrity.
        """
        try:
            # Define required columns
            required_cols = [self.index_col, self.grain_col, self.y_col, self.X_col]

            # Ensure all required columns exist
            for col in required_cols:
                if col not in self.df.columns:
                    raise ValueError(f"Missing required column: '{col}'")

            # Ensure target columns contain some non-null values
            for col in [self.y_col, self.X_col]:
                if self.df[col].dropna().empty:
                    raise ValueError(f"Column '{col}' contains only missing values.")

            # Convert index and grain columns to integer safely
            self.df[self.index_col] = self.df[self.index_col].astype(int)
            self.df[self.grain_col] = self.df[self.grain_col].astype(int)

            # Convert numerical target columns to float
            self.df[self.y_col] = pd.to_numeric(self.df[self.y_col], errors='coerce')
            self.df[self.X_col] = pd.to_numeric(self.df[self.X_col], errors='coerce')

            # Require at least 3 rows to allow interpolation
            if self.df.shape[0] < 3:
                raise ValueError("At least 3 observations are required for interpolation.")

            if self.verbose:
                self.logger.info("Input validation successful.")

        except Exception as e:
            raise ValueError(f"Input validation failed: {e}")

    def _create_full_index(self):
        """
        Creates a DataFrame with all combinations of index and grain.
        """
        try:
            # Get sorted list of unique values for index and grain
            all_indices = sorted(self.df[self.index_col].unique())
            all_grains = sorted(self.df[self.grain_col].unique())

            # Compute Cartesian product
            full_index = pd.MultiIndex.from_product(
                [all_indices, all_grains],
                names=[self.index_col, self.grain_col]
            )

            # Reindex the original DataFrame to include all possible combinations
            self.df_full = (
                self.df.set_index([self.index_col, self.grain_col])
                .reindex(full_index)
                .reset_index()
            )

            # Sort the completed DataFrame to maintain time order
            self.df_full = self.df_full.sort_values(
                by=[self.index_col, self.grain_col]
            ).reset_index(drop=True)

            # Calculate number of added rows
            added = self.df_full.shape[0] - self.df.shape[0]
            if self.verbose:
                self.logger.info(f"Full index created: {self.df_full.shape[0]} rows ({added} new).")

        except Exception as e:
            raise RuntimeError(f"Error while creating full index: {e}")

    def _impute_column(self, col_name):
        """
        Imputes missing values in a specified column using interpolation.

        Parameters
        ----------
        col_name : str
            Column name to interpolate.
        """
        try:
            # Count initial missing values
            before = self.df_full[col_name].isna().sum()

            # First round: use specified method with forward and backward fill
            self.df_full[col_name] = (
                self.df_full[col_name]
                .interpolate(method=self.interpolation_method, limit_direction='both')
                .ffill()
                .bfill()
            )

            # Apply fallback if NaNs remain
            if self.df_full[col_name].isna().any():
                if self.verbose:
                    self.logger.info(f"Fallback 'nearest' interpolation applied to '{col_name}'...")
                self.df_full[col_name] = (
                    self.df_full[col_name]
                    .interpolate(method='nearest', limit_direction='both')
                    .ffill()
                    .bfill()
                )

            # Count final missing values
            after = self.df_full[col_name].isna().sum()

            # Warn if some values still remain missing
            if after > 0:
                warnings.warn(
                    f"Column '{col_name}' still contains {after} missing values after fallback interpolation.",
                    UserWarning
                )

            if self.verbose:
                self.logger.info(f"Column '{col_name}' imputed: {before} → {after} missing values.")

        except Exception as e:
            raise RuntimeError(f"Error during imputation of '{col_name}': {e}")

    def _validate_output_no_nans(self):
        """
        Ensures that the imputed DataFrame contains no missing values in the target columns.
        """
        # Check remaining missing values in both columns
        missing = self.df_full[[self.y_col, self.X_col]].isna().sum()

        # Raise error if missing values still exist
        if missing.any():
            raise ValueError(f"Missing values remain after imputation:\n{missing}")

        if self.verbose:
            self.logger.info("All target columns fully imputed — no missing values remain.")