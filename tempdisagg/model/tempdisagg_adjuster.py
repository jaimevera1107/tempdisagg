import pandas as pd
import numpy as np
import warnings

from tempdisagg.postprocessing.post_estimation import PostEstimation
from tempdisagg.utils.logging_utils import VerboseLogger


class TempDisaggAdjuster:
    """
    Adjusts predicted values after disaggregation to ensure consistency with
    aggregation rules, correcting negative values if needed.

    This class expects the following attributes to be set before calling `adjust_output()`:
    - self.y_hat : np.ndarray
        Predicted values from the disaggregation model.
    - self.df_ : pd.DataFrame
        DataFrame containing the disaggregated results and predictions.
    - self.conversion : str
        Aggregation method used for the disaggregation ('sum', 'average', etc.).
    - self.n_pad_before : int (optional)
        Number of padded rows added before the original index.
    - self.n_pad_after : int (optional)
        Number of padded rows added after the original index.
    - self.verbose : bool
        Flag to control logging verbosity.
    """

    def __init__(self, verbose=False):
        """
        Initialize the TempDisaggAdjuster class with logging configuration.

        INPUT
        verbose : bool
            Whether to enable verbose logging messages.

        OUTPUT
        None
        """
        # Save verbosity setting
        self.verbose = verbose

        # Initialize logger using centralized factory
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

    def adjust_output(self, full=False):
        """
        Apply post-estimation adjustment to predicted values, ensuring non-negativity
        and consistency with the aggregation method. Optionally returns full or trimmed output.

        INPUT
        full : bool
            Whether to return the full prediction (with padding) or trimmed to original range.

        OUTPUT
        y_hat_adjusted : np.ndarray
            Adjusted prediction as a 2D numpy array, without negative values.

        RAISES
        RuntimeError
            If predictions are not available or post-estimation adjustment fails.
        ValueError
            If required attributes (`df_`, `conversion`) are not properly set.
        """
        # Validate that predictions exist
        if not hasattr(self, "y_hat") or self.y_hat is None:
            raise RuntimeError("Model must be fitted before calling `adjust_output()`.")

        # Validate that df_ exists and is a DataFrame
        if not hasattr(self, "df_") or not isinstance(self.df_, pd.DataFrame):
            raise ValueError("Attribute `df_` must be a pandas DataFrame before calling `adjust_output()`.")

        # Validate that conversion is a string
        if not hasattr(self, "conversion") or not isinstance(self.conversion, str):
            raise ValueError("Attribute `conversion` must be a string indicating the aggregation method used.")

        # Create post-estimation adjustment handler
        adjuster = PostEstimation(self.conversion)

        # Create a copy of df_ to avoid mutating the original data
        df_copy = self.df_.copy()

        # Add the raw predicted values to the copy
        try:
            df_copy["y_hat"] = self.y_hat.flatten()
        except Exception as e:
            raise RuntimeError(f"Could not assign predictions to DataFrame: {str(e)}")

        # Log adjustment start
        self.logger.info("Adjusting negative values in predicted output...")

        # Apply post-estimation adjustment
        try:
            adjusted_df = adjuster.adjust_negative_values(df_copy)
        except Exception as e:
            raise RuntimeError(f"Post-estimation adjustment failed: {str(e)}")

        # Extract adjusted values
        y_hat_adjusted = adjusted_df["y_hat"].to_numpy().reshape(-1, 1)

        # Retrieve padding information
        n_before = getattr(self, "n_pad_before", 0)
        n_after = getattr(self, "n_pad_after", 0)

        # Slice to original range if full=False
        if not full:
            if n_after == 0:
                y_hat_adjusted = y_hat_adjusted[n_before:]
            else:
                y_hat_adjusted = y_hat_adjusted[n_before:-n_after]

        # Log completion
        self.logger.info("Adjustment completed successfully.")

        return y_hat_adjusted