import numpy as np
import warnings

from tempdisagg.model.tempdisagg_fitter import ModelFitter
from tempdisagg.utils.logging_utils import VerboseLogger


class BaseDisaggModel:
    """
    Lightweight temporal disaggregation model used primarily for ensemble construction.
    Wraps ModelFitter and exposes a minimal API (.fit and .predict).

    This class stores fitted matrices and results for compatibility with other components.
    """

    def __init__(self, method, conversion="sum", verbose=False):
        """
        Initialize the base disaggregation model.

        INPUT
        method : str
            Disaggregation method to be used (e.g., 'chow_lin').
        conversion : str
            Aggregation rule ('sum', 'average', 'first', 'last').
        verbose : bool
            Whether to enable verbose logging messages.

        OUTPUT
        None
        """
        # Save initialization parameters
        self.method = method
        self.conversion = conversion
        self.verbose = verbose

        # Initialize centralized logger
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        # Instantiate model fitter
        self.fitter = ModelFitter(conversion=self.conversion, verbose=self.verbose)

        # Initialize placeholders for outputs and fitted components
        self.y_hat = None       # Predicted high-frequency series
        self.X = None           # Explanatory matrix
        self.C = None           # Conversion matrix
        self.y_l = None         # Low-frequency target
        self.beta = None        # Coefficients
        self.rho = None         # Autocorrelation parameter
        self.residuals = None   # Residuals of the model
        self.padding_info = None  # Padding metadata
        self.df_ = None         # Fitted DataFrame with prediction

    def fit(self, df):
        """
        Fit the model on the input DataFrame using the selected disaggregation method.

        INPUT
        df : pandas.DataFrame
            Input data with required columns ('y', 'X', index, grain, etc.)

        OUTPUT
        None

        RAISES
        ValueError
            If input is not a valid pandas DataFrame.
        RuntimeError
            If model fitting fails.
        """
        # Validate input is a DataFrame-like object
        if df is None or not hasattr(df, "copy"):
            raise ValueError("Input `df` must be a valid pandas DataFrame.")

        # Log the method being used
        self.logger.info(f"Fitting model using method: {self.method}")

        try:
            # Fit the model using the ModelFitter
            self.y_hat, self.padding_info, self.df_ = self.fitter.fit(df, method=self.method)

            # Extract results for the selected method
            result_dict = self.fitter.result_.get(self.method, {})

            # Store fitted matrices and parameters
            self.X = result_dict.get("X")
            self.C = result_dict.get("C")
            self.y_l = result_dict.get("y_l")
            self.beta = result_dict.get("beta")
            self.rho = result_dict.get("rho")
            self.residuals = result_dict.get("residuals")

            # Log successful completion
            self.logger.info("Model fitting completed successfully.")

        except Exception as e:
            raise RuntimeError(f"Model fitting failed for method '{self.method}': {str(e)}")

    def predict(self, full=True):
        """
        Return the predicted high-frequency series from the fitted model.

        INPUT
        full : bool
            Whether to return the full prediction (with padding).
            Currently not used, reserved for future use.

        OUTPUT
        y_hat : np.ndarray
            Predicted high-frequency series.

        RAISES
        RuntimeError
            If model has not been fitted.
        """
        # Check that predictions exist
        if self.y_hat is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        # Return predicted values (full always returned for now)
        return self.y_hat