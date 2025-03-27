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

    def __init__(self, method, conversion="sum", verbose=False,
                 use_retropolarizer=False, retro_method="linear_regression", retro_aux_col=None):
        """
        Initialize the base disaggregation model.

        INPUT
        method : str
            Disaggregation method to be used (e.g., 'chow_lin').
        conversion : str
            Aggregation rule ('sum', 'average', 'first', 'last').
        verbose : bool
            Whether to enable verbose logging messages.
        use_retropolarizer : bool
            Whether to use Retropolarizer instead of standard interpolation for y_col.
        retro_method : str
            Method used by Retropolarizer (e.g. 'linear_regression').
        retro_aux_col : str or None
            Auxiliary column to use as predictor for retropolarization. If None, uses X_col.

        OUTPUT
        None
        """
        self.method = method
        self.conversion = conversion
        self.verbose = verbose
        self.use_retropolarizer = use_retropolarizer
        self.retro_method = retro_method
        self.retro_aux_col = retro_aux_col

        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        self.fitter = ModelFitter(
            conversion=self.conversion,
            verbose=self.verbose,
            use_retropolarizer=self.use_retropolarizer,
            retro_method=self.retro_method,
            retro_aux_col=self.retro_aux_col
        )

        self.y_hat = None
        self.X = None
        self.C = None
        self.y_l = None
        self.beta = None
        self.rho = None
        self.residuals = None
        self.padding_info = None
        self.df_ = None

    def fit(self, df):
        if df is None or not hasattr(df, "copy"):
            raise ValueError("Input `df` must be a valid pandas DataFrame.")

        self.logger.info(f"Fitting model using method: {self.method}")

        try:
            self.y_hat, self.padding_info, self.df_ = self.fitter.fit(df, method=self.method)

            result_dict = self.fitter.result_.get(self.method, {})

            self.X = result_dict.get("X")
            self.C = result_dict.get("C")
            self.y_l = result_dict.get("y_l")
            self.beta = result_dict.get("beta")
            self.rho = result_dict.get("rho")
            self.residuals = result_dict.get("residuals")

            self.logger.info("Model fitting completed successfully.")

        except Exception as e:
            raise RuntimeError(f"Model fitting failed for method '{self.method}': {str(e)}")

    def predict(self, full=True):
        if self.y_hat is None:
            raise RuntimeError("Model must be fitted before calling predict().")

        return self.y_hat
