import numpy as np
import warnings

from tempdisagg.preprocessing.disagg_input_preparer import DisaggInputPreparer
from tempdisagg.model.models_handler import ModelsHandler
from tempdisagg.utils.logging_utils import VerboseLogger


class ModelFitter:
    """
    Fits a single temporal disaggregation model to a given dataset.

    This class handles preprocessing, method selection, fallback estimation, and
    stores internal components like predictions, coefficients, and residuals.
    """

    def __init__(
        self,
        conversion="sum",
        grain_col="Grain",
        index_col="Index",
        y_col="y",
        X_col="X",
        interpolation_method="linear",
        rho_min=-0.9,
        rho_max=0.99,
        fallback_method="fast",
        verbose=False,
        use_retropolarizer=False,
        retro_method="linear_regression",
        retro_aux_col=None
    ):
        """
        Initialize the ModelFitter.

        INPUT
        conversion : str
            Aggregation method ('sum', 'average', etc.).
        grain_col : str
            High-frequency index column name.
        index_col : str
            Low-frequency group column name.
        y_col : str
            Target variable for disaggregation.
        X_col : str
            Exogenous regressor column name.
        interpolation_method : str
            Strategy for imputing missing values.
        rho_min : float
            Lower bound for autocorrelation optimization.
        rho_max : float
            Upper bound for autocorrelation optimization.
        fallback_method : str
            Method used if the main one fails.
        verbose : bool
            Whether to enable logging messages.
        use_retropolarizer : bool
            Whether to use Retropolarizer instead of standard interpolation for y_col.
        retro_method : str
            Method used by Retropolarizer (e.g. 'linear_regression').
        retro_aux_col : str or None
            Auxiliary column to use as predictor for retropolarization. If None, uses X_col.

        OUTPUT
        None
        """
        self.conversion = conversion
        self.grain_col = grain_col
        self.index_col = index_col
        self.y_col = y_col
        self.X_col = X_col
        self.interpolation_method = interpolation_method
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.fallback_method = fallback_method
        self.verbose = verbose
        self.use_retropolarizer = use_retropolarizer
        self.retro_method = retro_method
        self.retro_aux_col = retro_aux_col

        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        self.base = DisaggInputPreparer(
            conversion=self.conversion,
            grain_col=self.grain_col,
            index_col=self.index_col,
            y_col=self.y_col,
            X_col=self.X_col,
            interpolation_method=self.interpolation_method,
            verbose=self.verbose,
            use_retropolarizer=self.use_retropolarizer,
            retro_method=self.retro_method,
            retro_aux_col=self.retro_aux_col
        )

        self.models = ModelsHandler(
            rho_min=self.rho_min,
            rho_max=self.rho_max,
            verbose=self.verbose
        )

        self.all_methods = {
            "ols": self.models.ols_estimation,
            "denton": self.models.denton_estimation,
            "chow-lin": self.models.chow_lin_estimation,
            "litterman": self.models.litterman_estimation,
            "fernandez": self.models.fernandez_estimation,
            "fast": self.models.fast_estimation,
            "chow-lin-opt": self.models.chow_lin_opt_estimation,
            "litterman-opt": self.models.litterman_opt_estimation,
            "chow-lin-ecotrim": self.models.chow_lin_minrss_ecotrim,
            "chow-lin-quilis": self.models.chow_lin_minrss_quilis,
            "denton-opt": self.models.denton_opt_estimation,
            "denton-colette": self.models.denton_cholette_estimation,
            "uniform": self.models.uniform_estimation,
        }

    def fit(self, df, method="chow-lin-opt"):
        """
        Fit the disaggregation model to the input data.

        INPUT
        df : pandas.DataFrame
            DataFrame with the required columns.
        method : str
            Name of the disaggregation method to use.

        OUTPUT
        y_hat : np.ndarray
            Predicted high-frequency series.
        padding_info : dict
            Information on padding applied before/after.
        df_full : pandas.DataFrame
            Completed DataFrame used in fitting.

        RAISES
        ValueError
            If method is invalid or prediction shape mismatches.
        RuntimeError
            If both primary and fallback estimations fail.
        """
        y_l, X, C, df_full, padding_info = self.base.prepare(df)

        if method not in self.all_methods:
            raise ValueError(f"Unknown method '{method}'.")

        self.logger.info(f"Fitting model using method: '{method}'...")

        result = self.all_methods[method](y_l, X, C)

        if result is None or "y_hat" not in result:
            warnings.warn(
                f"Estimation using method '{method}' failed. Using fallback '{self.fallback_method}'.",
                RuntimeWarning
            )
            fallback_func = self.all_methods.get(self.fallback_method)
            if fallback_func is None:
                raise RuntimeError(f"Fallback method '{self.fallback_method}' not found.")
            result = fallback_func(y_l, X, C)
            method = self.fallback_method

        y_hat = np.atleast_2d(result["y_hat"]).reshape(-1, 1)

        if y_hat.shape[0] != df_full.shape[0]:
            raise ValueError("Mismatch between `y_hat` and DataFrame length.")

        self.result_ = {
            method: {
                "beta": result.get("beta"),
                "rho": result.get("rho"),
                "residuals": result.get("residuals"),
                "Q": result.get("Q"),
                "vcov": result.get("vcov"),
                "X": X,
                "C": C,
                "y_l": y_l
            }
        }

        self.logger.info(f"Model fitting finished using method: '{method}'")
        return y_hat, padding_info, df_full

    def get_df(self):
        """
        Return completed DataFrame from preprocessing.

        OUTPUT
        df_full : pandas.DataFrame or None
            DataFrame returned by DisaggInputPreparer or None if not run yet.
        """
        return getattr(self.base, "df_full", None)
