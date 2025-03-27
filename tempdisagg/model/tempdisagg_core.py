import numpy as np
import warnings

from tempdisagg.preprocessing.disagg_input_preparer import DisaggInputPreparer
from tempdisagg.model.models_handler import ModelsHandler
from tempdisagg.utils.logging_utils import VerboseLogger


class TempDisaggModelCore:
    """
    Core engine for temporal disaggregation models.
    
    Handles data preparation, model estimation, fallback logic, and prediction slicing.
    Stores all relevant matrices and parameters for downstream analysis.
    """

    def __init__(
        self,
        conversion="sum",
        grain_col="Grain",
        index_col="Index",
        y_col="y",
        X_col="X",
        interpolation_method="linear",
        method="chow-lin-opt",
        rho_min=-0.9,
        rho_max=0.99,
        fallback_method="fast",
        verbose=False,
        use_retropolarizer=False,
        retro_method="linear_regression",
        retro_aux_col=None
    ):
        """
        Initialize the core disaggregation model.

        INPUT
        conversion : str
            Aggregation rule ('sum', 'average', etc.).
        grain_col : str
            High-frequency grain column.
        index_col : str
            Low-frequency group column.
        y_col : str
            Target variable for disaggregation.
        X_col : str
            Exogenous indicators column(s).
        interpolation_method : str
            Strategy to impute missing values.
        method : str
            Primary estimation method.
        rho_min : float
            Lower bound for rho (if optimized).
        rho_max : float
            Upper bound for rho (if optimized).
        fallback_method : str
            Fallback strategy if main method fails.
        verbose : bool
            Flag to activate logging messages.
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
        self.method = method
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.fallback_method = fallback_method
        self.verbose = verbose
        self.use_retropolarizer = use_retropolarizer
        self.retro_method = retro_method
        self.retro_aux_col = retro_aux_col

        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        self.rho = None
        self.beta = None
        self.residuals = None
        self.Q = None
        self.vcov = None
        self.y_hat = None
        self.y_l = None
        self.X = None
        self.C = None
        self.df_ = None
        self.results_ = {}

        self.n_pad_before = 0
        self.n_pad_after = 0
        self.padding_info = {}

        self.base = DisaggInputPreparer(
            conversion=self.conversion,
            grain_col=self.grain_col,
            index_col=self.index_col,
            y_col=self.y_col,
            X_col=self.X_col,
            verbose=self.verbose,
            interpolation_method=self.interpolation_method,
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
            "uniform": self.models.uniform_estimation
        }

    def fit(self, df):
        self.y_l, self.X, self.C, completed_df, self.padding_info = self.base.prepare(df)

        self.n_pad_before = self.padding_info.get("n_pad_before", 0)
        self.n_pad_after = self.padding_info.get("n_pad_after", 0)
        self.df_ = completed_df

        if self.method not in self.all_methods:
            raise ValueError(f"Unknown method '{self.method}'.")

        self.logger.info(f"Fitting model using method '{self.method}'...")

        result = self.all_methods[self.method](self.y_l, self.X, self.C)

        if result is None or "y_hat" not in result:
            warnings.warn(
                f"Estimation using method '{self.method}' failed. "
                f"Applying fallback with '{self.fallback_method}'.",
                RuntimeWarning
            )
            fallback_func = self.all_methods.get(self.fallback_method)
            if fallback_func is None:
                raise RuntimeError(f"Fallback method '{self.fallback_method}' not found.")
            result = fallback_func(self.y_l, self.X, self.C)
            if result is None or "y_hat" not in result:
                raise RuntimeError(f"Fallback estimation using '{self.fallback_method}' also failed.")
            self.method = self.fallback_method

        self.y_hat = np.atleast_2d(result["y_hat"]).reshape(-1, 1)
        self.beta = result.get("beta")
        self.rho = result.get("rho")
        self.residuals = result.get("residuals")
        self.Q = result.get("Q")
        self.vcov = result.get("vcov")

        if self.y_hat.shape[0] != self.df_.shape[0]:
            raise ValueError("Mismatch: y_hat and df_ have inconsistent lengths.")

        self.results_ = {
            self.method: {
                "beta": self.beta,
                "X": self.X,
                "rho": self.rho,
                "residuals": self.residuals,
                "weight": 1.0
            }
        }

        self.logger.info("Model fitting completed successfully.")
        return self

    def predict(self, full=True):
        if self.y_hat is None:
            raise RuntimeError("Model must be fitted before calling `predict()`.")

        if not full:
            if self.n_pad_after == 0:
                return self.y_hat[self.n_pad_before:]
            return self.y_hat[self.n_pad_before:-self.n_pad_after]
        return self.y_hat

    def fit_predict(self, df):
        self.fit(df)
        return self.predict()

    def get_params(self, deep=True):
        return {
            "conversion": self.conversion,
            "grain_col": self.grain_col,
            "index_col": self.index_col,
            "y_col": self.y_col,
            "X_col": self.X_col,
            "interpolation_method": self.interpolation_method,
            "method": self.method,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "fallback_method": self.fallback_method,
            "verbose": self.verbose,
            "use_retropolarizer": self.use_retropolarizer,
            "retro_method": self.retro_method,
            "retro_aux_col": self.retro_aux_col
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def to_dict(self):
        return {
            "method": self.method,
            "rho": self.rho,
            "beta": self.beta.tolist() if self.beta is not None else None,
            "score": None if self.y_hat is None else float(np.mean(np.abs(self.C @ self.y_hat - self.y_l))),
            "y_hat": self.y_hat.flatten().tolist() if self.y_hat is not None else None
        }
