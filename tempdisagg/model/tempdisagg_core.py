import numpy as np
import warnings

from tempdisagg.preprocessing.disagg_input_preparer import DisaggInputPreparer
from tempdisagg.model.models_handler import ModelsHandler

class TempDisaggModelCore:
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
        verbose=False
    ):
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

        self.rho = None
        self.beta = None
        self.residuals = None
        self.Q = None
        self.vcov = None
        self.y_hat = None
        self.y_l = None
        self.X = None
        self.C = None
        self.results_ = {}

        self.base = DisaggInputPreparer(
            conversion=self.conversion,
            grain_col=self.grain_col,
            index_col=self.index_col,
            y_col=self.y_col,
            X_col=self.X_col,
            verbose=self.verbose,
            interpolation_method=self.interpolation_method
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

    def fit(self, df):
        self.y_l, self.X, self.C, self.padding_info = self.base.prepare(df)
        self.n_pad_before = self.padding_info.get("n_pad_before", 0)
        self.n_pad_after = self.padding_info.get("n_pad_after", 0)
        self.df_ = self.base.df_full if hasattr(self.base, "df_full") else df


        if self.method not in self.all_methods:
            raise ValueError(f"Unknown method '{self.method}'.")

        result = self.all_methods[self.method](self.y_l, self.X, self.C)

        if result is None or "y_hat" not in result:
            warnings.warn(
                f"Estimation using method '{self.method}' failed. Applying fallback with '{self.fallback_method}' method.",
                RuntimeWarning
            )
            fallback_func = self.all_methods.get(self.fallback_method)
            if fallback_func is None:
                raise RuntimeError(f"Fallback method '{self.fallback_method}' not found.")
            fallback_result = fallback_func(self.y_l, self.X, self.C)
            if fallback_result is None or "y_hat" not in fallback_result:
                raise RuntimeError(f"Fallback estimation using '{self.fallback_method}' also failed.")
            else:
                self.method = self.fallback_method
                result = fallback_result

        self.y_hat = np.atleast_2d(result["y_hat"]).reshape(-1, 1)
        self.beta = result.get("beta")
        self.rho = result.get("rho")
        self.residuals = result.get("residuals")
        self.Q = result.get("Q")
        self.vcov = result.get("vcov")

        self.results_ = {
            self.method: {
                "beta": self.beta,
                "X": self.X,
                "rho": self.rho,
                "residuals": self.residuals,
                "weight": 1.0
            }
        }
        return self

    def predict(self, full=False):
        if self.y_hat is None:
            raise RuntimeError("Model must be fitted before prediction.")

        if full:
            return self.y_hat
        else:
            start = self.n_pad_before
            end = None if self.n_pad_after == 0 else -self.n_pad_after
            return self.y_hat[start:end].reshape(-1, 1)

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
            "verbose": self.verbose
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