import numpy as np
import matplotlib.pyplot as plt
import warnings

from tempdisagg.postprocessing.ensemble_prediction import EnsemblePrediction
from tempdisagg.preprocessing.disagg_input_preparer import DisaggInputPreparer
from tempdisagg.model.tempdisagg_base import BaseDisaggModel
from tempdisagg.utils.logging_utils import VerboseLogger


class EnsemblePredictor:
    """
    Runs ensemble predictions using multiple temporal disaggregation methods.
    Prepares inputs, fits multiple models, stores results and allows inspection and plotting.
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
        Initialize the ensemble predictor.

        INPUT
        conversion : str
            Aggregation rule to use (e.g., 'sum', 'average', etc.).
        grain_col : str
            High-frequency index column name.
        index_col : str
            Low-frequency group column name.
        y_col : str
            Name of the target variable to disaggregate.
        X_col : str
            Name of the indicator or exogenous variable column.
        interpolation_method : str
            Strategy for imputing missing values.
        rho_min : float
            Lower bound for rho parameter.
        rho_max : float
            Upper bound for rho parameter.
        fallback_method : str
            Method to use if one model fails.
        verbose : bool
            Whether to enable verbose logging.
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
        self.verbose = verbose

        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        self.base = DisaggInputPreparer(
            conversion=conversion,
            grain_col=grain_col,
            index_col=index_col,
            y_col=y_col,
            X_col=X_col,
            interpolation_method=interpolation_method,
            verbose=verbose,
            use_retropolarizer=use_retropolarizer,
            retro_method=retro_method,
            retro_aux_col=retro_aux_col
        )

        self.ensemble = None
        self.df_full = None
        self.padding_info = {}
        self.results_ = {}
        self.predictions = {}
        self.weights = None

    def fit(self, df, methods=None):
        y_l, X, C, df_full, padding_info = self.base.prepare(df)

        self.df_full = df_full
        self.padding_info = padding_info

        if methods is None:
            methods = [
                "ols", "denton", "chow-lin", "litterman", "fernandez", "fast",
                "chow-lin-opt", "litterman-opt", "chow-lin-ecotrim", "chow-lin-quilis",
                "denton-opt", "denton-colette", "uniform"
            ]
            self.logger.info(f"No methods specified. Using all available: {methods}")

        self.ensemble = EnsemblePrediction(
            model_class=BaseDisaggModel,
            conversion=self.conversion,
            methods=methods,
            verbose=self.verbose
        )

        self.logger.info("Fitting ensemble model...")
        y_hat = self.ensemble.run(df_full, y_l, C).reshape(-1, 1)

        self.results_ = {
            name: {
                "beta": m.beta,
                "X": m.X,
                "rho": m.rho,
                "residuals": m.residuals,
                "C": m.C,
                "y_l": m.y_l,
                "weight": self.ensemble.weights[i]
            }
            for i, (name, m) in enumerate(self.ensemble.models.items())
        }

        self.predictions = {
            name: m.y_hat.flatten()
            for name, m in self.ensemble.models.items()
        }

        self.weights = self.ensemble.weights

        self.logger.info("Ensemble fitting completed.")
        return y_hat, padding_info, df_full

    def predict(self):
        if self.ensemble is None:
            raise RuntimeError("Call `.fit()` before `.predict()`.")

        return self.ensemble.ensemble_predict().reshape(-1, 1)

    def plot(self, df=None):
        if self.weights is None or not self.predictions:
            warnings.warn("Run `.fit()` before calling `.plot()`.")
            return

        if df is None:
            if self.df_full is None:
                raise ValueError("No DataFrame available. Pass `df` or call `.fit()` first.")
            df_plot = self.df_full.copy()
        else:
            df_plot = df.copy()

        y_ens = self.predict().flatten()

        if len(df_plot) != len(y_ens):
            raise ValueError("Length mismatch between prediction and DataFrame.")

        df_plot["y_hat_ensemble"] = y_ens

        plt.figure(figsize=(12, 5))
        if "y" in df_plot.columns:
            plt.plot(df_plot.index, df_plot["y"], label="Observed y", linestyle="--", marker="o")

        for method, y_pred in self.predictions.items():
            plt.plot(df_plot.index, y_pred, label=f"{method}", alpha=0.3)

        plt.plot(df_plot.index, df_plot["y_hat_ensemble"], label="Ensemble Prediction", linewidth=2)

        plt.title("Temporal Disaggregation - Ensemble vs Individual Models")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def summary(self):
        if not self.results_:
            raise RuntimeError("Call `.fit()` before `.summary()`.")

        return {
            method: {
                "weight": round(info.get("weight", 0.0), 4),
                "rho": info.get("rho"),
                "beta": info.get("beta").tolist() if info.get("beta") is not None else None
            }
            for method, info in self.results_.items()
        }

    def summary_compact(self):
        summary = self.summary()
        print("Ensemble Summary:\n")
        print(f"{'Method':<25} {'Weight':<10} {'Rho':<10}")
        print("-" * 45)
        for method, values in summary.items():
            print(f"{method:<25} {values['weight']:<10} {values['rho']:<10}")
