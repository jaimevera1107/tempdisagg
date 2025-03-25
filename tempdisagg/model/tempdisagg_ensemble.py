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
        verbose=False
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

        OUTPUT
        None
        """
        # Store configuration
        self.conversion = conversion
        self.verbose = verbose

        # Initialize logger
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        # Prepare input manager
        self.base = DisaggInputPreparer(
            conversion=conversion,
            grain_col=grain_col,
            index_col=index_col,
            y_col=y_col,
            X_col=X_col,
            interpolation_method=interpolation_method,
            verbose=verbose
        )

        # Placeholders
        self.ensemble = None
        self.df_full = None
        self.padding_info = {}
        self.results_ = {}
        self.predictions = {}
        self.weights = None

    def fit(self, df, methods=None):
        """
        Fit an ensemble of disaggregation models.

        INPUT
        df : pandas.DataFrame
            Input DataFrame with target and indicators.
        methods : list or None
            List of methods to include in the ensemble.

        OUTPUT
        y_hat : np.ndarray
            Predicted high-frequency series.
        padding_info : dict
            Information on rows padded before and after.
        df_full : pandas.DataFrame
            Completed DataFrame used for fitting.
        """
        # Prepare matrices
        y_l, X, C, df_full, padding_info = self.base.prepare(df)

        # Save internal references
        self.df_full = df_full
        self.padding_info = padding_info

        # Initialize ensemble engine
        self.ensemble = EnsemblePrediction(
            model_class=BaseDisaggModel,
            conversion=self.conversion,
            methods=methods,
            verbose=self.verbose
        )

        # Run all models and generate prediction
        self.logger.info("Fitting ensemble model...")
        y_hat = self.ensemble.run(df_full, y_l, C).reshape(-1, 1)

        # Save results per method
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

        # Save predictions per model
        self.predictions = {
            name: m.y_hat.flatten()
            for name, m in self.ensemble.models.items()
        }

        # Save ensemble weights
        self.weights = self.ensemble.weights

        self.logger.info("Ensemble fitting completed.")
        return y_hat, padding_info, df_full

    def predict(self):
        """
        Return the ensemble prediction.

        OUTPUT
        y_hat : np.ndarray
            Final ensemble prediction as array.
        """
        if self.ensemble is None:
            raise RuntimeError("Call `.fit()` before `.predict()`.")

        return self.ensemble.ensemble_predict().reshape(-1, 1)

    def plot(self, df=None):
        """
        Plot ensemble prediction and individual models.

        INPUT
        df : pandas.DataFrame or None
            Optional DataFrame for plotting. If None, use internal `df_full`.

        OUTPUT
        None (displays a plot)
        """
        # Validate prediction availability
        if self.weights is None or not self.predictions:
            warnings.warn("Run `.fit()` before calling `.plot()`.")
            return

        # Use internal DataFrame if not provided
        if df is None:
            if self.df_full is None:
                raise ValueError("No DataFrame available. Pass `df` or call `.fit()` first.")
            df_plot = self.df_full.copy()
        else:
            df_plot = df.copy()

        # Get ensemble prediction
        y_ens = self.predict().flatten()

        # Validate alignment
        if len(df_plot) != len(y_ens):
            raise ValueError("Length mismatch between prediction and DataFrame.")

        # Assign predictions to DataFrame
        df_plot["y_hat_ensemble"] = y_ens

        # Plot observed series if available
        plt.figure(figsize=(12, 5))
        if "y" in df_plot.columns:
            plt.plot(df_plot.index, df_plot["y"], label="Observed y", linestyle="--", marker="o")

        # Plot individual model predictions
        for method, y_pred in self.predictions.items():
            plt.plot(df_plot.index, y_pred, label=f"{method}", alpha=0.3)

        # Plot ensemble line
        plt.plot(df_plot.index, df_plot["y_hat_ensemble"], label="Ensemble Prediction", linewidth=2)

        # Final touches
        plt.title("Temporal Disaggregation - Ensemble vs Individual Models")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def summary(self):
        """
        Return summary dictionary of weights and parameters per method.

        OUTPUT
        summary : dict
            Mapping of method to metadata (weights, rho, beta).
        """
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
        """
        Print compact summary table of model weights and rhos.

        OUTPUT
        None (prints table)
        """
        summary = self.summary()
        print("Ensemble Summary:\n")
        print(f"{'Method':<25} {'Weight':<10} {'Rho':<10}")
        print("-" * 45)
        for method, values in summary.items():
            print(f"{method:<25} {values['weight']:<10} {values['rho']:<10}")
