import numpy as np
import pandas as pd
import warnings
import logging

from scipy.optimize import minimize
from scipy.linalg import pinv
from scipy.stats import norm
import matplotlib.pyplot as plt

from tempdisagg.utils.logging_utils import VerboseLogger



class EnsemblePrediction:
    """
    Generate ensemble prediction from multiple temporal disaggregation models.
    Combines individual model outputs using optimal weights minimizing aggregation error.

    Parameters
    ----------
    model_class : class
        Class of the temporal disaggregation model to instantiate.
    conversion : str
        Aggregation rule used in temporal disaggregation ('sum', 'average', etc.).
    methods : list of str
        List of method names to ensemble.
    verbose : bool, default=False
        Whether to enable informative logging.
    """

    def __init__(self, model_class, conversion, methods, verbose=False):
        self.model_class = model_class
        self.conversion = conversion
        self.methods = methods
        self.verbose = verbose

        self.models = {}
        self.predictions = {}
        self.weights = None

        # Initialize logger
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

    def fit_predict_all(self, df):
        """
        Fit and predict all models in the ensemble.

        Parameters
        ----------
        df : pd.DataFrame
            Input DataFrame to pass to each model's .fit() method.

        Raises
        ------
        RuntimeError
            If none of the models produce valid predictions.
        """
        self.predictions = {}
        self.models = {}

        for method in self.methods:
            try:
                model = self.model_class(
                    method=method,
                    conversion=self.conversion,
                    verbose=self.verbose
                )
                model.fit(df)
                y_hat = model.predict().flatten()

                # Guardar predicción original para evitar sobrescritura por el ensemble
                model.original_y_hat = y_hat.copy()

                if not np.all(np.isnan(y_hat)):
                    self.predictions[method] = y_hat
                    self.models[method] = model

            except Exception as e:
                self.logger.warning(f"Method {method} failed: {e}")
                continue

        if not self.predictions:
            raise RuntimeError("No valid predictions found.")

    def evaluate_weights(self, y_l, C):
        """
        Optimize weights for combining predictions to minimize aggregated error.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        np.ndarray
            Optimal weights assigned to each model.

        Raises
        ------
        RuntimeError
            If the optimizer fails.
        """
        try:
            pred_matrix = np.column_stack([p for p in self.predictions.values()])
            y_l = np.atleast_2d(y_l).reshape(-1, 1)

            def objective(w):
                y_ens = pred_matrix @ w.reshape(-1, 1)
                y_agg = C @ y_ens
                return np.mean((y_agg - y_l) ** 2)

            init_w = np.ones(len(self.predictions)) / len(self.predictions)
            constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
            bounds = [(0, 1) for _ in init_w]

            result = minimize(objective, init_w, bounds=bounds, constraints=constraints)

            if not result.success:
                raise RuntimeError("Weight optimization failed.")

            self.weights = result.x
            return self.weights

        except Exception as e:
            raise RuntimeError(f"Ensemble weight evaluation failed: {e}")

    def ensemble_predict(self):
        """
        Combine predictions using optimized weights.

        Returns
        -------
        np.ndarray
            Ensemble prediction vector.

        Raises
        ------
        RuntimeError
            If weights are not yet evaluated.
        """
        if self.weights is None:
            raise RuntimeError("Weights must be evaluated before ensemble prediction.")

        pred_matrix = np.column_stack([p for p in self.predictions.values()])
        return pred_matrix @ self.weights

    def run(self, df, y_l, C):
        """
        Fit all models, evaluate ensemble weights, and return final prediction.

        Parameters
        ----------
        df : pd.DataFrame
            Input data.
        y_l : np.ndarray
            Low-frequency series.
        C : np.ndarray
            Conversion matrix.

        Returns
        -------
        np.ndarray
            Ensemble prediction vector.
        """
        self.fit_predict_all(df)
        self.evaluate_weights(y_l, C)
        return self.ensemble_predict()

    def to_model(self, base_model_class):
        """
        Convert ensemble results into a base model-like object for compatibility.

        Parameters
        ----------
        base_model_class : class
            Class to instantiate as the ensemble-compatible model.

        Returns
        -------
        object
            Fitted model object with ensemble predictions and metadata.
        """
        model = base_model_class(
            method="ensemble",
            conversion=self.conversion,
            verbose=self.verbose
        )
        model.y_hat = self.ensemble_predict().reshape(-1, 1)
        model.weights_ = self.weights
        model.ensemble_methods_ = list(self.predictions.keys())
        model.models = self.models
        return model
    
    def summary(self):
        """
        Print summary of individual model results using each model's own summary method.
        """
        if not self.models or self.weights is None:
            print("You must call .run(...) first.")
            return

        print("\nEnsemble Prediction Summary")
        print("=" * 60)

        for i, (method, model) in enumerate(self.models.items()):
            print(f"\nMethod: {method} | Weight: {self.weights[i]:.4f}")
            try:
                model.summary()
            except Exception as e:
                print(f"  Failed to summarize model '{method}': {str(e)}")

    def summary_compact(self, metric="mae"):
        """
        Print a compact table with beta, score and weight per model.

        Parameters
        ----------
        metric : str, default='mae'
            Evaluation metric to include ('mae', 'rmse', 'mse').
        """
        if not self.models or self.weights is None:
            print("You must call .run(...) first.")
            return

        rows = []
        for i, (method, model) in enumerate(self.models.items()):
            beta = model.beta
            score = None

            try:
                if model.y_l is not None and model.C is not None and model.y_hat is not None:
                    y_agg = model.C @ model.y_hat
                    y_l = model.y_l.flatten()
                    y_agg = y_agg.flatten()

                    if metric == "rmse":
                        score = np.sqrt(np.mean((y_l - y_agg) ** 2))
                    elif metric == "mse":
                        score = np.mean((y_l - y_agg) ** 2)
                    else:
                        score = np.mean(np.abs(y_l - y_agg))

                rows.append({
                    "Method": method,
                    "Coef.": beta.flatten()[0] if beta is not None else None,
                    "Score": score,
                    "Weight": self.weights[i]
                })
            except Exception as e:
                rows.append({
                    "Method": method,
                    "Coef.": None,
                    "Score": None,
                    "Weight": self.weights[i],
                    "Error": str(e)
                })

        df = pd.DataFrame(rows)
        print("\nCompact Ensemble Summary")
        print("=" * 60)
        print(df.to_string(index=False, float_format="%.6f"))
            

    def plot(self, df):
        """
        Plot ensemble prediction against individual model predictions.

        Parameters
        ----------
        df : pd.DataFrame
            Original DataFrame with optional 'y' column for observed values.
        """
        if self.weights is None or not self.predictions:
            print("Run .run(...) first.")
            return

        df_plot = df.copy()
        df_plot["y_hat_ensemble"] = self.ensemble_predict().flatten()

        plt.figure(figsize=(12, 5))
        if "y" in df_plot.columns:
            plt.plot(df_plot.index, df_plot["y"], label="Low-freq y (observed)", linestyle="--", marker="o")

        for i, (method, values) in enumerate(self.predictions.items()):
            plt.plot(df_plot.index, values, label=f"{method}", alpha=0.3)

        plt.plot(df_plot.index, df_plot["y_hat_ensemble"], label="Ensemble Prediction", linewidth=2)
        plt.title("Temporal Disaggregation - Ensemble vs Individual Models")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()