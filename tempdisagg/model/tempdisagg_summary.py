import numpy as np
import pandas as pd
from numpy.linalg import pinv
from scipy.stats import norm


class TempDisaggReporter:
    """
    Reporting and validation tools for temporal disaggregation models.
    Generates statistical summaries, validates aggregation consistency,
    and exposes model internals through properties.
    """

    def summary(self, metric="mae"):
        """
        Print detailed summary of model coefficients, statistics, and fit score.

        INPUT
        metric : str
            Error metric to compute score ('mae', 'rmse', 'mse').

        OUTPUT
        None
        """
        # Ensure the model has been fitted
        if not hasattr(self, "results_") or not self.results_:
            raise RuntimeError("Model must be fitted before calling summary().")

        print("\nTemporal Disaggregation Model Summary")
        print("=" * 50)

        # If ensemble object is present, delegate summary
        if hasattr(self, "ensemble_") and self.ensemble_ is not None:
            return self.ensemble_.summary()

        # Loop through each method in the results dictionary
        for method, res in self.results_.items():
            beta = res.get("beta")
            X = res.get("X")
            rho = res.get("rho")

            print(f"\nMethod: {method}")
            if rho is not None:
                print(f"Estimated rho: {rho:.4f}")

            # If beta and X are available, compute statistics
            if beta is not None and X is not None:
                try:
                    # Flatten beta and compute variance-covariance
                    beta = beta.flatten()
                    XTX_inv = pinv(X.T @ X)
                    std_err = np.sqrt(np.diag(XTX_inv))

                    # Compute t-stats and p-values
                    t_stat = beta / std_err
                    p_val = 2 * (1 - norm.cdf(np.abs(t_stat)))

                    # Significance stars
                    signif = ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "" for p in p_val]

                    # Assemble results into DataFrame
                    df = pd.DataFrame({
                        "Coef.": beta,
                        "Std.Err.": std_err,
                        "t-stat": t_stat,
                        "P>|t|": p_val,
                        "Signif.": signif
                    })

                    # Try to compute aggregation score if data is available
                    y_l = res.get("y_l")
                    C = res.get("C")
                    y_hat = res.get("y_hat", getattr(self, "y_hat", None))

                    if y_l is not None and C is not None and y_hat is not None:
                        try:
                            y_agg = C @ y_hat
                            score = TempDisaggReporter.compute_score(self, y_l.flatten(), y_agg.flatten(), metric)
                            df["Score"] = [score] + [np.nan] * (df.shape[0] - 1)
                        except Exception as e:
                            print(f"Score computation failed for method '{method}': {e}")

                    # Print the result table
                    print(df.to_string(index=False, float_format="%.9f"))

                except Exception as e:
                    print("Failed to compute summary statistics.")
                    print(f"Error: {e}")
            else:
                print("No coefficients estimated.")


    def validate_aggregation(self, tol=1e-6):
        """
        Check that the aggregated high-frequency prediction matches the low-frequency target.

        INPUT
        tol : float
            Maximum acceptable error between y_l and C @ y_hat.

        OUTPUT
        result : bool
            True if aggregation is valid, False otherwise.
        """
        if not hasattr(self, "y_hat") or self.y_hat is None:
            raise RuntimeError("Missing prediction (y_hat).")

        if not hasattr(self, "C") or self.C is None:
            raise RuntimeError("Missing conversion matrix (C).")

        if not hasattr(self, "y_l") or self.y_l is None:
            raise RuntimeError("Missing low-frequency target (y_l).")

        y_agg = self.C @ self.y_hat
        error = np.abs(self.y_l.flatten() - y_agg.flatten())
        max_err = np.max(error)

        if max_err > tol and getattr(self, "verbose", False):
            print(f"Max aggregation error: {max_err:.6f}")

        return bool(max_err <= tol)

    def get_docs(self):
        """
        Generate a text-based model summary.

        OUTPUT
        doc : str
            Summary text including method, rho, beta, and MAE score.
        """
        score = None
        try:
            if self.y_hat is not None:
                score = self.compute_score(self.y_l.flatten(), (self.C @ self.y_hat).flatten(), "mae")
        except Exception:
            pass

        return f"""
        Method: {getattr(self, 'method', 'N/A')}
        Rho: {getattr(self, 'rho', 'N/A')}
        Beta: {self.beta.flatten() if hasattr(self, 'beta') and self.beta is not None else 'N/A'}
        Score (MAE): {score:.6f}""" if score is not None else "Score: N/A"

    @property
    def coefficients(self):
        """
        Return estimated beta coefficients.
        """
        return getattr(self, "beta", None)

    @property
    def rho_estimate(self):
        """
        Return estimated rho parameter.
        """
        return getattr(self, "rho", None)

    @property
    def residuals_lowfreq(self):
        """
        Return model residuals at low frequency.
        """
        return getattr(self, "residuals", None)

    @property
    def prediction(self):
        """
        Return high-frequency prediction.
        """
        return getattr(self, "y_hat", None)

    @property
    def design_matrix(self):
        """
        Return exogenous matrix X used for disaggregation.
        """
        return getattr(self, "X", None)

    @property
    def conversion_matrix(self):
        """
        Return conversion matrix C used for aggregation constraints.
        """
        return getattr(self, "C", None)

    @property
    def disagg_results(self):
        """
        Return dictionary of results from the fitted model.
        """
        return getattr(self, "results_", None)

    def compute_score(self, y_true, y_pred, metric):
        """
        Internal scoring method.

        INPUT
        y_true : np.ndarray
        y_pred : np.ndarray
        metric : str ('rmse', 'mse', 'mae')

        OUTPUT
        score : float
        """
        if metric == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose from 'rmse', 'mse', 'mae'.")
