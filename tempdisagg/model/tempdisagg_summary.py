import numpy as np
import pandas as pd

from numpy.linalg import pinv
from scipy.stats import norm


class TempDisaggReporter:
    def summary(self, metric="mae"):
        if not self.results_:
            raise RuntimeError("Model must be fitted before calling summary().")

        print("\nTemporal Disaggregation Model Summary")
        print("=" * 50)

        if hasattr(self, "ensemble_") and self.ensemble_ is not None:
            return self.ensemble_.summary()

        for method, res in self.results_.items():
            beta = res.get("beta")
            X = res.get("X")
            rho = res.get("rho")

            print(f"\nMethod: {method}")
            if rho is not None:
                print(f"Estimated rho: {rho:.4f}")

            if beta is not None and X is not None:
                try:
                    beta = beta.flatten()
                    XTX_inv = pinv(X.T @ X)
                    std_err = np.sqrt(np.diag(XTX_inv))
                    t_stat = beta / std_err
                    p_val = 2 * (1 - norm.cdf(np.abs(t_stat)))
                    signif = ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "" for p in p_val]
                    df = pd.DataFrame({
                        "Coef.": beta,
                        "Std.Err.": std_err,
                        "t-stat": t_stat,
                        "P>|t|": p_val,
                        "Signif.": signif
                    })

                    y_l = res.get("y_l")
                    C = res.get("C")
                    y_hat = res.get("y_hat", getattr(self, "y_hat", None))

                    if y_l is not None and C is not None and y_hat is not None:
                        try:
                            y_agg = C @ y_hat
                            score = self._compute_score(y_l.flatten(), y_agg.flatten(), metric)
                            df["Score"] = [score] + [np.nan] * (df.shape[0] - 1)
                        except Exception as e:
                            print(f"Score computation failed: {e}")

                    print(df.to_string(index=False, float_format="%.8f"))

                except Exception as e:
                    print("Failed to compute summary statistics.")
                    print(f"Error: {e}")
            else:
                print("No coefficients estimated.")

    def validate_aggregation(self, tol=1e-6):
        if self.y_hat is None or self.C is None or self.y_l is None:
            raise RuntimeError("Model must be fitted before validation.")

        y_agg = self.C @ self.y_hat
        error = np.abs(self.y_l.flatten() - y_agg.flatten())
        max_err = np.max(error)
        if max_err > tol and self.verbose:
            print(f"Max aggregation error: {max_err:.6f}")
        return bool(max_err <= tol)

    def get_docs(self):
        score = None
        if self.y_hat is not None:
            score = self._compute_score(self.y_l.flatten(), (self.C @ self.y_hat).flatten(), "mae")

        return f"""
        Method: {self.method}
        Rho: {self.rho}
        Beta: {self.beta.flatten() if self.beta is not None else 'N/A'}
        Score (MAE): {score:.6f} if score is not None else 'N/A'
        """

    @property
    def coefficients(self):
        return self.beta

    @property
    def rho_estimate(self):
        return self.rho

    @property
    def residuals_lowfreq(self):
        return self.residuals

    @property
    def prediction(self):
        return self.y_hat

    @property
    def design_matrix(self):
        return self.X

    @property
    def conversion_matrix(self):
        return self.C

    @property
    def disagg_results(self):
        return self.results_

    def _compute_score(self, y_true, y_pred, metric):
        if metric == "rmse":
            return np.sqrt(np.mean((y_true - y_pred) ** 2))
        elif metric == "mse":
            return np.mean((y_true - y_pred) ** 2)
        elif metric == "mae":
            return np.mean(np.abs(y_true - y_pred))
        else:
            raise ValueError(f"Unknown metric '{metric}'. Choose from 'rmse', 'mse', 'mae'.")