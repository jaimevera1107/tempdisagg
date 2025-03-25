import numpy as np
import pandas as pd
import warnings

from scipy.linalg import pinv
from scipy.optimize import minimize_scalar
from scipy.stats import norm

from tempdisagg.utils.logging_utils import VerboseLogger



class RhoOptimizer:
    """
    Optimizes the autocorrelation parameter (rho) for temporal disaggregation models.

    Supports:
        - 'maxlog': Maximum likelihood estimation.
        - 'minrss': Minimum residual sum of squares.

    Returns full statistical outputs: rho, beta, residuals, Q matrix, variance-covariance matrix, and summary.
    """

    def __init__(self, rho_min=-0.9, rho_max=0.99, verbose=False):
        """
        Initialize the optimizer with bounds and verbosity.

        Parameters
        ----------
        rho_min : float
            Lower bound for rho (default = -0.9).
        rho_max : float
            Upper bound for rho (default = 0.99).
        verbose : bool
            Whether to enable informative logging.
        """
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.verbose = verbose

        # Create logger with verbosity control
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

    def power_matrix(self, n):
        """
        Construct a power matrix of absolute differences |i - j|.

        Parameters
        ----------
        n : int
            Dimension of the square matrix.

        Returns
        -------
        numpy.ndarray
            Power matrix of shape (n, n).
        """
        return np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

    def q_matrix(self, rho, power_matrix, epsilon=1e-6):
        """
        Generate Q matrix using autocorrelation parameter rho.

        Parameters
        ----------
        rho : float
            Autocorrelation coefficient.
        power_matrix : numpy.ndarray
            Power matrix from self.power_matrix().
        epsilon : float
            Stability constant to avoid division by zero.

        Returns
        -------
        numpy.ndarray
            Q matrix used for GLS estimation.
        """
        rho = np.clip(rho, -0.99, 0.99)
        return (1 / (1 - rho**2 + epsilon)) * (rho ** power_matrix)

    def optimize(self, y_l, X, C, method="maxlog"):
        """
        Optimize rho and estimate model parameters.

        Parameters
        ----------
        y_l : array-like
            Low-frequency response variable (n_low x 1).
        X : array-like
            High-frequency explanatory variable (n_high x 1).
        C : array-like
            Conversion matrix (n_low x n_high).
        method : str
            Optimization method: 'maxlog' or 'minrss'.

        Returns
        -------
        dict
            Dictionary with rho, Q, beta, residuals, std_errors, t_stats, p_values, and summary.
        """
        try:
            # Reshape and validate inputs
            y_l = np.atleast_2d(y_l).reshape(-1, 1)
            X = np.atleast_2d(X).reshape(-1, 1)
            C = np.atleast_2d(C)

            if C.shape[1] != X.shape[0]:
                raise ValueError("C columns must match X rows.")
            if C.shape[0] != y_l.shape[0]:
                raise ValueError("C rows must match y_l rows.")

            # Compute low-frequency regressor
            X_l = C @ X

            # Precompute power matrix
            power_mat = self.power_matrix(X.shape[0])

            # Store best result during optimization
            results = {}

            def objective(rho):
                """
                Objective function to minimize (depends on method).
                """
                if not (self.rho_min < rho < self.rho_max):
                    return np.inf

                try:
                    # Compute Q matrix and covariance
                    Q = self.q_matrix(rho, power_mat)
                    vcov = C @ Q @ C.T + np.eye(C.shape[0]) * 1e-8
                    inv_vcov = pinv(vcov)

                    # Estimate beta
                    XTX = X_l.T @ inv_vcov @ X_l
                    if XTX.shape[0] != XTX.shape[1]:
                        return np.inf

                    beta = pinv(XTX) @ X_l.T @ inv_vcov @ y_l
                    residuals = y_l - X_l @ beta

                    # Evaluate objective
                    if method == "maxlog":
                        det = np.linalg.det(vcov)
                        if det <= 0:
                            return np.inf
                        loglike = -0.5 * (np.log(det + 1e-8) + residuals.T @ inv_vcov @ residuals)
                        value = -loglike.item()
                    elif method == "minrss":
                        value = (residuals.T @ inv_vcov @ residuals).item()
                    else:
                        raise ValueError("Invalid method: use 'maxlog' or 'minrss'.")

                    # Update best result
                    if "best_val" not in results or value < results["best_val"]:
                        results.update({
                            "rho": rho,
                            "Q": Q,
                            "vcov": vcov,
                            "beta": beta,
                            "residuals": residuals,
                            "XTX": XTX,
                            "inv_vcov": inv_vcov,
                            "X_l": X_l,
                            "best_val": value
                        })

                    return value

                except Exception as e:
                    warnings.warn(f"Optimization failed at rho={rho:.4f}: {e}", RuntimeWarning)
                    return np.inf

            # Minimize scalar function within bounds
            result = minimize_scalar(objective, bounds=(self.rho_min, self.rho_max), method="bounded")

            if not result.success:
                raise RuntimeError(f"Rho optimization failed: {result.message}")

            if self.verbose:
                self.logger.info(f"Optimal rho found: {results['rho']:.6f}")

            # Compute inference statistics
            XTX_inv = pinv(results["XTX"])
            std_errors = np.sqrt(np.diag(XTX_inv))
            t_stats = results["beta"].flatten() / std_errors
            p_values = 2 * (1 - norm.cdf(np.abs(t_stats)))
            significance = ["***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "" for p in p_values]

            summary_df = pd.DataFrame({
                "Coef.": results["beta"].flatten(),
                "Std.Err.": std_errors,
                "t-stat": t_stats,
                "P>|t|": p_values,
                "Signif.": significance
            })

            return {
                "rho": results["rho"],
                "Q": results["Q"],
                "vcov": results["vcov"],
                "beta": results["beta"],
                "residuals": results["residuals"],
                "std_errors": std_errors,
                "t_stats": t_stats,
                "p_values": p_values,
                "significance": significance,
                "summary": summary_df
            }

        except Exception as e:
            raise RuntimeError(f"Optimization process failed: {e}")