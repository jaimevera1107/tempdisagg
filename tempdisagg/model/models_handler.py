import numpy as np
from scipy.linalg import toeplitz, pinv, solve
import warnings
import logging

from tempdisagg.utils.logging_utils import VerboseLogger
from tempdisagg.core.numeric_utils import NumericUtils
from tempdisagg.core.rho_optimizer import RhoOptimizer

class ModelsHandler:
    """
    Core handler for estimating temporal disaggregation models.
    Includes OLS, Denton, Chow-Lin, Litterman, Fernández and optimized variants.
    """

    def __init__(self, rho_min=-0.9, rho_max=0.99, verbose=False):
        self.min_rho_boundarie = rho_min
        self.max_rho_boundarie = rho_max
        self.verbose = verbose

        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        self.utils = NumericUtils()
        self.optimizer = RhoOptimizer(
            rho_min=self.min_rho_boundarie,
            rho_max=self.max_rho_boundarie,
            verbose=self.verbose
        )

    def preprocess_inputs(self, y_l, X, C):
        """
        Ensure input arrays have the correct shape and are compatible.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        tuple
            Tuple containing reshaped y_l, X, and original C.
        """

        y_l = np.atleast_2d(y_l).reshape(-1, 1)
        X = np.atleast_2d(X).reshape(-1, 1)
        return y_l, X, C

    def ols_estimation(self, y_l, X, C):
        """
        Perform OLS estimation for temporal disaggregation.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        dict or None
            Dictionary with keys 'y_hat' and 'beta' if successful, None otherwise.
        """

        try:
            y_l, X, C = self.preprocess_inputs(y_l, X, C)
            X_l = C @ X
            beta = pinv(X_l.T @ X_l) @ X_l.T @ y_l
            y_hat = X @ beta
            return {"y_hat": y_hat, "beta": beta}
        except Exception as e:
            self.logger.error(f"OLS estimation failed: {e}")
            return None

    def denton_estimation(self, y_l, X, C, h=1):
        """
        Perform Denton estimation using differencing of order h.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.
        h : int, default=1
            Order of differencing to penalize irregularity.

        Returns
        -------
        dict or None
            Dictionary with keys 'y_hat', 'beta', 'residuals', 'Q', and 'vcov' if successful, None otherwise.
        """

        try:
            y_l, X, C = self.preprocess_inputs(y_l, X, C)
            n = len(X)
            D = np.eye(n) - np.diag(np.ones(n - 1), -1)
            D_h = np.linalg.matrix_power(D, h) if h > 0 else np.eye(n)
            Sigma_D = pinv(D_h.T @ D_h)
            Q = C @ Sigma_D @ C.T
            inv_Q = pinv(Q)
            beta = pinv(X.T @ C.T @ inv_Q @ C @ X) @ X.T @ C.T @ inv_Q @ y_l
            p = X @ beta
            D_matrix = Sigma_D @ C.T @ inv_Q
            u_l = y_l - C @ p
            y_hat = p + D_matrix @ u_l
            return {"y_hat": y_hat, "beta": beta, "residuals": u_l, "Q": Sigma_D, "vcov": Q}
        except Exception as e:
            self.logger.error(f"Denton estimation failed: {e}")
            return None

    def chow_lin_estimation(self, y_l, X, C, rho=0.5):
        """
        Perform Chow-Lin estimation with fixed autocorrelation parameter rho.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.
        rho : float, default=0.5
            Autocorrelation coefficient (bounded internally).

        Returns
        -------
        dict or None
            Dictionary with keys 'y_hat', 'rho', 'beta', 'residuals', 'Q', and 'vcov' if successful, None otherwise.
        """

        try:
            y_l, X, C = self.preprocess_inputs(y_l, X, C)
            n = len(X)
            rho = np.clip(rho, self.min_rho_boundarie, self.max_rho_boundarie)
            power_mat = self.utils.power_matrix(n)
            Sigma_CL = self.utils.q_matrix(rho, power_mat)
            Q = C @ Sigma_CL @ C.T
            inv_Q = pinv(Q)
            beta = solve(X.T @ C.T @ inv_Q @ C @ X, X.T @ C.T @ inv_Q @ y_l).reshape(-1, 1)
            p = X @ beta
            D = Sigma_CL @ C.T @ inv_Q
            u_l = y_l - C @ p
            y_hat = p + D @ u_l
            return {"y_hat": y_hat, "rho": rho, "beta": beta, "residuals": u_l, "Q": Sigma_CL, "vcov": Q}
        except Exception as e:
            self.logger.error(f"Chow-Lin estimation failed: {e}")
            return None

    def litterman_estimation(self, y_l, X, C, rho=0.9):
        """
        Perform Litterman estimation with fixed rho value.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.
        rho : float, default=0.9
            Fixed autocorrelation coefficient for the prior.

        Returns
        -------
        dict or None
            Dictionary with keys 'y_hat', 'rho', 'beta', 'residuals', 'Q', and 'vcov' if successful, None otherwise.
        """

        try:
            y_l, X, C = self.preprocess_inputs(y_l, X, C)
            n = len(X)
            H = np.eye(n) - np.diag(np.ones(n - 1), -1) * rho
            Sigma_F = pinv(H.T @ H)
            Q = C @ Sigma_F @ C.T
            inv_Q = pinv(Q)
            beta = solve(X.T @ C.T @ inv_Q @ C @ X, X.T @ C.T @ inv_Q @ y_l)
            p = X @ beta
            D = Sigma_F @ C.T @ inv_Q
            u_l = y_l - C @ p
            y_hat = p + D @ u_l
            return {"y_hat": y_hat, "rho": rho, "beta": beta, "residuals": u_l, "Q": Sigma_F, "vcov": Q}
        except Exception as e:
            self.logger.error(f"Litterman estimation failed: {e}")
            return None

    def fernandez_estimation(self, y_l, X, C):
        """
        Perform Fernández estimation using second-order differencing.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        dict or None
            Dictionary with keys 'y_hat', 'beta', 'residuals', 'Q', and 'vcov' if successful, None otherwise.
        """

        try:
            y_l, X, C = self.preprocess_inputs(y_l, X, C)
            n = len(X)
            Delta = np.eye(n) - np.diag(np.ones(n - 1), -1)
            Sigma_F = np.linalg.inv(Delta.T @ Delta)
            Q = C @ Sigma_F @ C.T
            inv_Q = np.linalg.inv(Q)
            beta = solve(X.T @ C.T @ inv_Q @ C @ X, X.T @ C.T @ inv_Q @ y_l).reshape(-1, 1)
            p = X @ beta
            D = Sigma_F @ C.T @ inv_Q
            u_l = y_l - C @ p
            y_hat = p + D @ u_l
            return {"y_hat": y_hat, "beta": beta, "residuals": u_l, "Q": Sigma_F, "vcov": Q}
        except Exception as e:
            self.logger.error(f"Fernández estimation failed: {e}")
            return None

    def fast_estimation(self, y_l, X, C):
        """
        Perform fast estimation using Litterman method with fixed rho = 0.9.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        dict or None
            Dictionary with keys 'y_hat', 'rho', 'beta', 'residuals', 'Q', and 'vcov' if successful, None otherwise.
        """

        return self.litterman_estimation(y_l, X, C, rho=0.9)


    def chow_lin_minrss_ecotrim(self, y_l, X, C):
        """
        Perform Chow-Lin estimation with fixed rho = 0.75 (Ecotrim variant).

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        dict or None
            Estimation result or None if it fails.
        """
        try:
            return self.chow_lin_estimation(y_l=y_l, X=X, C=C, rho=0.75)
        except Exception as e:
            self.logger.error(f"Chow-Lin Ecotrim failed: {e}")
            return None

    def chow_lin_minrss_quilis(self, y_l, X, C):
        """
        Perform Chow-Lin estimation with fixed rho = 0.15 (Quilis variant).

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        dict or None
            Estimation result or None if it fails.
        """
        try:
            return self.chow_lin_estimation(y_l=y_l, X=X, C=C, rho=0.15)
        except Exception as e:
            self.logger.error(f"Chow-Lin Quilis failed: {e}")
            return None

    def chow_lin_opt_estimation(self, y_l, X, C):
        """
        Perform Chow-Lin estimation with optimized rho via log-likelihood.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        dict or None
            Dictionary with keys 'y_hat', 'rho', 'beta', 'residuals', 'Q', and 'vcov' if successful, None otherwise.
        """

        try:
            # Optimize rho using maximum log-likelihood
            result = self.optimizer.optimize(y_l=y_l, X=X, C=C, method="maxlog")
            rho_opt = result["rho"]

            # Use classic Chow-Lin with optimized rho
            return self.chow_lin_estimation(y_l=y_l, X=X, C=C, rho=rho_opt)

        except Exception as e:
            self.logger.error(f"Optimized Chow-Lin failed: {e}")
            return None
        
    def litterman_opt_estimation(self, y_l, X, C):
        """
        Perform Litterman estimation with optimized rho via RSS minimization.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series.
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        dict or None
            Dictionary with keys 'y_hat', 'rho', 'beta', 'residuals', 'Q', and 'vcov' if successful, None otherwise.
        """
        try:
            # Optimize rho using residual sum of squares minimization
            result = self.optimizer.optimize(y_l=y_l, X=X, C=C, method="minrss")
            rho_opt = result["rho"]

            # Use classic Litterman with optimized rho
            return self.litterman_estimation(y_l=y_l, X=X, C=C, rho=rho_opt)

        except Exception as e:
            self.logger.error(f"Optimized Litterman failed: {e}")
            return None 

    def denton_cholette_estimation(self,y_l, X, C, base_series=None, weights=None):
        """
        Perform Denton-Cholette estimation for smooth temporal disaggregation.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        C : np.ndarray
            Conversion matrix from high to low frequency.
        base_series : np.ndarray or None, optional
            Initial high-frequency proxy to be adjusted. If None, a uniform series is used.
        weights : np.ndarray or None, optional
            Optional weights for smoothness penalty. Must match length of high-frequency series.

        Returns
        -------
        dict or None
            Dictionary with keys 'y_hat', 'beta', 'rho', 'residuals', 'Q', and 'vcov'.
            Values are None for 'beta', 'rho', and 'vcov'.
        """
        try:
            y_l = np.atleast_2d(y_l).reshape(-1, 1)
            m = C.shape[1]

            if base_series is None:
                base_series = np.ones((m, 1))

            base_series = np.atleast_2d(base_series).reshape(-1, 1)

            if base_series.shape[0] != m:
                raise ValueError("Base series length must match number of high-frequency periods.")

            if weights is not None:
                weights = np.atleast_1d(weights).reshape(-1)
                if len(weights) != m:
                    raise ValueError("Weights must have same length as high-frequency base series.")
                W = np.diag(weights)
            else:
                W = np.eye(m)

            D = np.eye(m) - np.diag(np.ones(m - 1), -1)
            D = D[1:]
            penalty = D.T @ D
            P = W + penalty

            try:
                inv_P = pinv(P)
            except Exception:
                self.logger.warning("Falling back to pseudoinverse for Denton-Cholette.")
                inv_P = np.linalg.pinv(P)

            A = C @ inv_P @ C.T
            try:
                lambda_vec = solve(A, y_l - C @ base_series)
            except Exception as e:
                self.logger.error(f"Failed solving Denton-Cholette constraint system: {e}")
                return None

            y_hat = base_series + inv_P @ C.T @ lambda_vec
            residuals = y_l - C @ y_hat

            return {
                "y_hat": y_hat,
                "beta": None,
                "rho": None,
                "residuals": residuals,
                "Q": penalty,
                "vcov": None
            }

        except Exception as e:
            self.logger.error(f"Denton-Cholette estimation failed: {e}")
            return None
        

    def uniform_estimation(self, y_l, X, C):
        """
        Perform uniform disaggregation: evenly distributes each y_l across its corresponding high-frequency periods.

        Parameters
        ----------
        y_l : np.ndarray
            Low-frequency target series.
        X : np.ndarray
            High-frequency regressor series (ignored in this method).
        C : np.ndarray
            Conversion matrix from high to low frequency.

        Returns
        -------
        dict
            Dictionary with uniform 'y_hat' disaggregation.
        """
        try:
            y_l, _, C = self.preprocess_inputs(y_l, X, C)

            # Each row in C defines how low-freq y is mapped to high-freq. We want inverse mapping.
            n_high = C.shape[1]
            ones = np.ones((n_high, 1))
            C_inv = C.T @ pinv(C @ C.T)
            y_hat = C_inv @ y_l

            return {"y_hat": y_hat, "rho": None, "beta": None, "residuals": None, "Q": None, "vcov": None}

        except Exception as e:
            self.logger.error(f"Uniform estimation failed: {e}")
            return None
