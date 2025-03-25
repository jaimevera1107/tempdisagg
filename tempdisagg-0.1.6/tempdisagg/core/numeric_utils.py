import numpy as np
from scipy.linalg import inv, pinv
import warnings

class NumericUtils:
    """
    Utility class for numerical operations used in temporal disaggregation models.
    Includes methods for constructing power matrices and Q matrices with autocorrelation structure.
    """

    @staticmethod
    def power_matrix(n):
        """
        Create a matrix where each element (i, j) is |i - j|.

        Parameters
        ----------
        n : int
            Size of the square matrix to generate.

        Returns
        -------
        numpy.ndarray
            A square matrix of shape (n, n) with absolute differences of indices.
        """
        # Validate that n is a positive integer
        if not isinstance(n, int) or n <= 0:
            raise ValueError("Parameter 'n' must be a positive integer.")

        # Construct the power matrix using broadcasting
        return np.abs(np.subtract.outer(np.arange(n), np.arange(n)))

    @staticmethod
    def q_matrix(rho, power_matrix, epsilon=1e-6):
        """
        Construct a Q matrix for Chow-Lin type models using a power matrix.

        Parameters
        ----------
        rho : float
            Autocorrelation parameter, expected to be between -1 and 1.
        power_matrix : numpy.ndarray
            Matrix of absolute differences in time indices.
        epsilon : float, optional
            Small constant added for numerical stability (default is 1e-6).

        Returns
        -------
        numpy.ndarray
            Q matrix capturing autocorrelation structure.
        """
        # Clip rho to avoid instability
        rho = np.clip(rho, -0.99, 0.99)

        # Validate input power matrix
        if not isinstance(power_matrix, np.ndarray):
            raise TypeError("Parameter 'power_matrix' must be a NumPy ndarray.")

        if power_matrix.shape[0] != power_matrix.shape[1]:
            raise ValueError("Parameter 'power_matrix' must be a square matrix.")

        # Compute the Q matrix based on rho and power_matrix
        return (1 / (1 - rho ** 2 + epsilon)) * (rho ** power_matrix)

    @staticmethod
    def q_litterman(X, rho=0):
        """
        Construct the Q matrix for Litterman's method.

        Parameters
        ----------
        X : numpy.ndarray
            Regressor matrix with shape (n, k).
        rho : float, optional
            Autocorrelation parameter (default is 0).

        Returns
        -------
        numpy.ndarray
            The inverted Q matrix used for GLS estimation.
        """
        # Validate input matrix
        if not isinstance(X, np.ndarray):
            raise TypeError("Parameter 'X' must be a NumPy ndarray.")

        if X.ndim != 2:
            raise ValueError("Parameter 'X' must be a 2D array.")

        # Get the number of observations
        n = X.shape[0]

        # Construct H and D matrices for differencing
        H = np.eye(n) - np.diag(np.ones(n - 1), -1) * rho
        D = np.eye(n) - np.diag(np.ones(n - 1), -1)

        # Compute Q_Litterman
        Q_lit = D.T @ H.T @ H @ D

        try:
            # Attempt to invert Q_Lit with a small ridge for stability
            return inv(Q_lit + np.eye(n) * 1e-8)
        except np.linalg.LinAlgError:
            # Fallback to pseudoinverse if singular
            warnings.warn("Q_Lit matrix is singular. Using pseudo-inverse instead.", RuntimeWarning)
            return pinv(Q_lit)


    @staticmethod
    def validate_symmetric_positive_definite(matrix, tol=1e-8):
        """
        Validate whether a matrix is symmetric and positive definite.

        Parameters
        ----------
        matrix : numpy.ndarray
            The matrix to validate.
        tol : float
            Tolerance for symmetry check (default is 1e-8).

        Returns
        -------
        bool
            True if the matrix is symmetric and positive definite.

        Raises
        ------
        TypeError
            If input is not a NumPy array.
        ValueError
            If the matrix is not square or not symmetric within tolerance.
        np.linalg.LinAlgError
            If the matrix is not positive definite (Cholesky fails).
        """
        # Ensure matrix is a NumPy array
        if not isinstance(matrix, np.ndarray):
            raise TypeError("Input must be a NumPy ndarray.")

        # Ensure matrix is square
        if matrix.shape[0] != matrix.shape[1]:
            raise ValueError("Matrix must be square.")

        # Check symmetry within tolerance
        if not np.allclose(matrix, matrix.T, atol=tol):
            raise ValueError("Matrix is not symmetric within the given tolerance.")

        try:
            # Try Cholesky decomposition to verify positive definiteness
            _ = np.linalg.cholesky(matrix)
            return True
        except np.linalg.LinAlgError as e:
            raise np.linalg.LinAlgError("Matrix is not positive definite.") from e
