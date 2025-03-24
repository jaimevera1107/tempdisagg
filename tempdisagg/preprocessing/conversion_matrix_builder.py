import numpy as np
import warnings
from tempdisagg.utils.logging_utils import VerboseLogger


class ConversionMatrixBuilder:
    """
    Builds a conversion matrix for temporal disaggregation
    using a specified aggregation method.

    Supported methods:
        - "sum":     Equal weights summing to the total.
        - "average": Equal weights summing to 1.
        - "first":   Weight only for the first element.
        - "last":    Weight only for the last element.
    """

    def __init__(self, conversion, grain_col, index_col, verbose=False):
        """
        Initialize the ConversionMatrixBuilder.

        Parameters
        ----------
        conversion : str
            Aggregation method ("sum", "average", "first", or "last").
        grain_col : str
            High-frequency identifier column name.
        index_col : str
            Low-frequency identifier column name.
        verbose : bool
            Whether to enable informative logging.
        """
        # Store parameters
        self.conversion = conversion
        self.grain_col = grain_col
        self.index_col = index_col
        self.verbose = verbose

        # Initialize logger with verbosity control
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        # Validate conversion method upon initialization
        valid_methods = ["sum", "average", "first", "last"]
        if self.conversion not in valid_methods:
            raise ValueError(
                f"Invalid conversion method '{self.conversion}'. "
                f"Supported methods are: {valid_methods}."
            )

    def get_conversion_vector(self, size):
        """
        Generate a 1D conversion vector for a given group size.

        Parameters
        ----------
        size : int
            Number of high-frequency observations per low-frequency group.

        Returns
        -------
        numpy.ndarray
            Conversion vector of shape (size,).
        """
        # Validate size
        if size < 1:
            raise ValueError("Conversion vector size must be at least 1.")

        # Equal weights summing to the group total
        if self.conversion == "sum":
            return np.ones(size)

        # Equal weights summing to 1
        if self.conversion == "average":
            return np.ones(size) / size

        # First observation only
        if self.conversion == "first":
            vec = np.zeros(size)
            vec[0] = 1.0
            return vec

        # Last observation only
        if self.conversion == "last":
            vec = np.zeros(size)
            vec[-1] = 1.0
            return vec

        # Should never reach this point
        raise RuntimeError("Unexpected conversion method during vector generation.")

    def build(self, df):
        """
        Construct the conversion matrix from a DataFrame.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with both index and grain columns.

        Returns
        -------
        numpy.ndarray
            Conversion matrix of shape (n_low, n_high).
        """
        try:
            # Ensure required columns are present
            if self.index_col not in df.columns:
                raise KeyError(f"Column '{self.index_col}' not found in DataFrame.")
            if self.grain_col not in df.columns:
                raise KeyError(f"Column '{self.grain_col}' not found in DataFrame.")

            # Drop duplicates and sort (index_col, grain_col) combinations
            unique_combinations = (
                df[[self.index_col, self.grain_col]]
                .drop_duplicates()
                .sort_values([self.index_col, self.grain_col])
            )

            # Ensure no duplicates remain
            if unique_combinations.duplicated().any():
                raise ValueError("Duplicate (index, grain) combinations found. Check data integrity.")

            # Identify unique low-frequency groups
            unique_indexes = unique_combinations[self.index_col].unique()
            n_low = len(unique_indexes)
            n_high = len(df)

            # Validate minimum data requirement
            if n_low < 2:
                raise ValueError("At least two low-frequency groups are required to build the matrix.")

            # Log matrix dimensions
            if self.verbose:
                self.logger.info(f"Building conversion matrix: {n_low} x {n_high}")

            # Initialize empty conversion matrix
            C = np.zeros((n_low, n_high))

            # Fill matrix row-by-row based on index groupings
            for i, idx in enumerate(unique_indexes):
                # Boolean mask for rows matching current low-frequency group
                mask = (df[self.index_col] == idx).values

                # Apply conversion vector to selected columns
                C[i, mask] = self.get_conversion_vector(np.sum(mask))

            return C

        except Exception as e:
            raise RuntimeError(f"Failed to build conversion matrix: {str(e)}")