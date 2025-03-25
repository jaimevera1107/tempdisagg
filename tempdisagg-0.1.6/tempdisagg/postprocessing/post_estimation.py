import numpy as np
import pandas as pd
import warnings

from scipy.optimize import minimize

from tempdisagg.utils.logging_utils import VerboseLogger

class PostEstimation:
    """
    Post-estimation adjustment utility to correct negative values in predicted series
    after temporal disaggregation, ensuring consistency with the chosen aggregation rule.

    Supports the following aggregation types: 'sum', 'average', 'first', and 'last'.

    Parameters
    ----------
    conversion : str
        Aggregation rule used during disaggregation ('sum', 'average', 'first', 'last').
    index_col : str, default="Index"
        Name of the column indicating the low-frequency index or group.
    y_hat_name : str, default="y_hat"
        Name of the column containing predicted high-frequency values.
    verbose : bool, default=False
        Whether to enable informative logging messages.
    """

    def __init__(self, conversion, index_col="Index", y_hat_name="y_hat", verbose=False):
        self.conversion = conversion
        self.index_col = index_col
        self.y_hat_name = y_hat_name
        self.verbose = verbose

        # Initialize logger with verbosity control
        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=self.verbose).get_logger()

        # Validate conversion method
        allowed_conversions = ["sum", "average", "first", "last"]
        if self.conversion not in allowed_conversions:
            raise ValueError(
                f"Invalid conversion method '{self.conversion}'. "
                f"Must be one of {allowed_conversions}."
            )
        
    def _adjust_average_optimized(self, y_original):
        """
        Adjust a 1D array to be non-negative while preserving the mean,
        minimizing the squared deviation from original values.

        Parameters
        ----------
        y_original : np.ndarray
            Original series with possibly negative values.

        Returns
        -------
        np.ndarray
            Adjusted non-negative series with same mean.
        """
        y_original = np.asarray(y_original, dtype=float).flatten()
        n = len(y_original)
        target_sum = y_original.mean() * n

        # Objective: minimize squared deviation
        def objective(y):
            return np.sum((y - y_original) ** 2)

        # Constraint: sum(y) == sum(y_original)
        constraints = {'type': 'eq', 'fun': lambda y: np.sum(y) - target_sum}

        # Bounds: y_i >= 0
        bounds = [(0, None) for _ in range(n)]

        # Initial guess: clip negatives
        x0 = np.clip(y_original, 0, None)

        # Run optimization
        result = minimize(objective, x0, bounds=bounds, constraints=constraints, method='SLSQP')

        if not result.success:
            raise RuntimeError(f"Optimization failed: {result.message}")

        return result.x

    def adjust_negative_values(self, df):
        """
        Adjust negative predicted values in the disaggregated series,
        preserving the aggregate totals according to the conversion method.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing predicted values and low-frequency group identifiers.

        Returns
        -------
        pd.DataFrame
            Copy of the original DataFrame with adjusted predictions.

        Raises
        ------
        TypeError
            If input is not a pandas DataFrame.
        ValueError
            If required columns are missing or the DataFrame has fewer than 3 observations.
        """
        # Validate input type
        if not isinstance(df, pd.DataFrame):
            raise TypeError("Input must be a pandas DataFrame.")

        # Validate presence of required columns
        required_cols = [self.index_col, self.y_hat_name]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column '{col}' in input DataFrame.")

        # Validate sufficient data size
        if df.shape[0] < 3:
            raise ValueError("Input DataFrame must contain at least 3 observations.")

        # Check for NaNs in critical column
        if df[self.y_hat_name].isnull().any():
            raise ValueError(f"Column '{self.y_hat_name}' contains missing values (NaN).")

        # Make a copy to preserve original DataFrame
        df_adjusted = df.copy()

        # Identify groups with negative predictions
        negative_indexes = df_adjusted[df_adjusted[self.y_hat_name] < 0][self.index_col].unique()

        # Log if no adjustment needed
        if len(negative_indexes) == 0:
            self.logger.info("No negative predictions found. No adjustment needed.")
            return df_adjusted

        self.logger.info(f"Adjusting negative predictions for {len(negative_indexes)} group(s)...")

        # Iterate over each group needing adjustment
        for index in negative_indexes:
            try:
                group_mask = df_adjusted[self.index_col] == index
                group = df_adjusted[group_mask].reset_index(drop=True)

                # Extract predicted values as float array
                y_hat = group[self.y_hat_name].astype(float).values

                # Skip if no negatives remain
                if (y_hat >= 0).all():
                    continue

                # --- SUM CONVERSION: redistribute negatives proportionally across positive values
                if self.conversion == "sum":
                    negative_sum = np.abs(y_hat[y_hat < 0].sum())
                    positive_values = y_hat[y_hat > 0]
                    positive_sum = positive_values.sum()

                    if positive_sum > 0:
                        weights = positive_values / positive_sum
                        y_hat[y_hat > 0] -= negative_sum * weights
                        y_hat[y_hat < 0] = 0
                    else:
                        y_hat[:] = negative_sum / len(y_hat)

                # --- AVERAGE_OPTIMIZED: preserve mean and minimize deviation
                elif self.conversion == "average":
                    try:
                        y_hat = self._adjust_average_optimized(y_hat)
                    except Exception as e:
                        warnings.warn(
                            f"Optimization failed for group '{index}': {str(e)}. Falling back to zeroing negatives."
                        )
                        y_hat[y_hat < 0] = 0

                # --- FIRST CONVERSION: preserve or adjust the first value, correct remaining
                elif self.conversion == "first":
                    first_value = y_hat[0]
                    remaining_values = y_hat[1:]

                    if first_value < 0:
                        negative_sum = abs(first_value)
                        first_value = 0
                        if remaining_values.sum() > 0:
                            weights = remaining_values / remaining_values.sum()
                            remaining_values -= negative_sum * weights
                        else:
                            remaining_values[:] = negative_sum / len(remaining_values)
                    else:
                        if remaining_values.sum() < 0:
                            remaining_values[:] = 0
                        else:
                            negative_sum = np.abs(remaining_values[remaining_values < 0].sum())
                            positive_values = remaining_values[remaining_values > 0]
                            positive_sum = positive_values.sum()

                            if positive_sum > 0:
                                weights = positive_values / positive_sum
                                remaining_values[remaining_values > 0] -= negative_sum * weights

                            remaining_values[remaining_values < 0] = 0

                    y_hat[0] = first_value
                    y_hat[1:] = remaining_values


                # --- LAST CONVERSION: preserve or adjust the last value, correct previous
                elif self.conversion == "last":
                    last_value = y_hat[-1]
                    remaining_values = y_hat[:-1]

                    if last_value < 0:
                        negative_sum = abs(last_value)
                        last_value = 0
                        if remaining_values.sum() > 0:
                            weights = remaining_values / remaining_values.sum()
                            remaining_values -= negative_sum * weights
                        else:
                            remaining_values[:] = negative_sum / len(remaining_values)
                    else:
                        if remaining_values.sum() < 0:
                            remaining_values[:] = 0
                        else:
                            negative_sum = np.abs(remaining_values[remaining_values < 0].sum())
                            positive_values = remaining_values[remaining_values > 0]
                            positive_sum = positive_values.sum()

                            if positive_sum > 0:
                                weights = positive_values / positive_sum
                                remaining_values[remaining_values > 0] -= negative_sum * weights

                            remaining_values[remaining_values < 0] = 0

                    y_hat[:-1] = remaining_values
                    y_hat[-1] = last_value


                # Replace back into DataFrame
                df_adjusted.loc[group_mask, self.y_hat_name] = y_hat

            except Exception as e:
                warnings.warn(
                    f"Adjustment failed for index group '{index}': {str(e)}. "
                    f"Original values retained."
                )

        return df_adjusted