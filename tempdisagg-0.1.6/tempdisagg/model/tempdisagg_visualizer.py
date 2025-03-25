import matplotlib.pyplot as plt
import pandas as pd


class TempDisaggVisualizer:
    """
    Visualizer for disaggregated model results.

    Handles plotting for both single-method and ensemble outputs,
    and optionally includes adjusted predictions if available.
    """

    def plot(self, use_adjusted=False):
        """
        Plot disaggregated results versus observed low-frequency series.

        INPUT
        use_adjusted : bool
            Whether to include the adjusted prediction in the plot.

        OUTPUT
        None
        """
        # Check for ensemble and delegate if available
        if hasattr(self, "ensemble") and self.ensemble is not None:
            return self.ensemble.plot(self._df)

        # Validate presence of prediction
        if not hasattr(self, "y_hat") or self.y_hat is None:
            raise RuntimeError("Model must be fitted before plotting.")

        # Validate internal DataFrame
        if not hasattr(self, "_df") or self._df is None:
            raise RuntimeError("Internal DataFrame (_df) not found.")

        # Copy data for plotting
        df_plot = self._df.copy()

        # Assign disaggregated prediction
        df_plot["y_hat"] = self.y_hat.flatten()

        # Assign adjusted prediction if requested and available
        if use_adjusted:
            if not hasattr(self, "adjusted_") or self.adjusted_ is None:
                raise ValueError("No adjusted prediction found. Run `.adjust_output()` first.")
            df_plot["y_hat_adj"] = self.adjusted_.flatten()

        # Create figure
        plt.figure(figsize=(12, 5))

        # Plot low-frequency observed series if present
        if "y" in df_plot.columns:
            plt.plot(df_plot.index, df_plot["y"], label="Low-frequency y (observed)", linestyle="--", marker="o")

        # Plot disaggregated prediction
        plt.plot(df_plot.index, df_plot["y_hat"], label="Disaggregated y_hat", linewidth=2)

        # Plot adjusted version if available
        if use_adjusted:
            plt.plot(df_plot.index, df_plot["y_hat_adj"], label="Adjusted y_hat", linewidth=2)

        # Final plot formatting
        plt.title("Temporal Disaggregation Result")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
