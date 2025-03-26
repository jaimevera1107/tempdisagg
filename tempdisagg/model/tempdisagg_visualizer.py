import matplotlib.pyplot as plt
import pandas as pd


class TempDisaggVisualizer:
    """
    Visualizer for disaggregated model results.

    Handles plotting for both single-method and ensemble outputs,
    and optionally includes adjusted predictions if available.
    """

    def plot(self, use_adjusted=False, title="Temporal Disaggregation Result",
         xlim=None, ylim=None, figsize=(12, 5)):
        """
        Plot disaggregated results versus observed low-frequency series.

        INPUT
        use_adjusted : bool
            Whether to include the adjusted prediction in the plot.
        title : str
            Plot title.
        xlim : tuple or None
            Limits for the x-axis.
        ylim : tuple or None
            Limits for the y-axis.
        figsize : tuple
            Size of the figure.

        OUTPUT
        None
        """
        # Check for ensemble and delegate if available
        if hasattr(self, "ensemble") and self.ensemble is not None:
            return self.ensemble.plot(self._df)

        if not hasattr(self, "y_hat") or self.y_hat is None:
            raise RuntimeError("Model must be fitted before plotting.")

        if not hasattr(self, "_df") or self._df is None:
            raise RuntimeError("Internal DataFrame (_df) not found.")

        df_plot = self._df.copy()
        df_plot["y_hat"] = self.y_hat.flatten()

        if use_adjusted:
            if not hasattr(self, "adjusted_") or self.adjusted_ is None:
                raise ValueError("No adjusted prediction found. Run `.adjust_output()` first.")
            df_plot["y_hat_adj"] = self.adjusted_.flatten()

        plt.figure(figsize=figsize)

        if "y" in df_plot.columns:
            plt.plot(df_plot.index, df_plot["y"], label="Low-frequency y (observed)", linestyle="--", marker="o")

        plt.plot(df_plot.index, df_plot["y_hat"], label="Disaggregated y_hat", linewidth=2)

        if use_adjusted:
            plt.plot(df_plot.index, df_plot["y_hat_adj"], label="Adjusted y_hat", linewidth=2)

        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Value")

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
