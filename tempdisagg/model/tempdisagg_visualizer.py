import matplotlib.pyplot as plt
import pandas as pd

class TempDisaggVisualizer:
    def plot(self, df, use_adjusted=False):
        if hasattr(self, "ensemble_") and self.ensemble_ is not None:
            return self.ensemble_.plot(df)

        if self.y_hat is None:
            raise RuntimeError("Model must be fitted before plotting.")

        df_plot = df.copy()
        df_plot["y_hat"] = self.y_hat.flatten()

        if use_adjusted:
            if self.adjusted_ is None or "y_hat_adj" not in self.adjusted_.columns:
                raise ValueError("No adjusted prediction found. Run .adjust_output(df) first.")
            df_plot["y_hat_adj"] = self.adjusted_["y_hat_adj"].values

        plt.figure(figsize=(12, 5))

        if "y" in df_plot.columns:
            plt.plot(df_plot.index, df_plot["y"], label="Low-freq y (observed)", linestyle="--", marker="o")

        plt.plot(df_plot.index, df_plot["y_hat"], label="Disaggregated y_hat", linewidth=2)

        if use_adjusted:
            plt.plot(df_plot.index, df_plot["y_hat_adj"], label="Adjusted y_hat", linewidth=2)

        plt.title("Temporal Disaggregation Result")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
