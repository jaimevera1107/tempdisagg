import pandas as pd

from tempdisagg.postprocessing.post_estimation import PostEstimation


class TempDisaggAdjuster:
    def adjust_output(self, df):
        if self.y_hat is None:
            raise RuntimeError("Model must be fitted before calling adjust_output().")

        adjuster = PostEstimation(self.conversion)
        df_copy = df.copy()
        df_copy["y_hat"] = self.y_hat.flatten()
        self.adjusted_ = adjuster.adjust_negative_values(df_copy)
        return self.adjusted_
