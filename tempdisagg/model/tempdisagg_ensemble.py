import numpy as np

from tempdisagg.postprocessing.ensemble_prediction import EnsemblePrediction


class TempDisaggEnsemble:
    def ensemble(self, df, methods=None):
        if self.y_l is None or self.X is None or self.C is None:
            self.y_l, self.X, self.C, self.padding_info = self.base.prepare(df)
            self.n_pad_before = self.padding_info.get("n_pad_before", 0)
            self.n_pad_after = self.padding_info.get("n_pad_after", 0)
            self.df_ = self.base.df_full if hasattr(self.base, "df_full") else df


        if methods is None:
            methods = list(self.all_methods.keys())

        self.ensemble_ = EnsemblePrediction(
            model_class=self.__class__,
            conversion=self.conversion,
            methods=methods,
            verbose=self.verbose
        )
        self.y_hat = self.ensemble_.run(df, self.y_l, self.C).reshape(-1, 1)

        self.weights_ = self.ensemble_.weights
        self.ensemble_methods_ = list(self.ensemble_.predictions.keys())
        self.models = self.ensemble_.models
        self.results_ = {
            name: {
                "beta": m.beta,
                "X": m.X,
                "rho": m.rho,
                "residuals": m.residuals,
                "weight": self.weights_[i]
            }
            for i, (name, m) in enumerate(self.models.items())
        }

        return self.y_hat
