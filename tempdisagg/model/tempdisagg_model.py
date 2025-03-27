from tempdisagg.model.tempdisagg_fitter import ModelFitter
from tempdisagg.model.tempdisagg_ensemble import EnsemblePredictor
from tempdisagg.postprocessing.post_estimation import PostEstimation
from tempdisagg.utils.logging_utils import VerboseLogger
from tempdisagg.model.tempdisagg_visualizer import TempDisaggVisualizer
from tempdisagg.model.tempdisagg_summary import TempDisaggReporter
from tempdisagg.model.tempdisagg_adjuster import TempDisaggAdjuster
from tempdisagg.model.tempdisagg_core import TempDisaggModelCore


class TempDisaggModel:
    """
    High-level API for temporal disaggregation models.
    Supports single-method and ensemble fitting, prediction, adjustment, plotting, and reporting.
    """

    def __init__(
        self,
        method="chow-lin",
        conversion="sum",
        grain_col="Grain",
        index_col="Index",
        y_col="y",
        X_col="X",
        interpolation_method="linear",
        rho_min=-0.9,
        rho_max=0.99,
        fallback_method="fast",
        verbose=False,
        use_retropolarizer=False,
        retro_method="linear_regression",
        retro_aux_col=None
    ):
        """
        Initialize the disaggregation interface.

        INPUT
        method : str
            Disaggregation method to use.
        conversion : str
            Aggregation rule.
        grain_col, index_col, y_col, X_col : str
            Column names for time, group, target, and regressors.
        interpolation_method : str
            Imputation method.
        rho_min, rho_max : float
            Bounds for autocorrelation optimization.
        fallback_method : str
            Backup method if main estimation fails.
        verbose : bool
            Enables logging messages.
        use_retropolarizer : bool
            Whether to use Retropolarizer instead of standard interpolation for y_col.
        retro_method : str
            Method used by Retropolarizer (e.g. 'proportion', 'linear_regression', 'polynomial_regression', 'exponential_smoothing', 'mlp_regression').
        retro_aux_col : str or None
            Auxiliary column to use as predictor for retropolarization. If None, uses X_col.

        OUTPUT
        None
        """
        self.method = method
        self.conversion = conversion
        self.verbose = verbose
        self.is_ensemble = method == "ensemble"

        self.logger = VerboseLogger(f"{__name__}.{id(self)}", verbose=verbose).get_logger()

        self._y_hat = None
        self._adjusted = None
        self._padding_info = None
        self._df = None

        self.grain_col = grain_col
        self.index_col = index_col
        self.y_col = y_col
        self.X_col = X_col
        self.interpolation_method = interpolation_method
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.fallback_method = fallback_method

        self.fitter = ModelFitter(
            conversion=conversion,
            grain_col=grain_col,
            index_col=index_col,
            y_col=y_col,
            X_col=X_col,
            interpolation_method=interpolation_method,
            rho_min=rho_min,
            rho_max=rho_max,
            fallback_method=fallback_method,
            verbose=verbose,
            use_retropolarizer=use_retropolarizer,
            retro_method=retro_method,
            retro_aux_col=retro_aux_col
        )

        self.ensemble_predictor = EnsemblePredictor(
            conversion=conversion,
            grain_col=grain_col,
            index_col=index_col,
            y_col=y_col,
            X_col=X_col,
            interpolation_method=interpolation_method,
            rho_min=rho_min,
            rho_max=rho_max,
            fallback_method=fallback_method,
            verbose=verbose,
            use_retropolarizer=use_retropolarizer,
            retro_method=retro_method,
            retro_aux_col=retro_aux_col
        ) if self.is_ensemble else None

        self.adjuster = PostEstimation(conversion=conversion)

    def fit(self, df, methods=None):
        if self.is_ensemble:
            self._y_hat, self._padding_info, self._df = self.ensemble_predictor.fit(df, methods)
            self.ensemble = self.ensemble_predictor.ensemble
            self.results_ = self.ensemble_predictor.results_
        else:
            if methods is not None:
                raise ValueError("'methods' is only valid when method='ensemble'")
            self._y_hat, self._padding_info, self._df = self.fitter.fit(df, method=self.method)
            self.results_ = self.fitter.result_

        self.y_hat = self._y_hat
        self.adjusted_ = None
        self.logger.info("Model fitting complete.")
        return self

    def predict(self, full=True):
        if self._y_hat is None:
            raise RuntimeError("Model must be fitted before calling `predict()`.")
        return self._truncate(self._y_hat, full)

    def adjust_output(self, full=True):
        if self._y_hat is None:
            raise RuntimeError("Model must be fitted before calling `adjust_output()`.")

        df = self._df.copy()
        df["y_hat"] = self._y_hat.flatten()

        adjusted_df = self.adjuster.adjust_negative_values(df)
        self._adjusted = adjusted_df["y_hat"].to_numpy().reshape(-1, 1)
        return self._truncate(self._adjusted, full)

    def plot(self, use_adjusted=False, **kwargs):
        if self._df is None:
            raise RuntimeError("Model must be fitted before plotting.")

        if self.is_ensemble:
            return self.ensemble.plot(df=self._df)

        return TempDisaggVisualizer.plot(self, use_adjusted=use_adjusted, **kwargs)

    def summary(self, metric="mae"):
        return TempDisaggReporter.summary(self, metric=metric)

    def summary_compact(self):
        return TempDisaggReporter.summary_compact(self)

    def get_params(self, deep=True):
        return {
            "method": self.method,
            "conversion": self.conversion,
            "grain_col": self.grain_col,
            "index_col": self.index_col,
            "y_col": self.y_col,
            "X_col": self.X_col,
            "interpolation_method": self.interpolation_method,
            "rho_min": self.rho_min,
            "rho_max": self.rho_max,
            "fallback_method": self.fallback_method,
            "verbose": self.verbose
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self

    def _truncate(self, arr, full):
        if full or self._padding_info is None:
            return arr

        n_before = self._padding_info.get("n_pad_before", 0)
        n_after = self._padding_info.get("n_pad_after", 0)

        return arr[n_before: -n_after if n_after else None]
