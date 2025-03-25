# tempdisagg/__init__.py

"""
tempdisagg: Temporal disaggregation of low-frequency time series using various statistical methods.
"""

# Versioning and metadata
__version__ = "0.1.8"
__author__ = "Jaime Vera-Jaramillo"
__license__ = "MIT"

# Public API
from .core.temp_disagg_model import TempDisaggModel
from .core.model_fitter import ModelFitter
from .core.ensemble_predictor import EnsemblePredictor
from .core.temp_disagg_visualizer import TempDisaggVisualizer
from .core.temp_disagg_reporter import TempDisaggReporter
from .core.temp_disagg_adjuster import TempDisaggAdjuster

from .preprocessing.disagg_input_preparer import DisaggInputPreparer
from .preprocessing.input_preprocessor import InputPreprocessor
from .preprocessing.time_series_completer import TimeSeriesCompleter
from .preprocessing.conversion_matrix_builder import ConversionMatrixBuilder

from .model.models_handler import ModelsHandler
from .model.rho_optimizer import RhoOptimizer
from .model.numeric_utils import NumericUtils

from .postprocessing.post_estimation import PostEstimation
from .postprocessing.ensemble_prediction import EnsemblePrediction

from .utils.temporal_aggregator import TemporalAggregator

# Define what gets imported with `from tempdisagg import *`
__all__ = [
    "TempDisaggModel",
    "ModelFitter",
    "EnsemblePredictor",
    "TempDisaggVisualizer",
    "TempDisaggReporter",
    "TempDisaggAdjuster",
    "DisaggInputPreparer",
    "InputPreprocessor",
    "TimeSeriesCompleter",
    "ConversionMatrixBuilder",
    "ModelsHandler",
    "RhoOptimizer",
    "NumericUtils",
    "PostEstimation",
    "EnsemblePrediction",
    "TemporalAggregator",
]
