"""
tempdisagg: Temporal disaggregation of low-frequency time series using various statistical methods.
"""

# Public API
from .model.tempdisagg_model import TempDisaggModel
from .utils.retropolarizer import Retropolarizer
from .core.temporal_aggregator import TemporalAggregator

# Define what gets imported with `from tempdisagg import *`
__all__ = [
    "ConversionMatrixBuilder",
    "DisaggInputPreparer",
    "EnsemblePrediction",
    "EnsemblePredictor",
    "InputPreprocessor",
    "ModelsHandler",
    "ModelFitter",
    "NumericUtils",
    "PostEstimation",
    "Retropolarizer",
    "RhoOptimizer",
    "TempDisaggAdjuster",
    "TempDisaggModel",
    "TempDisaggReporter",
    "TempDisaggVisualizer",
    "TemporalAggregator",
    "TimeSeriesCompleter",
]

__version__ = "0.2.7"
