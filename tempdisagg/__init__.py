# tempdisagg/__init__.py

"""
tempdisagg: Temporal disaggregation of low-frequency time series using various statistical methods.
"""

# Public API
from .model.tempdisagg_model import TempDisaggModel

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


# Versioning and metadata
__version__ = "0.2.2"
__author__ = "Jaime Vera-Jaramillo"
__license__ = "MIT"