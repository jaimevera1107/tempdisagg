"""tempdisagg: temporal disaggregation of time series.

Methods follow the R package ``tempdisagg`` (Sax and Steiner, 2013):
Chow-Lin, Fernandez, Litterman, dynamic Chow-Lin, Denton, Denton-Cholette,
uniform and OLS, with any number of high-frequency indicators. Every method
is configurable through its specification class (:mod:`tempdisagg.methods`).

Quick use::

    from tempdisagg import disaggregate, TemporalDisaggregator
    series = disaggregate(y_annual, indicators)                 # automation
    result = TemporalDisaggregator("fernandez", include_intercept=False).fit(
        y_annual, indicators)                                    # analysis
"""

from tempdisagg.core.aggregation import Conversion, temporal_aggregate
from tempdisagg.core.constraints import LinearConstraint, aggregate_constraints, known_values
from tempdisagg.core.data import DataAligner, DisaggregationData
from tempdisagg.estimation.rho import RhoOptimizerSettings
from tempdisagg.evaluation.backtest import BacktestResult, Candidate, backtest
from tempdisagg.exceptions import (
    ArtificialDataWarning,
    ExactFitWarning,
    ExperimentalMethodWarning,
    IllConditionedWarning,
    IndicatorSignWarning,
    IneffectiveParameterWarning,
    InfeasibleConstraintError,
    InputValidationError,
    MissingTargetWarning,
    MultipleOptimaWarning,
    NumericalError,
    RedundantConstraintWarning,
    RhoBoundaryWarning,
    RhoNotIdentifiedWarning,
    SingularDesignError,
    TempDisaggError,
    TempDisaggWarning,
)
from tempdisagg.indicators import impulse, level_shift
from tempdisagg.legacy import Retropolarizer, TempDisaggModel
from tempdisagg.methods import (
    OLS,
    ChowLin,
    Denton,
    DynamicChowLin,
    Fernandez,
    Litterman,
    Uniform,
    available_methods,
    make_method,
)
from tempdisagg.model import TemporalDisaggregator, disaggregate
from tempdisagg.preprocessing.settings import CompletionSettings, PaddingSettings
from tempdisagg.ratios import RatioResult, disaggregate_ratio
from tempdisagg.reconciliation.cross_section import (
    CrossSectionalReconciler,
    CrossSectionalResult,
)
from tempdisagg.reconciliation.non_negative import NonNegativeReconciler
from tempdisagg.results import DisaggregationResult
from tempdisagg.splicing.series_splicer import SeriesSplicer, SplicingMethod, SplicingResult

__version__ = "0.6.2"

__all__ = [
    "OLS",
    "ArtificialDataWarning",
    "BacktestResult",
    "Candidate",
    "ChowLin",
    "CompletionSettings",
    "CrossSectionalReconciler",
    "CrossSectionalResult",
    "Conversion",
    "DataAligner",
    "Denton",
    "DisaggregationData",
    "DisaggregationResult",
    "DynamicChowLin",
    "ExactFitWarning",
    "ExperimentalMethodWarning",
    "Fernandez",
    "IllConditionedWarning",
    "IndicatorSignWarning",
    "IneffectiveParameterWarning",
    "InfeasibleConstraintError",
    "InputValidationError",
    "LinearConstraint",
    "Litterman",
    "MissingTargetWarning",
    "MultipleOptimaWarning",
    "NonNegativeReconciler",
    "NumericalError",
    "PaddingSettings",
    "RatioResult",
    "RedundantConstraintWarning",
    "Retropolarizer",
    "RhoBoundaryWarning",
    "RhoNotIdentifiedWarning",
    "RhoOptimizerSettings",
    "SeriesSplicer",
    "SingularDesignError",
    "SplicingMethod",
    "SplicingResult",
    "TempDisaggError",
    "TempDisaggModel",
    "TempDisaggWarning",
    "TemporalDisaggregator",
    "Uniform",
    "aggregate_constraints",
    "available_methods",
    "backtest",
    "disaggregate",
    "disaggregate_ratio",
    "impulse",
    "known_values",
    "level_shift",
    "make_method",
    "temporal_aggregate",
]
