"""Empirical evaluation of disaggregation models.

``backtest``             out-of-sample extrapolation of the low-frequency aggregate
``revision_analysis``    revisions caused by the arrival of new periods
``conformity``           co-movement of the estimate with its indicators
``common_factor_test``   Chow-Lin restriction inside the ADL(1,1) model
"""

from tempdisagg.evaluation.backtest import BacktestResult, Candidate, backtest
from tempdisagg.evaluation.conformity import conformity
from tempdisagg.evaluation.revisions import RevisionResult, revision_analysis
from tempdisagg.evaluation.specification import CommonFactorTest, common_factor_test

__all__ = [
    "BacktestResult",
    "Candidate",
    "CommonFactorTest",
    "RevisionResult",
    "backtest",
    "common_factor_test",
    "conformity",
    "revision_analysis",
]
