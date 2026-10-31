"""Residual structures for the regression-based (GLS) methods."""

from tempdisagg.covariance.factors import LowerTriangularFactor
from tempdisagg.covariance.structures import (
    AR1Covariance,
    CovarianceForm,
    DynamicStructure,
    RandomWalkAR1Covariance,
    ResidualStructure,
    WhiteNoiseCovariance,
)

__all__ = [
    "AR1Covariance",
    "CovarianceForm",
    "DynamicStructure",
    "LowerTriangularFactor",
    "RandomWalkAR1Covariance",
    "ResidualStructure",
    "WhiteNoiseCovariance",
]
