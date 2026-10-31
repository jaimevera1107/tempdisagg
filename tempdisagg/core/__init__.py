"""Core building blocks: aggregation, data alignment and linear algebra."""

from tempdisagg.core.aggregation import (
    Conversion,
    build_aggregation_matrix,
    temporal_aggregate,
)
from tempdisagg.core.data import DataAligner, DisaggregationData

__all__ = [
    "Conversion",
    "DataAligner",
    "DisaggregationData",
    "build_aggregation_matrix",
    "temporal_aggregate",
]
