"""Electricity pricing utilities package."""

__version__ = "0.1.0"

# Import commonly used functions for convenience
from .datatools import (
    create_timestamps,
    expected_periods,
    last_sunday_of_month,
    validate_timestamps,
    timestamp_to_settlement,
)

from .evaluation import (
    rmse,
    relative_rmse,
    mae,
    mape,
    r2_score,
)

from .features import (
    is_holiday,
    is_weekend,
)

__all__ = [
    # Processing utilities
    "create_timestamps",
    "expected_periods",
    "last_sunday_of_month",
    "validate_timestamps",
    "timestamp_to_settlement",
    # Evaluation metrics
    "rmse",
    "relative_rmse",
    "mae",
    "mape",
    "r2_score",
    # Feature utilities
    "is_holiday",
    "is_weekend",
]
