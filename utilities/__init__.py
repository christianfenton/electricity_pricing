"""Electricity pricing utilities package."""

__version__ = "0.1.0"

from .datatools import (
    create_timestamps,
    expected_periods,
    periods_in_date_range,
    periods_remaining,
    last_sunday_of_month,
    validate_timestamps,
    timestamp_to_settlement,
)

from .bootstrap import (
    cond_sieve_bootstrap,
    get_bootstrap_percentiles,
)

from .forecast import rolling_dayahead_forecast, forecast_dayahead

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
    "periods_in_date_range",
    "periods_remaining",
    "last_sunday_of_month",
    "validate_timestamps",
    "timestamp_to_settlement",

    # Bootstrapping
    "cond_sieve_bootstrap",
    "get_bootstrap_percentiles",

    # Forecasting
    "forecast_dayahead",
    "rolling_dayahead_forecast",

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
