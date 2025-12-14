"""Feature engineering utilities for time series forecasting."""

import pandas as pd
import holidays


def is_holiday(
        df: pd.DataFrame,
        date_column: str,
        country: str = "GB"
    ) -> pd.Series:
    """
    Create a binary indicator for public holidays.

    Args:
        df: DataFrame with date column
        date_column: Name of date column
        country: Country code for holidays (default: 'GB')

    Returns:
        Boolean Series indicating holidays
    """
    country_holidays = holidays.country_holidays(country)
    return df[date_column].apply(lambda x: x in country_holidays).astype(float)


def is_weekend(df: pd.DataFrame, date_column: str) -> pd.Series:
    """
    Create a binary indicator for weekends (Saturday and Sunday).

    Args:
        df: DataFrame with date column
        date_column: Name of date column

    Returns:
        Boolean Series indicating weekends
    """
    dow = df[date_column].dt.dayofweek  # type: ignore[attr-defined]
    return dow.isin([5, 6]).astype(float)
