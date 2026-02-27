"""Feature engineering utilities for time series forecasting."""

import numpy as np
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


def create_temporal_features(
    df: pd.DataFrame,
    date_column: str = "SETTLEMENT_DATE",
    period_column: str = "SETTLEMENT_PERIOD",
) -> pd.DataFrame:
    """
    Add temporal features to a DataFrame.

    Creates period, day-of-week, and month harmonics plus holiday/weekend
    indicators. Returns a copy with new columns added.

    Args:
        df: DataFrame with date and period columns
        date_column: Name of settlement date column
        period_column: Name of settlement period column

    Returns:
        DataFrame with added columns: period_sin, period_cos,
        dayofweek_sin, dayofweek_cos, month_sin, month_cos,
        is_holiday, is_weekend
    """
    out = df.copy()

    sp = out[period_column]
    out["period_sin"] = np.sin(4 * np.pi * (sp - 1) / 48)
    out["period_cos"] = np.cos(4 * np.pi * (sp - 1) / 48)

    dow = out[date_column].dt.dayofweek
    out["dayofweek_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dayofweek_cos"] = np.cos(2 * np.pi * dow / 7)

    month = out[date_column].dt.month
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12)

    out["is_holiday"] = is_holiday(out, date_column, country="GB")
    out["is_weekend"] = is_weekend(out, date_column)

    return out