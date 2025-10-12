"""Feature creation utilities for time series data."""

import pandas as pd
import numpy as np
import holidays


def is_holiday(df: pd.DataFrame, date_column: str, country: str = 'GB') -> pd.Series:
    """
    Return a `pandas.Series` indicating which dates are public holidays.

    Args:
        df: `pandas.DataFrame`
        date_column: Name of column containing dates
        country: Country code for holidays (default: 'GB')

    Returns:
        Binary series with 1.0 for holidays, 0.0 otherwise
    """
    country_holidays = holidays.country_holidays(country)
    return df[date_column].apply(lambda x: 1.0 if x in country_holidays else 0.0)


def is_weekend(df: pd.DataFrame, date_column: str) -> pd.Series:
    """
    Return a `pandas.Series` indicating which dates are weekends.

    Args:
        df: `pandas.DataFrame`
        date_column: Name of column containing dates

    Returns:
        Binary series with 1.0 for weekends, 0.0 for weekdays
    """
    return df[date_column].dt.dayofweek.isin([5, 6]).astype(float)


def add_intercept(df: pd.DataFrame, column_name='intercept', inplace=False) -> pd.DataFrame:
    """
    Add a column of ones to a dataframe (useful for linear regression with an intercept).

    Args:
        df: `pandas.DataFrame`
        column_name: Name of new column (default: 'intercept')
        inplace: If True, modifies dataframe in place. If False, returns a copy.

    Returns:
        DataFrame with intercept column added
    """
    if not inplace:
        df = df.copy()
    df[column_name] = np.ones(len(df))
    return df
