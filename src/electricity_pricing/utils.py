"""Feature creation utilities for time series data."""

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import holidays
from datetime import date
from typing import Tuple, Optional


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


def last_sunday_of_month(year: int, month: int) -> date:
    """
    Return the last Sunday of a given month.

    Args:
        year (int): Year
        month (int): Month
    """
    last_day = pd.Timestamp(year=year, month=month, day=1) + pd.offsets.MonthEnd(1)
    days_back = (last_day.dayofweek - 6) % 7  # Sunday is 6
    last_sunday = last_day - pd.Timedelta(days=days_back)
    return last_sunday.date()


def get_expected_periods(date: pd.Timestamp) -> int:
    """
    Return the expected number of 30-minute settlement periods for a given date in the UK.
    """
    year = date.year
    spring_dst = last_sunday_of_month(year, 3)
    autumn_dst = last_sunday_of_month(year, 10)
    if date.date() == spring_dst:
        return 46
    elif date.date() == autumn_dst:
        return 50
    else:
        return 48
    

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


def timeshift(series: pd.Series, shift: pd.Timedelta, name: Optional[str] = None) -> pd.Series:
    """
    Shift a timestamp-indexed series by an amount of time `shift`.

    This function handles daylight savings transitions correctly by using
    timestamp-based lookups rather than positional shifting.

    Args:
        series: `pandas.Series` with timestamps as the index
        shift: Time shift as a `pandas.Timedelta`
        name: Optional name for the returned series

    Returns:
        shifted_series: Series shifted by the specified timedelta

    Example:
        >>> # Shift price data back by 1 day
        >>> lagged_price = shift_series(df['price'], pd.Timedelta(days=-1), 'price_lag_1d')
    """
    assert np.all(ptypes.is_datetime64_any_dtype(series.index)), "Expected datetime type as index."
    inds = series.index
    inds_shifted = inds + shift
    lookup = series.to_dict()
    series_shifted = pd.Series([lookup.get(i, np.nan) for i in inds_shifted], index=inds, name=name)
    return series_shifted


def train_test_split(
    df: pd.DataFrame | pd.Series,
    train_range: pd.DatetimeIndex,
    test_range: pd.DatetimeIndex
) -> Tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """
    Split a time-indexed dataset into training and test sets.

    Args:
        df: `pandas.DataFrame` or `pandas.Series` with timestamps as indices
        train_range: DatetimeIndex specifying training timestamps (from pd.date_range)
        test_range: DatetimeIndex specifying test timestamps (from pd.date_range)

    Returns:
        df_train: Training set
        df_test: Test set

    Example:
        >>> train_range = pd.date_range("2021-01-01", "2024-01-02",
        ...                             freq="30min", tz="Europe/London", inclusive="left")
        >>> test_range = pd.date_range("2024-01-02", "2024-01-03",
        ...                            freq="30min", tz="Europe/London", inclusive="left")
        >>> X_train, X_test = train_test_split(features, train_range, test_range)
    """
    assert np.all(ptypes.is_datetime64_any_dtype(df.index)), "Expected datetime type as index."
    df_train = df[df.index.isin(train_range)].copy()
    df_test = df[df.index.isin(test_range)].copy()
    return df_train, df_test