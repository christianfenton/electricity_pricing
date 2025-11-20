"""Feature creation utilities for time series data."""

import numpy as np
import pandas as pd
import pandas.api.types as ptypes
import holidays
from datetime import date
from typing import Tuple, Optional


def is_holiday(
    df: pd.DataFrame, date_column: str, country: str = "GB"
) -> pd.Series:
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
    return df[date_column].apply(
        lambda x: 1.0 if x in country_holidays else 0.0
    )


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
    last_day = pd.Timestamp(
        year=year, month=month, day=1
    ) + pd.offsets.MonthEnd(1)
    days_back = (last_day.dayofweek - 6) % 7  # Sunday is 6
    last_sunday = last_day - pd.Timedelta(days=days_back)
    return last_sunday.date()


def get_expected_periods(date: pd.Timestamp) -> int:
    """
    Return the expected number of 30-minute settlement periods 
    for a given date in the UK.
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


def add_intercept(
    df: pd.DataFrame, column_name="intercept", inplace=False
) -> pd.DataFrame:
    """
    Add a column of ones to a dataframe

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


def timeshift(
    series: pd.Series, shift: pd.Timedelta, name: Optional[str] = None
) -> pd.Series:
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
    """
    assert np.all(ptypes.is_datetime64_any_dtype(series.index)), (
        "Expected datetime type as index."
    )
    inds = series.index
    inds_shifted = inds + shift
    lookup = series.to_dict()
    series_shifted = pd.Series(
        [lookup.get(i, np.nan) for i in inds_shifted], index=inds, name=name
    )
    return series_shifted


def create_timestamps(
    df: pd.DataFrame,
    date_column: str,
    period_column: str,
    tz: str = "Europe/London",
) -> pd.Series:
    """
    Create timezone-aware timestamps from settlement dates and periods.

    Args:
        df: `pandas.DataFrame` with settlement date and period columns
        date_column: Name of settlement date column
        period_column: Name of settlement period column
        tz: Timezone for timestamps (default: 'Europe/London')

    Returns:
        `pandas.Series`
    """
    base_timestamps = pd.to_datetime(df[date_column]).dt.tz_localize(tz)
    period_offsets = pd.to_timedelta((df[period_column] - 1) * 30, unit="m")
    timestamps = base_timestamps + period_offsets
    return pd.Series(timestamps, index=df.index, name="DATETIME")


def validate_timestamps(
    df: pd.DataFrame,
    datetime_column: str,
    date_column: str,
    period_column: str,
    tz: str = "Europe/London",
) -> bool:
    """
    Validate that timestamps match the (date, period) representation.

    Args:
        df: DataFrame with both timestamps and (date, period) columns
        datetime_column: Name of timestamp column to validate
        date_column: Name of settlement date column
        period_column: Name of settlement period column
        tz: Timezone to use for reconstruction
    """
    reconstructed = create_timestamps(
        df[[date_column, period_column]],
        date_column=date_column,
        period_column=period_column,
        tz=tz,
    )

    if datetime_column in df.columns:
        existing = df[datetime_column]
        mismatches = existing != reconstructed

        if mismatches.any():
            n_mismatches = mismatches.sum()
            first_mismatch_idx = mismatches.idxmax()
            raise ValueError(
                f"Timestamp inconsistency detected: {n_mismatches} mismatches found.\n"
                f"First mismatch at index {first_mismatch_idx}:\n"
                f"  Existing:      {existing.loc[first_mismatch_idx]}\n"
                f"  Reconstructed: {reconstructed.loc[first_mismatch_idx]}\n"
                f"  Date:          {df.loc[first_mismatch_idx, date_column]}\n"
                f"  Period:        {df.loc[first_mismatch_idx, period_column]}"
            )

    return True


def train_test_split(
    df: pd.DataFrame | pd.Series,
    train_range: pd.DatetimeIndex,
    test_range: pd.DatetimeIndex,
) -> Tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """
    Split a time-indexed dataset into training and test sets.

    Args:
        df: `pandas.DataFrame` or `pandas.Series` with timestamps as indices
        train_range: DatetimeIndex specifying training timestamps.
            Create this using `pandas.date_range`.
        test_range: DatetimeIndex specifying test timestamps.
            Create this using `pandas.date_range`.

    Returns:
        df_train: Training set
        df_test: Test set
    """
    assert np.all(ptypes.is_datetime64_any_dtype(df.index)), (
        "Expected datetime type as index."
    )
    df_train = df[df.index.isin(train_range)].copy()
    df_test = df[df.index.isin(test_range)].copy()
    return df_train, df_test
