"""Data utilities for electricity price forecasting."""

import numpy as np
import pandas as pd
from typing import Tuple, List


def timeshift(series: pd.Series, shift: pd.Timedelta, name: str | None = None) -> pd.Series:
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
    inds = series.index
    inds_shifted = inds + shift
    lookup = series.to_dict()
    series_shifted = pd.Series([lookup.get(i, np.nan) for i in inds_shifted], index=inds, name=name)
    return series_shifted



def train_test_split(
    df: pd.DataFrame | pd.Series,
    train_range: Tuple[pd.Timestamp, pd.Timestamp],
    test_range: Tuple[pd.Timestamp, pd.Timestamp]
) -> Tuple[pd.DataFrame | pd.Series, pd.DataFrame | pd.Series]:
    """
    Split a time-indexed dataset into training and test sets.

    Args:
        df: `pandas.DataFrame` or `pandas.Series` with timestamps as indices
        train_range: Tuple of (start_date, end_date) for training set
        test_range: Tuple of (start_date, end_date) for test set

    Returns:
        df_train: Training set with reset indices
        df_test: Test set with reset indices reset

    Example:
        >>> train_range = (pd.Timestamp("2021-01-01", tz="Europe/London"),
        ...                pd.Timestamp("2024-01-01", tz="Europe/London"))
        >>> test_range = (pd.Timestamp("2024-01-02", tz="Europe/London"),
        ...               pd.Timestamp("2024-12-01", tz="Europe/London"))
        >>> X_train, X_test = train_test_split(features, train_range, test_range)
    """
    train_start, train_end = train_range
    test_start, test_end = test_range

    mask_train = (df.index >= train_start) & (df.index <= train_end)
    mask_test = (df.index >= test_start) & (df.index <= test_end)

    df_train = df[mask_train].copy().reset_index(drop=True)
    df_test = df[mask_test].copy().reset_index(drop=True)

    return df_train, df_test