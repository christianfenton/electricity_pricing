"""
Data cleaning functions for electricity market data.

This module provides functions to clean and repair settlement period data,
including handling missing periods, duplicates, and DST-related issues.
"""

import pandas as pd
import numpy as np
from typing import Optional
from ..utils import get_expected_periods

def fill_missing_periods(
    df: pd.DataFrame,
    settlement_date: pd.Timestamp,
    date_column: str = 'SETTLEMENT_DATE',
    period_column: str = 'SETTLEMENT_PERIOD',
    method: str = 'interpolate'
) -> pd.DataFrame:
    """
    Fill in missing settlement periods for a specific date.

    Args:
        df: pandas.DataFrame
        settlement_date: The date to fix.
        date_column: Name of the date column. Default: 'SETTLEMENT_DATE'.
        period_column: Name of the settlement period column. Default: 'SETTLEMENT_PERIOD'.
        method: Method to fill missing values ('interpolate', 'forward_fill', or 'previous_day')

    Returns:
        pandas.DataFrame
    """
    date_normalised = pd.to_datetime(settlement_date).normalize()
    day_mask = df[date_column].dt.normalize() == date_normalised
    day_data = df[day_mask].copy()
    n_periods = len(day_data)
    expected_periods = get_expected_periods(date_normalised)

    if n_periods == expected_periods:
        return df.copy()

    # Create template with all expected periods
    template = pd.DataFrame({
        date_column: [date_normalised] * expected_periods, 
        period_column: range(1, expected_periods + 1)
        })

    day_data_filled = template.merge(day_data, on=[date_column, period_column], how='left')

    # Fill missing values based on method
    if method == 'interpolate':
        numeric_cols = day_data_filled.select_dtypes(include=[np.number]).columns
        day_data_filled[numeric_cols] = day_data_filled[numeric_cols].interpolate(method='linear', limit_direction='both')
        non_numeric_cols = [c for c in day_data_filled.columns if c not in numeric_cols]
        day_data_filled[non_numeric_cols] = day_data_filled[non_numeric_cols].ffill().bfill()

    elif method == 'forward_fill':
        day_data_filled = day_data_filled.ffill().bfill()

    elif method == 'previous_day':
        # Copy values from previous day
        prev_date = date_normalised - pd.Timedelta(days=1)
        prev_mask = df[date_column].dt.normalize() == prev_date
        prev_data = df[prev_mask].copy()

        if len(prev_data) > 0:
            # Match by settlement period
            prev_data = prev_data.set_index(period_column)
            for period in range(1, expected_periods + 1):
                if period in prev_data.index:
                    mask = day_data_filled[period_column] == period
                    for col in day_data_filled.columns:
                        if col not in [date_column, period_column] and pd.isna(day_data_filled.loc[mask, col]).any():
                            day_data_filled.loc[mask, col] = prev_data.loc[period, col]

    # Combine with other days
    df_clean = pd.concat([df[~day_mask].copy(), day_data_filled], ignore_index=True)

    return df_clean


def drop_excess_periods(
    df: pd.DataFrame,
    settlement_date: pd.Timestamp,
    date_column: str = 'SETTLEMENT_DATE',
    period_column: str = 'SETTLEMENT_PERIOD'
) -> pd.DataFrame:
    """
    Remove excess settlement periods for a specific date.

    Keeps the first N periods (where N is the expected number for that date).

    Args:
        df: DataFrame with settlement period data.
        settlement_date: The date to fix.
        date_column: Name of the date column. Default: 'SETTLEMENT_DATE'.
        period_column: Name of the settlement period column. Default: 'SETTLEMENT_PERIOD'.

    Returns:
        pandas.DataFrame
    """
    date_normalised = pd.to_datetime(settlement_date).normalize()
    day_mask = df[date_column].dt.normalize() == date_normalised
    day_data = df[day_mask].copy()
    n_periods = len(day_data)
    expected_periods = get_expected_periods(date_normalised)

    if n_periods <= expected_periods:
        return df.copy()

    day_data = day_data.sort_values(period_column).iloc[:expected_periods]
    df_clean = pd.concat([df[~day_mask].copy(), day_data], ignore_index=True)

    return df_clean


def clean_single_day(
    df: pd.DataFrame,
    settlement_date: pd.Timestamp,
    date_column: str = 'SETTLEMENT_DATE',
    period_column: str = 'SETTLEMENT_PERIOD',
    fill_method: str = 'interpolate'
) -> pd.DataFrame:
    """
    Clean data for a specific day (handle both missing and excess periods).

    Args:
        df: pandas.DataFrame
        settlement_date: The date to clean.
        date_column: Name of the date column. Default: 'SETTLEMENT_DATE'.
        period_column: Name of the settlement period column. Default: 'SETTLEMENT_PERIOD'.
        fill_method: Method to fill missing values. Default: 'interpolate'.

    Returns:
        DataFrame with cleaned data for the specified date
    """
    date_normalised = pd.to_datetime(settlement_date).normalize()
    day_mask = df[date_column].dt.normalize() == date_normalised
    day_data = df[day_mask].copy()
    n_periods = len(day_data)
    expected_periods = get_expected_periods(date_normalised)

    if n_periods < expected_periods:
        df = fill_missing_periods(df, settlement_date, date_column, period_column, fill_method)

    elif n_periods > expected_periods:
        df = drop_excess_periods(df, settlement_date, date_column, period_column)

    return df


def clean_dataset(
    df: pd.DataFrame,
    date_column: str = 'SETTLEMENT_DATE',
    period_column: str = 'SETTLEMENT_PERIOD',
    fill_method: str = 'interpolate',
    verbose: bool = False
) -> pd.DataFrame:
    """
    Comprehensive cleaning of data using settlement dates and periods to represent time.

    Steps:
    1. Remove duplicate (date, period) combinations
    2. Identify days with incorrect number of periods
    3. Fill missing periods or trim excess periods
    4. Sort by date and period

    Args:
        df: DataFrame with settlement period data
        date_column: Name of the date column. Default: 'SETTLEMENT_DATE'.
        period_column: Name of the settlement period column. Default: 'SETTLEMENT_PERIOD'.
        fill_method: Method to fill missing values. Default: 'interpolate'.
        verbose: If True, print progress messages

    Returns:
        pandas.DataFrame
    """
    df_clean = df.copy()

    # Remove duplicates
    n_before = len(df_clean)
    df_clean = df.drop_duplicates(subset=[date_column, period_column]).reset_index(drop=True)
    n_after = len(df_clean)
    if verbose and n_before > n_after:
        print(f"Removed {n_before - n_after} duplicate rows")

    # Find dates with an unexpected number of settlement periods
    daily_counts = df_clean.groupby(df_clean[date_column].dt.normalize()).size()
    irregular_days = []
    for date, count in daily_counts.items():
        expected = get_expected_periods(pd.Timestamp(date))
        if count != expected:
            irregular_days.append((date, count, expected))
    if verbose:
        print(f"Found {len(irregular_days)} days needing cleaning")

    # Clean each irregular date
    for date, actual, expected in irregular_days:
        if verbose:
            print(f"  Cleaning {date.date()}: {actual} --> {expected} periods")
        df_clean = clean_single_day(df_clean, date, date_column, period_column, fill_method)

    df_clean = df_clean.sort_values([date_column, period_column]).reset_index(drop=True)

    return df_clean
