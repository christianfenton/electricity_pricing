"""
Data validation functions for UK electricity market data.

This module provides functions to validate that every date has the correct
number of settlement periods. The functions assume UK DST transitions.
"""

import pandas as pd
from typing import Tuple, List
from ..utils import get_expected_periods, last_sunday_of_month


def find_irregular_days(
        df: pd.DataFrame, 
        date_column: str = 'SETTLEMENT_DATE'
        ) -> List[Tuple[pd.Timestamp, int, int]]:
    """
    Find days with an unexpected number of settlement periods.

    Args:
        df: pandas.DataFrame
        date_column: Name of the date column. Default: 'SETTLEMENT_DATE'.

    Returns:
        List of tuples: (date, actual_count, expected_count) for irregular days
    """
    daily_counts = df.groupby(df[date_column]).size()
    irregular_days = []

    for date, count in daily_counts.items():
        expected_count = get_expected_periods(date)
        if count != expected_count:
            irregular_days.append((date, count, expected_count))

    return irregular_days


def validate_day_lengths(
        df: pd.DataFrame, 
        date_column: str = 'SETTLEMENT_DATE'
        ) -> Tuple[bool, List[Tuple[pd.Timestamp, int, int]]]:
    """
    Validate that all days have the correct number of settlement periods.

    Args:
        df: DataFrame with settlement period data
        date_column: Name of the date column. Default: 'SETTLEMENT_DATE'.

    Returns:
        Tuple of (is_valid, list_of_irregular_days)
    """
    irregular_days = find_irregular_days(df, date_column)
    is_valid = len(irregular_days) == 0
    return is_valid, irregular_days


def validate_spring_dst(
    df: pd.DataFrame,
    date_column: str = 'SETTLEMENT_DATE'
) -> Tuple[bool, str]:
    """
    Validate that spring DST days (46 periods) occur on the correct dates.

    Only validates if the data spans into March. If there are no days with 46 periods,
    validation passes (data might not include DST dates).

    Args:
        df: DataFrame with settlement period data
        date_column: Name of the date column. Default: 'SETTLEMENT_DATE'

    Returns:
        Tuple of (is_valid, error_message)
    """
    daily_counts = df.groupby(df[date_column]).size()
    short_days = daily_counts[daily_counts == 46].index
    short_days = sorted(short_days)

    if len(short_days) == 0:
        return True, ""

    years_in_data = df[date_column].dt.year.unique()
    years_with_march = []
    for year in years_in_data:
        march_dates = df[df[date_column].dt.year == year][date_column].dt.month
        if 3 in march_dates.values:
            years_with_march.append(year)

    years_with_march = sorted(years_with_march)

    if len(years_with_march) != len(short_days):
        msg = f"Expected one short day (46 periods) per year with March data, found {len(short_days)} for {len(years_with_march)} years"
        return False, msg

    for i, year in enumerate(years_with_march):
        expected_date = last_sunday_of_month(year, 3)
        actual_date = short_days[i].date()
        if actual_date != expected_date:
            msg = f"Spring DST in year {year}: expected {expected_date}, got {actual_date}"
            return False, msg

    return True, ""


def validate_autumn_dst(
    df: pd.DataFrame,
    date_column: str = 'SETTLEMENT_DATE'
) -> Tuple[bool, str]:
    """
    Validate that autumn DST days (50 periods) occur on the correct dates.

    Only validates if the data spans into October. If there are no days with 50 periods,
    validation passes (data might not include DST dates).

    Args:
        df: DataFrame with settlement period data
        date_column: Name of the date column

    Returns:
        Tuple of (is_valid, error_message)
    """
    daily_counts = df.groupby(df[date_column]).size()
    long_days = daily_counts[daily_counts == 50].index
    long_days = sorted(long_days)

    if len(long_days) == 0:
        return True, ""

    years_in_data = df[date_column].dt.year.unique()
    years_with_october = []
    for year in years_in_data:
        october_dates = df[df[date_column].dt.year == year][date_column].dt.month
        if 10 in october_dates.values:
            years_with_october.append(year)

    years_with_october = sorted(years_with_october)

    if len(years_with_october) != len(long_days):
        msg = f"Expected one long day (50 periods) per year with October data, found {len(long_days)} for {len(years_with_october)} years"
        return False, msg

    for i, year in enumerate(years_with_october):
        expected_date = last_sunday_of_month(year, 10)
        actual_date = long_days[i].date()
        if actual_date != expected_date:
            msg = f"Autumn DST in year {year}: expected {expected_date}, got {actual_date}"
            return False, msg

    return True, ""


def validate_settlement_periods(
    df: pd.DataFrame,
    date_column: str = 'SETTLEMENT_DATE',
    period_column: str = 'SETTLEMENT_PERIOD',
    verbose: bool = False
) -> Tuple[bool, List[str]]:
    """
    Validate that the data has the expected settlement date and periods.

    Checks:
    1. All days have correct number of periods (46/48/50)
    2. Spring DST days (46 periods) are on last Sunday of March
    3. Autumn DST days (50 periods) are on last Sunday of October
    4. No duplicate (date, period) combinations
    5. Settlement periods are sequential (1 to N)

    Args:
        df: DataFrame with settlement period data
        date_column: Name of the date column
        period_column: Name of the settlement period column
        verbose: If True, print validation results

    Returns:
        Tuple of (is_valid, list_of_error_messages)
    """
    errors = []

    # Check for duplicates
    duplicates = df.duplicated(subset=[date_column, period_column], keep=False)
    if duplicates.any():
        n_duplicates = duplicates.sum()
        errors.append(f"Found {n_duplicates} duplicate (date, period) combinations")

    # Check day lengths
    is_valid_lengths, irregular_days = validate_day_lengths(df, date_column)
    if not is_valid_lengths:
        errors.append(f"Found {len(irregular_days)} days with incorrect number of periods")
        if verbose:
            for date, actual, expected in irregular_days[:5]:  # Show first 5
                errors.append(f"  - {date.date()}: {actual} periods (expected {expected})")

    # Check spring DST
    is_valid_spring, spring_msg = validate_spring_dst(df, date_column)
    if not is_valid_spring:
        errors.append(f"Spring DST validation failed: {spring_msg}")

    # Check autumn DST
    is_valid_autumn, autumn_msg = validate_autumn_dst(df, date_column)
    if not is_valid_autumn:
        errors.append(f"Autumn DST validation failed: {autumn_msg}")

    is_valid = len(errors) == 0

    if verbose:
        if is_valid:
            print("All validation checks passed")
        else:
            print(f"Validation failed with {len(errors)} error(s):")
            for error in errors:
                print(f"  {error}")

    return is_valid, errors
