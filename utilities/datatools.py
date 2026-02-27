"""Date and timestamp processing utilities for UK electricity settlement periods."""

import datetime as dt

import pandas as pd


def last_sunday_of_month(year: int, month: int) -> dt.date:
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


def expected_periods(date_: pd.Timestamp | dt.date) -> int:
    """
    Return the number of half-hourly settlement periods on the given date in the UK.

    Args:
        date: pandas.Timestamp or datetime.date type

    Returns:
        n (int): The number of settlement periods
    """
    year = date_.year
    spring_dst = last_sunday_of_month(year, 3)
    autumn_dst = last_sunday_of_month(year, 10)
    if date_.date() == spring_dst:
        return 46
    elif date_.date() == autumn_dst:
        return 50
    else:
        return 48
    

def periods_in_date_range(start_date: dt.date, end_date: dt.date) -> int:
    """
    Calculate total number of settlement periods in a date range (inclusive).

    Args:
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        Total number of 30-minute settlement periods
    """
    total = 0
    current = start_date
    while current <= end_date:
        total += expected_periods(current)
        current += dt.timedelta(days=1)
    return total


def periods_remaining(date: dt.date, period: int) -> int:
    """
    Return the number of settlement periods remaining in the day after the given period.

    Args:
        date: Settlement date
        period: Current settlement period (1-indexed)

    Returns:
        Number of periods remaining (including current period)
    """
    total_periods = expected_periods(date)
    return total_periods - period + 1


def create_timestamps(
    df: pd.DataFrame,
    date_column: str,
    period_column: str,
    tz: str = "Europe/London",
) -> pd.Series:
    """
    Create timezone-aware timestamps from settlement dates and periods.

    Args:
        df (pandas.DataFrame): Dataframe with settlement date and period columns
        date_column: Name of settlement date column
        period_column: Name of settlement period column
        tz: Timezone for timestamps (default: 'Europe/London')

    Returns:
        A pandas.Series.
    """
    base_timestamps = pd.to_datetime(df[date_column]).dt.tz_localize("Europe/London")
    period_offsets = pd.to_timedelta((df[period_column] - 1) * 30, unit="m")
    timestamps = (base_timestamps + period_offsets).dt.tz_convert(tz)
    return pd.Series(timestamps, index=df.index, name="DATETIME")


def validate_timestamps(
    df: pd.DataFrame,
    datetime_column: str,
    date_column: str,
    period_column: str,
    tz: str = "Europe/London",
) -> bool:
    """
    Validate that timestamps match the (date, settlement_period) representation.

    Args:
        df: DataFrame with both timestamps and (date, period) columns
        datetime_column: Name of timestamp column to validate
        date_column: Name of settlement date column
        period_column: Name of settlement period column
        tz: Timezone to use for reconstruction

    Returns True if checks pass, otherwise raises a ValueError.
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


def timestamp_to_settlement(timestamp: pd.Timestamp) -> tuple[dt.date, int]:
    """
    Convert a timestamp to settlement date and period.

    Args:
        timestamp: Timezone-aware timestamp

    Returns:
        Tuple of (settlement_date, settlement_period)
    """
    # Ensure timezone-aware in Europe/London
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize("Europe/London", ambiguous="NaT")
    else:
        timestamp = timestamp.tz_convert("Europe/London")

    # Get the date
    settlement_date = timestamp.date()

    # Get midnight of this date (timezone-aware)
    midnight = pd.Timestamp(settlement_date, tz="Europe/London")

    # Calculate elapsed time since midnight
    elapsed = timestamp - midnight

    # Convert to 30-minute periods (1-indexed)
    # elapsed.total_seconds() / (30 * 60) gives fractional periods
    settlement_period = int(elapsed.total_seconds() / (30 * 60)) + 1

    # Clamp to expected range for this date
    max_period = expected_periods(pd.Timestamp(settlement_date))
    settlement_period = min(settlement_period, max_period)
    settlement_period = max(settlement_period, 1)

    return settlement_date, settlement_period