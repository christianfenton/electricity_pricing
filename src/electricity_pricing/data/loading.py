"""
Data loading and merging functions for processed electricity market data.

This module provides functions to load and merge existing CSV files containing
electricity price data, generation mix, and day-ahead forecasts.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from datetime import date

from .validation import validate_settlement_periods
from .cleaning import clean_dataset


def load_electricity_data(
    file_path: str, verbose: bool = False
) -> pd.DataFrame:
    """
    Load electricity generation and price data from CSV.

    Args:
        file_path: Path to CSV file.
        verbose: Print progress messages. Default: False.

    Returns:
        pandas.DataFrame
    """
    if verbose:
        print(f"Loading electricity data from {file_path}...")

    # Determine which date columns to parse
    date_cols = []
    try:  # Check if file exists and peek at columns
        sample = pd.read_csv(file_path, nrows=0)
        for col in ["settlementDate", "SETTLEMENT_DATE", "startTime"]:
            if col in sample.columns:
                date_cols.append(col)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not peek at columns: {e}")

    # Load data
    df = pd.read_csv(file_path, parse_dates=date_cols)

    # Standardise column names
    if "settlementDate" in df.columns:
        df = df.rename(columns={"settlementDate": "SETTLEMENT_DATE"})
    if "settlementPeriod" in df.columns:
        df = df.rename(columns={"settlementPeriod": "SETTLEMENT_PERIOD"})

    # Drop startTime if present (helps avoid confusion arounds time zones)
    if "startTime" in df.columns:
        df = df.drop(columns=["startTime"])

    if verbose:
        print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")

    if verbose:
        print("  Cleaning data...")
    df = clean_dataset(df, verbose=verbose)

    return df


def load_forecast_data(file_path: str, verbose: bool = False) -> pd.DataFrame:
    """
    Load day-ahead forecast data from CSV.

    Args:
        file_path: Path to CSV file
        verbose: Print progress messages. Default: False.

    Returns:
        pandas.DataFrame
    """
    if verbose:
        print(f"Loading forecast data from {file_path}...")

    # Determine which columns to parse
    date_cols = []
    try:
        sample = pd.read_csv(file_path, nrows=0)
        for col in ["SETTLEMENT_DATE", "Forecast_Datetime", "DATETIME_GMT"]:
            if col in sample.columns:
                date_cols.append(col)
    except Exception as e:
        if verbose:
            print(f"Warning: Could not peek at columns: {e}")

    df = pd.read_csv(file_path, parse_dates=date_cols)

    if verbose:
        print(f"  Loaded {len(df)} rows with {len(df.columns)} columns")

    if verbose:
        print("  Cleaning data...")
    df = clean_dataset(df, verbose=verbose)

    return df


def merge_electricity_and_forecasts(
    df_electricity: pd.DataFrame,
    df_forecast: pd.DataFrame,
    validate_result: bool = True,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Merge electricity generation/price data with day-ahead forecasts.

    Args:
        df_electricity: pd.DataFrame with electricity data
        df_forecast: pd.DataFrame with forecast data
        how: Type of merge ('inner', 'left', 'outer')
        validate_result: Whether to validate the merged data
        verbose: If True, print progress messages

    Returns:
        pd.DataFrame
    """
    if verbose:
        print("Merging electricity and forecast data...")
        print(f"  Electricity data: {len(df_electricity)} rows")
        print(f"  Forecast data: {len(df_forecast)} rows")

    df_merged = pd.merge(
        df_electricity,
        df_forecast,
        on=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
        how="inner",
        suffixes=("_elec", "_forecast"),
    )

    if verbose:
        print(f"  Merged data: {len(df_merged)} rows")

    if validate_result:
        if verbose:
            print("  Validating merged data...")
        is_valid, errors = validate_settlement_periods(
            df_merged, verbose=verbose
        )
        if not is_valid and verbose:
            print(f"  Warning: Validation found {len(errors)} issue(s)")

    return df_merged


def load_and_merge_datasets(
    electricity_path: str,
    forecast_path: str,
    date_range: Optional[Tuple[date, date]] = None,
    validate: bool = True,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load and merge electricity and forecast datasets.

    This is the main function to use for creating a clean, merged dataset
    from existing CSV files.

    Args:
        electricity_path: Path to electricity data CSV
        forecast_path: Path to forecast data CSV
        date_range: Optional (start_date, end_date) tuple to filter data
        clean: Whether to clean data before merging
        validate: Whether to validate the result
        output_path: Optional path to save merged data
        verbose: If True, print progress messages

    Returns:
        pandas.DataFrame
    """
    if verbose:
        print("=" * 80)
        print("Loading and Merging Electricity Market Data")
        print("=" * 80)

    df_electricity = load_electricity_data(electricity_path, verbose=verbose)
    df_forecast = load_forecast_data(forecast_path, verbose=verbose)

    if date_range is not None:
        start_date, end_date = date_range
        if verbose:
            print(f"\nFiltering dates: {start_date} to {end_date}")

        for df in [df_electricity, df_forecast]:
            mask = (df["SETTLEMENT_DATE"].dt.date >= start_date) & (
                df["SETTLEMENT_DATE"].dt.date <= end_date
            )
            df = df[mask].copy()
            if verbose:
                print(f"  Filtered to {len(df)} rows")

        sdates_elec = df_electricity["SETTLEMENT_DATE"].dt.date
        mask_elec = (sdates_elec >= start_date) & (sdates_elec <= end_date)
        df_electricity = df_electricity[mask_elec].copy()

        sdates_fc = df_forecast["SETTLEMENT_DATE"].dt.date
        mask_fc = (sdates_fc >= start_date) & (sdates_fc <= end_date)
        df_forecast = df_forecast[mask_fc].copy()

    df_merged = merge_electricity_and_forecasts(
        df_electricity, df_forecast, validate_result=validate, verbose=verbose
    )

    if output_path:
        if verbose:
            print(f"\nSaving merged data to {output_path}...")
        df_merged.to_csv(output_path, index=False)
        if verbose:
            print(f"  Saved {len(df_merged)} rows")

    if verbose:
        print("\n" + "=" * 80)
        print(
            f"Load and merge complete: {len(df_merged)} rows, "
            +
            f"{len(df_merged.columns)} columns"
        )
        print("=" * 80)

    return df_merged
