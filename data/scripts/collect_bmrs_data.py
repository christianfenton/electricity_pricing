"""
Collect BMRS (Balancing Mechanism Reporting Service) data.

This script collects electricity generation, demand, and market price
data from the BMRS API and saves it as a CSV file.

Data collected:
- AGPT: Actual Generation Per Type (by fuel)
- FUELHH: Interconnector flows
- Demand: INDO/ITSO demand outturn
- MIP: Market Index Price and trading volume

Output columns:
- SETTLEMENT_DATE: Date (YYYY-MM-DD)
- SETTLEMENT_PERIOD: Period number (1-50, typically 48)
- Generation by fuel type: BIOMASS, COAL, GAS, NUCLEAR, OIL, SOLAR, WIND, OTHER
- INTER: Interconnector net flows
- INDO: Initial Demand Outturn
- ITSO: Initial Transmission System Demand Outturn
- ELECTRICITY_PRICE: Market price of electricity (£/MWh)
- TRADING_VOLUME: Market trading volume (MWh)

Usage:
```bash
python scripts/collect_bmrs_data.py --start-date 2021-01-01 --end-date 2025-01-01 --verbose
```
"""

import os
import argparse

from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np

from utilities import expected_periods

def collect_agpt_data(start_date, end_date, verbose=False):
    """
    Collect AGPT (Actual Generation Per Type) data from BMRS API.

    Handles 4-day API limit by chunking requests, which can be slow.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        verbose: If True, print progress messages

    Returns:
        DataFrame with generation data by fuel type
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/AGPT/"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + timedelta(days=3), end)

        params = {
            "publishDateTimeFrom": current_start.strftime("%Y-%m-%d 00:00"),
            "publishDateTimeTo": current_end.strftime("%Y-%m-%d 01:00"),
            "format": "json",
        }

        if verbose:
            print(
                f"Fetching AGPT data from {params['publishDateTimeFrom']} "
                f"to {params['publishDateTimeTo']}..."
            )

        response = requests.get(url, params=params)

        if response.ok:
            data = response.json()
            all_data.extend(data["data"])
        else:
            print(f"Error: {response.status_code} {response.text}")

        current_start = current_end

    df = pd.DataFrame(all_data)
    df["startTime"] = pd.to_datetime(df["startTime"])

    df_agpt = df.pivot_table(
        index=["settlementDate", "settlementPeriod", "startTime"],
        columns="psrType",
        values="quantity",
        aggfunc="first",
    ).reset_index()

    df_agpt = df_agpt.sort_values(
        ["settlementDate", "settlementPeriod"]
    ).reset_index(drop=True)

    # Group wind types
    df_agpt["WIND"] = df_agpt[["Wind Offshore", "Wind Onshore"]].sum(axis=1)
    df_agpt = df_agpt.drop(columns=["Wind Offshore", "Wind Onshore"])

    # Group hydro and other
    df_agpt["OTHER"] = df_agpt[
        ["Hydro Pumped Storage", "Hydro Run-of-river and poundage", "Other"]
    ].sum(axis=1)
    df_agpt = df_agpt.drop(
        columns=[
            "Hydro Pumped Storage",
            "Hydro Run-of-river and poundage",
            "Other",
        ]
    )

    # Rename columns to uppercase convention
    df_agpt = df_agpt.rename(
        columns={
            "settlementDate": "SETTLEMENT_DATE",
            "settlementPeriod": "SETTLEMENT_PERIOD",
            "startTime": "START_TIME",
            "Nuclear": "NUCLEAR",
            "Biomass": "BIOMASS",
            "Fossil Gas": "GAS",
            "Fossil Oil": "OIL",
            "Fossil Hard coal": "COAL",
            "Solar": "SOLAR",
        }
    )

    return df_agpt


def collect_fuelhh_data(start_date, end_date, verbose=False):
    """
    Collect FUELHH (Fuel Type Half-Hourly) data from BMRS API.

    Handles 4-day API limit by chunking requests.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        verbose: If True, print progress messages

    Returns:
        DataFrame with interconnector data
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/datasets/FUELHH/"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + timedelta(days=3), end)

        params = {
            "publishDateTimeFrom": current_start.strftime("%Y-%m-%d 00:00"),
            "publishDateTimeTo": current_end.strftime("%Y-%m-%d 01:00"),
            "format": "json",
        }

        if verbose:
            print(
                f"Fetching FUELHH data from {params['publishDateTimeFrom']} "
                f"to {params['publishDateTimeTo']}..."
            )

        response = requests.get(url, params=params)

        if response.ok:
            data = response.json()
            all_data.extend(data["data"])
        else:
            print(f"Error: {response.status_code} {response.text}")

        current_start = current_end

    df = pd.DataFrame(all_data)
    df["startTime"] = pd.to_datetime(df["startTime"])

    df_fuelhh = df.pivot_table(
        index=["settlementDate", "settlementPeriod", "startTime"],
        columns="fuelType",
        values="generation",
        aggfunc="first",
    ).reset_index()

    df_fuelhh = df_fuelhh.sort_values(
        ["settlementDate", "settlementPeriod"]
    ).reset_index(drop=True)

    # Group interconnectors
    interconnects = df_fuelhh.filter(regex="^INT")
    total_interconnect = interconnects.sum(axis=1)
    df_fuelhh = df_fuelhh.filter(regex="^(?!INT)")
    df_fuelhh["INTER"] = total_interconnect

    # Rename columns to uppercase convention
    df_fuelhh = df_fuelhh.rename(
        columns={
            "settlementDate": "SETTLEMENT_DATE",
            "settlementPeriod": "SETTLEMENT_PERIOD",
            "startTime": "START_TIME",
        }
    )

    # Keep only the columns we need
    df_fuelhh = df_fuelhh[
        ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "START_TIME", "INTER"]
    ]

    return df_fuelhh


def collect_demand_data(start_date, end_date, verbose=False):
    """
    Collect demand outturn data from BMRS API.

    Handles 4-day API limit by chunking requests.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        verbose: If True, print progress messages

    Returns:
        DataFrame with INDO and ITSO demand data
    """
    url = "https://data.elexon.co.uk/bmrs/api/v1/demand/outturn"

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + timedelta(days=3), end)

        params = {
            "settlementDateFrom": current_start.strftime("%Y-%m-%d"),
            "settlementDateTo": current_end.strftime("%Y-%m-%d"),
            "settlementPeriod": np.arange(1, 50).tolist(),
            "format": "json",
        }

        if verbose:
            print(
                f"Fetching demand data from {params['settlementDateFrom']} "
                f"to {params['settlementDateTo']}..."
            )

        response = requests.get(url, params=params)

        if response.ok:
            data = response.json()
            all_data.extend(data["data"])
        else:
            print(f"Error: {response.status_code} {response.text}")

        current_start = current_end

    df = pd.DataFrame(all_data)
    df["startTime"] = pd.to_datetime(df["startTime"])

    df_demand = df.pivot_table(
        index=["settlementDate", "settlementPeriod", "startTime"],
        values=[
            "initialDemandOutturn",
            "initialTransmissionSystemDemandOutturn",
        ],
    ).reset_index()

    df_demand = df_demand.sort_values(
        ["settlementDate", "settlementPeriod"]
    ).reset_index(drop=True)

    # Rename columns to uppercase convention
    df_demand = df_demand.rename(
        columns={
            "settlementDate": "SETTLEMENT_DATE",
            "settlementPeriod": "SETTLEMENT_PERIOD",
            "startTime": "START_TIME",
            "initialDemandOutturn": "INDO",
            "initialTransmissionSystemDemandOutturn": "ITSO",
        }
    )

    return df_demand


def collect_mip_data(start_date, end_date, verbose=False):
    """
    Collect electricity (market index) price data from BMRS API.
    Handles 4-day API limit by chunking requests.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        verbose: If True, print progress messages

    Returns:
        DataFrame with price and volume data
    """
    url = (
        "https://data.elexon.co.uk/bmrs/api/v1/balancing/pricing/market-index"
    )

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    all_data = []
    current_start = start

    while current_start < end:
        current_end = min(current_start + timedelta(days=4), end)

        params = {
            "from": current_start.strftime("%Y-%m-%d 00:00"),
            "to": current_end.strftime("%Y-%m-%d 00:00"),
            "dataProviders": ["APX"],
            "format": "json",
        }

        if verbose:
            print(
                f"Fetching price data from {params['from']} to {params['to']}..."
            )

        response = requests.get(url, params=params)

        if response.ok:
            data = response.json()
            all_data.extend(data["data"])
        else:
            print(f"Error: {response.status_code} {response.text}")

        current_start = current_end

    df = pd.DataFrame(all_data)
    df["startTime"] = pd.to_datetime(df["startTime"])

    df_price = df.pivot_table(
        index=["settlementDate", "settlementPeriod", "startTime"],
        values=["price", "volume"],
    ).reset_index()

    df_price = df_price.sort_values(
        ["settlementDate", "settlementPeriod"]
    ).reset_index(drop=True)

    # Rename columns to uppercase convention
    df_price.rename(
        columns={
            "settlementDate": "SETTLEMENT_DATE",
            "settlementPeriod": "SETTLEMENT_PERIOD",
            "startTime": "START_TIME",
            "price": "ELECTRICITY_PRICE",
            "volume": "TRADING_VOLUME",
        },
        inplace=True,
    )

    return df_price


def clean_dataset(df, verbose=False):
    """
    Clean dataset by handling missing/excess periods and removing duplicates.

    Args:
        df: DataFrame with SETTLEMENT_DATE and SETTLEMENT_PERIOD columns
        verbose: If True, print progress messages

    Returns:
        Cleaned DataFrame
    """
    df_clean = df.copy()

    # Ensure SETTLEMENT_DATE is datetime
    df_clean["SETTLEMENT_DATE"] = pd.to_datetime(df_clean["SETTLEMENT_DATE"])

    # Remove duplicates
    n_before = len(df_clean)
    df_clean = df_clean.drop_duplicates(
        subset=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]
    ).reset_index(drop=True)
    n_after = len(df_clean)
    if verbose and n_before > n_after:
        print(f"Removed {n_before - n_after} duplicate rows")

    # Find dates with an unexpected number of settlement periods
    daily_counts = df_clean.groupby(
        df_clean["SETTLEMENT_DATE"].dt.normalize()
    ).size()
    irregular_days = []
    for date, count in daily_counts.items():
        expected = expected_periods(pd.Timestamp(date))
        if count != expected:
            irregular_days.append((date, count, expected))

    if verbose:
        print(f"Found {len(irregular_days)} days needing cleaning")

    # Clean each irregular date
    for date, actual, expected in irregular_days:
        if verbose:
            print(f"  Cleaning {date.date()}: {actual} --> {expected} periods")

        date_normalized = pd.to_datetime(date).normalize()
        day_mask = (
            df_clean["SETTLEMENT_DATE"].dt.normalize() == date_normalized
        )
        day_data = df_clean[day_mask].copy()

        if actual < expected:
            # Fill missing periods by interpolating
            template = pd.DataFrame(
                {
                    "SETTLEMENT_DATE": [date_normalized] * expected,
                    "SETTLEMENT_PERIOD": range(1, expected + 1),
                }
            )
            day_data_filled = template.merge(
                day_data,
                on=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
                how="left",
            )
            numeric_cols = day_data_filled.select_dtypes(
                include=[np.number]
            ).columns
            day_data_filled[numeric_cols] = day_data_filled[
                numeric_cols
            ].interpolate(method="linear", limit_direction="both")
            df_clean = pd.concat(
                [df_clean[~day_mask], day_data_filled], ignore_index=True
            )

        elif actual > expected:
            # Drop excess periods (keep first N)
            day_data = day_data.sort_values("SETTLEMENT_PERIOD").iloc[
                :expected
            ]
            df_clean = pd.concat(
                [df_clean[~day_mask], day_data], ignore_index=True
            )

    # Sort by date and period
    df_clean = df_clean.sort_values(
        ["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"]
    ).reset_index(drop=True)

    return df_clean


def validate_dataset(df, verbose=False):
    """
    Validate that dataset has correct settlement periods for all dates.

    Args:
        df: DataFrame with SETTLEMENT_DATE and SETTLEMENT_PERIOD columns
        verbose: If True, print validation results

    Returns:
        True if validation passes, raises ValueError otherwise
    """
    errors = []

    # Ensure SETTLEMENT_DATE is datetime
    df_temp = df.copy()
    df_temp["SETTLEMENT_DATE"] = pd.to_datetime(df_temp["SETTLEMENT_DATE"])

    # Check for duplicates
    duplicates = df_temp.duplicated(
        subset=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"], keep=False
    )
    if duplicates.any():
        n_duplicates = duplicates.sum()
        errors.append(
            f"Found {n_duplicates} duplicate (date, period) combinations"
        )

    # Check day lengths
    daily_counts = df_temp.groupby(
        df_temp["SETTLEMENT_DATE"].dt.normalize()
    ).size()
    irregular_days = []
    for date, count in daily_counts.items():
        expected = expected_periods(pd.Timestamp(date))
        if count != expected:
            irregular_days.append((date, count, expected))

    if len(irregular_days) > 0:
        errors.append(
            f"Found {len(irregular_days)} days with incorrect number of periods"
        )
        if verbose:
            for date, actual, expected in irregular_days[:5]:
                errors.append(
                    f"  - {date.date()}: {actual} periods (expected {expected})"
                )

    if len(errors) > 0:
        error_msg = "Validation failed:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    if verbose:
        print("All validation checks passed")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect BMRS electricity data from API"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2021-01-01",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-01-01",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output CSV file path (default: data/processed/bmrs_data.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages",
    )

    args = parser.parse_args()

    # Set default output path
    if args.output_path is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        args.output_path = os.path.join(
            project_dir, "data", "processed", "bmrs_data.csv"
        )

    print("\n=== Collecting BMRS Data ===")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output: {args.output_path}\n")

    # Collect all datasets
    print("Collecting AGPT data...") if args.verbose else None
    df_agpt = collect_agpt_data(args.start_date, args.end_date, args.verbose)

    print("\nCollecting FUELHH data...") if args.verbose else None
    df_fuelhh = collect_fuelhh_data(
        args.start_date, args.end_date, args.verbose
    )

    print("\nCollecting demand data...") if args.verbose else None
    df_demand = collect_demand_data(
        args.start_date, args.end_date, args.verbose
    )

    print("\nCollecting market index price data...") if args.verbose else None
    df_mip = collect_mip_data(args.start_date, args.end_date, args.verbose)

    # Merge datasets
    print("\nMerging datasets...") if args.verbose else None
    df_merged = df_agpt.merge(
        df_fuelhh[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "INTER"]],
        on=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
        how="inner",
    )

    df_merged = df_merged.merge(
        df_demand[["SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "INDO", "ITSO"]],
        on=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
        how="inner",
    )

    df_merged = df_merged.merge(
        df_mip[
            [
                "SETTLEMENT_DATE",
                "SETTLEMENT_PERIOD",
                "ELECTRICITY_PRICE",
                "TRADING_VOLUME",
            ]
        ],
        on=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
        how="inner",
    )

    # Drop START_TIME column (not needed, can be reconstructed)
    df_merged = df_merged.drop(columns=["START_TIME"], errors="ignore")

    # Clean dataset
    print("\nCleaning dataset...") if args.verbose else None
    df_merged = clean_dataset(df_merged, verbose=args.verbose)

    # Validate dataset
    print("\nValidating dataset...") if args.verbose else None
    validate_dataset(df_merged, verbose=args.verbose)

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_merged.to_csv(args.output_path, index=False)

    print(f"\nSaved {len(df_merged)} records to {args.output_path}")
    print(
        f"""Date range: {df_merged['SETTLEMENT_DATE'].min()} 
        to {df_merged['SETTLEMENT_DATE'].max()}"""
    )
    print(f"Columns: {', '.join(df_merged.columns.tolist())}")


if __name__ == "__main__":
    main()
