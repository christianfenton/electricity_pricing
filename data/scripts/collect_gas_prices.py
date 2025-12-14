"""
Collect natural gas price data from ONS Excel file.

This script reads daily natural gas System Average Price (SAP) data
from the Office for National Statistics (ONS) Excel file and converts
it to a standardised format.

Input file expected:
- Excel file with sheet "Table 1 Daily SAP of Gas"
- Column "SAP actual day" with prices in pence/kWh

Output columns:
- SETTLEMENT_DATE: Date (YYYY-MM-DD)
- NATURAL_GAS_PRICE: Natural gas price (£/MWh)

Note: This produces daily values. When merged with settlement period data,
the daily price will be broadcast to all settlement periods for that day.

```bash
python scripts/collect_gas_prices.py --start-date 2021-01-01 --end-date 2025-01-01 --verbose
```
"""

import os
import argparse

import pandas as pd


def collect_gas_price_data(input_path, start_date, end_date):
    """
    Collect gas price data from locally stored Excel file.

    Returns daily gas prices that can be merged with settlement period data.
    The daily price will be broadcast to all settlement periods for that day.

    Args:
        input_path: Path to Excel file containing gas price data
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)

    Returns:
        DataFrame with SETTLEMENT_DATE and NATURAL_GAS_PRICE columns
    """
    df = pd.read_excel(
        input_path, sheet_name="Table 1 Daily SAP of Gas", header=5
    )

    # Filter date range and drop unnecessary columns
    df.drop(columns="SAP seven-day rolling average", inplace=True)
    df = df.loc[(df["Date"] >= start_date) & (df["Date"] <= end_date)].copy()

    # Convert price units from p/kWh to £/MWh
    PENCE_TO_POUNDS = 1 / 100
    KWH_TO_MWH = 1000
    df["NATURAL_GAS_PRICE"] = (
        df["SAP actual day"] * PENCE_TO_POUNDS * KWH_TO_MWH
    )

    # Convert Date to SETTLEMENT_DATE format (string YYYY-MM-DD)
    df["SETTLEMENT_DATE"] = pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")

    # Keep only the columns we need
    df = df[["SETTLEMENT_DATE", "NATURAL_GAS_PRICE"]]

    return df


def validate_gas_price_data(df, start_date, end_date, verbose=False):
    """
    Validate gas price dataset for completeness.

    Args:
        df: DataFrame with gas price data
        start_date: Expected start date (YYYY-MM-DD)
        end_date: Expected end date (YYYY-MM-DD)
        verbose: If True, print validation results

    Returns:
        True if validation passes, raises ValueError otherwise
    """
    errors = []

    # Check required columns
    required_cols = ["SETTLEMENT_DATE", "NATURAL_GAS_PRICE"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {', '.join(missing_cols)}")

    # Check for missing values
    n_missing_prices = df["NATURAL_GAS_PRICE"].isna().sum()
    if n_missing_prices > 0:
        errors.append(f"Found {n_missing_prices} missing price values")

    # Check for duplicate dates
    duplicates = df.duplicated(subset=["SETTLEMENT_DATE"], keep=False)
    if duplicates.any():
        n_duplicates = duplicates.sum()
        errors.append(f"Found {n_duplicates} duplicate dates")

    # Check date range coverage
    df_temp = df.copy()
    df_temp["SETTLEMENT_DATE"] = pd.to_datetime(df_temp["SETTLEMENT_DATE"])
    min_date = df_temp["SETTLEMENT_DATE"].min().strftime("%Y-%m-%d")
    max_date = df_temp["SETTLEMENT_DATE"].max().strftime("%Y-%m-%d")

    if min_date > start_date:
        errors.append(
            f"Data starts at {min_date}, but requested start date {start_date}"
        )

    if max_date < end_date:
        errors.append(
            f"Data ends at {max_date}, but requested end date {end_date}"
        )

    if len(errors) > 0:
        error_msg = "Validation failed:\n" + "\n".join(errors)
        raise ValueError(error_msg)

    if verbose:
        print("All validation checks passed")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Collect natural gas price data from ONS Excel file"
    )
    parser.add_argument(
        "--input-path",
        type=str,
        default=None,
        help="""
        Path to ONS gas price Excel file (default: data/raw/ons_gas_prices.xlsx)
        """,
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
        help="Output CSV file path (default: data/processed/gas_prices.csv)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress messages",
    )

    args = parser.parse_args()

    # Set default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    if args.input_path is None:
        args.input_path = os.path.join(
            project_dir, "data", "raw", "ons_gas_prices.xlsx"
        )

    if args.output_path is None:
        args.output_path = os.path.join(
            project_dir, "data", "processed", "gas_prices.csv"
        )

    print("\n=== Collecting Gas Price Data ===")
    print(f"Input file: {args.input_path}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Output: {args.output_path}\n")

    # Check input file exists
    if not os.path.exists(args.input_path):
        raise FileNotFoundError(f"Input file not found: {args.input_path}")

    # Collect gas price data
    print("Reading gas price data...") if args.verbose else None
    df_gas = collect_gas_price_data(
        args.input_path, args.start_date, args.end_date
    )

    # Validate dataset
    print("Validating dataset...") if args.verbose else None
    validate_gas_price_data(
        df_gas, args.start_date, args.end_date, verbose=args.verbose
    )

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_gas.to_csv(args.output_path, index=False)

    print(f"\nSaved {len(df_gas)} daily records to {args.output_path}")
    print(
        f"""Date range: {df_gas["SETTLEMENT_DATE"].min()} 
        to {df_gas["SETTLEMENT_DATE"].max()}"""
    )


if __name__ == "__main__":
    main()
