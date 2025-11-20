"""
Collect day-ahead forecast data from NESO API.

This script collects three types of day-ahead forecasts:
1. Demand forecasts (interpolated from cardinal points to 30-min grid)
2. Wind generation forecasts (metered wind only)
3. Embedded wind and solar forecasts (from locally stored CSV files)

Day-ahead demand forecasts:
https://www.neso.energy/data-portal/1-day-ahead-demand-forecast/historic_day_ahead_demand_forecasts

Day-ahead metered wind forecasts:
https://www.neso.energy/data-portal/day-ahead-wind-forecast

Embedded wind and solar forecasts:
https://www.neso.energy/data-portal/embedded-wind-and-solar-forecasts

Output columns (uppercase convention):
- SETTLEMENT_DATE: Date (date object or YYYY-MM-DD string)
- SETTLEMENT_PERIOD: Period number (1-50, typically 48)
- DEMAND_FORECAST: Forecasted demand (MW)
- WIND_FORECAST: Forecasted metered wind generation (MW)
- WIND_CAPACITY: Wind capacity (MW)
- EMBEDDED_WIND_FORECAST: Forecasted embedded wind generation (MW)
- EMBEDDED_SOLAR_FORECAST: Forecasted embedded solar generation (MW)

```bash
python scripts/collect_neso_forecasts.py --start-date 2021-01-02 --end-date 2025-01-01 --verbose
```
"""

import os
import argparse
import datetime
import requests
import pandas as pd
import numpy as np
from electricity_pricing.utils import get_expected_periods


def collect_demand_forecast(start_date, end_date, verbose=False):
    """
    Collect day-ahead demand forecasts from NESO API.

    Forecasts are provided as cardinal points (peaks/troughs) and are
    interpolated to a uniform 30-minute grid.

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        verbose: If True, print progress messages

    Returns:
        DataFrame with SETTLEMENT_DATE, SETTLEMENT_PERIOD, and DEMAND_FORECAST

    Definitions:
        Cardinal Point (CP):
            Electricity demand fluctuates during a day depending on 
            how much energy people, businesses and industries are using at 
            that moment in time. As this electricity demand goes up and down 
            we get characteristic peaks and troughs, with some of these peaks
            and troughs appearing every single day at similar times. These we 
            call cardinal points and are the points during the day that we 
            forecast demand for.
        Cardinal Point Type: 
            Fixed, Trough or Peak. Cardinal points (CPs) can 
            either be fixed (occur at a fixed time), trough (minimum demands 
            during a set period of the day) or peak(maximum demands during 
            a set period of the day). These are represented throught the 
            first letter of the point type (F,T or P).
        Cardinal Point Start Time: 
            The time when a particular cardinal point 
            (CP) starts during the day. This is given relative to the 
            timezone in effect in the UK at the forecast time and date.
        Cardinal Point End Time: 
            The time when a particular cardinal point (CP) ends during 
            the day. This is given relative to the timezone in effect 
            in the UK at the forecast time and date.
        Forecasting point 1 / Overnight minimum (OM): 
            Minimum national demand between half hour ending 00:30 and 07:30. 
        Forecasting point 2 / Daytime peak (DM): 
            Maximum national demand between half hour ending 08:00 and 13:00. 
        Forecasting point 3 / Daytime minimum (Dm): 
            Minimum national demand between half hour ending 13:30 and 16:30. 
        Forecasting point 4 / Evening peak (EM):
            Maximum national demand between half hour ending 17:00 and 24:00.
        Forecast Timestamp: The date and time at which the forecast was made.
    """
    url = "https://api.neso.energy/api/3/action/datastore_search_sql"
    resource_id = "9847e7bb-986e-49be-8138-717b25933fbb"

    query = f"""
    SELECT *
    FROM "{resource_id}"
    WHERE "TARGETDATE" >= '{start_date}' AND "TARGETDATE" <= '{end_date}'
    ORDER BY "TARGETDATE" ASC
    """

    params = {"sql": query}

    if verbose:
        print("Fetching demand forecast data from NESO API...")

    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception(f"Request failed with status {r.status_code}")

    demand_forecast = r.json()
    df = pd.DataFrame(demand_forecast["result"]["records"])

    # Drop unnecessary columns
    df.drop(columns=["_full_text", "_id"], inplace=True, errors="ignore")

    # Convert time data to proper format
    cp_st_time_str = df["CP_ST_TIME"].astype(str).str.zfill(4)
    cp_end_time_str = df["CP_END_TIME"].astype(str).str.zfill(4)

    # Identify rows with 2400 (need to be converted to 0000)
    cp_st_mask = df["CP_ST_TIME"] == "2400"
    cp_end_mask = df["CP_END_TIME"] == "2400"

    # Replace 2400 with 0000
    cp_st_time_str = cp_st_time_str.replace("2400", "0000")
    cp_end_time_str = cp_end_time_str.replace("2400", "0000")

    # Combine date and time
    df["CP_START"] = pd.to_datetime(
        df["TARGETDATE"].astype(str) + " " + cp_st_time_str,
        format="%Y-%m-%d %H%M",
    )
    df["CP_END"] = pd.to_datetime(
        df["TARGETDATE"].astype(str) + " " + cp_end_time_str,
        format="%Y-%m-%d %H%M",
    )

    # Localize to Europe/London timezone
    df["CP_START"] = df["CP_START"].dt.tz_localize(
        "Europe/London", ambiguous="NaT", nonexistent="NaT"
    )
    df["CP_END"] = df["CP_END"].dt.tz_localize(
        "Europe/London", ambiguous="NaT", nonexistent="NaT"
    )

    # Add one day where the original time was 2400
    df.loc[cp_st_mask, "CP_START"] += pd.Timedelta(days=1)
    df.loc[cp_end_mask, "CP_END"] += pd.Timedelta(days=1)

    # Calculate midpoint of cardinal point
    df["CP_MIDPOINT"] = df["CP_START"] + (df["CP_END"] - df["CP_START"]) / 2

    # Drop unnecessary columns
    df.drop(
        columns=[
            "CP_TYPE",
            "F_Point",
            "CARDINALPOINT",
            "CP_START",
            "CP_END",
            "CP_ST_TIME",
            "CP_END_TIME",
            "TARGETDATE",
            "FORECAST_TIMESTAMP",
        ],
        inplace=True,
        errors="ignore",
    )

    if (df["DAYSAHEAD"] == 1).all():
        df.drop(columns=["DAYSAHEAD"], inplace=True, errors="ignore")

    # Create uniform 30-minute index
    df.set_index("CP_MIDPOINT", inplace=True)
    start = df.index.min().floor("30min")
    end = df.index.max().ceil("30min")
    uniform_index = pd.date_range(
        start=start, end=end, freq="30min", tz="Europe/London"
    )

    # Drop duplicate indices (keep first forecast)
    df = df.loc[~df.index.duplicated(keep="first"), :]

    # Reindex to uniform spacing
    df = df.reindex(uniform_index, fill_value=np.nan)

    # Interpolate forecasts
    df["FORECASTDEMAND"] = df["FORECASTDEMAND"].interpolate(method="linear")

    # Add settlement date and period
    df["SETTLEMENT_DATE"] = df.index.tz_convert("Europe/London").date
    df = df.sort_index()
    df["SETTLEMENT_PERIOD"] = df.groupby("SETTLEMENT_DATE").cumcount() + 1

    # Reset index and rename
    df = df.reset_index(drop=True)
    df.rename(columns={"FORECASTDEMAND": "DEMAND_FORECAST"}, inplace=True)

    return df


def collect_wind_forecast(start_date, end_date, verbose=False):
    """
    Collect day-ahead wind generation forecasts from NESO API.

    These are forecasts for metered wind generation only (not including embedded).

    Args:
        start_date: Start date as string (YYYY-MM-DD)
        end_date: End date as string (YYYY-MM-DD)
        verbose: If True, print progress messages

    Returns:
        DataFrame with SETTLEMENT_DATE, SETTLEMENT_PERIOD, WIND_FORECAST, WIND_CAPACITY
    """
    url = "https://api.neso.energy/api/3/action/datastore_search_sql"
    resource_id = "7524ec65-f782-4258-aaf8-5b926c17b966"

    query = f"""
    SELECT *
    FROM "{resource_id}"
    WHERE "Datetime_GMT" >= '{start_date}' AND "Datetime_GMT" <= '{end_date}'
    ORDER BY "Datetime_GMT" ASC
    """

    params = {"sql": query}

    if verbose:
        print("Fetching wind forecast data from NESO API...")

    r = requests.get(url, params=params)
    if r.status_code != 200:
        raise Exception(f"Request failed with status {r.status_code}")

    wind_forecast = r.json()
    df = pd.DataFrame(wind_forecast["result"]["records"])

    # Drop unnecessary columns
    df.drop(
        columns=["_full_text", "_id", "Forecast_Timestamp", "Datetime_GMT"],
        inplace=True,
        errors="ignore",
    )

    # Rename to match conventions
    df.rename(
        columns={
            "Incentive_forecast": "WIND_FORECAST",
            "Settlement_period": "SETTLEMENT_PERIOD",
            "Date": "SETTLEMENT_DATE",
            "Capacity": "WIND_CAPACITY",
        },
        inplace=True,
    )

    # Convert date to proper format
    df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"]).dt.date

    return df


def collect_embedded_forecast(embedded_data_dir, years=None, verbose=False):
    """
    Collect embedded wind and solar forecasts from local CSV files.

    Args:
        embedded_data_dir: Directory containing NESO embedded forecast CSV files
        years: List of years to process (default: [2021, 2022, 2023, 2024])
        verbose: If True, print progress messages

    Returns:
        DataFrame with SETTLEMENT_DATE, SETTLEMENT_PERIOD, embedded forecasts
    """
    if years is None:
        years = [2021, 2022, 2023, 2024]

    dfs = []

    for year in years:
        load_path = os.path.join(
            embedded_data_dir, f"embedded_archive_{year}.csv"
        )

        if not os.path.exists(load_path):
            if verbose:
                print(f"Warning: Embedded data not found for year {year}")
            continue

        if verbose:
            print(f"Processing embedded forecasts for {year}...")

        df = pd.read_csv(load_path)

        # Combine separate date and time columns
        dates_str = pd.to_datetime(df["DATE_GMT"]).dt.date.astype("string")
        df["DATETIME_GMT"] = pd.to_datetime(
            dates_str + "T" + df["TIME_GMT"], format="ISO8601", utc=True
        )
        df.drop(columns=["DATE_GMT", "TIME_GMT"], inplace=True)
        df["Forecast_Datetime"] = pd.to_datetime(df["Forecast_Datetime"])
        df["SETTLEMENT_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"]).dt.date

        # Keep last forecast made on previous day before 09:00
        forecast_date = df["Forecast_Datetime"].dt.date
        target_date = df["SETTLEMENT_DATE"]
        forecast_hour = df["Forecast_Datetime"].dt.hour
        is_previous_day = (target_date - forecast_date) == pd.Timedelta(days=1)
        is_morning_forecast = (forecast_hour >= 0) & (forecast_hour < 9)
        df = df[is_previous_day & is_morning_forecast]

        # Remove duplicates, keeping last forecast before 09:00
        df = df.sort_values("Forecast_Datetime")
        df = df.drop_duplicates(subset="DATETIME_GMT", keep="last")
        df.reset_index(drop=True, inplace=True)

        # Drop unnecessary rows
        df.drop(columns=["Forecast_Datetime", "DATETIME_GMT"], inplace=True)

        dfs.append(df)

    if not dfs:
        raise ValueError("No embedded forecast data found")

    df_embedded = pd.concat(dfs, ignore_index=True)

    return df_embedded


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

    # Find dates with unexpected number of settlement periods
    daily_counts = df_clean.groupby(
        df_clean["SETTLEMENT_DATE"].dt.normalize()
    ).size()
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

        date_normalized = pd.to_datetime(date).normalize()
        day_mask = (
            df_clean["SETTLEMENT_DATE"].dt.normalize() == date_normalized
        )
        day_data = df_clean[day_mask].copy()

        if actual < expected:
            # Fill missing periods with interpolation
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
        expected = get_expected_periods(pd.Timestamp(date))
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
        description="Collect day-ahead forecast data from NESO API"
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2021-01-02",
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-01-01",
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--embedded-data-dir",
        type=str,
        default=None,
        help="Directory containing NESO embedded CSV files (default: data/raw/neso)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023, 2024],
        help="Years to process for embedded data (default: 2021 2022 2023 2024)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output CSV file path (default: data/processed/neso_forecasts.csv)",
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

    if args.embedded_data_dir is None:
        args.embedded_data_dir = os.path.join(
            project_dir, "data", "raw", "neso"
        )

    if args.output_path is None:
        args.output_path = os.path.join(
            project_dir, "data", "processed", "neso_forecasts.csv"
        )

    print("\n=== Collecting NESO Forecast Data ===")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Embedded data dir: {args.embedded_data_dir}")
    print(f"Output: {args.output_path}\n")

    # Collect demand forecasts
    print("Collecting demand forecast data...") if args.verbose else None
    df_demand = collect_demand_forecast(
        args.start_date, args.end_date, args.verbose
    )

    # Collect wind forecasts
    print("\nCollecting wind forecast data...") if args.verbose else None
    df_wind = collect_wind_forecast(
        args.start_date, args.end_date, args.verbose
    )

    # Collect embedded forecasts
    print("\nCollecting embedded forecast data...") if args.verbose else None
    df_embedded = collect_embedded_forecast(
        args.embedded_data_dir, years=args.years, verbose=args.verbose
    )

    # Merge datasets
    print("\nMerging datasets...") if args.verbose else None
    df_merged = pd.merge(
        df_demand,
        df_embedded,
        how="inner",
        on=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
    )
    df_merged = pd.merge(
        df_merged,
        df_wind,
        how="inner",
        on=["SETTLEMENT_DATE", "SETTLEMENT_PERIOD"],
    )

    # Filter date range
    print("\nFiltering to requested date range...") if args.verbose else None
    start_date_obj = datetime.datetime.strptime(
        args.start_date, "%Y-%m-%d"
    ).date()
    end_date_obj = datetime.datetime.strptime(args.end_date, "%Y-%m-%d").date()
    df_merged = df_merged[
        (df_merged["SETTLEMENT_DATE"] >= start_date_obj)
        & (df_merged["SETTLEMENT_DATE"] < end_date_obj)
    ]
    df_merged = df_merged.reset_index(drop=True)

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
