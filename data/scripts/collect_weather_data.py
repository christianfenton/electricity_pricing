"""
Collect weather data from Met Office CSV files.

This script processes Met Office hourly weather observations and solar irradiation
data from locally stored CSV files. Data is collected from multiple weather stations
across the UK.

Input files expected:
- data/raw/met_office/{location}/weather/hourly_weather_{location}_{year}.csv
- data/raw/met_office/{location}/solar_irradiation/solar_irradiation_{location}_{year}.csv

Output columns:
- ob_time: Observation timestamp (datetime)
- location: Weather station location (Heathrow, Crosby, Dyce)
- wind_speed: Wind speed (m/s)
- wind_direction: Wind direction (true degrees)
- visibility: Visibility (decametres)
- air_temperature: Air temperature (°C)
- glbl_irad_amt: Global solar irradiation (KJ/m²)

Note: This data uses hourly timestamps (ob_time), not settlement periods.

Usage:
```bash
python scripts/collect_weather_data.py --years 2021 2022 2023 2024 --verbose
```
"""

import os
import argparse

import pandas as pd
import numpy as np


def collect_weather_data(data_dir, locations=None, years=None, verbose=False):
    """
    Load and process weather data from Met Office CSV files.

    Args:
        data_dir: Directory containing Met Office data
        locations: List of location names (default: ["heathrow", "crosby", "dyce"])
        years: List of years to process (default: [2021, 2022, 2023, 2024])
        verbose: If True, print progress messages

    Returns:
        DataFrame with processed weather data from all locations and years
    """
    if locations is None:
        locations = ["heathrow", "crosby", "dyce"]
    if years is None:
        years = [2021, 2022, 2023, 2024]

    KNOT_TO_MS = 0.514  # 1 knot = 0.514 m/s
    dfs = []

    for location in locations:
        base_dir = os.path.join(data_dir, location)
        weather_dir = os.path.join(base_dir, "weather")
        irradiation_dir = os.path.join(base_dir, "solar_irradiation")
        dfs_location = []

        for year in years:
            weather_path = os.path.join(
                weather_dir, f"hourly_weather_{location}_{year}.csv"
            )

            if not os.path.exists(weather_path):
                if verbose:
                    print(
                        f"Warning: Weather data not found for {location} {year}"
                    )
                continue

            if verbose:
                print(f"Processing {location} {year}...")

            df_hw = pd.read_csv(weather_path, header=283, low_memory=False)
            df_hw = df_hw[
                [
                    "ob_time",
                    "wind_speed_unit_id",
                    "wind_speed",
                    "wind_direction",
                    "visibility",
                    "air_temperature",
                ]
            ]

            # Convert wind speed to m/s
            is_mps = df_hw["wind_speed_unit_id"].isin([0, 1])
            df_hw["wind_speed"] = df_hw["wind_speed"] * (
                is_mps + ~is_mps * KNOT_TO_MS
            )
            df_hw.drop(columns="wind_speed_unit_id", inplace=True)

            # Clean up data
            df_hw = df_hw.iloc[:-1]  # Drop last row ('end of data' line)
            df_hw["ob_time"] = pd.to_datetime(df_hw["ob_time"])

            # Load solar irradiation data if available
            irradiation_path = os.path.join(
                irradiation_dir, f"solar_irradiation_{location}_{year}.csv"
            )
            if os.path.exists(irradiation_path):
                df_rad = pd.read_csv(
                    irradiation_path, header=78, low_memory=False
                )
                df_rad = df_rad[["ob_end_time", "glbl_irad_amt"]].iloc[:-1]
                df_rad["ob_time"] = pd.to_datetime(df_rad["ob_end_time"])
                df_rad.drop(columns="ob_end_time", inplace=True)
                df_merged = df_hw.merge(df_rad, on="ob_time", how="left")
            else:

                if verbose:
                    print(
                        f"""Warning: No radiation data 
                        found for {location} {year}"""
                    )

                df_hw["glbl_irad_amt"] = np.nan
                df_merged = df_hw

            dfs_location.append(df_merged)

        if dfs_location:
            df_location = pd.concat(dfs_location, ignore_index=True)
            df_location["location"] = location.capitalize()
            dfs.append(df_location)

    if not dfs:
        raise ValueError("No weather data found")

    df_weather = pd.concat(dfs, ignore_index=True)

    # Sort by location and time
    df_weather = df_weather.sort_values(
        ["location", "ob_time"]
    ).reset_index(drop=True)

    return df_weather


def main():
    parser = argparse.ArgumentParser(
        description="Collect weather data from Met Office CSV files"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Directory containing Met Office data (default: data/raw/met_office)",
    )
    parser.add_argument(
        "--locations",
        type=str,
        nargs="+",
        default=["heathrow", "crosby", "dyce"],
        help="Weather station locations (default: heathrow crosby dyce)",
    )
    parser.add_argument(
        "--years",
        type=int,
        nargs="+",
        default=[2021, 2022, 2023, 2024],
        help="Years to process (default: 2021 2022 2023 2024)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output CSV file path (default: data/processed/weather_data.csv)",
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

    if args.data_dir is None:
        args.data_dir = os.path.join(project_dir, "data", "raw", "met_office")

    if args.output_path is None:
        args.output_path = os.path.join(
            project_dir, "data", "processed", "weather_data.csv"
        )

    print("\n=== Collecting Weather Data ===")
    print(f"Data directory: {args.data_dir}")
    print(f"Locations: {', '.join(args.locations)}")
    print(f"Years: {', '.join(map(str, args.years))}")
    print(f"Output: {args.output_path}\n")

    # Collect weather data
    df_weather = collect_weather_data(
        args.data_dir,
        locations=args.locations,
        years=args.years,
        verbose=args.verbose,
    )

    # Save to CSV
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    df_weather.to_csv(args.output_path, index=False)

    print(f"\nSaved {len(df_weather)} records to {args.output_path}")
    print(
        f"""Time range: {df_weather["ob_time"].min()} 
        to {df_weather["ob_time"].max()}"""
    )
    print(f"Locations: {', '.join(df_weather['location'].unique())}")
    print(f"Columns: {', '.join(df_weather.columns.tolist())}")


if __name__ == "__main__":
    main()
