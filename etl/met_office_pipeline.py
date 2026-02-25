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
python etl/met_office_pipeline.py --years 2021 2022 2023 2024 --verbose
```
"""

import os
import argparse

import pandas as pd
import numpy as np


KNOT_TO_MS = 0.514  # 1 knot = 0.514 m/s


class WeatherProcessor:
    """Load and process Met Office weather observations for a single location."""

    def __init__(self, data_dir, location, years, verbose=False):
        self.data_dir = data_dir
        self.location = location
        self.years = years
        self.verbose = verbose

    def collect(self) -> pd.DataFrame | None:
        """Load and combine weather data for all years at this location."""
        dfs = []

        for year in self.years:
            df = self._load_year(year)
            if df is not None:
                dfs.append(df)

        if not dfs:
            return None

        df_location = pd.concat(dfs, ignore_index=True)
        df_location["location"] = self.location.capitalize()
        return df_location

    def _load_year(self, year) -> pd.DataFrame | None:
        """Load weather and solar irradiation data for one year."""
        base_dir = os.path.join(self.data_dir, self.location)
        weather_dir = os.path.join(base_dir, "weather")
        weather_path = os.path.join(
            weather_dir, f"hourly_weather_{self.location}_{year}.csv"
        )

        if not os.path.exists(weather_path):
            if self.verbose:
                print(
                    f"Warning: Weather data not found for {self.location} {year}"
                )
            return None

        if self.verbose:
            print(f"Processing {self.location} {year}...")

        df = pd.read_csv(weather_path, header=283, low_memory=False)
        df = df[
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
        is_mps = df["wind_speed_unit_id"].isin([0, 1])
        df["wind_speed"] = df["wind_speed"] * (
            is_mps + ~is_mps * KNOT_TO_MS
        )
        df.drop(columns="wind_speed_unit_id", inplace=True)

        # Clean up data
        df = df.iloc[:-1]  # Drop last row ('end of data' line)
        df["ob_time"] = pd.to_datetime(df["ob_time"])

        # Merge solar irradiation if available
        df = self._merge_irradiation(df, year)

        return df

    def _merge_irradiation(self, df: pd.DataFrame, year) -> pd.DataFrame:
        """Merge solar irradiation data for a given year, if available."""
        base_dir = os.path.join(self.data_dir, self.location)
        irradiation_dir = os.path.join(base_dir, "solar_irradiation")
        irradiation_path = os.path.join(
            irradiation_dir, f"solar_irradiation_{self.location}_{year}.csv"
        )

        if os.path.exists(irradiation_path):
            df_rad = pd.read_csv(
                irradiation_path, header=78, low_memory=False
            )
            df_rad = df_rad[["ob_end_time", "glbl_irad_amt"]].iloc[:-1]
            df_rad["ob_time"] = pd.to_datetime(df_rad["ob_end_time"])
            df_rad.drop(columns="ob_end_time", inplace=True)
            return df.merge(df_rad, on="ob_time", how="left")

        if self.verbose:
            print(
                f"""Warning: No radiation data
                found for {self.location} {year}"""
            )

        df["glbl_irad_amt"] = np.nan
        return df


class DatasetAssembler:
    """Combine weather data from multiple locations."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def merge(self, dataframes: list[pd.DataFrame]) -> pd.DataFrame:
        """Concatenate location DataFrames and sort by location and time."""
        dfs = [df for df in dataframes if df is not None]

        if not dfs:
            raise ValueError("No weather data found")

        df = pd.concat(dfs, ignore_index=True)
        df = df.sort_values(["location", "ob_time"]).reset_index(drop=True)
        return df

    def save(self, df, output_path):
        """Save the dataset to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"\nSaved {len(df)} records to {output_path}")
        print(
            f"""Time range: {df["ob_time"].min()}
        to {df["ob_time"].max()}"""
        )
        print(f"Locations: {', '.join(df['location'].unique())}")
        print(f"Columns: {', '.join(df.columns.tolist())}")


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

    processors = [
        WeatherProcessor(args.data_dir, loc, args.years, args.verbose)
        for loc in args.locations
    ]
    assembler = DatasetAssembler(args.verbose)

    df = assembler.merge([p.collect() for p in processors])
    assembler.save(df, args.output_path)


if __name__ == "__main__":
    main()
