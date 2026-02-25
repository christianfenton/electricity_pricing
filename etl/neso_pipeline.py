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
- TARGET_DATE: Target date for forecast
- TARGET_PERIOD: Target period for forecast
- ISSUE_DATE: Date when forecast was issued
- ISSUE_PERIOD: Period when forecast was issued
- DEMAND_FORECAST: Forecasted demand (MW)
- WIND_FORECAST: Forecasted metered wind generation (MW)
- WIND_CAPACITY: Wind capacity (MW)
- EMBEDDED_WIND_FORECAST: Forecasted embedded wind generation (MW)
- EMBEDDED_SOLAR_FORECAST: Forecasted embedded solar generation (MW)

```bash
python etl/neso_pipeline.py --start-date 2021-01-02 --end-date 2025-01-01 --verbose
```
"""

import os
import argparse
import datetime as dt

import requests
import pandas as pd

from utilities import expected_periods, timestamp_to_settlement


class DemandForecastProcessor:
    """Collect and interpolate day-ahead demand forecasts from NESO API."""

    def __init__(self, start_date, end_date, verbose=False):
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose

    def collect(self) -> pd.DataFrame:
        """Fetch demand forecasts and interpolate to half-hourly resolution."""
        df = self._fetch()
        df = self._standardise_columns(df)
        df = self._interpolate_to_half_hourly(df)
        return df

    def _fetch(self) -> pd.DataFrame:
        """Fetch demand forecast data from NESO API."""
        url = "https://api.neso.energy/api/3/action/datastore_search_sql"
        resource_id = "9847e7bb-986e-49be-8138-717b25933fbb"

        try:
            pd.to_datetime(self.start_date, format="%Y-%m-%d")
            pd.to_datetime(self.end_date, format="%Y-%m-%d")
        except ValueError as e:
            raise ValueError(
                f"""Invalid date format. Expected YYYY-MM-DD,
            got start_date='{self.start_date}', end_date='{self.end_date}'"""
            ) from e

        query = f"""
        SELECT *
        FROM "{resource_id}"
        WHERE "TARGETDATE" >= '{self.start_date}'
        AND "TARGETDATE" <= '{self.end_date}' AND "DAYSAHEAD" = 1
        ORDER BY "TARGETDATE" ASC
        """

        if self.verbose:
            print("Fetching demand forecast data from NESO API...")

        response = requests.get(url, params={"sql": query})
        if response.status_code != 200:
            raise Exception(f"Request failed with status {response.status_code}")

        data = response.json()
        df = pd.DataFrame(data["result"]["records"])
        df.drop(columns=["_full_text", "_id"], inplace=True, errors="ignore")

        return df

    def _standardise_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rename columns and localise timestamps."""
        df.rename(
            columns={
                "FORECAST_TIMESTAMP": "ISSUE_TS",
                "TARGETDATE": "TARGET_DATE",
                "FORECASTDEMAND": "DEMAND_FORECAST",
            },
            inplace=True,
        )

        df["ISSUE_TS"] = pd.to_datetime(df["ISSUE_TS"]).dt.tz_localize(
            "Europe/London",
            ambiguous=True,
            nonexistent="shift_forward",
        )

        return df

    @staticmethod
    def _interpolate_to_half_hourly(df: pd.DataFrame) -> pd.DataFrame:
        """
        Expand cardinal-point demand forecasts onto a half-hourly grid.

        Arguments:
            df: A pandas.DataFrame with columns:
                - ISSUE_TS
                - TARGET_DATE
                - DAYSAHEAD
                - CP_ST_TIME
                - CP_END_TIME
                - DEMAND_FORECAST

        Returns:
            df: A pandas.DataFrame with columns:
                - ISSUE_DATE
                - ISSUE_PERIOD
                - TARGET_DATE
                - TARGET_PERIOD
                - DEMAND_FORECAST

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
        """

        # Create time-stamps from date and time strings
        start = df["CP_ST_TIME"].astype(str).str.zfill(4).replace("2400", "0000")
        end = df["CP_END_TIME"].astype(str).str.zfill(4).replace("2400", "0000")

        df["CP_START"] = pd.to_datetime(
            df["TARGET_DATE"].astype(str) + " " + start, format="%Y-%m-%d %H%M"
        ).dt.tz_localize(
            "Europe/London",
            ambiguous=True,
            nonexistent="shift_forward",
        )

        df["CP_END"] = pd.to_datetime(
            df["TARGET_DATE"].astype(str) + " " + end, format="%Y-%m-%d %H%M"
        ).dt.tz_localize(
            "Europe/London",
            ambiguous=True,
            nonexistent="shift_forward",
        )

        df.loc[df["CP_ST_TIME"] == "2400", "CP_START"] += pd.Timedelta(days=1)
        df.loc[df["CP_END_TIME"] == "2400", "CP_END"] += pd.Timedelta(days=1)

        # Expand to get half-hourly resolution between CP_START and CP_END
        rows = []
        for _, r in df.iterrows():
            idx = pd.date_range(
                r["CP_START"],
                r["CP_END"] - pd.Timedelta(minutes=30),
                freq="30min",
                tz="Europe/London",
            )
            rows.append(
                pd.DataFrame(
                    {
                        "TARGET_TS": idx,
                        "DEMAND_FORECAST": r["DEMAND_FORECAST"],
                        "ISSUE_TS": r["ISSUE_TS"],
                    }
                )
            )

        expanded = pd.concat(rows, ignore_index=True)

        # Choose earliest-issued forecast per target half-hour
        expanded = (
            expanded.sort_values("ISSUE_TS")
            .drop_duplicates(subset="TARGET_TS", keep="first")
            .set_index("TARGET_TS")
        )

        # Reindex to half-hourly resolved grid
        # Build per-day ranges to correctly handle DST transitions
        first_day = expanded.index.min().normalize()
        last_day = expanded.index.max().normalize()
        day_starts = pd.date_range(first_day, last_day, freq="D", tz="Europe/London")
        daily_ranges = []
        for day in day_starts:
            n_periods = expected_periods(day)
            day_index = pd.date_range(
                day, periods=n_periods, freq="30min", tz="Europe/London"
            )
            daily_ranges.append(day_index)
        hh_index = daily_ranges[0].append(daily_ranges[1:])

        out = expanded.reindex(hh_index)

        # Interpolate the forecast for each day to get half-hourly resolution
        out["TARGET_DATE"] = out.index.date
        out = out.groupby("TARGET_DATE", group_keys=False).apply(
            DemandForecastProcessor._interp_day, include_groups=False
        )

        # Compute settlement (date, time) pairs
        issue = out["ISSUE_TS"].apply(timestamp_to_settlement)
        out["ISSUE_DATE"] = issue.str[0]
        out["ISSUE_PERIOD"] = issue.str[1]
        out["TARGET_DATE"] = out.index.date
        out["TARGET_PERIOD"] = out.groupby("TARGET_DATE").cumcount() + 1

        # Keep only desired columns and reset index
        out = out.reset_index(drop=True)[
            [
                "ISSUE_DATE",
                "ISSUE_PERIOD",
                "TARGET_DATE",
                "TARGET_PERIOD",
                "DEMAND_FORECAST",
            ]
        ]

        return out

    @staticmethod
    def _interp_day(df: pd.DataFrame) -> pd.DataFrame:
        df["DEMAND_FORECAST"] = df["DEMAND_FORECAST"].interpolate(
            method="time", limit_direction="both"
        )
        df["ISSUE_TS"] = df["ISSUE_TS"].bfill().ffill()
        return df


class WindForecastProcessor:
    """Collect day-ahead metered wind generation forecasts from NESO API."""

    def __init__(self, start_date, end_date, verbose=False):
        self.start_date = start_date
        self.end_date = end_date
        self.verbose = verbose

    def collect(self) -> pd.DataFrame:
        """Fetch wind forecasts and transform to standard format."""
        df = self._fetch()
        df = self._transform(df)
        return df

    def _fetch(self) -> pd.DataFrame:
        """Fetch wind forecast data from NESO API."""
        url = "https://api.neso.energy/api/3/action/datastore_search_sql"
        resource_id = "7524ec65-f782-4258-aaf8-5b926c17b966"

        try:
            start_dt = pd.to_datetime(self.start_date, format="%Y-%m-%d")
            end_dt = pd.to_datetime(self.end_date, format="%Y-%m-%d")
        except ValueError as e:
            raise ValueError(
                f"Invalid date format. Expected YYYY-MM-DD, got start_date='{self.start_date}', "
                f"end_date='{self.end_date}'"
            ) from e

        # API uses Datetime_GMT (UTC). To get all London periods for a date,
        # we need to start from 23:00 UTC the previous day (covers BST when London is UTC+1)
        query_start = (start_dt - pd.Timedelta(hours=1)).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        query_end = (end_dt + pd.Timedelta(days=1)).strftime("%Y-%m-%d %H:%M:%S")

        query = f"""
        SELECT *
        FROM "{resource_id}"
        WHERE "Datetime_GMT" >= '{query_start}' AND "Datetime_GMT" < '{query_end}'
        ORDER BY "Datetime_GMT" ASC
        """

        if self.verbose:
            print("Fetching wind forecast data from NESO API...")

        r = requests.get(url, params={"sql": query})
        if r.status_code != 200:
            raise Exception(f"Request failed with status {r.status_code}")

        wind_forecast = r.json()
        df = pd.DataFrame(wind_forecast["result"]["records"])

        return df

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply column renames, settlement conversion, and date filtering."""
        # Convert Datetime_GMT (UTC) to London time and calculate TARGET date/period
        df["TARGET_TIMESTAMP"] = pd.to_datetime(
            df["Datetime_GMT"], utc=True
        ).dt.tz_convert("Europe/London")
        target_settlements = df["TARGET_TIMESTAMP"].apply(timestamp_to_settlement)
        df["TARGET_DATE"] = target_settlements.apply(lambda x: x[0])
        df["TARGET_PERIOD"] = target_settlements.apply(lambda x: x[1])

        df.rename(
            columns={
                "Incentive_forecast": "WIND_FORECAST",
                "Capacity": "WIND_CAPACITY",
            },
            inplace=True,
        )

        # Convert forecast issue timestamp (UTC) to London time settlement
        df["ISSUE_TIMESTAMP"] = pd.to_datetime(
            df["Forecast_Timestamp"], utc=True
        ).dt.tz_convert("Europe/London")
        forecast_settlements = df["ISSUE_TIMESTAMP"].apply(timestamp_to_settlement)
        df["ISSUE_DATE"] = forecast_settlements.apply(lambda x: x[0])
        df["ISSUE_PERIOD"] = forecast_settlements.apply(lambda x: x[1])

        # Filter to only include the requested date range
        start_dt = pd.to_datetime(self.start_date, format="%Y-%m-%d")
        end_dt = pd.to_datetime(self.end_date, format="%Y-%m-%d")
        df = df[
            (df["TARGET_DATE"] >= start_dt.date())
            & (df["TARGET_DATE"] < end_dt.date())
        ]

        cols = [
            "ISSUE_DATE",
            "ISSUE_PERIOD",
            "TARGET_DATE",
            "TARGET_PERIOD",
            "WIND_FORECAST",
            "WIND_CAPACITY",
        ]
        df = df[cols]

        return df


class EmbeddedForecastProcessor:
    """Collect embedded wind and solar forecasts from local CSV files."""

    def __init__(self, embedded_data_dir, years=None, verbose=False):
        self.embedded_data_dir = embedded_data_dir
        self.years = years if years is not None else [2021, 2022, 2023, 2024]
        self.verbose = verbose

    def collect(self) -> pd.DataFrame:
        """Load and combine embedded forecast CSVs for all years."""
        dfs = []

        for year in self.years:
            df = self._load_year(year)
            if df is not None:
                df = self._filter_day_ahead(df)
                dfs.append(df)

        if not dfs:
            raise ValueError("No embedded forecast data found")

        return pd.concat(dfs, ignore_index=True)

    def _load_year(self, year) -> pd.DataFrame | None:
        """Load and parse one year's embedded forecast CSV."""
        load_path = os.path.join(
            self.embedded_data_dir, f"embedded_archive_{year}.csv"
        )

        if not os.path.exists(load_path):
            if self.verbose:
                print(f"Warning: Embedded data not found for year {year}")
            return None

        if self.verbose:
            print(f"Processing embedded forecasts for {year}...")

        df = pd.read_csv(load_path)

        # Combine separate date and time columns
        dates_str = pd.to_datetime(df["DATE_GMT"]).dt.date.astype("string")
        df["DATETIME_GMT"] = pd.to_datetime(
            dates_str + "T" + df["TIME_GMT"], format="ISO8601", utc=True
        )
        df.drop(columns=["DATE_GMT", "TIME_GMT"], inplace=True)

        df["ISSUE_DATETIME"] = pd.to_datetime(df["Forecast_Datetime"])
        df["TARGET_DATE"] = pd.to_datetime(df["SETTLEMENT_DATE"]).dt.date
        df["TARGET_PERIOD"] = df["SETTLEMENT_PERIOD"]

        return df

    def _filter_day_ahead(self, df: pd.DataFrame) -> pd.DataFrame:
        """Keep last forecast made on previous day before 09:00."""
        issue_date = df["ISSUE_DATETIME"].dt.date
        issue_hour = df["ISSUE_DATETIME"].dt.hour
        target_date = df["TARGET_DATE"]
        is_previous_day = (target_date - issue_date) == pd.Timedelta(days=1)
        is_morning_forecast = (issue_hour >= 0) & (issue_hour < 9)
        df = df[is_previous_day & is_morning_forecast]

        # Remove duplicates, keeping last forecast before 09:00
        df = df.sort_values("ISSUE_DATETIME")
        df = df.drop_duplicates(subset="DATETIME_GMT", keep="last")
        df.reset_index(drop=True, inplace=True)

        # Convert forecast datetime to settlement date/period
        forecast_settlements = df["ISSUE_DATETIME"].apply(
            timestamp_to_settlement
        )
        df["ISSUE_DATE"] = forecast_settlements.apply(lambda x: x[0])
        df["ISSUE_PERIOD"] = forecast_settlements.apply(lambda x: x[1])

        columns = [
            "TARGET_DATE",
            "TARGET_PERIOD",
            "ISSUE_DATE",
            "ISSUE_PERIOD",
            "EMBEDDED_WIND_FORECAST",
            "EMBEDDED_SOLAR_FORECAST",
        ]
        df = df[columns]

        return df


class DatasetAssembler:
    """Merge, clean, validate, and save the combined forecast dataset."""

    def __init__(self, verbose=False):
        self.verbose = verbose

    def merge(self, df_demand, df_wind, df_embedded) -> pd.DataFrame:
        """Merge demand, wind, and embedded forecasts into a single DataFrame."""
        if self.verbose:
            print("\nMerging datasets...")

        # Merge demand and wind
        df = pd.merge(
            df_demand,
            df_wind,
            how="inner",
            on=["ISSUE_DATE", "TARGET_DATE", "TARGET_PERIOD"],
        )
        df["ISSUE_PERIOD"] = df[["ISSUE_PERIOD_x", "ISSUE_PERIOD_y"]].max(axis=1)
        df.drop(columns=["ISSUE_PERIOD_x", "ISSUE_PERIOD_y"], inplace=True)

        # Merge with embedded
        df = pd.merge(
            df,
            df_embedded,
            how="inner",
            on=["TARGET_DATE", "TARGET_PERIOD", "ISSUE_DATE"],
        )
        df["ISSUE_PERIOD"] = df[["ISSUE_PERIOD_x", "ISSUE_PERIOD_y"]].max(axis=1)
        df.drop(columns=["ISSUE_PERIOD_x", "ISSUE_PERIOD_y"], inplace=True)

        return df

    def filter_date_range(self, df, start_date, end_date) -> pd.DataFrame:
        """Filter to only include the requested date range."""
        if self.verbose:
            print("\nFiltering to requested date range...")

        start_date_obj = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date_obj = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        df = df[
            (df["TARGET_DATE"] >= start_date_obj)
            & (df["TARGET_DATE"] < end_date_obj)
        ]
        return df.reset_index(drop=True)

    def clean(self, df) -> pd.DataFrame:
        """Handle missing/excess periods and remove duplicates."""
        if self.verbose:
            print("\nCleaning dataset...")

        df_clean = df.copy()

        df_clean["TARGET_DATE"] = pd.to_datetime(df_clean["TARGET_DATE"])

        # Remove duplicates
        n_before = len(df_clean)
        df_clean = df_clean.drop_duplicates(
            subset=["TARGET_DATE", "TARGET_PERIOD"]
        )
        df_clean = df_clean.reset_index(drop=True)
        n_after = len(df_clean)
        if self.verbose and n_before > n_after:
            print(f"Removed {n_before - n_after} duplicate rows")

        # Find dates with unexpected number of settlement periods
        daily_counts = df_clean.groupby(
            df_clean["TARGET_DATE"].dt.normalize()
        ).size()
        irregular_days = []
        for date, count in daily_counts.items():
            expected = expected_periods(pd.Timestamp(date))
            if count != expected:
                irregular_days.append((date, count, expected))

        if self.verbose:
            print(f"Found {len(irregular_days)} days needing cleaning")

        # Clean each irregular date
        for date, actual, expected in irregular_days:
            if self.verbose:
                print(f"  Cleaning {date.date()}: {actual} --> {expected} periods")

            date_normalized = pd.to_datetime(date).normalize()
            day_mask = df_clean["TARGET_DATE"].dt.normalize() == date_normalized
            day_data = df_clean[day_mask].copy()

            if actual < expected:
                template = pd.DataFrame(
                    {
                        "TARGET_DATE": [date_normalized] * expected,
                        "TARGET_PERIOD": range(1, expected + 1),
                    }
                )
                day_data_filled = template.merge(
                    day_data,
                    on=["TARGET_DATE", "TARGET_PERIOD"],
                    how="left",
                )
                day_data_filled = day_data_filled.ffill()
                df_clean = pd.concat(
                    [df_clean[~day_mask], day_data_filled], ignore_index=True
                )
            elif actual > expected:
                day_data = day_data.sort_values("TARGET_PERIOD").iloc[:expected]
                df_clean = pd.concat(
                    [df_clean[~day_mask], day_data], ignore_index=True
                )

        df_clean = df_clean.sort_values(
            ["TARGET_DATE", "TARGET_PERIOD"]
        ).reset_index(drop=True)

        return df_clean

    def validate(self, df) -> bool:
        """Validate that dataset has correct settlement periods for all dates."""
        if self.verbose:
            print("\nValidating dataset...")

        errors = []

        df_temp = df.copy()
        df_temp["TARGET_DATE"] = pd.to_datetime(df_temp["TARGET_DATE"])

        # Check for duplicates
        duplicates = df_temp.duplicated(
            subset=["TARGET_DATE", "TARGET_PERIOD"], keep=False
        )
        if duplicates.any():
            n_duplicates = duplicates.sum()
            errors.append(
                f"Found {n_duplicates} duplicate (date, period) combinations"
            )

        # Check day lengths
        daily_counts = df_temp.groupby(
            df_temp["TARGET_DATE"].dt.normalize()
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
            if self.verbose:
                for date, actual, expected in irregular_days[:5]:
                    errors.append(
                        f"  - {date.date()}: {actual} periods (expected {expected})"
                    )

        if len(errors) > 0:
            error_msg = "Validation failed:\n" + "\n".join(errors)
            raise ValueError(error_msg)

        if self.verbose:
            print("All validation checks passed")

        return True

    def save(self, df, output_path):
        """Save the dataset to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)

        print(f"\nSaved {len(df)} records to {output_path}")
        print(
            f"""Date range: {df["TARGET_DATE"].min()}
        to {df["TARGET_DATE"].max()}"""
        )
        print(f"Columns: {', '.join(df.columns.tolist())}")


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

    demand = DemandForecastProcessor(args.start_date, args.end_date, args.verbose)
    wind = WindForecastProcessor(args.start_date, args.end_date, args.verbose)
    embedded = EmbeddedForecastProcessor(
        args.embedded_data_dir, args.years, args.verbose
    )
    assembler = DatasetAssembler(args.verbose)

    df = assembler.merge(demand.collect(), wind.collect(), embedded.collect())
    df = assembler.filter_date_range(df, args.start_date, args.end_date)
    df = assembler.clean(df)
    assembler.validate(df)
    assembler.save(df, args.output_path)


if __name__ == "__main__":
    main()
