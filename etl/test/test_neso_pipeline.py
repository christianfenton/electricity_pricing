import datetime as dt
import pandas as pd
import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from neso_pipeline import (
    DemandForecastProcessor,
    WindForecastProcessor,
    EmbeddedForecastProcessor,
    DatasetAssembler,
)


class TestDemandInterpolation:

    # Cardinal-point segments covering a full day
    SEGMENTS = [
        (0, 600, 20000),       # overnight
        (600, 900, 25000),     # morning ramp
        (900, 1600, 28000),    # daytime
        (1600, 1900, 33000),   # evening peak
        (1900, 2400, 27000),   # evening decline
    ]

    OUTPUT_COLUMNS = {
        "ISSUE_DATE", "ISSUE_PERIOD", "TARGET_DATE", "TARGET_PERIOD",
        "DEMAND_FORECAST",
    }

    @staticmethod
    def _make_cardinal_points(target_date, issue_ts, segments):
        """Build a DataFrame of cardinal-point rows for _interpolate_to_half_hourly.

        Args:
            target_date: Date string (YYYY-MM-DD).
            issue_ts: Timezone-aware Timestamp for the forecast issue time.
            segments: List of (start_hhmm, end_hhmm, demand_mw) tuples.
                Times are integers in HHMM format (e.g. 600 for 06:00).
                Use 2400 for midnight at end-of-day.
        """
        rows = []
        for st, end, demand in segments:
            rows.append(
                {
                    "ISSUE_TS": issue_ts,
                    "TARGET_DATE": target_date,
                    "DAYSAHEAD": 1,
                    "CP_ST_TIME": st,
                    "CP_END_TIME": end,
                    "DEMAND_FORECAST": demand,
                }
            )
        return pd.DataFrame(rows)

    def test_normal_day_produces_48_periods(self):
        """A normal (non-DST) day should produce exactly 48 half-hourly periods."""
        issue_ts = pd.Timestamp("2024-06-14 08:00", tz="Europe/London")
        df = self._make_cardinal_points("2024-06-15", issue_ts, self.SEGMENTS)

        result = DemandForecastProcessor.interpolate(df)

        assert set(result.columns) == self.OUTPUT_COLUMNS
        assert len(result) == 48
        assert list(result["TARGET_PERIOD"]) == list(range(1, 49))
        assert (result["TARGET_DATE"] == dt.date(2024, 6, 15)).all()

    def test_autumn_dst_produces_50_periods(self):
        """Autumn DST (clocks fall back) should produce 50 half-hourly periods."""
        issue_ts = pd.Timestamp("2024-10-26 08:00", tz="Europe/London")
        df = self._make_cardinal_points("2024-10-27", issue_ts, self.SEGMENTS)

        result = DemandForecastProcessor.interpolate(df)

        assert len(result) == 50
        assert list(result["TARGET_PERIOD"]) == list(range(1, 51))
        assert (result["TARGET_DATE"] == dt.date(2024, 10, 27)).all()

    def test_spring_dst_produces_46_periods(self):
        """Spring DST (clocks spring forward) should produce 46 half-hourly periods."""
        issue_ts = pd.Timestamp("2024-03-30 08:00", tz="Europe/London")
        df = self._make_cardinal_points("2024-03-31", issue_ts, self.SEGMENTS)

        result = DemandForecastProcessor.interpolate(df)

        assert len(result) == 46
        assert list(result["TARGET_PERIOD"]) == list(range(1, 47))
        assert (result["TARGET_DATE"] == dt.date(2024, 3, 31)).all()

    def test_no_nan_on_normal_day(self):
        """No NaN values should appear for a normal day."""
        issue_ts = pd.Timestamp("2024-06-14 08:00", tz="Europe/London")
        df = self._make_cardinal_points("2024-06-15", issue_ts, self.SEGMENTS)

        result = DemandForecastProcessor.interpolate(df)

        assert result.isna().sum().sum() == 0

    def test_no_nan_on_autumn_dst(self):
        """No NaN values should appear on an Autumn DST transition day."""
        issue_ts = pd.Timestamp("2024-10-26 08:00", tz="Europe/London")
        df = self._make_cardinal_points("2024-10-27", issue_ts, self.SEGMENTS)

        result = DemandForecastProcessor.interpolate(df)

        assert result.isna().sum().sum() == 0

    def test_no_nan_on_spring_dst(self):
        """No NaN values should appear on a Spring DST transition day."""
        issue_ts = pd.Timestamp("2024-03-30 08:00", tz="Europe/London")
        df = self._make_cardinal_points("2024-03-31", issue_ts, self.SEGMENTS)

        result = DemandForecastProcessor.interpolate(df)

        assert result.isna().sum().sum() == 0

    def test_multi_day_span_with_dst(self):
        """A span including a DST transition should produce correct period counts."""
        issue_ts = pd.Timestamp("2024-10-25 08:00", tz="Europe/London")
        dfs = []
        for date in ["2024-10-26", "2024-10-27", "2024-10-28"]:
            dfs.append(self._make_cardinal_points(date, issue_ts, self.SEGMENTS))
        df = pd.concat(dfs, ignore_index=True)

        result = DemandForecastProcessor.interpolate(df)

        counts = result.groupby("TARGET_DATE").size()
        assert counts[dt.date(2024, 10, 26)] == 48
        assert counts[dt.date(2024, 10, 27)] == 50
        assert counts[dt.date(2024, 10, 28)] == 48
        assert result.isna().sum().sum() == 0

    def test_issue_date_populated(self):
        """ISSUE_DATE and ISSUE_PERIOD should be populated from the issue timestamp."""
        issue_ts = pd.Timestamp("2024-06-14 08:00", tz="Europe/London")
        df = self._make_cardinal_points("2024-06-15", issue_ts, self.SEGMENTS)

        result = DemandForecastProcessor.interpolate(df)

        assert (result["ISSUE_DATE"] == dt.date(2024, 6, 14)).all()
        assert (result["ISSUE_PERIOD"] == 17).all()  # 08:00 = period 17

    def test_demand_values_at_midpoints(self):
        """CP demand values should be exact at the midpoint of each window."""
        issue_ts = pd.Timestamp("2024-06-14 08:00", tz="Europe/London")
        df = self._make_cardinal_points("2024-06-15", issue_ts, self.SEGMENTS)

        result = DemandForecastProcessor.interpolate(df)

        midpoint_periods = {7: 20000, 16: 25000, 26: 28000, 36: 33000, 44: 27000}
        for period, expected_demand in midpoint_periods.items():
            actual = result.loc[
                result["TARGET_PERIOD"] == period, "DEMAND_FORECAST"
            ].iloc[0]
            assert actual == expected_demand, (
                f"Period {period}: expected {expected_demand}, got {actual}"
            )

    def test_demand_curve_smooth_between_cps(self):
        """Values between CPs should transition smoothly (no step jumps)."""
        issue_ts = pd.Timestamp("2024-06-14 08:00", tz="Europe/London")
        df = self._make_cardinal_points("2024-06-15", issue_ts, self.SEGMENTS)

        result = DemandForecastProcessor.interpolate(df)
        demand = result["DEMAND_FORECAST"].values

        # Between overnight midpoint (P7, 20000) and morning midpoint (P16, 25000),
        # PCHIP should produce values strictly between the two CP values
        transition = demand[7:15]  # periods 8-15 (between P7 and P16)
        assert all(v > 20000 for v in transition), "Morning ramp should be above overnight"
        assert all(v < 25000 for v in transition), "Morning ramp should be below morning CP"

    def test_2400_fixed_cp_does_not_crash(self):
        """A Fixed CP at 2400 (midnight boundary) should not crash PCHIP."""
        issue_ts = pd.Timestamp("2024-06-14 08:00", tz="Europe/London")
        segments_with_2400 = [
            (0, 600, 20000),
            (600, 900, 25000),
            (900, 1600, 28000),
            (1600, 1900, 33000),
            (1900, 2400, 27000),
            (2400, 2400, 22000),   # Fixed CP at midnight boundary
        ]
        df = self._make_cardinal_points("2024-06-15", issue_ts, segments_with_2400)

        result = DemandForecastProcessor.interpolate(df)

        assert len(result) == 48
        assert (result["TARGET_DATE"] == dt.date(2024, 6, 15)).all()
        assert result["DEMAND_FORECAST"].isna().sum() == 0

    def test_fixed_cardinal_point(self):
        """A Fixed CP (CP_START == CP_END) should not crash and should be interpolated through."""
        issue_ts = pd.Timestamp("2024-06-14 08:00", tz="Europe/London")
        segments_with_fixed = [
            (0, 600, 20000),
            (600, 600, 24000),     # Fixed CP at 06:00
            (600, 900, 25000),
            (900, 1600, 28000),
            (1600, 1900, 33000),
            (1900, 2400, 27000),
        ]
        df = self._make_cardinal_points("2024-06-15", issue_ts, segments_with_fixed)

        result = DemandForecastProcessor.interpolate(df)

        assert len(result) == 48
        assert result.isna().sum().sum() == 0


class TestParseCPTimestamps:
    """Test DemandForecastProcessor._parse_cp_timestamps."""

    @staticmethod
    def _make_row(target_date, st_time, end_time):
        return pd.DataFrame([{
            "TARGET_DATE": target_date,
            "CP_ST_TIME": st_time,
            "CP_END_TIME": end_time,
            "DEMAND_FORECAST": 25000,
            "ISSUE_TS": pd.Timestamp("2024-06-14 08:00", tz="Europe/London"),
        }])

    def test_normal_timestamps(self):
        """HHMM ints (0, 600, 1600) produce correct tz-aware timestamps."""
        df = self._make_row("2024-06-15", 0, 600)
        result = DemandForecastProcessor.parse_timestamps(df)

        assert result["CP_START"].iloc[0] == pd.Timestamp("2024-06-15 00:00", tz="Europe/London")
        assert result["CP_END"].iloc[0] == pd.Timestamp("2024-06-15 06:00", tz="Europe/London")

    def test_2400_becomes_next_day_midnight(self):
        """CP_END_TIME=2400 should become next day 00:00."""
        df = self._make_row("2024-06-15", 1900, 2400)
        result = DemandForecastProcessor.parse_timestamps(df)

        assert result["CP_END"].iloc[0] == pd.Timestamp("2024-06-16 00:00", tz="Europe/London")

    def test_fixed_cp_start_equals_end(self):
        """CP_ST_TIME == CP_END_TIME produces CP_START == CP_END."""
        df = self._make_row("2024-06-15", 600, 600)
        result = DemandForecastProcessor.parse_timestamps(df)

        assert result["CP_START"].iloc[0] == result["CP_END"].iloc[0]
        assert result["CP_START"].iloc[0] == pd.Timestamp("2024-06-15 06:00", tz="Europe/London")


class TestBuildHalfHourlyGrid:
    """Test DemandForecastProcessor._build_half_hourly_grid."""

    def test_normal_day_48_slots(self):
        """A single normal day should produce a 48-entry index."""
        grid = DemandForecastProcessor._build_half_hourly_grid(
            [dt.date(2024, 6, 15)]
        )

        assert len(grid) == 48

    def test_autumn_dst_50_slots(self):
        """Autumn DST day (Oct 27 2024) should produce 50 entries."""
        grid = DemandForecastProcessor._build_half_hourly_grid(
            [dt.date(2024, 10, 27)]
        )

        assert len(grid) == 50

    def test_spring_dst_46_slots(self):
        """Spring DST day (Mar 31 2024) should produce 46 entries."""
        grid = DemandForecastProcessor._build_half_hourly_grid(
            [dt.date(2024, 3, 31)]
        )

        assert len(grid) == 46

    def test_multi_day_grid(self):
        """A 3-day span including DST should produce correct total entries."""
        grid = DemandForecastProcessor._build_half_hourly_grid(
            [dt.date(2024, 10, 26), dt.date(2024, 10, 27), dt.date(2024, 10, 28)]
        )

        # Oct 26 (48) + Oct 27 DST (50) + Oct 28 (48) = 146
        assert len(grid) == 146


class TestWindProcessor:
    """Test WindForecastProcessor column renames and settlement conversion."""

    @staticmethod
    def _make_raw_wind(n_periods=3):
        """Build a DataFrame mimicking raw wind API response."""
        rows = []
        base = pd.Timestamp("2024-06-14 23:00:00", tz="UTC")
        for i in range(n_periods):
            rows.append({
                "Datetime_GMT": (base + pd.Timedelta(minutes=30 * i)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
                "Incentive_forecast": 3000.0 + i * 10,
                "Capacity": 15000.0,
                "Forecast_Timestamp": "2024-06-14T08:00:00Z",
            })
        return pd.DataFrame(rows)

    def test_output_columns(self):
        """Transform should produce the expected output columns."""
        proc = WindForecastProcessor.__new__(WindForecastProcessor)
        proc.start_date = "2024-06-14"
        proc.end_date = "2024-06-16"

        df = self._make_raw_wind()
        result = proc._transform(df)

        expected = {
            "ISSUE_DATE", "ISSUE_PERIOD", "TARGET_DATE", "TARGET_PERIOD",
            "WIND_FORECAST", "WIND_CAPACITY",
        }
        assert set(result.columns) == expected

    def test_wind_forecast_renamed(self):
        """Incentive_forecast should be renamed to WIND_FORECAST."""
        proc = WindForecastProcessor.__new__(WindForecastProcessor)
        proc.start_date = "2024-06-14"
        proc.end_date = "2024-06-16"

        df = self._make_raw_wind(n_periods=1)
        result = proc._transform(df)

        assert result["WIND_FORECAST"].iloc[0] == 3000.0

    def test_target_date_is_date_object(self):
        """TARGET_DATE should be a Python date object."""
        proc = WindForecastProcessor.__new__(WindForecastProcessor)
        proc.start_date = "2024-06-14"
        proc.end_date = "2024-06-16"

        df = self._make_raw_wind(n_periods=1)
        result = proc._transform(df)

        assert isinstance(result["TARGET_DATE"].iloc[0], dt.date)


class TestEmbeddedForecastProcessor:
    """Test EmbeddedForecastProcessor._filter_day_ahead logic."""

    @staticmethod
    def _make_embedded_df():
        """Build a DataFrame mimicking loaded embedded forecast data."""
        return pd.DataFrame({
            "DATETIME_GMT": pd.to_datetime([
                "2024-06-15 00:00", "2024-06-15 00:30",
                "2024-06-15 00:00", "2024-06-15 00:30",
                "2024-06-15 00:00",
            ], utc=True),
            "ISSUE_DATETIME": pd.to_datetime([
                "2024-06-14 06:00", "2024-06-14 06:00",   # day-ahead, morning
                "2024-06-14 08:30", "2024-06-14 08:30",   # day-ahead, morning (later)
                "2024-06-15 06:00",                        # same-day (not day-ahead)
            ]),
            "TARGET_DATE": [
                dt.date(2024, 6, 15), dt.date(2024, 6, 15),
                dt.date(2024, 6, 15), dt.date(2024, 6, 15),
                dt.date(2024, 6, 15),
            ],
            "TARGET_PERIOD": [1, 2, 1, 2, 1],
            "EMBEDDED_WIND_FORECAST": [100, 110, 120, 130, 999],
            "EMBEDDED_SOLAR_FORECAST": [50, 55, 60, 65, 999],
            "SETTLEMENT_DATE": ["2024-06-15"] * 5,
            "SETTLEMENT_PERIOD": [1, 2, 1, 2, 1],
            "Forecast_Datetime": [
                "2024-06-14 06:00", "2024-06-14 06:00",
                "2024-06-14 08:30", "2024-06-14 08:30",
                "2024-06-15 06:00",
            ],
        })

    def test_keeps_only_previous_day_morning(self):
        """Only forecasts issued on the previous day before 09:00 should survive."""
        proc = EmbeddedForecastProcessor.__new__(EmbeddedForecastProcessor)
        df = self._make_embedded_df()
        result = proc._filter_day_ahead(df)

        # Same-day forecast (2024-06-15 06:00 for target 2024-06-15) should be excluded
        assert len(result) == 2  # 2 unique half-hours

    def test_keeps_latest_forecast(self):
        """When multiple morning forecasts exist, keep the latest one."""
        proc = EmbeddedForecastProcessor.__new__(EmbeddedForecastProcessor)
        df = self._make_embedded_df()
        result = proc._filter_day_ahead(df)

        # The 08:30 forecast (120, 130) should win over the 06:00 forecast (100, 110)
        assert result["EMBEDDED_WIND_FORECAST"].iloc[0] == 120
        assert result["EMBEDDED_WIND_FORECAST"].iloc[1] == 130

    def test_output_columns(self):
        """Output should contain exactly the expected columns."""
        proc = EmbeddedForecastProcessor.__new__(EmbeddedForecastProcessor)
        df = self._make_embedded_df()
        result = proc._filter_day_ahead(df)

        expected = {
            "TARGET_DATE", "TARGET_PERIOD",
            "ISSUE_DATE", "ISSUE_PERIOD",
            "EMBEDDED_WIND_FORECAST", "EMBEDDED_SOLAR_FORECAST",
        }
        assert set(result.columns) == expected


class TestNESODatasetAssembler:
    """Test NESO DatasetAssembler merge, clean, and validate logic."""

    @staticmethod
    def _make_forecast_df(date, periods, columns):
        """Build a simple forecast DataFrame."""
        data = {
            "ISSUE_DATE": [dt.date(2024, 6, 14)] * periods,
            "ISSUE_PERIOD": [17] * periods,
            "TARGET_DATE": [date] * periods,
            "TARGET_PERIOD": list(range(1, periods + 1)),
        }
        for col, val in columns.items():
            data[col] = [val] * periods
        return pd.DataFrame(data)

    def test_merge_combines_all_sources(self):
        """Merge should join demand, wind, and embedded on shared keys."""
        date = dt.date(2024, 6, 15)
        n = 48
        df_demand = self._make_forecast_df(date, n, {"DEMAND_FORECAST": 25000})
        df_wind = self._make_forecast_df(date, n, {"WIND_FORECAST": 3000, "WIND_CAPACITY": 15000})
        df_embedded = self._make_forecast_df(date, n, {
            "EMBEDDED_WIND_FORECAST": 500,
            "EMBEDDED_SOLAR_FORECAST": 200,
        })

        assembler = DatasetAssembler()
        result = assembler.merge(df_demand, df_wind, df_embedded)

        assert len(result) == n
        assert "DEMAND_FORECAST" in result.columns
        assert "WIND_FORECAST" in result.columns
        assert "EMBEDDED_SOLAR_FORECAST" in result.columns

    def test_filter_date_range(self):
        """filter_date_range should exclude dates outside the range."""
        dates = [dt.date(2024, 6, 14), dt.date(2024, 6, 15), dt.date(2024, 6, 16)]
        rows = []
        for d in dates:
            for sp in range(1, 49):
                rows.append({
                    "TARGET_DATE": d,
                    "TARGET_PERIOD": sp,
                    "VALUE": 100,
                })
        df = pd.DataFrame(rows)

        assembler = DatasetAssembler()
        result = assembler.filter_date_range(df, "2024-06-15", "2024-06-16")

        assert set(result["TARGET_DATE"].unique()) == {dt.date(2024, 6, 15)}

    def test_clean_removes_duplicates(self):
        """Clean should drop duplicate (date, period) rows."""
        df = pd.DataFrame({
            "TARGET_DATE": ["2024-06-15"] * 49,
            "TARGET_PERIOD": list(range(1, 49)) + [1],
            "VALUE": [100.0] * 49,
        })
        assembler = DatasetAssembler()
        result = assembler.clean(df)

        assert len(result) == 48

    def test_validate_passes_for_correct_data(self):
        """Validate should return True for a well-formed dataset."""
        df = pd.DataFrame({
            "TARGET_DATE": ["2024-06-15"] * 48,
            "TARGET_PERIOD": list(range(1, 49)),
        })
        assembler = DatasetAssembler()
        assert assembler.validate(df) is True

    def test_validate_raises_on_wrong_period_count(self):
        """Validate should raise ValueError for wrong number of periods."""
        df = pd.DataFrame({
            "TARGET_DATE": ["2024-06-15"] * 47,
            "TARGET_PERIOD": list(range(1, 48)),
        })
        assembler = DatasetAssembler()
        with pytest.raises(ValueError, match="incorrect number of periods"):
            assembler.validate(df)