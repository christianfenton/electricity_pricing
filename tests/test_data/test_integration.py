"""
Integration tests for the full data pipeline.

These tests verify that the complete pipeline works end-to-end,
from loading raw CSV files to producing clean, merged datasets.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from datetime import date

from electricity_pricing.data.loading import (
    load_electricity_data,
    load_forecast_data,
    merge_electricity_and_forecasts,
    load_and_merge_datasets,
)
from electricity_pricing.data.validation import validate_settlement_periods


@pytest.fixture
def sample_electricity_csv(tmp_path):
    """Create a temporary electricity data CSV file."""
    # Create sample data with some imperfections
    dates = ["2021-01-01"] * 47 + ["2021-01-02"] * 48 + ["2021-01-03"] * 49
    periods = list(range(1, 48)) + list(range(1, 49)) + list(range(1, 50))

    df = pd.DataFrame(
        {
            "settlementDate": dates,
            "settlementPeriod": periods,
            "marketIndexPrice": np.random.rand(len(dates)) * 100,
            "BIOMASS": np.random.rand(len(dates)) * 1000,
            "GAS": np.random.rand(len(dates)) * 10000,
            "COAL": np.random.rand(len(dates)) * 100,
            "NUCLEAR": np.random.rand(len(dates)) * 5000,
            "WIND": np.random.rand(len(dates)) * 3000,
            "naturalGasPrice": np.random.rand(len(dates)) * 20,
        }
    )

    file_path = tmp_path / "electricity_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


@pytest.fixture
def sample_forecast_csv(tmp_path):
    """Create a temporary forecast data CSV file."""
    # Create sample data with some imperfections
    dates = ["2021-01-01"] * 47 + ["2021-01-02"] * 48 + ["2021-01-03"] * 49
    periods = list(range(1, 48)) + list(range(1, 49)) + list(range(1, 50))

    df = pd.DataFrame(
        {
            "SETTLEMENT_DATE": dates,
            "SETTLEMENT_PERIOD": periods,
            "DEMAND_FORECAST": np.random.rand(len(dates)) * 30000,
            "WIND_FORECAST": np.random.rand(len(dates)) * 5000,
            "EMBEDDED_WIND_FORECAST": np.random.rand(len(dates)) * 1000,
            "EMBEDDED_SOLAR_FORECAST": np.random.rand(len(dates)) * 500,
            "WIND_CAPACITY": np.ones(len(dates)) * 15000,
            "EMBEDDED_WIND_CAPACITY": np.ones(len(dates)) * 6000,
            "EMBEDDED_SOLAR_CAPACITY": np.ones(len(dates)) * 13000,
        }
    )

    file_path = tmp_path / "forecast_data.csv"
    df.to_csv(file_path, index=False)
    return str(file_path)


class TestLoadDatasets:
    def test_load_electricity(self, sample_electricity_csv):
        df = load_electricity_data(sample_electricity_csv)
        daily_counts = df.groupby(df["SETTLEMENT_DATE"]).size()
        assert all(count == 48 for count in daily_counts)

    def test_load_forecast(self, sample_forecast_csv):
        df = load_forecast_data(sample_forecast_csv)
        daily_counts = df.groupby(df["SETTLEMENT_DATE"]).size()
        assert all(count == 48 for count in daily_counts)

    def test_column_names(self, sample_electricity_csv):
        df = load_electricity_data(sample_electricity_csv)
        assert "SETTLEMENT_DATE" in df.columns
        assert "SETTLEMENT_PERIOD" in df.columns
        assert "settlementDate" not in df.columns
        assert "settlementPeriod" not in df.columns


class TestMergeDatasets:
    def test_merge(self, sample_electricity_csv, sample_forecast_csv):
        df_elec = load_electricity_data(sample_electricity_csv)
        df_forecast = load_forecast_data(sample_forecast_csv)
        df_merged = merge_electricity_and_forecasts(df_elec, df_forecast)
        assert len(df_merged) > 0
        assert "marketIndexPrice" in df_merged.columns
        assert "DEMAND_FORECAST" in df_merged.columns
        assert "BIOMASS" in df_merged.columns
        assert "WIND_FORECAST" in df_merged.columns


class TestLoadAndMergeDatasets:
    def test_load_and_merge(self, sample_electricity_csv, sample_forecast_csv):
        """Test the complete load and merge pipeline."""
        df_merged = load_and_merge_datasets(
            electricity_path=sample_electricity_csv,
            forecast_path=sample_forecast_csv,
            date_range=(date(2021, 1, 1), date(2021, 1, 2)),
            validate=True,
            verbose=False,
        )

        # Basic checks
        assert len(df_merged) > 0
        assert "SETTLEMENT_DATE" in df_merged.columns
        assert "SETTLEMENT_PERIOD" in df_merged.columns

        # Check that data from both sources is present
        assert "marketIndexPrice" in df_merged.columns
        assert "DEMAND_FORECAST" in df_merged.columns

        # Check date range
        assert df_merged["SETTLEMENT_DATE"].min().date() >= date(2021, 1, 1)
        assert df_merged["SETTLEMENT_DATE"].max().date() <= date(2021, 1, 2)

        # Validate data quality
        is_valid, errors = validate_settlement_periods(df_merged)
        assert is_valid, f"Pipeline output validation failed: {errors}"

        # Verify all days have correct number of periods
        daily_counts = df_merged.groupby(df_merged["SETTLEMENT_DATE"]).size()
        assert all(count == 48 for count in daily_counts)


class TestMessyData:
    def test_dst_handling(self, tmp_path):
        """Test handling of DST transitions."""
        # data with spring and autumn DST transitions
        dates = (
            ["2021-03-28"] * 46  # Spring DST (46 periods)
            + ["2021-06-15"] * 48  # Normal day
            + ["2021-10-31"] * 50
        )  # Autumn DST (50 periods)

        df_elec = pd.DataFrame(
            {
                "settlementDate": dates,
                "settlementPeriod": list(range(1, 47))
                + list(range(1, 49))
                + list(range(1, 51)),
                "marketIndexPrice": np.random.rand(len(dates)) * 100,
                "GAS": np.random.rand(len(dates)) * 10000,
            }
        )

        df_forecast = pd.DataFrame(
            {
                "SETTLEMENT_DATE": dates,
                "SETTLEMENT_PERIOD": list(range(1, 47))
                + list(range(1, 49))
                + list(range(1, 51)),
                "DEMAND_FORECAST": np.random.rand(len(dates)) * 30000,
            }
        )

        # Save to temporary files
        elec_path = tmp_path / "elec.csv"
        forecast_path = tmp_path / "forecast.csv"
        df_elec.to_csv(elec_path, index=False)
        df_forecast.to_csv(forecast_path, index=False)

        df_merged = load_and_merge_datasets(
            str(elec_path), str(forecast_path), validate=True, verbose=False
        )

        spring_dst_data = df_merged[
            df_merged["SETTLEMENT_DATE"] == pd.Timestamp("2021-03-28")
        ]
        autumn_dst_data = df_merged[
            df_merged["SETTLEMENT_DATE"] == pd.Timestamp("2021-10-31")
        ]
        normal_day_data = df_merged[
            df_merged["SETTLEMENT_DATE"] == pd.Timestamp("2021-06-15")
        ]
        assert len(spring_dst_data) == 46
        assert len(autumn_dst_data) == 50
        assert len(normal_day_data) == 48

    def test_missing_and_duplicate_handling(self, tmp_path):
        """Test handling of missing periods and duplicates simultaneously."""
        # messy electricity data
        dates_elec = ["2021-01-01"] * 46 + [
            "2021-01-02"
        ] * 49  # missing and extra periods
        periods_elec = (
            list(range(1, 47)) + [1, 1] + list(range(2, 49))
        )  # duplicates

        df_elec = pd.DataFrame(
            {
                "settlementDate": dates_elec,
                "settlementPeriod": periods_elec,
                "marketIndexPrice": np.random.rand(len(dates_elec)) * 100,
                "GAS": np.random.rand(len(dates_elec)) * 10000,
            }
        )

        # clean forecast data
        dates_forecast = ["2021-01-01"] * 48 + ["2021-01-02"] * 48
        df_forecast = pd.DataFrame(
            {
                "SETTLEMENT_DATE": dates_forecast,
                "SETTLEMENT_PERIOD": list(range(1, 49)) * 2,
                "DEMAND_FORECAST": np.random.rand(len(dates_forecast)) * 30000,
            }
        )

        # save to temp files
        elec_path = tmp_path / "elec.csv"
        forecast_path = tmp_path / "forecast.csv"
        df_elec.to_csv(elec_path, index=False)
        df_forecast.to_csv(forecast_path, index=False)

        df_merged = load_and_merge_datasets(
            str(elec_path), str(forecast_path), validate=True, verbose=False
        )

        for day_date in ["2021-01-01", "2021-01-02"]:
            day_data = df_merged[
                df_merged["SETTLEMENT_DATE"] == pd.Timestamp(day_date)
            ]
            assert len(day_data) == 48, (
                f"Day {day_date} should have 48 periods"
            )
            assert not day_data.duplicated(subset=["SETTLEMENT_PERIOD"]).any()


# if __name__ == '__main__':
#     pytest.main([__file__, '-v'])
