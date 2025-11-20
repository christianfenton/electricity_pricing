"""
Unit tests for data cleaning functions.
"""

import pytest
import pandas as pd
import numpy as np

from electricity_pricing.data.cleaning import (
    fill_missing_periods,
    drop_excess_periods,
    clean_single_day,
    clean_dataset,
)


class TestFillMissingPeriods:
    """Tests for `fill_missing_periods`."""

    def test_fill_one_missing_period_interpolate(self):
        """Test filling one missing period with interpolation."""
        # data with period 24 missing
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 47),
                "SETTLEMENT_PERIOD": list(range(1, 24)) + list(range(25, 49)),
                "value": list(range(1, 24)) + list(range(25, 49)),
            }
        )

        df_filled = fill_missing_periods(
            df, pd.Timestamp("2021-01-01"), method="interpolate"
        )
        assert (
            len(
                df_filled[
                    df_filled["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")
                ]
            )
            == 48
        )

        mask = (df_filled["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")) & (
            df_filled["SETTLEMENT_PERIOD"] == 24
        )
        period_24 = df_filled[mask]
        assert len(period_24) == 1
        assert (
            period_24["value"].values[0] == 24
        )  # Should be interpolated between 23 and 25

    def test_fill_multiple_missing_periods(self):
        """Test filling multiple missing periods."""
        # Create data with periods 10-15 missing
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 42),
                "SETTLEMENT_PERIOD": list(range(1, 10)) + list(range(16, 49)),
                "value": list(range(1, 10)) + list(range(16, 49)),
            }
        )
        df_filled = fill_missing_periods(df, pd.Timestamp("2021-01-01"))
        assert (
            len(
                df_filled[
                    df_filled["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")
                ]
            )
            == 48
        )

    def test_fill_spring_dst_missing_period(self):
        """Test filling missing period on spring DST day."""
        # Spring DST should have 46 periods, create with 45
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-03-28"] * 45),
                "SETTLEMENT_PERIOD": list(range(1, 24)) + list(range(25, 47)),
                "value": range(45),
            }
        )
        df_filled = fill_missing_periods(df, pd.Timestamp("2021-03-28"))
        assert (
            len(
                df_filled[
                    df_filled["SETTLEMENT_DATE"] == pd.Timestamp("2021-03-28")
                ]
            )
            == 46
        )

    def test_already_complete_no_change(self):
        """Test that complete data is not modified."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 48),
                "SETTLEMENT_PERIOD": range(1, 49),
                "value": range(48),
            }
        )
        df_filled = fill_missing_periods(df, pd.Timestamp("2021-01-01"))
        pd.testing.assert_frame_equal(df_filled, df)


class TestTrimExcessPeriods:
    """Tests for trim_excess_periods function."""

    def test_trim_one_extra_period(self):
        """Test trimming one extra period."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 49),
                "SETTLEMENT_PERIOD": range(1, 50),
                "value": range(49),
            }
        )
        df_trimmed = drop_excess_periods(df, pd.Timestamp("2021-01-01"))
        day_data = df_trimmed[
            df_trimmed["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")
        ]
        assert len(day_data) == 48
        assert day_data["SETTLEMENT_PERIOD"].max() == 48

    def test_trim_multiple_extra_periods(self):
        """Test trimming multiple extra periods."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 52),
                "SETTLEMENT_PERIOD": range(1, 53),
                "value": range(52),
            }
        )
        df_trimmed = drop_excess_periods(df, pd.Timestamp("2021-01-01"))
        day_data = df_trimmed[
            df_trimmed["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")
        ]
        assert len(day_data) == 48

    def test_trim_autumn_dst_excess(self):
        """Test trimming excess on autumn DST day."""
        # Autumn DST should have 50 periods, create with 52
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-10-31"] * 52),
                "SETTLEMENT_PERIOD": range(1, 53),
                "value": range(52),
            }
        )
        df_trimmed = drop_excess_periods(df, pd.Timestamp("2021-10-31"))
        day_data = df_trimmed[
            df_trimmed["SETTLEMENT_DATE"] == pd.Timestamp("2021-10-31")
        ]
        assert len(day_data) == 50

    def test_already_correct_no_change(self):
        """Test that correct data is not modified."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 48),
                "SETTLEMENT_PERIOD": range(1, 49),
                "value": range(48),
            }
        )
        df_trimmed = drop_excess_periods(df, pd.Timestamp("2021-01-01"))
        pd.testing.assert_frame_equal(df_trimmed, df)


class TestCleanSingleDay:
    """Tests for `clean_single_day`."""

    def test_clean_day_with_missing_periods(self):
        """Test cleaning a day with missing periods."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(
                    ["2021-01-01"] * 46 + ["2021-01-02"] * 48
                ),
                "SETTLEMENT_PERIOD": list(range(1, 47)) + list(range(1, 49)),
                "value": range(94),
            }
        )
        df_clean = clean_single_day(df, pd.Timestamp("2021-01-01"))
        day_data = df_clean[
            df_clean["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")
        ]
        assert len(day_data) == 48

    def test_clean_day_with_excess_periods(self):
        """Test cleaning a day with excess periods."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(
                    ["2021-01-01"] * 50 + ["2021-01-02"] * 48
                ),
                "SETTLEMENT_PERIOD": list(range(1, 51)) + list(range(1, 49)),
                "value": range(98),
            }
        )
        df_clean = clean_single_day(df, pd.Timestamp("2021-01-01"))
        day_data = df_clean[
            df_clean["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")
        ]
        assert len(day_data) == 48

    def test_clean_correct_day_no_change(self):
        """Test that a correct day is not modified."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(
                    ["2021-01-01"] * 48 + ["2021-01-02"] * 48
                ),
                "SETTLEMENT_PERIOD": list(range(1, 49)) * 2,
                "value": range(96),
            }
        )
        df_clean = clean_single_day(df, pd.Timestamp("2021-01-01"))
        pd.testing.assert_frame_equal(df_clean, df)


class TestCleanDataset:
    """Tests for `clean_dataset`."""

    def test_clean_with_duplicates(self):
        """Test cleaning data with duplicates."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 49),
                "SETTLEMENT_PERIOD": [1, 1]
                + list(range(2, 49)),  # Period 1 duplicated
                "value": range(49),
            }
        )
        df_clean = clean_dataset(df)
        day_data = df_clean[
            df_clean["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")
        ]
        assert len(day_data) == 48
        assert not day_data.duplicated(subset=["SETTLEMENT_PERIOD"]).any()

    def test_clean_with_missing_periods(self):
        """Test cleaning data with missing periods."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(
                    ["2021-01-01"] * 46 + ["2021-01-02"] * 47
                ),
                "SETTLEMENT_PERIOD": list(range(1, 47)) + list(range(1, 48)),
                "value": range(93),
            }
        )
        df_clean = clean_dataset(df)
        assert (
            len(
                df_clean[
                    df_clean["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")
                ]
            )
            == 48
        )
        assert (
            len(
                df_clean[
                    df_clean["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-02")
                ]
            )
            == 48
        )

    def test_clean_with_excess_periods(self):
        """Test cleaning data with excess periods."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(
                    ["2021-01-01"] * 50 + ["2021-01-02"] * 49
                ),
                "SETTLEMENT_PERIOD": list(range(1, 51)) + list(range(1, 50)),
                "value": range(99),
            }
        )
        df_clean = clean_dataset(df)
        assert (
            len(
                df_clean[
                    df_clean["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-01")
                ]
            )
            == 48
        )
        assert (
            len(
                df_clean[
                    df_clean["SETTLEMENT_DATE"] == pd.Timestamp("2021-01-02")
                ]
            )
            == 48
        )

    def test_clean_dst_days(self):
        """Test cleaning with DST days."""
        # Create dataset with spring and autumn DST, each with wrong number of periods
        dates = (
            ["2021-03-28"] * 47  # Spring DST (should be 46)
            + ["2021-10-31"] * 49
        )  # Autumn DST (should be 50)
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(dates),
                "SETTLEMENT_PERIOD": list(range(1, 48)) + list(range(1, 50)),
                "value": range(96),
            }
        )
        df_clean = clean_dataset(df)
        assert (
            len(
                df_clean[
                    df_clean["SETTLEMENT_DATE"] == pd.Timestamp("2021-03-28")
                ]
            )
            == 46
        )
        assert (
            len(
                df_clean[
                    df_clean["SETTLEMENT_DATE"] == pd.Timestamp("2021-10-31")
                ]
            )
            == 50
        )

    def test_clean_sorted_output(self):
        """Test that output is sorted by date and period."""
        # Create unsorted data
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(
                    ["2021-01-02"] * 48 + ["2021-01-01"] * 48
                ),
                "SETTLEMENT_PERIOD": list(range(1, 49)) * 2,
                "value": range(96),
            }
        )
        df_clean = clean_dataset(df)

        # Check sorting
        assert df_clean["SETTLEMENT_DATE"].iloc[0] == pd.Timestamp(
            "2021-01-01"
        )
        assert df_clean["SETTLEMENT_DATE"].iloc[-1] == pd.Timestamp(
            "2021-01-02"
        )
        assert df_clean["SETTLEMENT_PERIOD"].iloc[0] == 1
        assert df_clean["SETTLEMENT_PERIOD"].iloc[47] == 48

    def test_clean_verbose_output(self, capsys):
        """Test verbose mode prints output."""
        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(["2021-01-01"] * 47),
                "SETTLEMENT_PERIOD": range(1, 48),
                "value": range(47),
            }
        )
        clean_dataset(df, verbose=True)
        captured = capsys.readouterr()
        assert len(captured.out) > 0  # Should print something


class TestCleaningIntegration:
    """Integration tests for cleaning functions."""

    def test_real_world_scenario(self):
        """Test cleaning with realistic messy data."""
        # Create data with multiple issues:
        # - Duplicates
        # - Missing periods on some days
        # - Excess periods on other days
        # - Unsorted
        dates = (
            ["2021-01-01"] * 48  # Normal day (correct)
            + ["2021-01-02"] * 46  # Missing 2 periods
            + ["2021-01-03"] * 51  # 3 excess periods
            + ["2021-01-04"] * 49
        )  # 1 excess + 1 duplicate
        periods = (
            list(range(1, 49))  # Day 1
            + list(range(1, 47))  # Day 2 (missing 47, 48)
            + list(range(1, 52))  # Day 3
            + [1, 1]
            + list(range(2, 49))
        )  # Day 4 (period 1 duplicated)

        df = pd.DataFrame(
            {
                "SETTLEMENT_DATE": pd.to_datetime(dates),
                "SETTLEMENT_PERIOD": periods,
                "value": np.random.rand(len(dates)) * 100,
            }
        )

        # Clean the data
        df_clean = clean_dataset(df, verbose=False)

        # Verify all days have exactly 48 periods
        for date in ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-04"]:
            day_data = df_clean[
                df_clean["SETTLEMENT_DATE"] == pd.Timestamp(date)
            ]
            assert len(day_data) == 48, (
                f"Day {date} should have 48 periods, got {len(day_data)}"
            )

            # Verify periods are 1-48
            assert set(day_data["SETTLEMENT_PERIOD"]) == set(range(1, 49))

            # Verify no duplicates
            assert not day_data.duplicated(subset=["SETTLEMENT_PERIOD"]).any()

        # Verify sorting
        assert df_clean["SETTLEMENT_DATE"].is_monotonic_increasing
        for date in df_clean["SETTLEMENT_DATE"].unique():
            day_data = df_clean[df_clean["SETTLEMENT_DATE"] == date]
            assert day_data["SETTLEMENT_PERIOD"].is_monotonic_increasing
