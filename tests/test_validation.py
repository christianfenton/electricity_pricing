"""
Unit tests for data validation functions.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date

from electricity_pricing.data.validation import (
    last_sunday_of_month,
    get_expected_periods,
    find_irregular_days,
    validate_day_lengths,
    validate_spring_dst,
    validate_autumn_dst,
    validate_settlement_periods
)


class TestLastSundayOfMonth:
    """Tests for `last_sunday_of_month`."""

    def test_march_2021(self):
        result = last_sunday_of_month(2021, 3)
        assert result == date(2021, 3, 28)

    def test_october_2021(self):
        result = last_sunday_of_month(2021, 10)
        assert result == date(2021, 10, 31)

    def test_march_2022(self):
        result = last_sunday_of_month(2022, 3)
        assert result == date(2022, 3, 27)

    def test_october_2022(self):
        result = last_sunday_of_month(2022, 10)
        assert result == date(2022, 10, 30)

    def test_returns_date_object(self):
        result = last_sunday_of_month(2021, 3)
        assert isinstance(result, date)


class TestGetExpectedPeriods:
    """Tests for `get_expected_periods`."""

    def test_spring_dst_46_periods(self):
        result = get_expected_periods(pd.Timestamp('2021-03-28'))
        assert result == 46

    def test_autumn_dst_50_periods(self):
        result = get_expected_periods(pd.Timestamp('2021-10-31'))
        assert result == 50

    def test_normal_day_48_periods(self):
        result = get_expected_periods(pd.Timestamp('2021-06-15'))
        assert result == 48

    def test_multiple_years(self):
        assert get_expected_periods(pd.Timestamp('2022-03-27')) == 46
        assert get_expected_periods(pd.Timestamp('2022-10-30')) == 50
        assert get_expected_periods(pd.Timestamp('2023-03-26')) == 46
        assert get_expected_periods(pd.Timestamp('2023-10-29')) == 50


class TestFindIrregularDays:
    """Tests for `find_irregular_days`."""

    def test_all_days_correct(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-01-01'] * 48 + ['2021-01-02'] * 48),
            'SETTLEMENT_PERIOD': list(range(1, 49)) * 2
        })
        irregular = find_irregular_days(df)
        assert len(irregular) == 0

    def test_too_few(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-01-01'] * 47 + ['2021-01-02'] * 48),
            'SETTLEMENT_PERIOD': list(range(1, 48)) + list(range(1, 49))
        })
        irregular = find_irregular_days(df)
        assert len(irregular) == 1
        assert irregular[0][1] == 47  # actual count
        assert irregular[0][2] == 48  # expected count

    def test_too_many(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-01-01'] * 49 + ['2021-01-02'] * 48),
            'SETTLEMENT_PERIOD': list(range(1, 50)) + list(range(1, 49))
        })
        irregular = find_irregular_days(df)
        assert len(irregular) == 1
        assert irregular[0][1] == 49  # actual count
        assert irregular[0][2] == 48  # expected count

    def test_spring_dst(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-03-28'] * 46),
            'SETTLEMENT_PERIOD': range(1, 47)
        })
        irregular = find_irregular_days(df)
        assert len(irregular) == 0

    def test_autumn_dst(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-10-31'] * 50),
            'SETTLEMENT_PERIOD': range(1, 51)
        })
        irregular = find_irregular_days(df)
        assert len(irregular) == 0


class TestValidateDayLengths:
    """Tests for `validate_day_lengths`."""

    def test_valid_data(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-01-01'] * 48 + ['2021-01-02'] * 48),
            'SETTLEMENT_PERIOD': list(range(1, 49)) * 2
        })
        is_valid, irregular = validate_day_lengths(df)
        assert is_valid is True
        assert len(irregular) == 0

    def test_invalid_data(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-01-01'] * 47 + ['2021-01-02'] * 49),
            'SETTLEMENT_PERIOD': list(range(1, 48)) + list(range(1, 50))
        })
        is_valid, irregular = validate_day_lengths(df)
        assert is_valid is False
        assert len(irregular) == 2


class TestValidateSpringDST:
    """Tests for `validate_spring_dst`."""

    def test_correct_spring_dst_single_year(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-03-28'] * 46 + ['2021-06-15'] * 48),
            'SETTLEMENT_PERIOD': list(range(1, 47)) + list(range(1, 49))
        })
        is_valid, msg = validate_spring_dst(df)
        assert is_valid is True
        assert msg == ""

    def test_wrong_spring_dst_date(self):
        # Create data with 46 periods on wrong date
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-03-27'] * 46 + ['2021-06-15'] * 48),
            'SETTLEMENT_PERIOD': list(range(1, 47)) + list(range(1, 49))
        })
        is_valid, msg = validate_spring_dst(df)
        assert is_valid is False
        assert "2021-03-28" in msg  # Expected date
        assert "2021-03-27" in msg  # Actual date

    def test_multiple_spring_dst_days(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-03-27'] * 46 + ['2021-03-28'] * 46),
            'SETTLEMENT_PERIOD': list(range(1, 47)) * 2
        })
        is_valid, msg = validate_spring_dst(df)
        assert is_valid is False
        assert "2" in msg


class TestValidateAutumnDST:
    """Tests for `validate_autumn_dst`."""

    def test_correct_autumn_dst_single_year(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-10-31'] * 50 + ['2021-06-15'] * 48),
            'SETTLEMENT_PERIOD': list(range(1, 51)) + list(range(1, 49))
        })
        is_valid, msg = validate_autumn_dst(df)
        assert is_valid is True
        assert msg == ""

    def test_wrong_autumn_dst_date(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-10-30'] * 50 + ['2021-06-15'] * 48),
            'SETTLEMENT_PERIOD': list(range(1, 51)) + list(range(1, 49))
        })
        is_valid, msg = validate_autumn_dst(df)
        assert is_valid is False
        assert "2021-10-31" in msg  # Expected date
        assert "2021-10-30" in msg  # Actual date


class TestValidateSettlementPeriods:
    """Tests for `validate_settlement_periods`."""

    def test_valid_data(self):
        # Create one year of data with correct DST days
        dates = []
        periods = []

        # Normal days
        for day in range(1, 27):  # Jan 1-26
            dates.extend([f'2021-01-{day:02d}'] * 48)
            periods.extend(list(range(1, 49)))

        # Spring DST
        dates.extend(['2021-03-28'] * 46)
        periods.extend(list(range(1, 47)))

        # More normal days
        for day in range(29, 32):  # Mar 29-31
            dates.extend([f'2021-03-{day:02d}'] * 48)
            periods.extend(list(range(1, 49)))

        # Autumn DST
        dates.extend(['2021-10-31'] * 50)
        periods.extend(list(range(1, 51)))

        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(dates),
            'SETTLEMENT_PERIOD': periods
        })

        is_valid, errors = validate_settlement_periods(df)
        assert is_valid is True
        assert len(errors) == 0

    def test_with_duplicates(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-01-01'] * 49),
            'SETTLEMENT_PERIOD': [1] * 2 + list(range(2, 49))  # Period 1 appears twice
        })
        is_valid, errors = validate_settlement_periods(df)
        assert is_valid is False
        assert any('duplicate' in err.lower() for err in errors)

    def test_with_wrong_period_counts(self):
        df = pd.DataFrame({
            'SETTLEMENT_DATE': pd.to_datetime(['2021-01-01'] * 47),  # Should be 48
            'SETTLEMENT_PERIOD': range(1, 48)
        })
        is_valid, errors = validate_settlement_periods(df)
        assert is_valid is False
        assert any('incorrect number' in err.lower() for err in errors)


# if __name__ == '__main__':
#     pytest.main([__file__, '-v'])
