import pandas as pd
import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from bmrs_pipeline import (
    AGPTProcessor,
    FUELHHProcessor,
    DemandOutturnProcessor,
    MIPProcessor,
    DatasetAssembler,
)


class TestAGPTTransform:
    """Test AGPTProcessor._transform pivots and groups fuel types correctly."""

    @staticmethod
    def _make_raw_agpt(periods=3):
        """Build a DataFrame mimicking raw AGPT API response rows."""
        fuel_types = [
            "Biomass", "Fossil Gas", "Fossil Hard coal", "Fossil Oil",
            "Nuclear", "Solar", "Wind Offshore", "Wind Onshore",
            "Hydro Pumped Storage", "Hydro Run-of-river and poundage", "Other",
        ]
        base = pd.Timestamp("2024-06-14 23:00", tz="UTC")
        rows = []
        for sp in range(1, periods + 1):
            ts = base + pd.Timedelta(minutes=(sp - 1) * 30)
            for fuel in fuel_types:
                rows.append({
                    "settlementDate": "2024-06-15",
                    "settlementPeriod": sp,
                    "startTime": ts,
                    "psrType": fuel,
                    "quantity": 100.0,
                })
        return pd.DataFrame(rows)

    def test_output_columns(self):
        """Transform should produce the expected uppercase column names."""
        proc = AGPTProcessor.__new__(AGPTProcessor)
        df = self._make_raw_agpt()
        result = proc._transform(df)

        expected_cols = {
            "SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "START_TIME",
            "BIOMASS", "GAS", "COAL", "OIL", "NUCLEAR", "SOLAR",
            "WIND", "OTHER",
        }
        assert expected_cols.issubset(set(result.columns))

    def test_wind_grouped(self):
        """Wind Offshore + Wind Onshore should be summed into WIND."""
        proc = AGPTProcessor.__new__(AGPTProcessor)
        df = self._make_raw_agpt(periods=1)
        result = proc._transform(df)

        assert "WIND" in result.columns
        assert "Wind Offshore" not in result.columns
        assert result["WIND"].iloc[0] == 200.0  # 100 + 100

    def test_other_grouped(self):
        """Hydro types + Other should be summed into OTHER."""
        proc = AGPTProcessor.__new__(AGPTProcessor)
        df = self._make_raw_agpt(periods=1)
        result = proc._transform(df)

        assert "OTHER" in result.columns
        assert "Hydro Pumped Storage" not in result.columns
        assert result["OTHER"].iloc[0] == 300.0  # 100 + 100 + 100

    def test_sorted_by_date_and_period(self):
        """Output should be sorted by settlement date and period."""
        proc = AGPTProcessor.__new__(AGPTProcessor)
        df = self._make_raw_agpt(periods=5)
        result = proc._transform(df)

        periods = result["SETTLEMENT_PERIOD"].tolist()
        assert periods == sorted(periods)

    def test_one_row_per_period(self):
        """Each settlement period should produce exactly one row."""
        proc = AGPTProcessor.__new__(AGPTProcessor)
        n = 4
        df = self._make_raw_agpt(periods=n)
        result = proc._transform(df)

        assert len(result) == n


class TestFUELHHTransform:
    """Test FUELHHProcessor._transform sums interconnectors correctly."""

    @staticmethod
    def _make_raw_fuelhh():
        """Build a DataFrame mimicking raw FUELHH API response rows."""
        fuel_types = {
            "INTFR": 500.0,
            "INTIRL": 200.0,
            "INTNED": -100.0,
            "INTNSL": 50.0,
            "CCGT": 8000.0,
        }
        rows = []
        for fuel, gen in fuel_types.items():
            rows.append({
                "settlementDate": "2024-06-15",
                "settlementPeriod": 1,
                "startTime": pd.Timestamp("2024-06-14 23:00", tz="UTC"),
                "fuelType": fuel,
                "generation": gen,
            })
        return pd.DataFrame(rows)

    def test_interconnectors_summed(self):
        """All INT* columns should be summed into a single INTER column."""
        proc = FUELHHProcessor.__new__(FUELHHProcessor)
        df = self._make_raw_fuelhh()
        result = proc._transform(df)

        assert "INTER" in result.columns
        assert result["INTER"].iloc[0] == 650.0  # 500 + 200 - 100 + 50

    def test_output_columns(self):
        """Transform should produce exactly the expected columns."""
        proc = FUELHHProcessor.__new__(FUELHHProcessor)
        df = self._make_raw_fuelhh()
        result = proc._transform(df)

        assert set(result.columns) == {
            "SETTLEMENT_DATE", "SETTLEMENT_PERIOD", "START_TIME", "INTER",
        }

    def test_non_interconnector_fuels_dropped(self):
        """Non-INT fuel types (e.g. CCGT) should not appear in output."""
        proc = FUELHHProcessor.__new__(FUELHHProcessor)
        df = self._make_raw_fuelhh()
        result = proc._transform(df)

        assert "CCGT" not in result.columns


class TestDemandOutturnTransform:
    """Test DemandOutturnProcessor._transform renames columns correctly."""

    @staticmethod
    def _make_raw_demand(periods=3):
        """Build a DataFrame mimicking raw demand outturn API response."""
        base = pd.Timestamp("2024-06-14 23:00", tz="UTC")
        rows = []
        for sp in range(1, periods + 1):
            rows.append({
                "settlementDate": "2024-06-15",
                "settlementPeriod": sp,
                "startTime": base + pd.Timedelta(minutes=(sp - 1) * 30),
                "initialDemandOutturn": 25000.0 + sp * 100,
                "initialTransmissionSystemDemandOutturn": 26000.0 + sp * 100,
            })
        return pd.DataFrame(rows)

    def test_output_columns(self):
        """Transform should rename to INDO and ITSO."""
        proc = DemandOutturnProcessor.__new__(DemandOutturnProcessor)
        df = self._make_raw_demand()
        result = proc._transform(df)

        assert "INDO" in result.columns
        assert "ITSO" in result.columns
        assert "initialDemandOutturn" not in result.columns

    def test_values_preserved(self):
        """Original demand values should be preserved through the transform."""
        proc = DemandOutturnProcessor.__new__(DemandOutturnProcessor)
        df = self._make_raw_demand(periods=2)
        result = proc._transform(df)

        assert result["INDO"].iloc[0] == 25100.0
        assert result["ITSO"].iloc[1] == 26200.0


class TestMIPTransform:
    """Test MIPProcessor._transform renames price/volume correctly."""

    @staticmethod
    def _make_raw_mip(periods=3):
        """Build a DataFrame mimicking raw MIP API response."""
        base = pd.Timestamp("2024-06-14 23:00", tz="UTC")
        rows = []
        for sp in range(1, periods + 1):
            rows.append({
                "settlementDate": "2024-06-15",
                "settlementPeriod": sp,
                "startTime": base + pd.Timedelta(minutes=(sp - 1) * 30),
                "price": 50.0 + sp,
                "volume": 1000.0 + sp * 10,
            })
        return pd.DataFrame(rows)

    def test_output_columns(self):
        """Transform should rename price/volume to uppercase convention."""
        proc = MIPProcessor.__new__(MIPProcessor)
        df = self._make_raw_mip()
        result = proc._transform(df)

        assert "ELECTRICITY_PRICE" in result.columns
        assert "TRADING_VOLUME" in result.columns
        assert "price" not in result.columns

    def test_values_preserved(self):
        """Price and volume values should be preserved through the transform."""
        proc = MIPProcessor.__new__(MIPProcessor)
        df = self._make_raw_mip(periods=2)
        result = proc._transform(df)

        assert result["ELECTRICITY_PRICE"].iloc[0] == 51.0
        assert result["TRADING_VOLUME"].iloc[1] == 1020.0


class TestBMRSDatasetAssembler:
    """Test DatasetAssembler merge, clean, and validate logic."""

    @staticmethod
    def _make_dataset(date, periods, value_col, value):
        """Build a simple settlement-period DataFrame."""
        return pd.DataFrame({
            "SETTLEMENT_DATE": [date] * periods,
            "SETTLEMENT_PERIOD": list(range(1, periods + 1)),
            value_col: [value] * periods,
        })

    def test_merge_joins_on_date_period(self):
        """Merge should inner-join all four datasets on date/period."""
        date = "2024-06-15"
        n = 48
        df_agpt = self._make_dataset(date, n, "GAS", 5000)
        df_agpt["START_TIME"] = pd.Timestamp("2024-06-14 23:00")
        df_fuelhh = self._make_dataset(date, n, "INTER", 300)
        df_fuelhh["START_TIME"] = pd.Timestamp("2024-06-14 23:00")
        df_demand = self._make_dataset(date, n, "INDO", 25000)
        df_demand["ITSO"] = 26000
        df_demand["START_TIME"] = pd.Timestamp("2024-06-14 23:00")
        df_mip = self._make_dataset(date, n, "ELECTRICITY_PRICE", 50)
        df_mip["TRADING_VOLUME"] = 1000
        df_mip["START_TIME"] = pd.Timestamp("2024-06-14 23:00")

        assembler = DatasetAssembler()
        result = assembler.merge(df_agpt, df_fuelhh, df_demand, df_mip)

        assert len(result) == n
        assert "GAS" in result.columns
        assert "INTER" in result.columns
        assert "INDO" in result.columns
        assert "ELECTRICITY_PRICE" in result.columns
        assert "START_TIME" not in result.columns

    def test_clean_removes_duplicates(self):
        """Clean should remove duplicate (date, period) rows."""
        df = pd.DataFrame({
            "SETTLEMENT_DATE": ["2024-06-15"] * 49,
            "SETTLEMENT_PERIOD": list(range(1, 49)) + [1],  # period 1 duplicated
            "VALUE": [100.0] * 49,
        })
        assembler = DatasetAssembler()
        result = assembler.clean(df)

        assert len(result) == 48

    def test_clean_fills_missing_periods(self):
        """Clean should fill in missing periods for a day with too few."""
        # 47 periods (missing period 48)
        df = pd.DataFrame({
            "SETTLEMENT_DATE": ["2024-06-15"] * 47,
            "SETTLEMENT_PERIOD": list(range(1, 48)),
            "VALUE": [100.0] * 47,
        })
        assembler = DatasetAssembler()
        result = assembler.clean(df)

        assert len(result) == 48
        assert list(result["SETTLEMENT_PERIOD"]) == list(range(1, 49))

    def test_clean_trims_excess_periods(self):
        """Clean should trim excess periods for a day with too many."""
        df = pd.DataFrame({
            "SETTLEMENT_DATE": ["2024-06-15"] * 50,
            "SETTLEMENT_PERIOD": list(range(1, 51)),
            "VALUE": [100.0] * 50,
        })
        assembler = DatasetAssembler()
        result = assembler.clean(df)

        assert len(result) == 48

    def test_validate_passes_for_correct_data(self):
        """Validate should return True for a well-formed dataset."""
        df = pd.DataFrame({
            "SETTLEMENT_DATE": ["2024-06-15"] * 48,
            "SETTLEMENT_PERIOD": list(range(1, 49)),
        })
        assembler = DatasetAssembler()
        assert assembler.validate(df) is True

    def test_validate_raises_on_duplicates(self):
        """Validate should raise ValueError if there are duplicate periods."""
        df = pd.DataFrame({
            "SETTLEMENT_DATE": ["2024-06-15"] * 49,
            "SETTLEMENT_PERIOD": list(range(1, 49)) + [1],
        })
        assembler = DatasetAssembler()
        with pytest.raises(ValueError, match="duplicate"):
            assembler.validate(df)

    def test_validate_raises_on_wrong_period_count(self):
        """Validate should raise ValueError if a day has wrong period count."""
        df = pd.DataFrame({
            "SETTLEMENT_DATE": ["2024-06-15"] * 47,
            "SETTLEMENT_PERIOD": list(range(1, 48)),
        })
        assembler = DatasetAssembler()
        with pytest.raises(ValueError, match="incorrect number of periods"):
            assembler.validate(df)