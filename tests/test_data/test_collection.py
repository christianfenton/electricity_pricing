"""
Tests for data collection.
"""

import pytest
import os
import pandas as pd
from datetime import datetime

from electricity_pricing.data.collectors import (
    collect_agpt_data,
    collect_fuelhh_data,
    collect_demand_data,
    collect_mip_data,
)


# These tests require API access and may be slow
class TestBMRSCollectors:
    """Integration tests for BMRS API collectors."""

    def test_collect_agpt_data(self):
        """Test AGPT data collection."""
        df = collect_agpt_data("2024-01-01", "2024-01-02")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "settlementDate" in df.columns
        assert "WIND" in df.columns
        assert "NUCLEAR" in df.columns

    def test_collect_fuelhh_data(self):
        """Test FUELHH data collection."""
        df = collect_fuelhh_data("2024-01-01", "2024-01-02")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "INTER" in df.columns

    def test_collect_demand_data(self):
        """Test demand data collection."""
        df = collect_demand_data("2024-01-01", "2024-01-02")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "INDO" in df.columns
        assert "ITSO" in df.columns

    def test_collect_mip_data(self):
        """Test price data collection."""
        df = collect_mip_data("2024-01-01", "2024-01-02")

        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "marketIndexPrice" in df.columns
        assert "marketIndexTradingVolume" in df.columns
