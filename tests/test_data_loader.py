"""Tests for src.data_loader – schema, row count, severity distribution."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.data_loader import generate_synthetic, load_accidents, load_real


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def synthetic_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic data once for all tests in this module."""
    return generate_synthetic(n=50_000, seed=42, save=False)


@pytest.fixture(scope="module")
def accidents_df(synthetic_data: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return synthetic_data[0]


@pytest.fixture(scope="module")
def hotspots_df(synthetic_data: tuple[pd.DataFrame, pd.DataFrame]) -> pd.DataFrame:
    return synthetic_data[1]


# ── Schema tests ─────────────────────────────────────────────────────────────

class TestSchema:
    """Verify that the generated DataFrame matches the standard schema."""

    def test_standard_columns_present(self, accidents_df: pd.DataFrame) -> None:
        """All standard columns must exist in the output."""
        for col in config.STANDARD_COLUMNS:
            assert col in accidents_df.columns, f"Missing column: {col}"

    def test_no_extra_columns(self, accidents_df: pd.DataFrame) -> None:
        """No unexpected columns beyond the standard schema."""
        extra = set(accidents_df.columns) - set(config.STANDARD_COLUMNS)
        assert extra == set(), f"Unexpected columns: {extra}"

    def test_severity_dtype_is_integer(self, accidents_df: pd.DataFrame) -> None:
        """Severity codes should be integers 1, 2, or 3."""
        valid = accidents_df["severity"].dropna()
        assert set(valid.unique()).issubset({1, 2, 3})

    def test_hotspots_schema(self, hotspots_df: pd.DataFrame) -> None:
        """Hotspot centres table must have expected columns."""
        expected = {"latitude", "longitude", "spatial_std", "name"}
        assert set(hotspots_df.columns) == expected


# ── Row-count tests ──────────────────────────────────────────────────────────

class TestRowCount:
    """Verify record counts are in expected range."""

    def test_base_count_with_duplicates(self, accidents_df: pd.DataFrame) -> None:
        """Row count should be > n due to injected duplicates (~1 %)."""
        assert len(accidents_df) > 50_000
        # Should not exceed n + 2 % (duplicates are ~1 %)
        assert len(accidents_df) < 52_000

    def test_hotspot_count(self, hotspots_df: pd.DataFrame) -> None:
        """There should be exactly 9 true hotspot centres."""
        assert len(hotspots_df) == 9


# ── Severity distribution tests ──────────────────────────────────────────────

class TestSeverityDistribution:
    """Ensure severity is heavily imbalanced as specified."""

    def test_fatal_roughly_one_percent(self, accidents_df: pd.DataFrame) -> None:
        """Fatal (1) should be approximately 1 % (tolerance: 0.3 – 3 %)."""
        pct = (accidents_df["severity"] == 1).mean()
        assert 0.003 <= pct <= 0.03, f"Fatal pct = {pct:.4f}"

    def test_serious_roughly_fourteen_percent(self, accidents_df: pd.DataFrame) -> None:
        """Serious (2) should be approximately 14 % (tolerance: 8 – 22 %)."""
        pct = (accidents_df["severity"] == 2).mean()
        assert 0.08 <= pct <= 0.22, f"Serious pct = {pct:.4f}"

    def test_slight_is_majority(self, accidents_df: pd.DataFrame) -> None:
        """Slight (3) should be the vast majority (> 75 %)."""
        pct = (accidents_df["severity"] == 3).mean()
        assert pct > 0.75, f"Slight pct = {pct:.4f}"


# ── Data-quality tests (verifying injected noise) ────────────────────────────

class TestInjectedNoise:
    """Confirm that the synthetic data contains deliberate quality issues."""

    def test_has_missing_coordinates(self, accidents_df: pd.DataFrame) -> None:
        """Some latitude/longitude values should be NaN."""
        assert accidents_df["latitude"].isna().sum() > 0
        assert accidents_df["longitude"].isna().sum() > 0

    def test_has_out_of_bounds_coords(self, accidents_df: pd.DataFrame) -> None:
        """Some coordinates should be outside valid geographic bounds."""
        oob_lat = (accidents_df["latitude"].dropna().abs() > 90).sum()
        oob_lon = (accidents_df["longitude"].dropna().abs() > 180).sum()
        assert (oob_lat + oob_lon) > 0

    def test_has_duplicate_rows(self, accidents_df: pd.DataFrame) -> None:
        """There should be duplicate accident_id values."""
        assert accidents_df["accident_id"].duplicated().sum() > 0

    def test_has_null_weather(self, accidents_df: pd.DataFrame) -> None:
        """Weather column should contain some NaN values."""
        assert accidents_df["weather"].isna().sum() > 0

    def test_has_null_road_surface(self, accidents_df: pd.DataFrame) -> None:
        """Road surface column should contain some NaN values."""
        assert accidents_df["road_surface"].isna().sum() > 0


# ── load_real tests ──────────────────────────────────────────────────────────

class TestLoadReal:
    """Test CSV loading with column mapping."""

    def test_maps_columns_correctly(self, tmp_path: Path) -> None:
        """Columns should be renamed to standard schema."""
        csv = tmp_path / "sample.csv"
        pd.DataFrame({
            "Accident_Index": ["A001", "A002"],
            "Date": ["2023-01-01", "2023-01-02"],
            "Time": ["08:30", "17:45"],
            "Latitude": [51.5, 51.51],
            "Longitude": [-0.12, -0.13],
            "Accident_Severity": [3, 2],
            "Number_of_Vehicles": [2, 1],
            "Number_of_Casualties": [1, 2],
            "Road_Type": ["Single carriageway", "Dual carriageway"],
            "Speed_limit": [30, 40],
            "Light_Conditions": ["Daylight", "Dark"],
            "Weather_Conditions": ["Fine", "Rain"],
            "Road_Surface_Conditions": ["Dry", "Wet"],
            "Junction_Detail": ["T junction", "Crossroads"],
            "Urban_or_Rural_Area": ["Urban", "Rural"],
        }).to_csv(csv, index=False)

        df = load_real(csv)
        for col in config.STANDARD_COLUMNS:
            assert col in df.columns
        assert len(df) == 2

    def test_missing_source_columns_become_nan(self, tmp_path: Path) -> None:
        """Standard columns missing from source should be NaN-filled."""
        csv = tmp_path / "minimal.csv"
        pd.DataFrame({
            "Latitude": [51.5],
            "Longitude": [-0.12],
        }).to_csv(csv, index=False)

        df = load_real(csv)
        assert df["severity"].isna().all()

    def test_file_not_found_raises(self) -> None:
        """Loading a non-existent file should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_real(Path("nonexistent.csv"))


# ── load_accidents unified entry point ───────────────────────────────────────

class TestLoadAccidents:
    """Test the unified load_accidents entry point."""

    def test_synthetic_source(self) -> None:
        """Synthetic source should return a DataFrame with standard columns."""
        df = load_accidents(source="synthetic", n=500, save_synthetic=False)
        assert len(df) > 500  # duplicates injected
        for col in config.STANDARD_COLUMNS:
            assert col in df.columns

    def test_invalid_source_raises(self) -> None:
        """Unknown source should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown source"):
            load_accidents(source="parquet")

    def test_real_without_path_raises(self) -> None:
        """Real source without a path should raise ValueError."""
        with pytest.raises(ValueError, match="path is required"):
            load_accidents(source="real")
