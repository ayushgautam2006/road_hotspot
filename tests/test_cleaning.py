"""Tests for src.cleaning – individual steps and full pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.cleaning import (
    CATEGORICAL_COLUMNS,
    clean_accidents,
    impute_missing_categoricals,
    parse_datetime,
    remove_duplicates,
    remove_missing_coordinates,
    remove_oob_coordinates,
    standardise_categoricals,
)
from src.data_loader import generate_synthetic


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def raw_df() -> pd.DataFrame:
    """Small synthetic dataset for cleaning tests."""
    df, _ = generate_synthetic(n=5_000, seed=42, save=False)
    return df


@pytest.fixture(scope="module")
def cleaned_result(raw_df: pd.DataFrame) -> tuple:
    """Run the full pipeline once for the module."""
    return clean_accidents(raw_df, save=False)


@pytest.fixture(scope="module")
def clean_df(cleaned_result: tuple) -> pd.DataFrame:
    return cleaned_result[0]


@pytest.fixture(scope="module")
def report(cleaned_result: tuple) -> dict:
    return cleaned_result[1]


# ── Unit tests for individual steps ──────────────────────────────────────────

class TestRemoveDuplicates:
    """remove_duplicates() unit tests."""

    def test_removes_exact_dupes(self) -> None:
        df = pd.DataFrame({"a": [1, 1, 2], "b": [3, 3, 4]})
        result, n = remove_duplicates(df)
        assert n == 1
        assert len(result) == 2

    def test_no_dupes_is_noop(self) -> None:
        df = pd.DataFrame({"a": [1, 2, 3]})
        result, n = remove_duplicates(df)
        assert n == 0
        assert len(result) == 3


class TestRemoveMissingCoordinates:
    """remove_missing_coordinates() unit tests."""

    def test_drops_nan_lat(self) -> None:
        df = pd.DataFrame({"latitude": [1.0, np.nan, 3.0],
                           "longitude": [1.0, 2.0, 3.0]})
        result, n = remove_missing_coordinates(df)
        assert n == 1 and len(result) == 2

    def test_drops_nan_lon(self) -> None:
        df = pd.DataFrame({"latitude": [1.0, 2.0],
                           "longitude": [np.nan, 2.0]})
        result, n = remove_missing_coordinates(df)
        assert n == 1


class TestRemoveOOBCoordinates:
    """remove_oob_coordinates() unit tests."""

    def test_removes_invalid_lat(self) -> None:
        df = pd.DataFrame({"latitude": [51.5, 120.0, -91.0],
                           "longitude": [-0.1, -0.2, -0.3]})
        result, n = remove_oob_coordinates(df)
        assert n == 2 and len(result) == 1

    def test_custom_range(self) -> None:
        df = pd.DataFrame({"latitude": [51.5, 55.0],
                           "longitude": [-0.1, -0.2]})
        result, n = remove_oob_coordinates(df, lat_range=(51.0, 52.0))
        assert n == 1


class TestStandardiseCategoricals:
    """standardise_categoricals() unit tests."""

    def test_lowercases(self) -> None:
        df = pd.DataFrame({"weather": ["RAIN", "Fine"]})
        result = standardise_categoricals(df)
        assert list(result["weather"]) == ["rain", "fine"]

    def test_strips_whitespace(self) -> None:
        df = pd.DataFrame({"weather": [" rain ", "  fog  "]})
        result = standardise_categoricals(df)
        assert list(result["weather"]) == ["rain", "fog"]

    def test_replaces_spaces_with_underscores(self) -> None:
        df = pd.DataFrame({"weather": ["high wind"]})
        result = standardise_categoricals(df)
        assert result["weather"].iloc[0] == "high_wind"

    def test_preserves_nan(self) -> None:
        df = pd.DataFrame({"weather": [np.nan, "rain"]})
        result = standardise_categoricals(df)
        assert pd.isna(result["weather"].iloc[0])
        assert result["weather"].iloc[1] == "rain"


class TestParseDatetime:
    """parse_datetime() unit tests."""

    def test_creates_datetime_column(self) -> None:
        df = pd.DataFrame({"date": ["2023-01-15"], "time": ["08:30"]})
        result = parse_datetime(df)
        assert "datetime" in result.columns
        assert result["datetime"].iloc[0].hour == 8
        assert result["datetime"].iloc[0].minute == 30

    def test_invalid_time_produces_nat(self) -> None:
        df = pd.DataFrame({"date": ["2023-01-15"], "time": ["BADTIME"]})
        result = parse_datetime(df)
        assert pd.isna(result["datetime"].iloc[0])


class TestImputeMissing:
    """impute_missing_categoricals() unit tests."""

    def test_fills_with_unknown(self) -> None:
        df = pd.DataFrame({"weather": [np.nan, "rain", np.nan]})
        result, imputed = impute_missing_categoricals(df)
        assert (result["weather"] == "unknown").sum() == 2
        assert imputed["weather"] == 2

    def test_no_nulls_remain(self) -> None:
        df = pd.DataFrame({"weather": [np.nan], "road_surface": [np.nan]})
        result, _ = impute_missing_categoricals(df)
        assert result["weather"].isna().sum() == 0
        assert result["road_surface"].isna().sum() == 0

    def test_custom_fill_value(self) -> None:
        df = pd.DataFrame({"weather": [np.nan]})
        result, _ = impute_missing_categoricals(df, fill_value="missing")
        assert result["weather"].iloc[0] == "missing"


# ── Integration tests for the full pipeline ──────────────────────────────────

class TestCleanAccidents:
    """End-to-end tests on the full cleaning pipeline."""

    def test_no_duplicate_rows(self, clean_df: pd.DataFrame) -> None:
        assert clean_df.duplicated().sum() == 0

    def test_no_missing_coords(self, clean_df: pd.DataFrame) -> None:
        assert clean_df["latitude"].isna().sum() == 0
        assert clean_df["longitude"].isna().sum() == 0

    def test_no_oob_coords(self, clean_df: pd.DataFrame) -> None:
        assert clean_df["latitude"].between(-90, 90).all()
        assert clean_df["longitude"].between(-180, 180).all()

    def test_has_datetime_column(self, clean_df: pd.DataFrame) -> None:
        assert "datetime" in clean_df.columns
        assert pd.api.types.is_datetime64_any_dtype(clean_df["datetime"])

    def test_no_null_categoricals(self, clean_df: pd.DataFrame) -> None:
        for col in CATEGORICAL_COLUMNS:
            assert clean_df[col].isna().sum() == 0, f"{col} still has nulls"

    def test_categoricals_are_lowercase(self, clean_df: pd.DataFrame) -> None:
        for col in ["weather", "road_surface", "light_conditions"]:
            for val in clean_df[col].dropna().unique():
                assert val == val.lower(), f"{col}: '{val}' not lowercase"
                assert val == val.strip(), f"{col}: '{val}' has whitespace"

    def test_categoricals_have_no_spaces(self, clean_df: pd.DataFrame) -> None:
        for col in ["weather", "road_surface"]:
            for val in clean_df[col].dropna().unique():
                assert " " not in val, f"{col}: '{val}' contains spaces"

    def test_fewer_rows_after_cleaning(
        self, raw_df: pd.DataFrame, clean_df: pd.DataFrame,
    ) -> None:
        assert len(clean_df) < len(raw_df)

    def test_report_has_required_keys(self, report: dict) -> None:
        expected = {
            "initial_rows", "duplicates_removed", "missing_coords_removed",
            "oob_coords_removed", "categoricals_imputed", "final_rows",
            "rows_removed_total",
        }
        assert expected.issubset(report.keys())

    def test_report_rows_add_up(self, report: dict) -> None:
        removed = (
            report["duplicates_removed"]
            + report["missing_coords_removed"]
            + report["oob_coords_removed"]
        )
        assert removed == report["rows_removed_total"]

    def test_report_initial_gt_final(self, report: dict) -> None:
        assert report["initial_rows"] > report["final_rows"]
