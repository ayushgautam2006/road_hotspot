"""Tests for src.features – no NaNs, no leakage, correct shapes & values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.cleaning import clean_accidents
from src.data_loader import generate_synthetic
from src.features import (
    DROP_COLUMNS,
    ID_AND_RAW_COLUMNS,
    LEAKAGE_COLUMNS,
    ONE_HOT_COLUMNS,
    add_interaction_features,
    add_spatial_features,
    add_temporal_features,
    build_features,
    encode_categoricals,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    """Small cleaned synthetic dataset for feature tests."""
    raw, _ = generate_synthetic(n=3_000, seed=42, save=False)
    df, _ = clean_accidents(raw, save=False)
    return df


@pytest.fixture(scope="module")
def feature_result(clean_df: pd.DataFrame) -> tuple:
    return build_features(clean_df)


@pytest.fixture(scope="module")
def X(feature_result: tuple) -> pd.DataFrame:
    return feature_result[0]


@pytest.fixture(scope="module")
def y_binary(feature_result: tuple) -> pd.Series:
    return feature_result[1]


@pytest.fixture(scope="module")
def y_multiclass(feature_result: tuple) -> pd.Series:
    return feature_result[2]


# ── Core contract: no NaNs ──────────────────────────────────────────────────

class TestNoNaNs:
    """X must be completely NaN-free."""

    def test_no_nans_in_X(self, X: pd.DataFrame) -> None:
        nan_counts = X.isna().sum()
        bad = nan_counts[nan_counts > 0]
        assert bad.empty, f"NaN columns: {bad.to_dict()}"

    def test_no_nans_in_y_binary(self, y_binary: pd.Series) -> None:
        assert y_binary.isna().sum() == 0

    def test_no_nans_in_y_multiclass(self, y_multiclass: pd.Series) -> None:
        assert y_multiclass.isna().sum() == 0


# ── Core contract: no leakage ───────────────────────────────────────────────

class TestNoLeakage:
    """No target-derived or identifier columns may appear in X."""

    def test_severity_not_in_X(self, X: pd.DataFrame) -> None:
        assert "severity" not in X.columns

    def test_num_casualties_not_in_X(self, X: pd.DataFrame) -> None:
        assert "num_casualties" not in X.columns

    def test_no_leakage_columns(self, X: pd.DataFrame) -> None:
        for col in LEAKAGE_COLUMNS:
            assert col not in X.columns, f"Leakage column found: {col}"

    def test_no_id_or_raw_columns(self, X: pd.DataFrame) -> None:
        for col in ID_AND_RAW_COLUMNS:
            assert col not in X.columns, f"Raw/id column found: {col}"


# ── Temporal features ───────────────────────────────────────────────────────

class TestTemporalFeatures:
    """Verify ranges and types of temporal features."""

    def test_hour_range(self, X: pd.DataFrame) -> None:
        assert X["hour"].between(0, 23).all()

    def test_weekday_range(self, X: pd.DataFrame) -> None:
        assert X["weekday"].between(0, 6).all()

    def test_month_range(self, X: pd.DataFrame) -> None:
        assert X["month"].between(1, 12).all()

    def test_is_weekend_binary(self, X: pd.DataFrame) -> None:
        assert set(X["is_weekend"].unique()).issubset({0, 1})

    def test_is_rush_hour_binary(self, X: pd.DataFrame) -> None:
        assert set(X["is_rush_hour"].unique()).issubset({0, 1})

    def test_is_night_binary(self, X: pd.DataFrame) -> None:
        assert set(X["is_night"].unique()).issubset({0, 1})

    def test_weekend_matches_weekday(self, X: pd.DataFrame) -> None:
        weekend_rows = X[X["is_weekend"] == 1]
        assert (weekend_rows["weekday"] >= 5).all()


# ── Interaction features ────────────────────────────────────────────────────

class TestInteractionFeatures:
    """Interaction columns must be binary."""

    def test_night_and_unlit_binary(self, X: pd.DataFrame) -> None:
        assert set(X["night_and_unlit"].unique()).issubset({0, 1})

    def test_wet_and_high_speed_binary(self, X: pd.DataFrame) -> None:
        assert set(X["wet_and_high_speed"].unique()).issubset({0, 1})

    def test_junction_and_urban_binary(self, X: pd.DataFrame) -> None:
        assert set(X["junction_and_urban"].unique()).issubset({0, 1})


# ── Spatial features ────────────────────────────────────────────────────────

class TestSpatialFeatures:
    """Grid cell count must be non-negative (LOO)."""

    def test_cell_accident_count_non_negative(self, X: pd.DataFrame) -> None:
        assert (X["cell_accident_count"] >= 0).all()

    def test_cell_accident_count_integer(self, X: pd.DataFrame) -> None:
        assert X["cell_accident_count"].dtype in (np.int64, np.int32, int)

    def test_grid_cell_dropped(self, X: pd.DataFrame) -> None:
        assert "grid_cell" not in X.columns


# ── Encoded categoricals ────────────────────────────────────────────────────

class TestEncodedCategoricals:
    """Raw categorical columns must be gone; OHE columns present."""

    def test_no_raw_categorical_columns(self, X: pd.DataFrame) -> None:
        for col in ONE_HOT_COLUMNS:
            assert col not in X.columns, f"Raw column still present: {col}"

    def test_has_one_hot_columns(self, X: pd.DataFrame) -> None:
        ohe = [c for c in X.columns if c.startswith("road_type_")
               or c.startswith("weather_")
               or c.startswith("light_conditions_")]
        assert len(ohe) > 0, "No OHE columns found"

    def test_one_hot_values_are_binary(self, X: pd.DataFrame) -> None:
        prefixes = tuple(f"{p}_" for p in ONE_HOT_COLUMNS)
        ohe = [c for c in X.columns if c.startswith(prefixes)]
        for col in ohe:
            assert set(X[col].unique()).issubset({0, 1}), f"{col} not binary"

    def test_speed_limit_kept_numeric(self, X: pd.DataFrame) -> None:
        assert "speed_limit" in X.columns
        assert pd.api.types.is_numeric_dtype(X["speed_limit"])


# ── Target vectors ──────────────────────────────────────────────────────────

class TestTargets:
    """Binary and multiclass targets must be consistent."""

    def test_y_binary_values(self, y_binary: pd.Series) -> None:
        assert set(y_binary.unique()).issubset({0, 1})

    def test_y_multiclass_values(self, y_multiclass: pd.Series) -> None:
        assert set(y_multiclass.unique()).issubset({1, 2, 3})

    def test_binary_matches_multiclass(
        self, y_binary: pd.Series, y_multiclass: pd.Series,
    ) -> None:
        expected = (y_multiclass.isin([1, 2])).astype(int)
        pd.testing.assert_series_equal(
            y_binary.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_lengths_match(
        self, X: pd.DataFrame, y_binary: pd.Series, y_multiclass: pd.Series,
    ) -> None:
        assert len(X) == len(y_binary) == len(y_multiclass)


# ── All-numeric check ───────────────────────────────────────────────────────

class TestAllNumeric:
    """Every column in X must be numeric (int or float)."""

    def test_all_columns_numeric(self, X: pd.DataFrame) -> None:
        non_numeric = [
            c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])
        ]
        assert non_numeric == [], f"Non-numeric columns: {non_numeric}"
