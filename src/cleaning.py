"""Data-cleaning pipeline for accident records.

Handles duplicate removal, coordinate validation, categorical
standardisation, datetime parsing, and missing-value imputation.
Every step logs how many rows were affected so the caller gets a
full cleaning report.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

# ── Columns that undergo categorical cleaning ────────────────────────────────
CATEGORICAL_COLUMNS: List[str] = [
    config.COL_ROAD_TYPE,
    config.COL_LIGHT_CONDITIONS,
    config.COL_WEATHER,
    config.COL_ROAD_SURFACE,
    config.COL_JUNCTION_TYPE,
    config.COL_URBAN_RURAL,
]


# ─────────────────────────────────────────────────────────────────────────────
# Individual cleaning steps (each independently importable and testable)
# ─────────────────────────────────────────────────────────────────────────────

def _standardise_label(value: object) -> object:
    """Normalise one categorical value: strip → lower → spaces to underscores."""
    if pd.isna(value):
        return value
    s = str(value).strip().lower()
    s = re.sub(r"\s+", "_", s)
    return s


def remove_duplicates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop exact-duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, int]
        Cleaned DataFrame and count of rows removed.
    """
    before = len(df)
    df = df.drop_duplicates()
    return df, before - len(df)


def remove_missing_coordinates(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """Drop rows where latitude or longitude is null.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, int]
        Cleaned DataFrame and count of rows removed.
    """
    before = len(df)
    df = df.dropna(subset=[config.COL_LATITUDE, config.COL_LONGITUDE])
    return df, before - len(df)


def remove_oob_coordinates(
    df: pd.DataFrame,
    lat_range: Tuple[float, float] = (-90.0, 90.0),
    lon_range: Tuple[float, float] = (-180.0, 180.0),
) -> Tuple[pd.DataFrame, int]:
    """Keep only rows whose coordinates fall within valid bounds.

    Parameters
    ----------
    df : pd.DataFrame
    lat_range : tuple[float, float]
    lon_range : tuple[float, float]

    Returns
    -------
    tuple[pd.DataFrame, int]
        Filtered DataFrame and count of rows removed.
    """
    before = len(df)
    valid = (
        df[config.COL_LATITUDE].between(*lat_range)
        & df[config.COL_LONGITUDE].between(*lon_range)
    )
    df = df[valid]
    return df, before - len(df)


def standardise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise categorical columns: lowercase, strip, underscore-join.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        DataFrame with cleaned categorical labels.
    """
    df = df.copy()
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(_standardise_label)
    return df


def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Combine *date* and *time* columns into a single ``datetime`` column.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``config.COL_DATE`` and ``config.COL_TIME``.

    Returns
    -------
    pd.DataFrame
        DataFrame with an added ``datetime`` column.
    """
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df[config.COL_DATE].astype(str) + " " + df[config.COL_TIME].astype(str),
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    return df


def impute_missing_categoricals(
    df: pd.DataFrame,
    fill_value: str = "unknown",
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Fill NaN values in categorical columns with an explicit label.

    Parameters
    ----------
    df : pd.DataFrame
    fill_value : str
        Value used to replace NaNs (default ``"unknown"``).

    Returns
    -------
    tuple[pd.DataFrame, dict[str, int]]
        DataFrame with imputed values and a mapping of
        ``{column_name: count_imputed}``.
    """
    df = df.copy()
    imputed: Dict[str, int] = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            null_count = int(df[col].isna().sum())
            if null_count > 0:
                df[col] = df[col].fillna(fill_value)
                imputed[col] = null_count
    return df, imputed


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning report
# ─────────────────────────────────────────────────────────────────────────────

def _print_report(report: Dict[str, Any]) -> None:
    """Print a human-readable cleaning summary to stdout."""
    print("=" * 60)
    print("  CLEANING REPORT")
    print("=" * 60)
    print(f"  Initial rows:              {report['initial_rows']:>8,}")
    print(f"  Duplicates removed:        {report['duplicates_removed']:>8,}")
    print(f"  Missing coords removed:    {report['missing_coords_removed']:>8,}")
    print(f"  Out-of-bounds removed:     {report['oob_coords_removed']:>8,}")
    if report.get("categoricals_imputed"):
        print("  Categoricals imputed:")
        for col, n in report["categoricals_imputed"].items():
            print(f"    {col:25s}  {n:>6,}")
    print("  " + "-" * 40)
    print(f"  Total rows removed:        {report['rows_removed_total']:>8,}")
    print(f"  Final rows:                {report['final_rows']:>8,}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def clean_accidents(
    df: pd.DataFrame,
    save: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Execute the full cleaning pipeline and return cleaned data + report.

    Steps (in order):

    1. Remove exact duplicate rows
    2. Remove rows with missing coordinates
    3. Remove rows with out-of-bounds coordinates
    4. Standardise categorical labels (case, whitespace, underscores)
    5. Parse *date* + *time* into a ``datetime`` column
    6. Impute remaining null categoricals with ``"unknown"``

    Parameters
    ----------
    df : pd.DataFrame
        Raw accident data whose columns match ``config.STANDARD_COLUMNS``.
    save : bool
        If ``True``, persist the cleaned DataFrame to
        ``config.CLEAN_ACCIDENTS_CSV``.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        ``(cleaned_df, report)`` where *report* contains row counts for
        every cleaning step.
    """
    report: Dict[str, Any] = {"initial_rows": len(df)}

    # Step 1 ── duplicates
    df, n = remove_duplicates(df)
    report["duplicates_removed"] = n
    logger.info("Removed %d duplicate rows", n)

    # Step 2 ── missing coordinates
    df, n = remove_missing_coordinates(df)
    report["missing_coords_removed"] = n
    logger.info("Removed %d rows with missing coordinates", n)

    # Step 3 ── out-of-bounds coordinates
    df, n = remove_oob_coordinates(df)
    report["oob_coords_removed"] = n
    logger.info("Removed %d rows with out-of-bounds coordinates", n)

    # Step 4 ── categorical standardisation
    df = standardise_categoricals(df)

    # Step 5 ── datetime parsing
    df = parse_datetime(df)

    # Step 6 ── impute missing categoricals
    df, imputed = impute_missing_categoricals(df)
    report["categoricals_imputed"] = imputed

    # Step 7 ── second dedup pass (standardisation can create new dupes)
    df, n2 = remove_duplicates(df)
    report["duplicates_removed"] += n2
    if n2:
        logger.info("Removed %d additional duplicates after standardisation", n2)

    # Summary
    report["final_rows"] = len(df)
    report["rows_removed_total"] = report["initial_rows"] - report["final_rows"]

    _print_report(report)

    # Reset index
    df = df.reset_index(drop=True)

    # Persist
    if save:
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.CLEAN_ACCIDENTS_CSV, index=False)
        logger.info("Saved cleaned data → %s", config.CLEAN_ACCIDENTS_CSV)

    return df, report
