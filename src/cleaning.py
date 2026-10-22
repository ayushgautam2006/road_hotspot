"""Data-cleaning pipeline for accident records.

Handles missing values, outlier removal, type coercion, coordinate
validation, and deduplication.
"""

import pandas as pd


def drop_missing_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where latitude or longitude is null.

    Parameters
    ----------
    df : pd.DataFrame
        Accident data.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    raise NotImplementedError


def impute_missing_values(
    df: pd.DataFrame,
    strategy: str = "mode",
) -> pd.DataFrame:
    """Fill missing categorical/numeric values using the chosen strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Accident data with potential nulls.
    strategy : str
        Imputation strategy – ``"mode"``, ``"median"``, or ``"drop"``.

    Returns
    -------
    pd.DataFrame
        DataFrame with imputed values.
    """
    raise NotImplementedError


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact-duplicate accident records.

    Parameters
    ----------
    df : pd.DataFrame
        Accident data.

    Returns
    -------
    pd.DataFrame
        Deduplicated DataFrame.
    """
    raise NotImplementedError


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cast columns to their expected dtypes (dates, numerics, categories).

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially cleaned data.

    Returns
    -------
    pd.DataFrame
        DataFrame with corrected dtypes.
    """
    raise NotImplementedError


def filter_geographic_bounds(
    df: pd.DataFrame,
    lat_range: tuple[float, float] = (-90.0, 90.0),
    lon_range: tuple[float, float] = (-180.0, 180.0),
) -> pd.DataFrame:
    """Keep only rows whose coordinates fall within valid bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Accident data.
    lat_range : tuple[float, float]
        (min_lat, max_lat).
    lon_range : tuple[float, float]
        (min_lon, max_lon).

    Returns
    -------
    pd.DataFrame
        Geographically filtered DataFrame.
    """
    raise NotImplementedError


def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full cleaning pipeline in sequence.

    Steps: drop missing coords → remove duplicates → coerce types →
    impute missing → filter bounds.

    Parameters
    ----------
    df : pd.DataFrame
        Raw accident data.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.
    """
    raise NotImplementedError
