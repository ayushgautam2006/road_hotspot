"""Data loading utilities for raw accident datasets.

Supports CSV and GeoJSON ingestion and basic validation before handing
data off to the cleaning pipeline.
"""

from pathlib import Path
from typing import Optional

import geopandas as gpd
import pandas as pd


def load_csv(filepath: Path, **kwargs) -> pd.DataFrame:
    """Read a CSV file into a pandas DataFrame.

    Parameters
    ----------
    filepath : Path
        Absolute or relative path to the CSV file.
    **kwargs
        Additional keyword arguments forwarded to ``pd.read_csv``.

    Returns
    -------
    pd.DataFrame
        Raw accident data.
    """
    raise NotImplementedError


def load_geojson(filepath: Path, **kwargs) -> gpd.GeoDataFrame:
    """Read a GeoJSON file into a GeoDataFrame.

    Parameters
    ----------
    filepath : Path
        Path to the GeoJSON file.
    **kwargs
        Additional keyword arguments forwarded to ``gpd.read_file``.

    Returns
    -------
    gpd.GeoDataFrame
        Raw accident data with geometry column.
    """
    raise NotImplementedError


def load_from_directory(
    directory: Path,
    file_pattern: str = "*.csv",
) -> pd.DataFrame:
    """Concatenate all matching files in a directory into one DataFrame.

    Parameters
    ----------
    directory : Path
        Directory containing one or more data files.
    file_pattern : str
        Glob pattern to filter files (default ``"*.csv"``).

    Returns
    -------
    pd.DataFrame
        Combined raw accident data.
    """
    raise NotImplementedError


def validate_columns(
    df: pd.DataFrame,
    required_columns: Optional[list[str]] = None,
) -> bool:
    """Check that the DataFrame contains all required columns.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : list[str] | None
        Column names to look for.  Defaults to the essential columns
        defined in ``config``.

    Returns
    -------
    bool
        ``True`` if all required columns are present.

    Raises
    ------
    ValueError
        If any required column is missing.
    """
    raise NotImplementedError
