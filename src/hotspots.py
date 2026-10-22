"""Spatial clustering to identify accident hotspots.

Uses HDBSCAN on haversine-converted coordinates with parameters drawn
from ``config``.
"""

from typing import Any

import numpy as np
import pandas as pd


def prepare_coordinates(df: pd.DataFrame) -> np.ndarray:
    """Extract latitude/longitude and convert to radians for haversine.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain latitude and longitude columns.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 2)`` with coordinates in radians.
    """
    raise NotImplementedError


def cluster_accidents(
    df: pd.DataFrame,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_radius_km: float | None = None,
) -> pd.DataFrame:
    """Run HDBSCAN clustering on accident locations.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched accident data.
    min_cluster_size : int | None
        Override ``config.MIN_CLUSTER_SIZE``.
    min_samples : int | None
        Override ``config.MIN_SAMPLES``.
    cluster_radius_km : float | None
        Override ``config.CLUSTER_RADIUS_KM``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``cluster_label`` column.
    """
    raise NotImplementedError


def summarise_clusters(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statistics per cluster (count, mean severity, centroid).

    Parameters
    ----------
    df : pd.DataFrame
        Clustered accident data (must contain ``cluster_label``).

    Returns
    -------
    pd.DataFrame
        One row per cluster with summary statistics.
    """
    raise NotImplementedError


def get_cluster_model(
    df: pd.DataFrame,
) -> Any:
    """Fit and return the HDBSCAN model object for inspection.

    Parameters
    ----------
    df : pd.DataFrame
        Accident data.

    Returns
    -------
    Any
        Fitted HDBSCAN clusterer instance.
    """
    raise NotImplementedError
