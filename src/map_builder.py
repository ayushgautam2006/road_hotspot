"""Interactive map generation with Folium.

Builds choropleth / heatmap / marker-cluster maps for accident
hotspots and exports them as standalone HTML files.
"""

from pathlib import Path
from typing import Any

import folium
import pandas as pd


def create_base_map(
    location: list[float] | None = None,
    zoom_start: int | None = None,
) -> folium.Map:
    """Create a Folium base map centred on the given location.

    Parameters
    ----------
    location : list[float] | None
        ``[lat, lon]``.  Defaults to ``config.MAP_DEFAULT_LOCATION``.
    zoom_start : int | None
        Initial zoom.  Defaults to ``config.MAP_DEFAULT_ZOOM``.

    Returns
    -------
    folium.Map
        Base Folium map object.
    """
    raise NotImplementedError


def add_heatmap_layer(
    m: folium.Map,
    df: pd.DataFrame,
) -> folium.Map:
    """Overlay an accident-density heatmap on the map.

    Parameters
    ----------
    m : folium.Map
        Base map.
    df : pd.DataFrame
        Accident data with latitude/longitude.

    Returns
    -------
    folium.Map
        Map with heatmap layer added.
    """
    raise NotImplementedError


def add_cluster_markers(
    m: folium.Map,
    cluster_summary: pd.DataFrame,
) -> folium.Map:
    """Add circle markers for each hotspot cluster centroid.

    Marker size and colour encode priority score.

    Parameters
    ----------
    m : folium.Map
        Base map.
    cluster_summary : pd.DataFrame
        Cluster summary with centroids and priority scores.

    Returns
    -------
    folium.Map
        Map with cluster markers added.
    """
    raise NotImplementedError


def add_accident_markers(
    m: folium.Map,
    df: pd.DataFrame,
) -> folium.Map:
    """Add individual accident point markers (optionally clustered).

    Parameters
    ----------
    m : folium.Map
        Base map.
    df : pd.DataFrame
        Accident-level data.

    Returns
    -------
    folium.Map
        Map with individual markers.
    """
    raise NotImplementedError


def save_map(m: folium.Map, filepath: Path | None = None) -> Path:
    """Save the Folium map to an HTML file.

    Parameters
    ----------
    m : folium.Map
        Map to save.
    filepath : Path | None
        Destination path.  Defaults to ``config.MAPS_OUTPUT_DIR / "map.html"``.

    Returns
    -------
    Path
        Path where the map was saved.
    """
    raise NotImplementedError


def build_hotspot_map(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> folium.Map:
    """High-level helper: build a complete hotspot map in one call.

    Combines base map, heatmap, and cluster markers.

    Parameters
    ----------
    df : pd.DataFrame
        Accident-level data.
    cluster_summary : pd.DataFrame
        Cluster summary with centroids and priority scores.

    Returns
    -------
    folium.Map
        Fully assembled map.
    """
    raise NotImplementedError
