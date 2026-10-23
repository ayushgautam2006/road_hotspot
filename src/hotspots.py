"""Spatial clustering to identify and evaluate road accident hotspots.

Uses HDBSCAN (or DBSCAN) on haversine-converted coordinates with hyper-parameters
configured via ``src.config``. Provides cluster summarisation (centroids, counts,
severity-weighted scoring, dominant environmental conditions, spatial extents),
parameter sweeping with silhouette quality evaluation, 2D Kernel Density Estimation
(KDE) surface generation, and validation against ground-truth hotspot locations.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend safe for all environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, QhullError
from scipy.stats import gaussian_kde
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import haversine_distances

from src import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Coordinate preparation & Haversine utilities
# ─────────────────────────────────────────────────────────────────────────────

def prepare_coordinates(df: pd.DataFrame) -> np.ndarray:
    """Extract latitude/longitude and convert to radians for haversine calculations.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``config.COL_LATITUDE`` and ``config.COL_LONGITUDE`` columns.

    Returns
    -------
    np.ndarray
        Array of shape ``(n, 2)`` containing ``[latitude_rad, longitude_rad]``.

    Raises
    ------
    KeyError
        If latitude or longitude columns are missing.
    ValueError
        If coordinates contain null / NaN values.
    """
    for col in (config.COL_LATITUDE, config.COL_LONGITUDE):
        if col not in df.columns:
            raise KeyError(f"Missing required coordinate column '{col}' in DataFrame")

    coords = df[[config.COL_LATITUDE, config.COL_LONGITUDE]].to_numpy(dtype=float)
    if np.isnan(coords).any():
        raise ValueError("Coordinates array contains NaN values; clean data before clustering")

    return np.radians(coords)


def haversine_distance_km(
    coords_rad1: np.ndarray,
    coords_rad2: np.ndarray,
) -> np.ndarray:
    """Compute pairwise haversine distance in kilometres between two coordinate sets.

    Parameters
    ----------
    coords_rad1 : np.ndarray
        Array of shape ``(n, 2)`` in radians ``[lat, lon]``.
    coords_rad2 : np.ndarray
        Array of shape ``(m, 2)`` in radians ``[lat, lon]``.

    Returns
    -------
    np.ndarray
        Distance matrix of shape ``(n, m)`` in kilometres.
    """
    return haversine_distances(coords_rad1, coords_rad2) * config.EARTH_RADIUS_KM


# ─────────────────────────────────────────────────────────────────────────────
# 2. Clustering models
# ─────────────────────────────────────────────────────────────────────────────

def get_cluster_model(
    df: pd.DataFrame,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_radius_km: float | None = None,
    algorithm: str = "hdbscan",
) -> Any:
    """Fit and return the clustering estimator object for inspection.

    Supports both ``"hdbscan"`` and ``"dbscan"``.

    Parameters
    ----------
    df : pd.DataFrame
        Accident dataset with latitude and longitude.
    min_cluster_size : int | None
        Minimum points to form a cluster. Defaults to ``config.MIN_CLUSTER_SIZE``.
    min_samples : int | None
        Minimum samples in neighborhood. Defaults to ``config.MIN_SAMPLES``.
    cluster_radius_km : float | None
        Neighborhood radius in kilometres. Defaults to ``config.CLUSTER_RADIUS_KM``.
    algorithm : str
        Clustering algorithm to use: ``"hdbscan"`` (default) or ``"dbscan"``.

    Returns
    -------
    Any
        Fitted clustering estimator instance.
    """
    coords_rad = prepare_coordinates(df)
    mcs = min_cluster_size if min_cluster_size is not None else config.MIN_CLUSTER_SIZE
    ms = min_samples if min_samples is not None else config.MIN_SAMPLES
    radius_km = cluster_radius_km if cluster_radius_km is not None else config.CLUSTER_RADIUS_KM
    eps_rad = radius_km / config.EARTH_RADIUS_KM

    algo_norm = algorithm.lower().strip()
    if algo_norm == "dbscan":
        model = DBSCAN(
            eps=eps_rad,
            min_samples=ms,
            metric="haversine",
            algorithm="ball_tree",
        )
        model.fit(coords_rad)
        return model

    if algo_norm == "hdbscan":
        # Attempt with cluster_selection_epsilon; if tree traversal encounters
        # sklearn's known Cython recursion bug with certain deep trees, fall back to 0.0
        try:
            model = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_epsilon=eps_rad,
                metric="haversine",
                copy=True,
            )
            model.fit(coords_rad)
            return model
        except TypeError as err:
            logger.warning(
                "HDBSCAN with cluster_selection_epsilon encountered Cython scalar error (%s); "
                "falling back to cluster_selection_epsilon=0.0",
                err,
            )
            fallback_model = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_epsilon=0.0,
                metric="haversine",
                copy=True,
            )
            fallback_model.fit(coords_rad)
            return fallback_model

    raise ValueError(f"Unsupported algorithm '{algorithm}'. Must be 'hdbscan' or 'dbscan'.")


def cluster_accidents(
    df: pd.DataFrame,
    min_cluster_size: int | None = None,
    min_samples: int | None = None,
    cluster_radius_km: float | None = None,
    algorithm: str = "hdbscan",
) -> pd.DataFrame:
    """Run spatial clustering on accident locations and assign cluster labels.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned or feature-enriched accident data.
    min_cluster_size : int | None
        Override ``config.MIN_CLUSTER_SIZE``.
    min_samples : int | None
        Override ``config.MIN_SAMPLES``.
    cluster_radius_km : float | None
        Override ``config.CLUSTER_RADIUS_KM``.
    algorithm : str
        ``"hdbscan"`` or ``"dbscan"``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with an added ``cluster_label`` column (-1 for noise).
    """
    if len(df) == 0:
        df_empty = df.copy()
        df_empty[config.COL_CLUSTER_LABEL] = pd.Series(dtype=int)
        return df_empty

    model = get_cluster_model(
        df=df,
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_radius_km=cluster_radius_km,
        algorithm=algorithm,
    )

    df_out = df.copy()
    df_out[config.COL_CLUSTER_LABEL] = model.labels_
    return df_out


# ─────────────────────────────────────────────────────────────────────────────
# 3. Cluster Summarisation
# ─────────────────────────────────────────────────────────────────────────────

def _get_dominant_condition(series: pd.Series, fallback: str = "unknown") -> str:
    """Return the most frequent non-null value in a series."""
    valid = series.dropna()
    if len(valid) == 0:
        return fallback
    mode_vals = valid.mode()
    return str(mode_vals.iloc[0]) if len(mode_vals) > 0 else fallback


def _get_hour_band(df: pd.DataFrame) -> pd.Series:
    """Extract an hour band string for each row in the DataFrame."""
    if "hour" in df.columns:
        hours = df["hour"].fillna(12).astype(int)
    elif "datetime" in df.columns:
        hours = pd.to_datetime(df["datetime"], errors="coerce").dt.hour.fillna(12).astype(int)
    elif "time" in df.columns:
        hours = (
            df["time"]
            .astype(str)
            .str.split(":")
            .str[0]
            .str.extract(r"(\d+)", expand=False)
            .fillna(12)
            .astype(int)
        )
    else:
        return pd.Series(["unknown"] * len(df), index=df.index)

    # Discretise into 4 canonical hour bands
    # Night: 22-05, Morning: 06-11, Afternoon: 12-17, Evening: 18-21
    bands = pd.cut(
        hours,
        bins=[-1, 5, 11, 17, 21, 24],
        labels=["night (22-05)", "morning (06-11)", "afternoon (12-17)", "evening (18-21)", "night (22-05)"],
        ordered=False,
    )
    return bands.astype(str)


def compute_convex_hull(points_deg: np.ndarray) -> List[List[float]]:
    """Compute 2D convex hull coordinates [lat, lon] for a set of points.

    Parameters
    ----------
    points_deg : np.ndarray
        Array of shape ``(n, 2)`` in degrees ``[latitude, longitude]``.

    Returns
    -------
    list[list[float]]
        Ordered list of vertex coordinates enclosing the cluster points.
    """
    unique_pts = np.unique(points_deg, axis=0)
    if len(unique_pts) < 3:
        return unique_pts.tolist()

    try:
        hull = ConvexHull(unique_pts)
        return unique_pts[hull.vertices].tolist()
    except (QhullError, ValueError):
        # Fallback for degenerate collinear cases
        return unique_pts.tolist()


def summarise_clusters(
    df: pd.DataFrame,
    severity_weights: Dict[str, float] | None = None,
    save: bool = True,
    save_path: Path | str | None = None,
) -> pd.DataFrame:
    """Aggregate comprehensive statistics per cluster.

    Excludes noise points (``cluster_label == -1``). For each valid cluster:
    * Centroid (latitude, longitude)
    * Accident count
    * Counts of fatal, serious, and slight accidents
    * Average severity
    * Severity-weighted score (weights configurable via ``config.SEVERITY_WEIGHTS``)
    * Dominant conditions: most common hour band, weather, and road type
    * Spatial extent: maximum cluster radius (km and metres) and convex hull vertices

    Parameters
    ----------
    df : pd.DataFrame
        Clustered accident data containing ``config.COL_CLUSTER_LABEL``.
    severity_weights : dict[str, float] | None
        Override weights for severity scoring. Defaults to ``config.SEVERITY_WEIGHTS``.
    save : bool
        Whether to save the summary to CSV. Defaults to ``True``.
    save_path : Path | str | None
        File path for CSV output. Defaults to ``config.HOTSPOTS_CSV``.

    Returns
    -------
    pd.DataFrame
        One row per cluster sorted by ``severity_weighted_score`` descending.
    """
    if config.COL_CLUSTER_LABEL not in df.columns:
        raise KeyError(f"DataFrame must contain '{config.COL_CLUSTER_LABEL}' column")

    weights = severity_weights or config.SEVERITY_WEIGHTS
    w_fatal = float(weights.get("fatal", 10.0))
    w_serious = float(weights.get("serious", 3.0))
    w_slight = float(weights.get("slight", 1.0))

    cols = [
        "cluster_label",
        "centroid_lat",
        "centroid_lon",
        "accident_count",
        "fatal_count",
        "serious_count",
        "slight_count",
        "average_severity",
        "severity_weighted_score",
        "dominant_hour_band",
        "dominant_weather",
        "dominant_road_type",
        "radius_km",
        "radius_m",
        "convex_hull",
    ]

    # Filter out noise
    valid_clusters = df[df[config.COL_CLUSTER_LABEL] != -1]
    if len(valid_clusters) == 0:
        empty_df = pd.DataFrame(columns=cols)
        if save:
            out_file = Path(save_path or config.HOTSPOTS_CSV)
            out_file.parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(out_file, index=False)
        return empty_df

    # Prepare hour band series if not present
    df_annotated = valid_clusters.copy()
    df_annotated["hour_band"] = _get_hour_band(df_annotated)

    # Standardise severity representation for consistent counting
    sev_raw = df_annotated[config.COL_SEVERITY]
    if pd.api.types.is_numeric_dtype(sev_raw):
        # 1 = fatal, 2 = serious, 3 = slight
        sev_numeric = sev_raw.astype(float)
        is_fatal = (sev_raw == 1)
        is_serious = (sev_raw == 2)
        is_slight = (sev_raw == 3)
    else:
        sev_str = sev_raw.astype(str).str.lower().str.strip()
        is_fatal = sev_str.isin(["1", "fatal"])
        is_serious = sev_str.isin(["2", "serious"])
        is_slight = sev_str.isin(["3", "slight"])
        # Map string to numeric severity for average calculation
        sev_numeric = sev_str.map({"fatal": 1.0, "serious": 2.0, "slight": 3.0, "1": 1.0, "2": 2.0, "3": 3.0}).fillna(3.0)

    df_annotated["_is_fatal"] = is_fatal.astype(int)
    df_annotated["_is_serious"] = is_serious.astype(int)
    df_annotated["_is_slight"] = is_slight.astype(int)
    df_annotated["_sev_num"] = sev_numeric

    rows = []
    for cluster_id, grp in df_annotated.groupby(config.COL_CLUSTER_LABEL):
        c_lat = float(grp[config.COL_LATITUDE].mean())
        c_lon = float(grp[config.COL_LONGITUDE].mean())
        count = len(grp)

        n_fatal = int(grp["_is_fatal"].sum())
        n_serious = int(grp["_is_serious"].sum())
        n_slight = int(grp["_is_slight"].sum())
        avg_sev = round(float(grp["_sev_num"].mean()), 3)

        score = round((n_fatal * w_fatal) + (n_serious * w_serious) + (n_slight * w_slight), 2)

        dom_hour = _get_dominant_condition(grp["hour_band"])
        dom_weather = _get_dominant_condition(grp[config.COL_WEATHER]) if config.COL_WEATHER in grp.columns else "unknown"
        dom_road = _get_dominant_condition(grp[config.COL_ROAD_TYPE]) if config.COL_ROAD_TYPE in grp.columns else "unknown"

        # Radii and spatial hull
        pts_deg = grp[[config.COL_LATITUDE, config.COL_LONGITUDE]].to_numpy(dtype=float)
        pts_rad = np.radians(pts_deg)
        c_rad = np.radians(np.array([[c_lat, c_lon]]))
        dists_km = haversine_distance_km(pts_rad, c_rad).flatten()
        max_r_km = round(float(np.max(dists_km)), 4)
        max_r_m = round(max_r_km * 1000.0, 1)

        hull_coords = compute_convex_hull(pts_deg)

        rows.append({
            "cluster_label": int(cluster_id),
            "centroid_lat": round(c_lat, 6),
            "centroid_lon": round(c_lon, 6),
            "accident_count": count,
            "fatal_count": n_fatal,
            "serious_count": n_serious,
            "slight_count": n_slight,
            "average_severity": avg_sev,
            "severity_weighted_score": score,
            "dominant_hour_band": dom_hour,
            "dominant_weather": dom_weather,
            "dominant_road_type": dom_road,
            "radius_km": max_r_km,
            "radius_m": max_r_m,
            "convex_hull": json.dumps(hull_coords),
        })

    summary_df = pd.DataFrame(rows)
    # Rank clusters by priority score descending
    summary_df = summary_df.sort_values("severity_weighted_score", ascending=False).reset_index(drop=True)

    if save:
        out_file = Path(save_path or config.HOTSPOTS_CSV)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_file, index=False)
        logger.info("Saved %d cluster summaries → %s", len(summary_df), out_file)

    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# 4. Parameter Sweep & Quality Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_parameter_sweep(
    df: pd.DataFrame,
    param_grid: List[Dict[str, Any]] | None = None,
    algorithm: str = "hdbscan",
    sample_size: int | None = 10000,
    save: bool = True,
    save_csv_path: Path | str | None = None,
    save_plot_path: Path | str | None = None,
) -> pd.DataFrame:
    """Run hyperparameter sweep reporting cluster counts, noise fraction, and silhouette score.

    Parameters
    ----------
    df : pd.DataFrame
        Accident dataset.
    param_grid : list[dict[str, Any]] | None
        List of dicts specifying hyperparameter combinations to evaluate.
        If ``None``, a sensible grid over ``min_cluster_size`` and ``cluster_radius_km`` is evaluated.
    algorithm : str
        ``"hdbscan"`` or ``"dbscan"``.
    sample_size : int | None
        Optional subsample size for very large datasets to accelerate sweep.
    save : bool
        If True, save sweep results CSV and diagnostic plot.
    save_csv_path : Path | str | None
        Path to output CSV. Defaults to ``config.HOTSPOT_SWEEP_CSV``.
    save_plot_path : Path | str | None
        Path to output PNG plot. Defaults to ``config.HOTSPOT_SWEEP_PLOT``.

    Returns
    -------
    pd.DataFrame
        Table of evaluated parameter settings with clustering metrics.
    """
    if param_grid is None:
        param_grid = [
            {"min_cluster_size": 15, "cluster_radius_km": 0.3},
            {"min_cluster_size": 20, "cluster_radius_km": 0.4},
            {"min_cluster_size": 25, "cluster_radius_km": 0.5},
            {"min_cluster_size": 35, "cluster_radius_km": 0.5},
            {"min_cluster_size": 50, "cluster_radius_km": 0.75},
        ]

    # Optionally take a reproducible subsample for speed if dataset is huge
    if sample_size and len(df) > sample_size:
        eval_df = df.sample(n=sample_size, random_state=config.RANDOM_SEED).reset_index(drop=True)
    else:
        eval_df = df

    coords_rad = prepare_coordinates(eval_df)
    n_total = len(coords_rad)

    results: List[Dict[str, Any]] = []

    for i, p in enumerate(param_grid):
        mcs = p.get("min_cluster_size", config.MIN_CLUSTER_SIZE)
        radius = p.get("cluster_radius_km", config.CLUSTER_RADIUS_KM)
        ms = p.get("min_samples", mcs // 2 if mcs > 4 else 2)

        try:
            model = get_cluster_model(
                eval_df,
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_radius_km=radius,
                algorithm=algorithm,
            )
            labels = model.labels_
        except Exception as err:
            logger.warning("Clustering failed for param setting %s: %s", p, err)
            continue

        unique_labels = set(labels)
        n_clusters = len(unique_labels - {-1})
        noise_count = int(np.sum(labels == -1))
        noise_frac = round(noise_count / n_total, 4)

        # Silhouette-like quality measure on non-noise points
        clustered_mask = (labels != -1)
        n_clustered = int(np.sum(clustered_mask))

        if n_clusters >= 2 and n_clustered > n_clusters:
            # Silhouette on up to 2000 points for speed and stability
            eval_sample = min(2000, n_clustered)
            try:
                score = round(float(
                    silhouette_score(
                        coords_rad[clustered_mask],
                        labels[clustered_mask],
                        metric="haversine",
                        sample_size=eval_sample,
                        random_state=config.RANDOM_SEED,
                    )
                ), 4)
            except Exception:
                score = np.nan
        else:
            score = np.nan

        results.append({
            "setting_id": i + 1,
            "min_cluster_size": mcs,
            "cluster_radius_km": radius,
            "min_samples": ms,
            "num_clusters": n_clusters,
            "noise_count": noise_count,
            "noise_fraction": noise_frac,
            "silhouette_score": score,
        })

    res_df = pd.DataFrame(results)

    if save:
        csv_file = Path(save_csv_path or config.HOTSPOT_SWEEP_CSV)
        csv_file.parent.mkdir(parents=True, exist_ok=True)
        res_df.to_csv(csv_file, index=False)
        logger.info("Saved parameter sweep results → %s", csv_file)

        plot_file = Path(save_plot_path or config.HOTSPOT_SWEEP_PLOT)
        plot_file.parent.mkdir(parents=True, exist_ok=True)
        plot_parameter_sweep(res_df, save_path=plot_file)

    return res_df


def plot_parameter_sweep(
    sweep_df: pd.DataFrame,
    save_path: Path | str | None = None,
) -> None:
    """Generate diagnostic multi-panel plot of hyperparameter sweep metrics.

    Parameters
    ----------
    sweep_df : pd.DataFrame
        Output of ``run_parameter_sweep``.
    save_path : Path | str | None
        Target path for image output.
    """
    if len(sweep_df) == 0:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), dpi=200)

    x_labels = [
        f"mcs={r.min_cluster_size}\nr={r.cluster_radius_km}km"
        for _, r in sweep_df.iterrows()
    ]
    x_pos = np.arange(len(sweep_df))

    # 1. Number of clusters
    axes[0].plot(x_pos, sweep_df["num_clusters"], marker="o", color="#1f77b4", linewidth=2, markersize=7)
    axes[0].set_title("Number of Clusters", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Clusters")
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(x_labels, fontsize=9)
    axes[0].grid(True, linestyle="--", alpha=0.5)

    # 2. Noise fraction
    axes[1].plot(x_pos, sweep_df["noise_fraction"] * 100, marker="s", color="#ff7f0e", linewidth=2, markersize=7)
    axes[1].set_title("Noise Fraction (%)", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Noise %")
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(x_labels, fontsize=9)
    axes[1].grid(True, linestyle="--", alpha=0.5)

    # 3. Silhouette quality score
    clean_scores = sweep_df["silhouette_score"].fillna(0.0)
    axes[2].bar(x_pos, clean_scores, color="#2ca02c", alpha=0.7, width=0.5)
    axes[2].set_title("Silhouette Quality Score", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Silhouette Score")
    axes[2].set_xticks(x_pos)
    axes[2].set_xticklabels(x_labels, fontsize=9)
    axes[2].grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    out_target = Path(save_path or config.HOTSPOT_SWEEP_PLOT)
    out_target.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_target, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved parameter sweep diagnostic plot → %s", out_target)


# ─────────────────────────────────────────────────────────────────────────────
# 5. KDE Density Surface (Heatmap-ready grid)
# ─────────────────────────────────────────────────────────────────────────────

def compute_kde_grid(
    df: pd.DataFrame,
    grid_size: int = 100,
    lat_range: Tuple[float, float] | None = None,
    lon_range: Tuple[float, float] | None = None,
    sample_size: int = 10000,
    save: bool = True,
    save_path: Path | str | None = None,
) -> pd.DataFrame:
    """Compute 2D Gaussian Kernel Density Estimation (KDE) surface on a regular grid.

    Produces a ready-to-render heatmap density surface across the accident domain.

    Parameters
    ----------
    df : pd.DataFrame
        Accident dataset with latitude and longitude.
    grid_size : int
        Number of steps along each axis (produces ``grid_size * grid_size`` points).
    lat_range : tuple[float, float] | None
        Min and max latitude. Defaults to bounds in data or ``config.CITY_LAT_RANGE``.
    lon_range : tuple[float, float] | None
        Min and max longitude. Defaults to bounds in data or ``config.CITY_LON_RANGE``.
    sample_size : int
        Maximum points to sample for KDE bandwidth estimation for speed.
    save : bool
        If True, persists grid to CSV.
    save_path : Path | str | None
        Output CSV path. Defaults to ``config.KDE_GRID_CSV``.

    Returns
    -------
    pd.DataFrame
        Table with ``latitude``, ``longitude``, ``density``, and ``normalized_density``.
    """
    valid = df.dropna(subset=[config.COL_LATITUDE, config.COL_LONGITUDE])
    if len(valid) == 0:
        return pd.DataFrame(columns=["latitude", "longitude", "density", "normalized_density"])

    # Determine spatial bounding box
    if lat_range is None:
        lat_range = (float(valid[config.COL_LATITUDE].min()), float(valid[config.COL_LATITUDE].max()))
    if lon_range is None:
        lon_range = (float(valid[config.COL_LONGITUDE].min()), float(valid[config.COL_LONGITUDE].max()))

    # Subsample if dataset is large for rapid KDE fitting
    if len(valid) > sample_size:
        sample_pts = valid.sample(n=sample_size, random_state=config.RANDOM_SEED)
    else:
        sample_pts = valid

    lats = sample_pts[config.COL_LATITUDE].to_numpy(dtype=float)
    lons = sample_pts[config.COL_LONGITUDE].to_numpy(dtype=float)

    # 2D KDE estimator
    kde = gaussian_kde(np.vstack([lats, lons]))

    # Generate evaluation meshgrid
    lat_lin = np.linspace(lat_range[0], lat_range[1], grid_size)
    lon_lin = np.linspace(lon_range[0], lon_range[1], grid_size)
    lat_mesh, lon_mesh = np.meshgrid(lat_lin, lon_lin)

    mesh_coords = np.vstack([lat_mesh.ravel(), lon_mesh.ravel()])
    densities = kde(mesh_coords)

    max_dens = densities.max() if len(densities) > 0 and densities.max() > 0 else 1.0
    norm_densities = densities / max_dens

    grid_df = pd.DataFrame({
        "latitude": np.round(lat_mesh.ravel(), 6),
        "longitude": np.round(lon_mesh.ravel(), 6),
        "density": densities,
        "normalized_density": np.round(norm_densities, 5),
    })

    if save:
        out_file = Path(save_path or config.KDE_GRID_CSV)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        grid_df.to_csv(out_file, index=False)
        logger.info("Saved KDE heatmap grid (%d points) → %s", len(grid_df), out_file)

    return grid_df


# ─────────────────────────────────────────────────────────────────────────────
# 6. Validation against Ground-Truth Hotspots
# ─────────────────────────────────────────────────────────────────────────────

def validate_hotspots(
    discovered_clusters: pd.DataFrame,
    true_hotspots_path: Path | str | None = None,
    recovery_threshold_km: float | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Compare discovered cluster centroids against ground-truth hotspots.

    Calculates how many true hotspots are recovered (within tolerance distance)
    and computes the mean and median distance error in metres.

    Parameters
    ----------
    discovered_clusters : pd.DataFrame
        Table with ``centroid_lat``, ``centroid_lon``, and ``cluster_label``.
    true_hotspots_path : Path | str | None
        Path to ``true_hotspots.csv``. Defaults to ``config.TRUE_HOTSPOTS_CSV``.
    recovery_threshold_km : float | None
        Distance threshold (km) within which a true hotspot is considered
        recovered. Defaults to ``config.HOTSPOT_VALIDATION_RADIUS_KM`` (1.0 km).

    Returns
    -------
    tuple[dict[str, Any], pd.DataFrame]
        ``(summary, match_details_df)`` where summary has recovery counts and
        distance statistics, and match_details_df contains per-hotspot metrics.
    """
    threshold_km = recovery_threshold_km or config.HOTSPOT_VALIDATION_RADIUS_KM
    true_path = Path(true_hotspots_path or config.TRUE_HOTSPOTS_CSV)

    if not true_path.exists():
        raise FileNotFoundError(f"True hotspots file not found: {true_path}")

    true_df = pd.read_csv(true_path)
    if len(true_df) == 0:
        return {"total_true_hotspots": 0, "recovered_count": 0, "recovery_rate": 0.0}, pd.DataFrame()

    if len(discovered_clusters) == 0:
        details = true_df.copy()
        details["matched_cluster_label"] = -1
        details["distance_m"] = np.nan
        details["recovered"] = False
        summary = {
            "total_true_hotspots": len(true_df),
            "recovered_count": 0,
            "recovery_rate": 0.0,
            "mean_distance_error_m": np.nan,
            "median_distance_error_m": np.nan,
        }
        return summary, details

    # Radian coordinates
    true_rad = np.radians(true_df[[config.COL_LATITUDE, config.COL_LONGITUDE]].to_numpy())
    clust_rad = np.radians(discovered_clusters[["centroid_lat", "centroid_lon"]].to_numpy())

    dist_matrix_km = haversine_distance_km(true_rad, clust_rad)

    details_list = []
    recovered_errors_m = []

    for i, (_, row) in enumerate(true_df.iterrows()):
        row_dists = dist_matrix_km[i]
        nearest_idx = int(np.argmin(row_dists))
        min_d_km = float(row_dists[nearest_idx])
        min_d_m = round(min_d_km * 1000.0, 1)
        matched_label = int(discovered_clusters.iloc[nearest_idx]["cluster_label"])
        is_recovered = bool(min_d_km <= threshold_km)

        if is_recovered:
            recovered_errors_m.append(min_d_m)

        details_list.append({
            "name": row.get("name", f"True_Hotspot_{i}"),
            "true_lat": float(row[config.COL_LATITUDE]),
            "true_lon": float(row[config.COL_LONGITUDE]),
            "matched_cluster_label": matched_label,
            "distance_m": min_d_m,
            "distance_km": round(min_d_km, 3),
            "recovered": is_recovered,
        })

    details_df = pd.DataFrame(details_list)
    recovered_cnt = len(recovered_errors_m)
    tot_cnt = len(true_df)
    rec_rate = round(recovered_cnt / tot_cnt, 4) if tot_cnt > 0 else 0.0

    mean_err_m = round(float(np.mean(recovered_errors_m)), 1) if recovered_cnt > 0 else np.nan
    med_err_m = round(float(np.median(recovered_errors_m)), 1) if recovered_cnt > 0 else np.nan

    summary = {
        "total_true_hotspots": tot_cnt,
        "recovered_count": recovered_cnt,
        "recovery_rate": rec_rate,
        "recovery_threshold_km": threshold_km,
        "mean_distance_error_m": mean_err_m,
        "median_distance_error_m": med_err_m,
    }

    _print_validation_report(summary, details_df)
    return summary, details_df


def _print_validation_report(summary: Dict[str, Any], details: pd.DataFrame) -> None:
    """Print formatted validation recovery report."""
    print("=" * 65)
    print("  HOTSPOT VALIDATION REPORT (vs. Ground-Truth)")
    print("=" * 65)
    print(f"  Total True Hotspots:       {summary['total_true_hotspots']:>6}")
    print(f"  Recovered Hotspots:        {summary['recovered_count']:>6}  ({summary['recovery_rate']:.1%})")
    print(f"  Recovery Threshold:        {summary['recovery_threshold_km']:>6.1f} km")
    if not np.isnan(summary.get("mean_distance_error_m", np.nan)):
        print(f"  Mean Distance Error:       {summary['mean_distance_error_m']:>6.1f} m")
        print(f"  Median Distance Error:     {summary['median_distance_error_m']:>6.1f} m")
    print("  " + "-" * 50)
    print(f"  {'Hotspot Name':22s} {'Cluster':>8s} {'Distance':>10s} {'Status':>10s}")
    for _, r in details.iterrows():
        status = "RECOVERED" if r["recovered"] else "MISSED"
        print(f"  {r['name']:22s} {r['matched_cluster_label']:>8d} {r['distance_m']:>8.1f} m {status:>10s}")
    print("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# 7. End-to-End Pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_hotspot_pipeline(
    df: pd.DataFrame | None = None,
    min_cluster_size: int | None = None,
    cluster_radius_km: float | None = None,
    algorithm: str = "hdbscan",
    run_sweep: bool = True,
    sweep_sample_size: int | None = 10000,
    save: bool = True,
) -> Dict[str, Any]:
    """Run the complete hotspot analysis pipeline end-to-end.

    Steps:
    1. Load cleaned accidents dataset if not passed.
    2. Optional parameter sweep with diagnostic plot.
    3. Spatial clustering with haversine distance.
    4. Comprehensive cluster summarisation.
    5. KDE density surface generation.
    6. Validation against ground truth.

    Parameters
    ----------
    df : pd.DataFrame | None
        Input accident data. If ``None``, loads ``config.CLEAN_ACCIDENTS_CSV``.
    min_cluster_size : int | None
        Clustering parameter.
    cluster_radius_km : float | None
        Clustering parameter.
    algorithm : str
        Clustering algorithm (``"hdbscan"`` or ``"dbscan"``).
    run_sweep : bool
        Whether to run parameter sweep first.
    sweep_sample_size : int | None
        Sample size for parameter sweep.
    save : bool
        Whether to save processed artifacts.

    Returns
    -------
    dict[str, Any]
        Dictionary of results containing clustered data, hotspots summary,
        KDE grid, parameter sweep results, and validation metrics.
    """
    if df is None:
        if not config.CLEAN_ACCIDENTS_CSV.exists():
            raise FileNotFoundError(
                f"Cleaned accidents CSV not found: {config.CLEAN_ACCIDENTS_CSV}. "
                "Run data cleaning first."
            )
        df = pd.read_csv(config.CLEAN_ACCIDENTS_CSV)

    sweep_results = None
    if run_sweep:
        logger.info("Running hyperparameter sweep...")
        sweep_results = run_parameter_sweep(
            df,
            algorithm=algorithm,
            sample_size=sweep_sample_size,
            save=save,
        )

    logger.info("Clustering accident data...")
    clustered_df = cluster_accidents(
        df,
        min_cluster_size=min_cluster_size,
        cluster_radius_km=cluster_radius_km,
        algorithm=algorithm,
    )

    logger.info("Summarising clusters...")
    hotspots_df = summarise_clusters(clustered_df, save=save)

    logger.info("Computing KDE density surface...")
    kde_df = compute_kde_grid(df, save=save)

    val_summary, val_details = None, None
    if config.TRUE_HOTSPOTS_CSV.exists():
        logger.info("Validating discovered hotspots against true hotspots...")
        val_summary, val_details = validate_hotspots(hotspots_df)

    return {
        "clustered_data": clustered_df,
        "hotspots": hotspots_df,
        "kde_grid": kde_df,
        "sweep_results": sweep_results,
        "validation_summary": val_summary,
        "validation_details": val_details,
    }
