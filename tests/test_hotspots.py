"""Tests for src.hotspots – spatial clustering, summarisation, sweep, KDE, and validation.

Uses a hand-made dataset with two obvious spatial clusters and background noise.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest

from src import config
from src.hotspots import (
    cluster_accidents,
    compute_convex_hull,
    compute_kde_grid,
    get_cluster_model,
    haversine_distance_km,
    prepare_coordinates,
    run_parameter_sweep,
    summarise_clusters,
    validate_hotspots,
)


# ─────────────────────────────────────────────────────────────────────────────
# Hand-made dataset fixture: 2 obvious tight clusters + outlier noise
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def two_cluster_data() -> Tuple[pd.DataFrame, Tuple[float, float], Tuple[float, float]]:
    """Create a hand-made dataset with two tight spatial clusters and 3 noise points.

    Cluster 1: centered at (51.5074, -0.1278) [London central]
               25 points, mostly severe accidents in rain at roundabout in morning.
    Cluster 2: centered at (51.6500, 0.1500) [Far North-East, ~25 km away]
               25 points, mostly slight accidents in fine weather on single carriageway in afternoon.
    Noise:     3 isolated points far from both clusters.
    """
    rng = np.random.default_rng(config.RANDOM_SEED)

    c1_center = (51.5074, -0.1278)
    c2_center = (51.6500, 0.1500)

    # 25 points in Cluster 1 (spread ~50-100 metres)
    c1_lats = c1_center[0] + rng.normal(0, 0.0005, size=25)
    c1_lons = c1_center[1] + rng.normal(0, 0.0005, size=25)
    # Severity: 5 fatal (1), 8 serious (2), 12 slight (3)
    c1_sev = [1] * 5 + [2] * 8 + [3] * 12

    # 25 points in Cluster 2
    c2_lats = c2_center[0] + rng.normal(0, 0.0005, size=25)
    c2_lons = c2_center[1] + rng.normal(0, 0.0005, size=25)
    # Severity: 2 fatal (1), 3 serious (2), 20 slight (3)
    c2_sev = [1] * 2 + [2] * 3 + [3] * 20

    # 3 Noise points far from both
    noise_lats = [51.2000, 51.8500, 51.3000]
    noise_lons = [-0.5000, 0.4500, 0.3500]
    noise_sev = [3, 2, 3]

    lats = np.concatenate([c1_lats, c2_lats, noise_lats])
    lons = np.concatenate([c1_lons, c2_lons, noise_lons])
    sevs = c1_sev + c2_sev + noise_sev

    n_total = len(lats)
    df = pd.DataFrame({
        config.COL_ACCIDENT_ID: [f"ACC_{i:04d}" for i in range(n_total)],
        config.COL_LATITUDE: lats,
        config.COL_LONGITUDE: lons,
        config.COL_SEVERITY: sevs,
        config.COL_DATE: ["2023-05-15"] * n_total,
        config.COL_TIME: ["08:30"] * 25 + ["14:15"] * 25 + ["23:00"] * 3,
        config.COL_WEATHER: ["rain"] * 20 + ["fog"] * 5 + ["fine"] * 20 + ["other"] * 5 + ["fine"] * 3,
        config.COL_ROAD_TYPE: ["roundabout"] * 22 + ["one_way"] * 3 + ["single_carriageway"] * 23 + ["slip_road"] * 2 + ["unknown"] * 3,
        config.COL_LIGHT_CONDITIONS: ["daylight"] * n_total,
        config.COL_ROAD_SURFACE: ["wet"] * 25 + ["dry"] * 28,
        config.COL_JUNCTION_TYPE: ["roundabout"] * 25 + ["not_junction"] * 28,
        config.COL_URBAN_RURAL: ["urban"] * n_total,
        config.COL_CASUALTIES: [1] * n_total,
        config.COL_VEHICLES: [2] * n_total,
        config.COL_SPEED_LIMIT: [30] * n_total,
    })

    return df, c1_center, c2_center


# ─────────────────────────────────────────────────────────────────────────────
# Test coordinate preparation & haversine
# ─────────────────────────────────────────────────────────────────────────────

def test_prepare_coordinates_shape_and_radians(two_cluster_data) -> None:
    """prepare_coordinates must extract radians array with shape (n, 2)."""
    df, _, _ = two_cluster_data
    coords_rad = prepare_coordinates(df)

    assert isinstance(coords_rad, np.ndarray)
    assert coords_rad.shape == (len(df), 2)
    # Radians for London latitude should be ~0.9 radians (51.5 * pi / 180)
    assert np.all((coords_rad[:, 0] > 0.8) & (coords_rad[:, 0] < 1.0))
    # Radians for longitude should be close to 0
    assert np.all((coords_rad[:, 1] > -0.05) & (coords_rad[:, 1] < 0.05))


def test_prepare_coordinates_missing_column() -> None:
    """Raises KeyError when latitude or longitude is missing."""
    df = pd.DataFrame({"latitude": [51.5]})
    with pytest.raises(KeyError, match="Missing required coordinate column"):
        prepare_coordinates(df)


def test_prepare_coordinates_nan_raises() -> None:
    """Raises ValueError when coordinates contain NaN."""
    df = pd.DataFrame({"latitude": [51.5, np.nan], "longitude": [-0.1, -0.1]})
    with pytest.raises(ValueError, match="contains NaN"):
        prepare_coordinates(df)


def test_haversine_distance_km() -> None:
    """haversine_distance_km should return known distance between two coordinates."""
    # 1 degree of latitude is roughly 111.19 km
    p1 = np.radians([[51.0, 0.0]])
    p2 = np.radians([[52.0, 0.0]])
    dist = haversine_distance_km(p1, p2)[0, 0]
    assert 111.0 < dist < 112.0


# ─────────────────────────────────────────────────────────────────────────────
# Test clustering logic
# ─────────────────────────────────────────────────────────────────────────────

def test_cluster_accidents_finds_two_clusters(two_cluster_data) -> None:
    """Clustering must detect exactly 2 clusters and flag noise as -1."""
    df, _, _ = two_cluster_data
    # Use min_cluster_size=10, radius=1.0 km
    clustered = cluster_accidents(
        df,
        min_cluster_size=10,
        cluster_radius_km=1.0,
        algorithm="hdbscan",
    )

    assert config.COL_CLUSTER_LABEL in clustered.columns
    labels = clustered[config.COL_CLUSTER_LABEL].to_numpy()

    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) == 2, f"Expected 2 clusters, got {unique_clusters}"

    # Cluster sizes should be 25 each
    for c in unique_clusters:
        assert (labels == c).sum() == 25

    # Outliers should be marked as noise (-1)
    noise_count = (labels == -1).sum()
    assert noise_count == 3


def test_cluster_accidents_dbscan_option(two_cluster_data) -> None:
    """DBSCAN algorithm option should also identify the 2 clusters."""
    df, _, _ = two_cluster_data
    clustered = cluster_accidents(
        df,
        min_cluster_size=10,
        cluster_radius_km=1.0,
        algorithm="dbscan",
    )

    labels = clustered[config.COL_CLUSTER_LABEL].to_numpy()
    unique_clusters = set(labels) - {-1}
    assert len(unique_clusters) == 2
    assert (labels == -1).sum() == 3


def test_get_cluster_model_fit(two_cluster_data) -> None:
    """get_cluster_model returns a fitted estimator object with labels_ attribute."""
    df, _, _ = two_cluster_data
    model = get_cluster_model(df, min_cluster_size=10, algorithm="hdbscan")
    assert hasattr(model, "labels_")
    assert len(model.labels_) == len(df)


# ─────────────────────────────────────────────────────────────────────────────
# Test cluster summarisation
# ─────────────────────────────────────────────────────────────────────────────

def test_summarise_clusters_metrics_and_schema(two_cluster_data, tmp_path: Path) -> None:
    """summarise_clusters must calculate accurate statistics, scores, and dominant values."""
    df, c1_center, c2_center = two_cluster_data
    clustered = cluster_accidents(df, min_cluster_size=10, cluster_radius_km=1.0)

    csv_path = tmp_path / "test_hotspots.csv"
    summary = summarise_clusters(clustered, save=True, save_path=csv_path)

    assert csv_path.exists()
    assert len(summary) == 2

    # Check required schema columns
    expected_cols = [
        "cluster_label", "centroid_lat", "centroid_lon", "accident_count",
        "fatal_count", "serious_count", "slight_count", "average_severity",
        "severity_weighted_score", "dominant_hour_band", "dominant_weather",
        "dominant_road_type", "radius_km", "radius_m", "convex_hull",
    ]
    for col in expected_cols:
        assert col in summary.columns

    # Verify counts
    assert list(summary["accident_count"]) == [25, 25]

    # Cluster 1 has: fatal=5, serious=8, slight=12
    # Score with weights (10, 3, 1): 5*10 + 8*3 + 12*1 = 50 + 24 + 12 = 86
    # Cluster 2 has: fatal=2, serious=3, slight=20
    # Score with weights (10, 3, 1): 2*10 + 3*3 + 20*1 = 20 + 9 + 20 = 49
    # Since summary is sorted descending by score:
    c_high = summary.iloc[0]
    c_low = summary.iloc[1]

    assert c_high["severity_weighted_score"] == 86.0
    assert c_high["fatal_count"] == 5
    assert c_high["serious_count"] == 8
    assert c_high["slight_count"] == 12
    assert c_high["dominant_weather"] == "rain"
    assert c_high["dominant_road_type"] == "roundabout"
    assert c_high["dominant_hour_band"] == "morning (06-11)"

    assert c_low["severity_weighted_score"] == 49.0
    assert c_low["fatal_count"] == 2
    assert c_low["serious_count"] == 3
    assert c_low["slight_count"] == 20
    assert c_low["dominant_weather"] == "fine"
    assert c_low["dominant_road_type"] == "single_carriageway"
    assert c_low["dominant_hour_band"] == "afternoon (12-17)"

    # Centroid verification: should be extremely close to ground truth center
    assert abs(c_high["centroid_lat"] - c1_center[0]) < 0.002
    assert abs(c_high["centroid_lon"] - c1_center[1]) < 0.002

    # Spatial extents
    assert c_high["radius_km"] > 0
    assert c_high["radius_m"] > 0
    hull_parsed = json.loads(c_high["convex_hull"])
    assert isinstance(hull_parsed, list)
    assert len(hull_parsed) >= 3


def test_compute_convex_hull() -> None:
    """compute_convex_hull handles degenerate and regular point sets."""
    # 2 points
    pts_2 = np.array([[51.5, -0.1], [51.6, -0.2]])
    hull_2 = compute_convex_hull(pts_2)
    assert len(hull_2) == 2

    # Triangle
    pts_3 = np.array([[51.5, -0.1], [51.6, -0.1], [51.55, 0.0]])
    hull_3 = compute_convex_hull(pts_3)
    assert len(hull_3) == 3


# ─────────────────────────────────────────────────────────────────────────────
# Test parameter sweep and plotting
# ─────────────────────────────────────────────────────────────────────────────

def test_run_parameter_sweep(two_cluster_data, tmp_path: Path) -> None:
    """run_parameter_sweep must evaluate settings and write CSV & plot."""
    df, _, _ = two_cluster_data

    custom_grid = [
        {"min_cluster_size": 5, "cluster_radius_km": 0.5},
        {"min_cluster_size": 15, "cluster_radius_km": 1.0},
    ]
    csv_file = tmp_path / "sweep.csv"
    plot_file = tmp_path / "sweep.png"

    sweep_df = run_parameter_sweep(
        df,
        param_grid=custom_grid,
        algorithm="hdbscan",
        save=True,
        save_csv_path=csv_file,
        save_plot_path=plot_file,
    )

    assert csv_file.exists()
    assert plot_file.exists()
    assert len(sweep_df) == 2

    for col in [
        "setting_id", "min_cluster_size", "cluster_radius_km",
        "num_clusters", "noise_count", "noise_fraction", "silhouette_score",
    ]:
        assert col in sweep_df.columns

    # In both settings with these clusters, 2 clusters should be found
    assert sweep_df["num_clusters"].iloc[0] == 2
    assert sweep_df["noise_fraction"].iloc[0] < 0.20
    assert sweep_df["silhouette_score"].iloc[0] > 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Test KDE density surface
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_kde_grid(two_cluster_data, tmp_path: Path) -> None:
    """compute_kde_grid produces a regular density grid saved to CSV."""
    df, _, _ = two_cluster_data
    grid_csv = tmp_path / "kde.csv"

    grid_df = compute_kde_grid(
        df,
        grid_size=15,
        save=True,
        save_path=grid_csv,
    )

    assert grid_csv.exists()
    assert len(grid_df) == 15 * 15
    assert set(grid_df.columns) == {"latitude", "longitude", "density", "normalized_density"}
    assert (grid_df["density"] >= 0).all()
    assert np.isclose(grid_df["normalized_density"].max(), 1.0)


# ─────────────────────────────────────────────────────────────────────────────
# Test Ground-Truth Validation
# ─────────────────────────────────────────────────────────────────────────────

def test_validate_hotspots(two_cluster_data, tmp_path: Path) -> None:
    """validate_hotspots should match discovered centroids to known true hotspots."""
    df, c1_center, c2_center = two_cluster_data
    clustered = cluster_accidents(df, min_cluster_size=10, cluster_radius_km=1.0)
    summary = summarise_clusters(clustered, save=False)

    # Hand-made true hotspots file matching c1 and c2
    true_csv = tmp_path / "test_true.csv"
    true_data = pd.DataFrame([
        {"latitude": c1_center[0], "longitude": c1_center[1], "spatial_std": 0.004, "name": "True_City"},
        {"latitude": c2_center[0], "longitude": c2_center[1], "spatial_std": 0.005, "name": "True_NorthEast"},
    ])
    true_data.to_csv(true_csv, index=False)

    val_summary, val_details = validate_hotspots(
        summary,
        true_hotspots_path=true_csv,
        recovery_threshold_km=1.0,
    )

    assert val_summary["total_true_hotspots"] == 2
    assert val_summary["recovered_count"] == 2
    assert val_summary["recovery_rate"] == 1.0
    # Mean error should be very small (< 200m)
    assert val_summary["mean_distance_error_m"] < 200.0

    assert len(val_details) == 2
    assert all(val_details["recovered"])


# ─────────────────────────────────────────────────────────────────────────────
# Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

def test_empty_dataframe_handling() -> None:
    """Empty input data should return empty outputs without crashing."""
    empty_df = pd.DataFrame(columns=[config.COL_LATITUDE, config.COL_LONGITUDE, config.COL_SEVERITY])

    clustered = cluster_accidents(empty_df)
    assert len(clustered) == 0
    assert config.COL_CLUSTER_LABEL in clustered.columns

    summary = summarise_clusters(clustered, save=False)
    assert len(summary) == 0
    assert "cluster_label" in summary.columns

    grid = compute_kde_grid(empty_df, save=False)
    assert len(grid) == 0


def test_all_noise_handling() -> None:
    """When all points are noise, summarise_clusters should return an empty summary."""
    df = pd.DataFrame({
        config.COL_LATITUDE: [51.5],
        config.COL_LONGITUDE: [-0.1],
        config.COL_SEVERITY: [3],
        config.COL_CLUSTER_LABEL: [-1],
    })
    summary = summarise_clusters(df, save=False)
    assert len(summary) == 0
