"""Centralised configuration for the road_hotspots project.

Every tunable value, file path, column-name constant, and hyper-parameter
lives here so that no magic numbers or strings are scattered across modules.
"""

from pathlib import Path
from typing import Dict, List

# ── Project root (two levels up from this file) ──────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# ── Directory paths ───────────────────────────────────────────────────────────
RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
MAPS_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "maps"
FIGURES_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "figures"
MODELS_OUTPUT_DIR: Path = PROJECT_ROOT / "outputs" / "models"

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED: int = 42

# ── Column-name constants ────────────────────────────────────────────────────
COL_LATITUDE: str = "latitude"
COL_LONGITUDE: str = "longitude"
COL_SEVERITY: str = "severity"
COL_DATE: str = "date"
COL_TIME: str = "time"
COL_ROAD_TYPE: str = "road_type"
COL_WEATHER: str = "weather"
COL_LIGHT_CONDITIONS: str = "light_conditions"
COL_ROAD_SURFACE: str = "road_surface"
COL_CASUALTIES: str = "number_of_casualties"
COL_VEHICLES: str = "number_of_vehicles"
COL_SPEED_LIMIT: str = "speed_limit"
COL_CLUSTER_LABEL: str = "cluster_label"
COL_PRIORITY_SCORE: str = "priority_score"

# ── Clustering hyper-parameters ──────────────────────────────────────────────
CLUSTER_RADIUS_KM: float = 0.5
"""Maximum geographic radius (km) used to define neighbourhood in HDBSCAN."""

MIN_CLUSTER_SIZE: int = 5
"""Minimum number of accidents to form a cluster."""

MIN_SAMPLES: int = 3
"""HDBSCAN min_samples parameter – controls cluster density."""

# ── Severity weights ─────────────────────────────────────────────────────────
SEVERITY_WEIGHTS: Dict[str, float] = {
    "fatal": 5.0,
    "serious": 3.0,
    "slight": 1.0,
}
"""Mapping of severity category → numeric weight for priority scoring."""

# ── Priority scoring parameters ──────────────────────────────────────────────
PRIORITY_RECENCY_DECAY: float = 0.1
"""Exponential decay rate applied to older accidents in priority scoring."""

PRIORITY_WEIGHT_FREQUENCY: float = 0.4
"""Weight of accident frequency component in composite priority score."""

PRIORITY_WEIGHT_SEVERITY: float = 0.4
"""Weight of severity component in composite priority score."""

PRIORITY_WEIGHT_TREND: float = 0.2
"""Weight of temporal-trend component in composite priority score."""

# ── Model hyper-parameters ───────────────────────────────────────────────────
LIGHTGBM_PARAMS: Dict[str, object] = {
    "objective": "multiclass",
    "num_class": 3,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "n_estimators": 300,
    "max_depth": 6,
    "random_state": RANDOM_SEED,
    "verbose": -1,
}
"""Default LightGBM parameters for severity classification."""

TEST_SIZE: float = 0.2
"""Fraction of data held out for model evaluation."""

# ── Feature-engineering constants ─────────────────────────────────────────────
TIME_BINS: List[str] = ["night", "morning", "afternoon", "evening"]
"""Labels for discretised time-of-day bins."""

TIME_BIN_EDGES: List[int] = [0, 6, 12, 18, 24]
"""Hour boundaries for time-of-day bins (left-inclusive)."""

# ── Map defaults ──────────────────────────────────────────────────────────────
MAP_DEFAULT_LOCATION: List[float] = [51.5074, -0.1278]
"""Default map centre [lat, lon] – London, UK."""

MAP_DEFAULT_ZOOM: int = 11
"""Default zoom level for Folium maps."""
