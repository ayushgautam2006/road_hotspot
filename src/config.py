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

# ── Standard schema ──────────────────────────────────────────────────────────
STANDARD_COLUMNS: List[str] = [
    "accident_id", "date", "time", "latitude", "longitude", "severity",
    "num_vehicles", "num_casualties", "road_type", "speed_limit",
    "light_conditions", "weather", "road_surface", "junction_type",
    "urban_rural",
]
"""Canonical column order for every DataFrame that enters the pipeline."""

# ── Column-name constants ────────────────────────────────────────────────────
COL_ACCIDENT_ID: str = "accident_id"
COL_LATITUDE: str = "latitude"
COL_LONGITUDE: str = "longitude"
COL_SEVERITY: str = "severity"
COL_DATE: str = "date"
COL_TIME: str = "time"
COL_ROAD_TYPE: str = "road_type"
COL_WEATHER: str = "weather"
COL_LIGHT_CONDITIONS: str = "light_conditions"
COL_ROAD_SURFACE: str = "road_surface"
COL_CASUALTIES: str = "num_casualties"
COL_VEHICLES: str = "num_vehicles"
COL_SPEED_LIMIT: str = "speed_limit"
COL_JUNCTION_TYPE: str = "junction_type"
COL_URBAN_RURAL: str = "urban_rural"
COL_CLUSTER_LABEL: str = "cluster_label"
COL_PRIORITY_SCORE: str = "priority_score"

# ── Severity code mapping ────────────────────────────────────────────────────
SEVERITY_CODES: Dict[int, str] = {1: "fatal", 2: "serious", 3: "slight"}
"""Integer severity code → human-readable label."""

# ── Canonical category values (used by synthetic generator) ──────────────────
ROAD_TYPES: List[str] = [
    "single_carriageway", "dual_carriageway", "roundabout",
    "one_way", "slip_road", "unknown",
]
LIGHT_CONDITIONS_VALUES: List[str] = [
    "daylight", "dark_lit", "dark_unlit", "dawn_dusk",
]
WEATHER_VALUES: List[str] = [
    "fine", "rain", "fog", "snow", "high_wind", "other",
]
ROAD_SURFACE_VALUES: List[str] = [
    "dry", "wet", "ice", "snow", "flood",
]
JUNCTION_TYPES: List[str] = [
    "not_junction", "crossroads", "t_junction", "roundabout",
    "slip_road", "other",
]
URBAN_RURAL_VALUES: List[str] = ["urban", "rural"]
SPEED_LIMITS: List[int] = [20, 30, 40, 50, 60, 70]

# ── Default column mapping for real data (UK STATS19 example) ────────────────
DEFAULT_COLUMN_MAPPING: Dict[str, str] = {
    "Accident_Index": "accident_id",
    "Date": "date",
    "Time": "time",
    "Latitude": "latitude",
    "Longitude": "longitude",
    "Accident_Severity": "severity",
    "Number_of_Vehicles": "num_vehicles",
    "Number_of_Casualties": "num_casualties",
    "Road_Type": "road_type",
    "Speed_limit": "speed_limit",
    "Light_Conditions": "light_conditions",
    "Weather_Conditions": "weather",
    "Road_Surface_Conditions": "road_surface",
    "Junction_Detail": "junction_type",
    "Urban_or_Rural_Area": "urban_rural",
}
"""Source column → standard column for UK STATS19 CSVs."""

# ── Clustering hyper-parameters ──────────────────────────────────────────────
CLUSTER_RADIUS_KM: float = 0.5
"""Maximum geographic radius (km) used to define neighbourhood in clustering."""

MIN_CLUSTER_SIZE: int = 25
"""Minimum number of accidents to form a cluster."""

MIN_SAMPLES: int = 10
"""Minimum samples parameter – controls cluster density."""

EARTH_RADIUS_KM: float = 6371.0088
"""Mean Earth radius in kilometres used for haversine conversions."""

HOTSPOT_VALIDATION_RADIUS_KM: float = 1.0
"""Maximum distance threshold (km) to consider a true hotspot recovered."""

# ── Severity weights ─────────────────────────────────────────────────────────
SEVERITY_WEIGHTS: Dict[str, float] = {
    "fatal": 10.0,
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

# ── Synthetic-data file paths ─────────────────────────────────────────────────
RAW_ACCIDENTS_CSV: Path = RAW_DATA_DIR / "accidents_raw.csv"
TRUE_HOTSPOTS_CSV: Path = RAW_DATA_DIR / "true_hotspots.csv"
CLEAN_ACCIDENTS_CSV: Path = PROCESSED_DATA_DIR / "accidents_clean.csv"
HOTSPOTS_CSV: Path = PROCESSED_DATA_DIR / "hotspots.csv"
KDE_GRID_CSV: Path = PROCESSED_DATA_DIR / "kde_grid.csv"
HOTSPOT_SWEEP_CSV: Path = MODELS_OUTPUT_DIR / "hotspot_param_sweep.csv"
HOTSPOT_SWEEP_PLOT: Path = FIGURES_OUTPUT_DIR / "hotspot_param_sweep.png"

# ── Geographic bounds for the fictional city (greater London area) ───────────
CITY_LAT_RANGE: List[float] = [51.35, 51.65]
CITY_LON_RANGE: List[float] = [-0.35, 0.15]
