"""Data loading utilities for road accident datasets.

Provides two data sources:

1. ``generate_synthetic`` – creates realistic synthetic accident data with
   hidden hotspot clusters, factor-dependent severity, rush-hour / weekend
   time patterns, and deliberate data-quality issues for testing the
   cleaning pipeline.
2. ``load_real`` – reads a CSV file and maps its columns to the standard
   schema using a caller-supplied (or default) column mapping.

The top-level ``load_accidents`` function is the single entry point that the
rest of the pipeline should use.  It returns a DataFrame whose columns match
``config.STANDARD_COLUMNS`` regardless of the underlying source.
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src import config


# ─────────────────────────────────────────────────────────────────────────────
# True hotspot centres for the fictional city  (lat, lon, σ, name, weight)
# ─────────────────────────────────────────────────────────────────────────────
_HOTSPOT_CENTRES: List[Tuple[float, float, float, str, float]] = [
    (51.5074, -0.1278, 0.0045, "City Centre",       0.14),
    (51.5155, -0.0922, 0.0040, "Liverpool Street",   0.11),
    (51.5014, -0.1419, 0.0030, "Westminster",        0.08),
    (51.5225, -0.1553, 0.0055, "Euston/Kings Cross", 0.12),
    (51.4700, -0.0800, 0.0065, "Brixton/Camberwell", 0.10),
    (51.5400, -0.0100, 0.0050, "Stratford",          0.09),
    (51.4613, -0.1156, 0.0035, "Oval/Kennington",    0.07),
    (51.5558, -0.1085, 0.0055, "Finsbury Park",      0.10),
    (51.4850, -0.1750, 0.0045, "Chelsea/Fulham",     0.07),
]
"""Nine hidden hotspot locations; *weight* controls the fraction of
cluster-assigned accidents that originate from each centre."""

_HOTSPOT_FRACTION: float = 0.65
"""Fraction of total accidents generated from hotspot clusters (rest is
uniform background noise)."""


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _generate_dates(
    rng: np.random.Generator,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return arrays of dates and times with rush-hour / weekend patterns.

    Weekday accidents are concentrated at 07-09 and 16-19 (rush hours).
    Weekend accidents have a flatter distribution shifted towards late
    evening / night.
    """
    # Random dates spanning ~3 years
    start = np.datetime64("2021-01-01")
    days_span = 3 * 365
    day_offsets = rng.integers(0, days_span, size=n)
    dates = start + day_offsets.astype("timedelta64[D]")

    # Day-of-week: 0 = Monday … 6 = Sunday
    # numpy weekday: Monday=0
    dow = ((dates - np.datetime64("2021-01-04")) // np.timedelta64(1, "D")) % 7

    hours = np.empty(n, dtype=int)
    minutes = rng.integers(0, 60, size=n)

    for i in range(n):
        if dow[i] < 5:  # weekday
            r = rng.random()
            if r < 0.25:
                hours[i] = rng.choice([7, 8])          # morning rush
            elif r < 0.50:
                hours[i] = rng.choice([16, 17, 18])     # evening rush
            elif r < 0.70:
                hours[i] = rng.integers(9, 16)          # midday
            elif r < 0.85:
                hours[i] = rng.integers(19, 23)         # evening
            else:
                hours[i] = rng.integers(0, 7)           # night
        else:  # weekend
            r = rng.random()
            if r < 0.30:
                hours[i] = rng.integers(20, 24)         # late evening
            elif r < 0.50:
                hours[i] = rng.integers(0, 5)           # night
            elif r < 0.80:
                hours[i] = rng.integers(10, 18)         # daytime
            else:
                hours[i] = rng.integers(5, 10)          # morning

    time_strings = np.array(
        [f"{h:02d}:{m:02d}" for h, m in zip(hours, minutes)]
    )
    date_strings = dates.astype(str)
    return date_strings, time_strings


def _assign_severity(
    rng: np.random.Generator,
    speed_limits: np.ndarray,
    light_conds: np.ndarray,
    weather: np.ndarray,
    road_surface: np.ndarray,
    junction_type: np.ndarray,
) -> np.ndarray:
    """Compute severity codes (1/2/3) influenced by realistic risk factors.

    Returns an array calibrated so the overall distribution is approximately
    1 % fatal, 14 % serious, 85 % slight.
    """
    n = len(speed_limits)
    # Risk score starts at 0; each factor adds to it
    risk = np.zeros(n, dtype=float)

    # Speed: higher → riskier
    speed_risk = {20: 0.0, 30: 0.3, 40: 0.8, 50: 1.4, 60: 2.2, 70: 3.0}
    for spd, r in speed_risk.items():
        risk[speed_limits == spd] += r

    # Light
    light_risk = {"daylight": 0.0, "dark_lit": 0.6, "dark_unlit": 1.5, "dawn_dusk": 0.3}
    for val, r in light_risk.items():
        risk[light_conds == val] += r

    # Weather
    weather_risk = {"fine": 0.0, "rain": 0.4, "fog": 0.9, "snow": 1.2, "high_wind": 0.5, "other": 0.2}
    for val, r in weather_risk.items():
        risk[weather == val] += r

    # Road surface
    surface_risk = {"dry": 0.0, "wet": 0.4, "ice": 1.3, "snow": 1.0, "flood": 0.8}
    for val, r in surface_risk.items():
        risk[road_surface == val] += r

    # Junction
    junction_risk = {
        "not_junction": 0.0, "crossroads": 0.6, "t_junction": 0.5,
        "roundabout": 0.3, "slip_road": 0.4, "other": 0.3,
    }
    for val, r in junction_risk.items():
        risk[junction_type == val] += r

    # Map risk → severity probabilities  [fatal, serious, slight]
    # Calibrated so weighted average ≈ [0.01, 0.14, 0.85]
    severity = np.empty(n, dtype=int)
    for i in range(n):
        s = risk[i]
        if s < 1.0:
            probs = [0.002, 0.06, 0.938]
        elif s < 2.0:
            probs = [0.005, 0.10, 0.895]
        elif s < 3.0:
            probs = [0.012, 0.17, 0.818]
        elif s < 4.0:
            probs = [0.030, 0.28, 0.690]
        elif s < 5.0:
            probs = [0.055, 0.38, 0.565]
        else:
            probs = [0.090, 0.45, 0.460]
        severity[i] = rng.choice([1, 2, 3], p=probs)

    return severity


def _inject_noise(
    rng: np.random.Generator,
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Deliberately degrade data quality for pipeline testing.

    Introduces:
    - ~2 % missing coordinates (set to NaN)
    - ~0.3 % out-of-bounds coordinates
    - ~1 % duplicated rows
    - Inconsistent category labels in weather / road_surface
    - ~3 % nulls scattered in weather and road_surface
    """
    n = len(df)
    df = df.copy()

    # ── Missing coordinates ──────────────────────────────────────────────
    miss_mask = rng.random(n) < 0.02
    df.loc[miss_mask, "latitude"] = np.nan
    # A few missing longitudes too (different rows)
    miss_lon = rng.random(n) < 0.008
    df.loc[miss_lon, "longitude"] = np.nan

    # ── Out-of-bounds coordinates ────────────────────────────────────────
    oob_mask = rng.random(n) < 0.003
    df.loc[oob_mask, "latitude"] = rng.uniform(90, 180, size=oob_mask.sum())
    oob_mask2 = rng.random(n) < 0.003
    df.loc[oob_mask2, "longitude"] = rng.uniform(200, 360, size=oob_mask2.sum())

    # ── Duplicate rows ───────────────────────────────────────────────────
    dup_idx = rng.choice(n, size=int(n * 0.01), replace=False)
    dups = df.iloc[dup_idx].copy()
    df = pd.concat([df, dups], ignore_index=True)

    # ── Inconsistent labels ──────────────────────────────────────────────
    # Randomly mangle ~5 % of weather values
    mangle_mask = rng.random(len(df)) < 0.05
    mangled = df.loc[mangle_mask, "weather"].copy()
    transforms = [str.upper, str.title, lambda s: f" {s} ", lambda s: s.replace("_", " ")]
    for idx in mangled.index:
        val = mangled[idx]
        if isinstance(val, str):
            fn = rng.choice(transforms)  # type: ignore[arg-type]
            mangled[idx] = fn(val)
    df.loc[mangle_mask, "weather"] = mangled

    # Same for road_surface ~3 %
    mangle_rs = rng.random(len(df)) < 0.03
    mangled_rs = df.loc[mangle_rs, "road_surface"].copy()
    for idx in mangled_rs.index:
        val = mangled_rs[idx]
        if isinstance(val, str):
            fn = rng.choice(transforms)  # type: ignore[arg-type]
            mangled_rs[idx] = fn(val)
    df.loc[mangle_rs, "road_surface"] = mangled_rs

    # ── Nulls in weather / road_surface ──────────────────────────────────
    null_w = rng.random(len(df)) < 0.03
    df.loc[null_w, "weather"] = np.nan
    null_rs = rng.random(len(df)) < 0.02
    df.loc[null_rs, "road_surface"] = np.nan

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_synthetic(
    n: int = 50_000,
    seed: int = config.RANDOM_SEED,
    save: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate realistic synthetic accident data for a fictional city.

    Creates *n* accident records distributed across 9 hidden hotspot
    clusters plus uniform background noise.  Severity depends on speed
    limit, lighting, weather, road surface, and junction type.  Time
    patterns mimic real rush-hour and weekend distributions.  The
    returned data deliberately contains quality issues (missing values,
    duplicates, inconsistent labels) for testing the cleaning pipeline.

    Parameters
    ----------
    n : int
        Number of base records to generate (before duplicate injection).
    seed : int
        Random seed for reproducibility.
    save : bool
        If ``True``, write the raw accidents CSV and true-hotspot centres
        CSV to ``config.RAW_DATA_DIR``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(accidents_df, hotspots_df)`` where *accidents_df* has columns
        matching ``config.STANDARD_COLUMNS`` (plus injected noise) and
        *hotspots_df* lists the 9 true cluster centres.
    """
    rng = np.random.default_rng(seed)

    # ── Spatial coordinates ──────────────────────────────────────────────
    n_cluster = int(n * _HOTSPOT_FRACTION)
    n_background = n - n_cluster

    weights = np.array([h[4] for h in _HOTSPOT_CENTRES])
    weights /= weights.sum()
    cluster_assignments = rng.choice(len(_HOTSPOT_CENTRES), size=n_cluster, p=weights)

    lats_c = np.empty(n_cluster)
    lons_c = np.empty(n_cluster)
    for idx, (lat, lon, sigma, _, _) in enumerate(_HOTSPOT_CENTRES):
        mask = cluster_assignments == idx
        count = mask.sum()
        lats_c[mask] = rng.normal(lat, sigma, size=count)
        lons_c[mask] = rng.normal(lon, sigma, size=count)

    # Background: uniform over city bounds
    lat_lo, lat_hi = config.CITY_LAT_RANGE
    lon_lo, lon_hi = config.CITY_LON_RANGE
    lats_bg = rng.uniform(lat_lo, lat_hi, size=n_background)
    lons_bg = rng.uniform(lon_lo, lon_hi, size=n_background)

    lats = np.concatenate([lats_c, lats_bg])
    lons = np.concatenate([lons_c, lons_bg])

    # Shuffle so cluster / background aren't contiguous
    order = rng.permutation(n)
    lats = lats[order]
    lons = lons[order]

    # ── Dates and times ──────────────────────────────────────────────────
    dates, times = _generate_dates(rng, n)

    # ── Road attributes ──────────────────────────────────────────────────
    road_types = rng.choice(
        config.ROAD_TYPES,
        size=n,
        p=[0.40, 0.25, 0.10, 0.10, 0.10, 0.05],
    )
    speed_limits = rng.choice(
        config.SPEED_LIMITS,
        size=n,
        p=[0.08, 0.40, 0.20, 0.12, 0.12, 0.08],
    )
    light_conds = rng.choice(
        config.LIGHT_CONDITIONS_VALUES,
        size=n,
        p=[0.55, 0.25, 0.10, 0.10],
    )
    weather = rng.choice(
        config.WEATHER_VALUES,
        size=n,
        p=[0.55, 0.25, 0.06, 0.03, 0.06, 0.05],
    )
    road_surface = rng.choice(
        config.ROAD_SURFACE_VALUES,
        size=n,
        p=[0.50, 0.35, 0.05, 0.05, 0.05],
    )
    junction_type = rng.choice(
        config.JUNCTION_TYPES,
        size=n,
        p=[0.40, 0.15, 0.20, 0.10, 0.08, 0.07],
    )
    urban_rural = rng.choice(
        config.URBAN_RURAL_VALUES,
        size=n,
        p=[0.75, 0.25],
    )

    # ── Severity (factor-dependent) ──────────────────────────────────────
    severity = _assign_severity(
        rng, speed_limits, light_conds, weather, road_surface, junction_type,
    )

    # ── Vehicles & casualties ────────────────────────────────────────────
    num_vehicles = rng.choice([1, 2, 3, 4, 5], size=n, p=[0.25, 0.50, 0.15, 0.07, 0.03])
    # Casualties loosely correlated with severity
    num_casualties = np.ones(n, dtype=int)
    for i in range(n):
        if severity[i] == 1:  # fatal
            num_casualties[i] = rng.choice([1, 2, 3, 4], p=[0.60, 0.25, 0.10, 0.05])
        elif severity[i] == 2:  # serious
            num_casualties[i] = rng.choice([1, 2, 3], p=[0.65, 0.25, 0.10])
        else:  # slight
            num_casualties[i] = rng.choice([1, 2], p=[0.80, 0.20])

    # ── Assemble clean DataFrame ─────────────────────────────────────────
    accident_ids = [uuid.uuid4().hex[:12].upper() for _ in range(n)]

    df = pd.DataFrame({
        "accident_id":     accident_ids,
        "date":            dates,
        "time":            times,
        "latitude":        lats,
        "longitude":       lons,
        "severity":        severity,
        "num_vehicles":    num_vehicles,
        "num_casualties":  num_casualties,
        "road_type":       road_types,
        "speed_limit":     speed_limits,
        "light_conditions": light_conds,
        "weather":         weather,
        "road_surface":    road_surface,
        "junction_type":   junction_type,
        "urban_rural":     urban_rural,
    })

    # ── Inject messy data ────────────────────────────────────────────────
    df = _inject_noise(rng, df)

    # ── Hotspot-centres reference table ──────────────────────────────────
    hotspots_df = pd.DataFrame(
        [(lat, lon, sigma, name) for lat, lon, sigma, name, _ in _HOTSPOT_CENTRES],
        columns=["latitude", "longitude", "spatial_std", "name"],
    )

    # ── Persist to disk ─────────────────────────────────────────────────
    if save:
        config.RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.RAW_ACCIDENTS_CSV, index=False)
        hotspots_df.to_csv(config.TRUE_HOTSPOTS_CSV, index=False)

    return df, hotspots_df


def load_real(
    path: Path | str,
    column_mapping: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Read a real-world CSV and map columns to the standard schema.

    Any source column not present in the file is silently skipped.
    Missing standard columns are filled with ``NaN``.

    Parameters
    ----------
    path : Path | str
        Path to the CSV file.
    column_mapping : dict[str, str] | None
        ``{source_col: standard_col}`` mapping.  Defaults to
        ``config.DEFAULT_COLUMN_MAPPING`` (UK STATS19 layout).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns matching ``config.STANDARD_COLUMNS``.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if column_mapping is None:
        column_mapping = config.DEFAULT_COLUMN_MAPPING

    raw = pd.read_csv(path, low_memory=False)

    # Rename only columns that actually exist in the source file
    rename_map = {k: v for k, v in column_mapping.items() if k in raw.columns}
    df = raw.rename(columns=rename_map)

    # Keep only standard columns; add missing ones as NaN
    for col in config.STANDARD_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    df = df[config.STANDARD_COLUMNS]
    return df


def load_accidents(
    source: str = "synthetic",
    *,
    n: int = 50_000,
    seed: int = config.RANDOM_SEED,
    path: Optional[Path | str] = None,
    column_mapping: Optional[Dict[str, str]] = None,
    save_synthetic: bool = True,
) -> pd.DataFrame:
    """Unified entry point – returns a DataFrame in the standard schema.

    Parameters
    ----------
    source : str
        ``"synthetic"`` or ``"real"``.
    n : int
        Number of records for synthetic generation.
    seed : int
        Random seed (synthetic only).
    path : Path | str | None
        CSV path (required when ``source="real"``).
    column_mapping : dict | None
        Column mapping (real only).
    save_synthetic : bool
        Whether to persist synthetic data to disk.

    Returns
    -------
    pd.DataFrame
        Accident data with columns ``config.STANDARD_COLUMNS``.

    Raises
    ------
    ValueError
        If *source* is not ``"synthetic"`` or ``"real"``, or if *path*
        is missing when ``source="real"``.
    """
    if source == "synthetic":
        df, _ = generate_synthetic(n=n, seed=seed, save=save_synthetic)
        return df

    if source == "real":
        if path is None:
            raise ValueError("path is required when source='real'")
        return load_real(path, column_mapping)

    raise ValueError(f"Unknown source: {source!r}. Use 'synthetic' or 'real'.")
