"""Feature-engineering transformations for accident data.

Derives temporal, spatial, interaction, and encoded features consumed
by the severity model.  Every feature is documented and no feature is
derived from the severity label of the same row.

**Leakage prevention**:

* ``num_casualties`` is excluded from *X* because in the synthetic
  dataset it is generated directly from severity.
* ``cell_accident_count`` uses a **leave-one-out** approach: the count
  for each row equals the total accidents in its grid cell *minus one*,
  so the row's own presence is never included.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from src import config

# ── Grid constants (~250 m cells at London latitude) ─────────────────────────
GRID_LAT_STEP: float = 0.00225  # ~250 m
GRID_LON_STEP: float = 0.00362  # ~250 m at ≈51.5° N

# ── Temporal thresholds ──────────────────────────────────────────────────────
RUSH_HOURS: set = {7, 8, 9, 16, 17, 18}
NIGHT_HOURS: set = {22, 23, 0, 1, 2, 3, 4, 5}
HIGH_SPEED_THRESHOLD: int = 50

# ── Columns that must NOT appear in the final feature matrix X ───────────────
LEAKAGE_COLUMNS: List[str] = [
    "severity",        # target variable
    "num_casualties",  # generated from severity in synthetic data
]

ID_AND_RAW_COLUMNS: List[str] = [
    "accident_id",
    "date",
    "time",
    "datetime",
    "latitude",
    "longitude",
    "grid_cell",
]

DROP_COLUMNS: List[str] = LEAKAGE_COLUMNS + ID_AND_RAW_COLUMNS

# ── Categorical columns to one-hot encode ────────────────────────────────────
ONE_HOT_COLUMNS: List[str] = [
    "road_type",
    "light_conditions",
    "weather",
    "road_surface",
    "junction_type",
    "urban_rural",
]


# ─────────────────────────────────────────────────────────────────────────────
# Individual feature-engineering steps
# ─────────────────────────────────────────────────────────────────────────────

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract temporal columns from *datetime*.

    New columns
    -----------
    hour : int (0-23)
    weekday : int (0=Monday … 6=Sunday)
    month : int (1-12)
    is_weekend : binary
    is_rush_hour : binary (7-9 AM, 4-6 PM)
    is_night : binary (10 PM – 5 AM)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain a ``datetime`` column.

    Returns
    -------
    pd.DataFrame
        Input with six new temporal columns.
    """
    df = df.copy()
    dt = df["datetime"]
    df["hour"] = dt.dt.hour
    df["weekday"] = dt.dt.dayofweek
    df["month"] = dt.dt.month
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    df["is_rush_hour"] = df["hour"].isin(RUSH_HOURS).astype(int)
    df["is_night"] = df["hour"].isin(NIGHT_HOURS).astype(int)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary interaction features from combinations of risk factors.

    New columns
    -----------
    night_and_unlit : 1 if night-time AND dark unlit road
    wet_and_high_speed : 1 if wet/icy/flooded surface AND speed ≥ 50
    junction_and_urban : 1 if at a junction AND urban area

    Parameters
    ----------
    df : pd.DataFrame
        Must already have ``is_night`` (call ``add_temporal_features``
        first), ``light_conditions``, ``road_surface``, ``speed_limit``,
        ``junction_type``, and ``urban_rural``.

    Returns
    -------
    pd.DataFrame
        Input with three new interaction columns.
    """
    df = df.copy()

    df["night_and_unlit"] = (
        (df["is_night"] == 1) & (df["light_conditions"] == "dark_unlit")
    ).astype(int)

    df["wet_and_high_speed"] = (
        df["road_surface"].isin(["wet", "ice", "flood"])
        & (df["speed_limit"] >= HIGH_SPEED_THRESHOLD)
    ).astype(int)

    df["junction_and_urban"] = (
        (df["junction_type"] != "not_junction") & (df["urban_rural"] == "urban")
    ).astype(int)

    return df


def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Assign each accident to a ~250 m grid cell and compute a
    leave-one-out historical accident count.

    **Leakage prevention**: the count for each row equals the total
    number of accidents in its grid cell *minus one* (the row itself).

    New columns
    -----------
    grid_cell : str – temporary cell identifier (dropped later)
    cell_accident_count : int – LOO count of other accidents in the cell

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``latitude`` and ``longitude``.

    Returns
    -------
    pd.DataFrame
        Input with ``grid_cell`` and ``cell_accident_count`` added.
    """
    df = df.copy()

    grid_lat = np.floor(df["latitude"] / GRID_LAT_STEP).astype(int)
    grid_lon = np.floor(df["longitude"] / GRID_LON_STEP).astype(int)
    df["grid_cell"] = grid_lat.astype(str) + "_" + grid_lon.astype(str)

    # Leave-one-out count: total in cell minus the current row
    cell_total = df.groupby("grid_cell")["grid_cell"].transform("count")
    df["cell_accident_count"] = (cell_total - 1).astype(int)

    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode unordered categorical columns.

    ``speed_limit`` is kept as a numeric feature because its integer
    values already carry meaningful ordinal spacing (20 → 30 → … → 70).

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
        Input with raw categorical columns replaced by binary dummies.
    """
    df = df.copy()
    present = [c for c in ONE_HOT_COLUMNS if c in df.columns]
    df = pd.get_dummies(df, columns=present, prefix=present, dtype=int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────────────────────

def build_features(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Build the feature matrix and target vectors from cleaned data.

    Pipeline order:

    1. Temporal features (hour, weekday, month, binary flags)
    2. Interaction features (night × unlit, wet × speed, junction × urban)
    3. Spatial features (grid cell + leave-one-out count)
    4. One-hot encoding of unordered categoricals
    5. Drop all identifier, raw, and leakage columns

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned accident data (output of ``cleaning.clean_accidents``).
        Must contain a ``datetime`` column and the columns listed in
        ``config.STANDARD_COLUMNS``.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series, pd.Series]
        ``(X, y_binary, y_multiclass)`` where:

        * **X** – feature matrix with no NaNs and no leakage columns.
        * **y_binary** – ``1`` if fatal or serious (severity 1 or 2),
          ``0`` if slight (severity 3).
        * **y_multiclass** – original 3-class severity code (1 / 2 / 3).

    Raises
    ------
    ValueError
        If the resulting *X* contains any NaN values.
    """
    df = df.copy()

    # ── Engineer features ────────────────────────────────────────────────
    df = add_temporal_features(df)
    df = add_interaction_features(df)
    df = add_spatial_features(df)
    df = encode_categoricals(df)

    # ── Targets ──────────────────────────────────────────────────────────
    y_multiclass = df["severity"].copy()
    y_multiclass.name = "severity"

    y_binary = (df["severity"].isin([1, 2])).astype(int)
    y_binary.name = "fatal_or_serious"

    # ── Drop non-feature columns ─────────────────────────────────────────
    to_drop = [c for c in DROP_COLUMNS if c in df.columns]
    X = df.drop(columns=to_drop)

    # ── Safety check ─────────────────────────────────────────────────────
    nan_cols = X.columns[X.isna().any()].tolist()
    if nan_cols:
        raise ValueError(
            f"Feature matrix contains NaNs in columns: {nan_cols}"
        )

    return X, y_binary, y_multiclass
