"""Feature-engineering transformations for accident data.

Derives temporal, spatial, and categorical features consumed by the
severity model and the priority scorer.
"""

import pandas as pd


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract hour, day-of-week, month, and weekend flag from datetime.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned accident data containing a date/time column.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional temporal feature columns.
    """
    raise NotImplementedError


def add_time_bin(df: pd.DataFrame) -> pd.DataFrame:
    """Bin the hour of day into categorical time-of-day periods.

    Uses ``config.TIME_BIN_EDGES`` and ``config.TIME_BINS``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain an ``hour`` column (call ``add_time_features`` first).

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``time_bin`` column.
    """
    raise NotImplementedError


def encode_categoricals(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """One-hot or ordinal-encode categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        Accident data.
    columns : list[str] | None
        Columns to encode.  ``None`` encodes all ``object``/``category``
        columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with encoded features.
    """
    raise NotImplementedError


def add_severity_weight(df: pd.DataFrame) -> pd.DataFrame:
    """Map severity labels to numeric weights from ``config.SEVERITY_WEIGHTS``.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain the severity column.

    Returns
    -------
    pd.DataFrame
        DataFrame with a ``severity_weight`` column.
    """
    raise NotImplementedError


def run_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """Execute the full feature-engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned accident data.

    Returns
    -------
    pd.DataFrame
        Feature-enriched DataFrame.
    """
    raise NotImplementedError
