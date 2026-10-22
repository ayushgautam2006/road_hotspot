"""Priority scoring for accident hotspot clusters.

Combines frequency, severity, and temporal-trend components into a
single composite score used to rank clusters for intervention.
"""

import pandas as pd


def compute_frequency_score(cluster_summary: pd.DataFrame) -> pd.Series:
    """Normalised accident count per cluster.

    Parameters
    ----------
    cluster_summary : pd.DataFrame
        Output of ``hotspots.summarise_clusters``.

    Returns
    -------
    pd.Series
        Frequency scores in [0, 1].
    """
    raise NotImplementedError


def compute_severity_score(cluster_summary: pd.DataFrame) -> pd.Series:
    """Weighted-average severity per cluster, normalised.

    Parameters
    ----------
    cluster_summary : pd.DataFrame
        Cluster-level summary with severity weights.

    Returns
    -------
    pd.Series
        Severity scores in [0, 1].
    """
    raise NotImplementedError


def compute_trend_score(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> pd.Series:
    """Temporal trend component – recent accidents score higher.

    Applies ``config.PRIORITY_RECENCY_DECAY`` exponential weighting.

    Parameters
    ----------
    df : pd.DataFrame
        Accident-level data with dates and cluster labels.
    cluster_summary : pd.DataFrame
        Cluster-level summary.

    Returns
    -------
    pd.Series
        Trend scores in [0, 1].
    """
    raise NotImplementedError


def compute_priority_scores(
    df: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Composite priority score combining frequency, severity, and trend.

    Uses weights from ``config.PRIORITY_WEIGHT_*``.

    Parameters
    ----------
    df : pd.DataFrame
        Accident-level data.
    cluster_summary : pd.DataFrame
        Cluster-level summary from ``hotspots.summarise_clusters``.

    Returns
    -------
    pd.DataFrame
        ``cluster_summary`` with an added ``priority_score`` column,
        sorted descending.
    """
    raise NotImplementedError


def rank_clusters(scored: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Return the top-N highest-priority clusters.

    Parameters
    ----------
    scored : pd.DataFrame
        Output of ``compute_priority_scores``.
    top_n : int
        Number of clusters to return.

    Returns
    -------
    pd.DataFrame
        Top clusters by priority score.
    """
    raise NotImplementedError
