"""Smoke tests for config module – ensure it imports and has expected values."""

from src.config import (
    CLUSTER_RADIUS_KM,
    MIN_CLUSTER_SIZE,
    PROJECT_ROOT,
    RANDOM_SEED,
    SEVERITY_WEIGHTS,
)


def test_random_seed() -> None:
    """Random seed should be 42."""
    assert RANDOM_SEED == 42


def test_project_root_exists() -> None:
    """PROJECT_ROOT should resolve to an existing directory."""
    assert PROJECT_ROOT.exists()


def test_severity_weights_keys() -> None:
    """Severity weights must contain fatal, serious, slight."""
    assert set(SEVERITY_WEIGHTS.keys()) == {"fatal", "serious", "slight"}


def test_cluster_params_positive() -> None:
    """Clustering parameters must be positive."""
    assert CLUSTER_RADIUS_KM > 0
    assert MIN_CLUSTER_SIZE > 0
