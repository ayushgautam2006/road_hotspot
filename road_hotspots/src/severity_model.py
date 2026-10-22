"""Severity classification model (LightGBM) with SHAP explanations.

Trains a multi-class classifier to predict accident severity and
provides feature-importance explanations via SHAP.
"""

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd


def split_data(
    df: pd.DataFrame,
    target_col: str | None = None,
    test_size: float | None = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-enriched accident data.
    target_col : str | None
        Column to predict.  Defaults to ``config.COL_SEVERITY``.
    test_size : float | None
        Fraction held out.  Defaults to ``config.TEST_SIZE``.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        ``(X_train, X_test, y_train, y_test)``.
    """
    raise NotImplementedError


def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    params: Dict[str, Any] | None = None,
) -> Any:
    """Train a LightGBM classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    params : dict | None
        Override ``config.LIGHTGBM_PARAMS``.

    Returns
    -------
    Any
        Fitted LightGBM model.
    """
    raise NotImplementedError


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, Any]:
    """Compute classification metrics (accuracy, F1, confusion matrix).

    Parameters
    ----------
    model : Any
        Trained classifier.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        True labels.

    Returns
    -------
    dict[str, Any]
        Dictionary of metric names → values.
    """
    raise NotImplementedError


def explain_model(
    model: Any,
    X: pd.DataFrame,
) -> np.ndarray:
    """Generate SHAP values for the given data.

    Parameters
    ----------
    model : Any
        Trained LightGBM model.
    X : pd.DataFrame
        Feature matrix to explain.

    Returns
    -------
    np.ndarray
        SHAP values array.
    """
    raise NotImplementedError


def save_model(model: Any, filepath: Path | None = None) -> Path:
    """Persist the trained model to disk.

    Parameters
    ----------
    model : Any
        Trained model to save.
    filepath : Path | None
        Destination path.  Defaults to ``config.MODELS_OUTPUT_DIR``.

    Returns
    -------
    Path
        Path where the model was saved.
    """
    raise NotImplementedError


def load_model(filepath: Path) -> Any:
    """Load a previously saved model from disk.

    Parameters
    ----------
    filepath : Path
        Path to the saved model file.

    Returns
    -------
    Any
        Loaded model object.
    """
    raise NotImplementedError
