"""
LightGBM training utilities for temporal baseball pitch prediction.

This module demonstrates the main tree-based modeling strategy used in
the project:

1. Temporal validation by season
2. Binary control_success prediction
3. Multiclass auxiliary prediction for miss direction
4. Multi-seed ensembling
5. Out-of-fold prediction generation for later ensemble/calibration
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd

from sklearn.metrics import log_loss


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

@dataclass
class GBDTConfig:
    objective: str = "binary"

    learning_rate: float = 0.03
    n_estimators: int = 1200

    num_leaves: int = 31
    max_depth: int = -1

    min_child_samples: int = 100

    subsample: float = 0.85
    colsample_bytree: float = 0.85

    reg_alpha: float = 0.1
    reg_lambda: float = 2.0

    n_jobs: int = -1


DEFAULT_SEEDS = [17, 43, 101, 211]


# ---------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    validation_season: int,
):
    """
    Create a forward-looking temporal split.

    Training:
        seasons strictly before validation_season

    Validation:
        validation_season

    This avoids random row-level leakage between seasons.
    """
    train_mask = df["season"] < validation_season
    valid_mask = df["season"] == validation_season

    return train_mask, valid_mask


# ---------------------------------------------------------------------
# Binary LightGBM
# ---------------------------------------------------------------------

def train_binary_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    config: GBDTConfig | None = None,
):
    """
    Train one binary LightGBM model.
    """
    if config is None:
        config = GBDTConfig()

    model = lgb.LGBMClassifier(
        objective="binary",
        learning_rate=config.learning_rate,
        n_estimators=config.n_estimators,
        num_leaves=config.num_leaves,
        max_depth=config.max_depth,
        min_child_samples=config.min_child_samples,
        subsample=config.subsample,
        colsample_bytree=config.colsample_bytree,
        reg_alpha=config.reg_alpha,
        reg_lambda=config.reg_lambda,
        random_state=seed,
        n_jobs=config.n_jobs,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    pred = model.predict_proba(X_valid)[:, 1]

    return model, pred


def train_binary_seed_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seeds: Iterable[int] = DEFAULT_SEEDS,
):
    """
    Train several binary models using different random seeds.

    Averaging multiple seeds reduces model variance while preserving
    the same feature representation and objective.
    """
    models = []
    predictions = []

    for seed in seeds:
        model, pred = train_binary_model(
            X_train,
            y_train,
            X_valid,
            y_valid,
            seed=seed,
        )

        models.append(model)
        predictions.append(pred)

    ensemble_pred = np.mean(
        np.column_stack(predictions),
        axis=1,
    )

    return models, ensemble_pred


# ---------------------------------------------------------------------
# Multiclass auxiliary target
# ---------------------------------------------------------------------

MULTICLASS_LABELS = {
    0: "success",
    1: "middle_miss",
    2: "reverse_miss",
    3: "far_miss",
}


def train_multiclass_model(
    X_train: np.ndarray,
    y_train_4class: np.ndarray,
    X_valid: np.ndarray,
    y_valid_4class: np.ndarray,
    seed: int,
):
    """
    Train a 4-class auxiliary LightGBM model.

    Class 0 represents control success.
    Other classes distinguish different miss directions.

    The final probability used for the main task is the predicted
    probability of class 0.
    """
    model = lgb.LGBMClassifier(
        objective="multiclass",
        num_class=4,

        learning_rate=0.03,
        n_estimators=1200,

        num_leaves=31,
        min_child_samples=100,

        subsample=0.85,
        colsample_bytree=0.85,

        reg_lambda=2.0,

        random_state=seed,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train_4class,
        eval_set=[
            (
                X_valid,
                y_valid_4class,
            )
        ],
        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=False,
            ),
            lgb.log_evaluation(period=0),
        ],
    )

    class_prob = model.predict_proba(X_valid)

    success_prob = class_prob[:, 0]

    return model, success_prob


def train_multiclass_seed_ensemble(
    X_train: np.ndarray,
    y_train_4class: np.ndarray,
    X_valid: np.ndarray,
    y_valid_4class: np.ndarray,
    seeds: Iterable[int] = DEFAULT_SEEDS,
):
    """
    Train a multi-seed ensemble for the 4-class auxiliary task.
    """
    models = []
    predictions = []

    for seed in seeds:
        model, pred = train_multiclass_model(
            X_train,
            y_train_4class,
            X_valid,
            y_valid_4class,
            seed,
        )

        models.append(model)
        predictions.append(pred)

    ensemble_pred = np.mean(
        np.column_stack(predictions),
        axis=1,
    )

    return models, ensemble_pred


# ---------------------------------------------------------------------
# Temporal OOF generation
# ---------------------------------------------------------------------

def generate_temporal_oof(
    df: pd.DataFrame,
    X: np.ndarray,
    y_binary: np.ndarray,
    y_multiclass: np.ndarray,
    validation_seasons: list[int],
):
    """
    Generate temporal out-of-fold predictions.

    For every validation season:

        train = all earlier seasons
        valid = current season

    Both binary and multiclass ensembles are trained independently.
    """
    oof_binary = np.full(
        len(df),
        np.nan,
        dtype=np.float32,
    )

    oof_multiclass = np.full(
        len(df),
        np.nan,
        dtype=np.float32,
    )

    fold_results = []

    for season in validation_seasons:

        train_mask, valid_mask = temporal_split(
            df,
            validation_season=season,
        )

        train_idx = np.where(train_mask)[0]
        valid_idx = np.where(valid_mask)[0]

        if len(train_idx) == 0 or len(valid_idx) == 0:
            continue

        X_train = X[train_idx]
        X_valid = X[valid_idx]

        y_train = y_binary[train_idx]
        y_valid = y_binary[valid_idx]

        y4_train = y_multiclass[train_idx]
        y4_valid = y_multiclass[valid_idx]

        # ---------------------------------------------------------
        # Binary model
        # ---------------------------------------------------------

        _, pred_binary = train_binary_seed_ensemble(
            X_train,
            y_train,
            X_valid,
            y_valid,
        )

        # ---------------------------------------------------------
        # Multiclass auxiliary model
        # ---------------------------------------------------------

        _, pred_multiclass = train_multiclass_seed_ensemble(
            X_train,
            y4_train,
            X_valid,
            y4_valid,
        )

        oof_binary[valid_idx] = pred_binary
        oof_multiclass[valid_idx] = pred_multiclass

        # ---------------------------------------------------------
        # Diagnostics
        # ---------------------------------------------------------

        binary_logloss = log_loss(
            y_valid,
            np.clip(
                pred_binary,
                1e-6,
                1 - 1e-6,
            ),
        )

        multiclass_as_binary_logloss = log_loss(
            y_valid,
            np.clip(
                pred_multiclass,
                1e-6,
                1 - 1e-6,
            ),
        )

        correlation = np.corrcoef(
            pred_binary,
            pred_multiclass,
        )[0, 1]

        fold_results.append(
            {
                "season": season,
                "n_train": len(train_idx),
                "n_valid": len(valid_idx),
                "binary_logloss": binary_logloss,
                "multiclass_success_logloss":
                    multiclass_as_binary_logloss,
                "prediction_correlation": correlation,
            }
        )

    return {
        "binary": oof_binary,
        "multiclass": oof_multiclass,
        "fold_results": pd.DataFrame(fold_results),
    }
