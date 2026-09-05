"""
LightGBM training utilities for pitch-level control prediction.

Main ideas
----------
1. Temporal validation by season
2. Binary control_success prediction
3. Multi-seed LightGBM ensembling
4. Out-of-fold predictions for ensemble evaluation

This is a cleaned portfolio version of the tree-based pipeline used
for the project's best-performing V7 ensemble.
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
    """
    LightGBM configuration used for the binary control-success model.

    The exact competition pipeline contained several model variants.
    This portfolio version keeps the shared training logic explicit
    and easy to reproduce.
    """

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
# Temporal validation
# ---------------------------------------------------------------------

def temporal_split(
    df: pd.DataFrame,
    validation_season: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split data chronologically by season.

    Training
    --------
    All seasons strictly before `validation_season`.

    Validation
    ----------
    Rows belonging to `validation_season`.

    Notes
    -----
    A random row-level split can leak information across seasons and
    produce overly optimistic validation results when the evaluation
    distribution is concentrated in a later season.

    Temporal validation was therefore used to better approximate the
    competition's hidden-test setting.
    """
    train_mask = (
        df["season"].to_numpy()
        < validation_season
    )

    valid_mask = (
        df["season"].to_numpy()
        == validation_season
    )

    return train_mask, valid_mask


# ---------------------------------------------------------------------
# Single LightGBM model
# ---------------------------------------------------------------------

def build_binary_model(
    seed: int,
    config: GBDTConfig | None = None,
) -> lgb.LGBMClassifier:
    """
    Construct one binary LightGBM classifier.
    """
    if config is None:
        config = GBDTConfig()

    return lgb.LGBMClassifier(
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


def train_binary_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seed: int,
    config: GBDTConfig | None = None,
):
    """
    Train one LightGBM model and return validation probabilities.

    Parameters
    ----------
    X_train:
        Training feature matrix.

    y_train:
        Binary control_success labels.

    X_valid:
        Validation feature matrix.

    y_valid:
        Validation labels.

    seed:
        Random seed for this ensemble member.

    config:
        Optional LightGBM configuration.

    Returns
    -------
    model:
        Fitted LightGBM classifier.

    prediction:
        Probability of control_success for validation rows.
    """
    model = build_binary_model(
        seed=seed,
        config=config,
    )

    model.fit(
        X_train,
        y_train,

        eval_set=[
            (
                X_valid,
                y_valid,
            )
        ],

        eval_metric="binary_logloss",

        callbacks=[
            lgb.early_stopping(
                stopping_rounds=100,
                verbose=False,
            ),

            lgb.log_evaluation(
                period=0
            ),
        ],
    )

    prediction = (
        model.predict_proba(
            X_valid
        )[:, 1]
        .astype(np.float32)
    )

    return model, prediction


# ---------------------------------------------------------------------
# Multi-seed ensemble
# ---------------------------------------------------------------------

def train_seed_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    config: GBDTConfig | None = None,
):
    """
    Train several LightGBM models using different random seeds.

    The individual models use the same target and feature space.
    Predictions are averaged to reduce variance.

    Returns
    -------
    models:
        List of fitted LightGBM classifiers.

    ensemble_prediction:
        Mean validation probability across all seeds.

    member_predictions:
        Matrix with shape:
            (n_validation_rows, n_models)
    """
    models = []
    predictions = []

    for seed in seeds:

        model, prediction = train_binary_model(
            X_train=X_train,
            y_train=y_train,

            X_valid=X_valid,
            y_valid=y_valid,

            seed=seed,
            config=config,
        )

        models.append(model)
        predictions.append(prediction)

    member_predictions = np.column_stack(
        predictions
    )

    ensemble_prediction = np.mean(
        member_predictions,
        axis=1,
    ).astype(np.float32)

    return (
        models,
        ensemble_prediction,
        member_predictions,
    )


# ---------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------

def evaluate_predictions(
    y_true: np.ndarray,
    prediction: np.ndarray,
) -> dict:
    """
    Compute simple validation diagnostics.

    The competition leaderboard metric is intentionally not reproduced
    here because this module focuses on the modeling pipeline itself.
    """
    prediction = np.clip(
        prediction,
        1e-6,
        1.0 - 1e-6,
    )

    return {
        "log_loss": float(
            log_loss(
                y_true,
                prediction,
            )
        ),

        "prediction_mean": float(
            np.mean(prediction)
        ),

        "prediction_std": float(
            np.std(prediction)
        ),

        "target_mean": float(
            np.mean(y_true)
        ),
    }


# ---------------------------------------------------------------------
# Temporal OOF
# ---------------------------------------------------------------------

def generate_temporal_oof(
    df: pd.DataFrame,
    X: np.ndarray,
    y: np.ndarray,
    validation_seasons: Iterable[int],
    seeds: Iterable[int] = DEFAULT_SEEDS,
    config: GBDTConfig | None = None,
) -> dict:
    """
    Generate temporal out-of-fold predictions.

    For each validation season:

        train = all earlier seasons
        valid = current season

    This creates predictions that more closely resemble inference on
    future-season data than a conventional random split.

    Returns
    -------
    dict with:

        oof
            Out-of-fold probability for each row.

        fold_results
            Season-level validation diagnostics.

        models
            Fitted models grouped by validation season.
    """
    oof = np.full(
        len(df),
        np.nan,
        dtype=np.float32,
    )

    fold_results = []
    fold_models = {}

    for season in validation_seasons:

        train_mask, valid_mask = temporal_split(
            df,
            validation_season=season,
        )

        train_idx = np.flatnonzero(
            train_mask
        )

        valid_idx = np.flatnonzero(
            valid_mask
        )

        # Skip impossible folds.
        if (
            len(train_idx) == 0
            or len(valid_idx) == 0
        ):
            continue

        X_train = X[train_idx]
        X_valid = X[valid_idx]

        y_train = y[train_idx]
        y_valid = y[valid_idx]

        (
            models,
            prediction,
            member_predictions,
        ) = train_seed_ensemble(
            X_train=X_train,
            y_train=y_train,

            X_valid=X_valid,
            y_valid=y_valid,

            seeds=seeds,
            config=config,
        )

        oof[valid_idx] = prediction

        diagnostics = evaluate_predictions(
            y_true=y_valid,
            prediction=prediction,
        )

        # Prediction diversity between seed members.
        if member_predictions.shape[1] > 1:

            corr = np.corrcoef(
                member_predictions,
                rowvar=False,
            )

            upper = corr[
                np.triu_indices(
                    corr.shape[0],
                    k=1,
                )
            ]

            mean_seed_correlation = float(
                np.mean(upper)
            )

        else:
            mean_seed_correlation = 1.0

        fold_results.append(
            {
                "season": int(season),

                "n_train": int(
                    len(train_idx)
                ),

                "n_valid": int(
                    len(valid_idx)
                ),

                **diagnostics,

                "mean_seed_correlation":
                    mean_seed_correlation,
            }
        )

        fold_models[int(season)] = models

    return {
        "oof": oof,

        "fold_results": pd.DataFrame(
            fold_results
        ),

        "models": fold_models,
    }


# ---------------------------------------------------------------------
# Full-data training
# ---------------------------------------------------------------------

def train_final_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    seeds: Iterable[int] = DEFAULT_SEEDS,
    config: GBDTConfig | None = None,
):
    """
    Train the final LightGBM seed ensemble on all available training rows.

    Unlike temporal OOF training, no validation set is used here.
    Early stopping is therefore disabled and the configured number of
    boosting rounds is used directly.
    """
    if config is None:
        config = GBDTConfig()

    models = []

    for seed in seeds:

        model = build_binary_model(
            seed=seed,
            config=config,
        )

        model.fit(
            X,
            y,
            callbacks=[
                lgb.log_evaluation(
                    period=0
                )
            ],
        )

        models.append(model)

    return models


def predict_ensemble(
    models: list[lgb.LGBMClassifier],
    X: np.ndarray,
) -> np.ndarray:
    """
    Average probabilities from trained LightGBM ensemble members.
    """
    predictions = [
        model.predict_proba(X)[:, 1]
        for model in models
    ]

    return np.mean(
        np.column_stack(predictions),
        axis=1,
    ).astype(np.float32)
