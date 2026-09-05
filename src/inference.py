"""
Inference utilities for the V7 modeling pipeline.

Pipeline
--------
1. Transform each pitch using train-derived feature artifacts.
2. Convert engineered features into the model matrix.
3. Generate LightGBM ensemble probabilities.
4. Generate neural-network farm probabilities.
5. Blend the two model families.
6. Build a submission dataframe while preserving row_id order.

Important
---------
Evaluation features are designed to depend only on:

    current evaluation row
    +
    artifacts derived from training data

and never on statistics computed from other evaluation rows.

The original competition pipeline contained additional season-decomposition
features and exported neural-network artifacts. This public portfolio module
documents the core inference architecture rather than reproducing every
competition artifact exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ensemble import blend_v7
from feature_engineering import to_model_matrix


ID_COLUMN = "row_id"
TARGET_COLUMN = "control_success"


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_feature_matrix(
    X: np.ndarray,
) -> np.ndarray:
    """
    Validate the model feature matrix.

    Parameters
    ----------
    X:
        Two-dimensional numerical model matrix.

    Returns
    -------
    np.ndarray
        Contiguous float32 matrix.
    """

    X = np.asarray(
        X,
        dtype=np.float32,
    )

    if X.ndim != 2:
        raise ValueError(
            "Feature matrix must be two-dimensional."
        )

    if len(X) == 0:
        raise ValueError(
            "Feature matrix contains no rows."
        )

    if not np.all(np.isfinite(X)):
        raise ValueError(
            "Feature matrix contains NaN or infinite values."
        )

    return np.ascontiguousarray(X)


def validate_prediction(
    prediction: np.ndarray,
    expected_rows: int,
    name: str,
) -> np.ndarray:
    """
    Ensure that one finite probability exists for every input row.
    """

    prediction = np.asarray(
        prediction,
        dtype=np.float64,
    ).reshape(-1)

    if len(prediction) != expected_rows:
        raise ValueError(
            f"{name} returned {len(prediction)} rows; "
            f"expected {expected_rows}."
        )

    if not np.all(
        np.isfinite(prediction)
    ):
        raise ValueError(
            f"{name} contains NaN or infinite predictions."
        )

    return np.clip(
        prediction,
        1e-6,
        1.0 - 1e-6,
    )


# ---------------------------------------------------------------------
# LightGBM inference
# ---------------------------------------------------------------------

def predict_lightgbm_ensemble(
    models,
    X: np.ndarray,
) -> np.ndarray:
    """
    Average probabilities from multiple LightGBM models.

    Supported model interfaces
    --------------------------
    1. sklearn-style models exposing ``predict_proba``
    2. native LightGBM Booster objects exposing ``predict``

    Parameters
    ----------
    models:
        Iterable of trained LightGBM models.

    X:
        Numerical model feature matrix.

    Returns
    -------
    np.ndarray
        Averaged LightGBM probability.
    """

    X = validate_feature_matrix(X)

    predictions = []

    for model in models:

        if hasattr(
            model,
            "predict_proba",
        ):
            prediction = (
                model
                .predict_proba(X)[:, 1]
            )

        elif hasattr(
            model,
            "predict",
        ):
            prediction = (
                model.predict(X)
            )

        else:
            raise TypeError(
                "LightGBM model must expose "
                "'predict_proba' or 'predict'."
            )

        prediction = validate_prediction(
            prediction=prediction,
            expected_rows=len(X),
            name="LightGBM model",
        )

        predictions.append(
            prediction
        )

    if not predictions:
        raise ValueError(
            "No LightGBM models were supplied."
        )

    ensemble_prediction = np.mean(
        np.column_stack(
            predictions
        ),
        axis=1,
    )

    return ensemble_prediction.astype(
        np.float32
    )


# ---------------------------------------------------------------------
# Neural-network inference
# ---------------------------------------------------------------------

def probability_to_logit(
    probability: np.ndarray,
) -> np.ndarray:
    """
    Convert probability into log-odds.
    """

    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    probability = np.clip(
        probability,
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(
        probability
        / (
            1.0
            - probability
        )
    )


def predict_nn_farm(
    members,
    X: np.ndarray,
    anchor_probability: np.ndarray,
) -> np.ndarray:
    """
    Average predictions from neural-network farm members.

    Each public inference member is expected to expose:

        member.predict(X, anchor_logit)

    The original competition submission used exported neural-network
    parameters and a lightweight NumPy forward pass.

    Parameters
    ----------
    members:
        Iterable of NN inference members.

    X:
        Model feature matrix.

    anchor_probability:
        Historical / smoothed control-success probability used as the
        neural-network logit anchor.

    Returns
    -------
    np.ndarray
        Averaged NN-farm probability.
    """

    X = validate_feature_matrix(X)

    anchor_probability = validate_prediction(
        prediction=anchor_probability,
        expected_rows=len(X),
        name="Anchor probability",
    )

    anchor_logit = (
        probability_to_logit(
            anchor_probability
        )
    )

    predictions = []

    for member in members:

        if not hasattr(
            member,
            "predict",
        ):
            raise TypeError(
                "NN member must expose a "
                "'predict(X, anchor_logit)' method."
            )

        prediction = member.predict(
            X,
            anchor_logit,
        )

        prediction = validate_prediction(
            prediction=prediction,
            expected_rows=len(X),
            name="NN member",
        )

        predictions.append(
            prediction
        )

    if not predictions:
        raise ValueError(
            "No neural-network members were supplied."
        )

    ensemble_prediction = np.mean(
        np.column_stack(
            predictions
        ),
        axis=1,
    )

    return ensemble_prediction.astype(
        np.float32
    )


# ---------------------------------------------------------------------
# Final V7 model blend
# ---------------------------------------------------------------------

def predict_v7(
    X: np.ndarray,
    lightgbm_models,
    nn_members,
    anchor_probability: np.ndarray,
) -> dict[str, np.ndarray]:
    """
    Generate model-family and final ensemble probabilities.

    Best public-leaderboard blend:

        25% LightGBM
        +
        75% NN farm

    Returns
    -------
    dict
        ``lightgbm``, ``neural_network`` and ``final`` predictions.
    """

    X = validate_feature_matrix(X)

    lgb_prediction = (
        predict_lightgbm_ensemble(
            models=lightgbm_models,
            X=X,
        )
    )

    nn_prediction = (
        predict_nn_farm(
            members=nn_members,
            X=X,
            anchor_probability=(
                anchor_probability
            ),
        )
    )

    final_prediction = blend_v7(
        lgb_prediction,
        nn_prediction,
    )

    return {
        "lightgbm":
            lgb_prediction,

        "neural_network":
            nn_prediction,

        "final":
            final_prediction,
    }


# ---------------------------------------------------------------------
# Anchor resolution
# ---------------------------------------------------------------------

def resolve_anchor_probability(
    engineered: pd.DataFrame,
    anchor_probability: np.ndarray | None = None,
    anchor_column: str | None = None,
) -> np.ndarray:
    """
    Resolve the NN anchor probability.

    The original V7 model used a smoothed season-level success estimate
    as the logit anchor.

    In the public portfolio code, the exact original season-decomposition
    feature pipeline is not fully reproduced. Therefore, the anchor may be
    supplied in either of two ways:

    1. Directly through ``anchor_probability``.
    2. By naming an existing engineered feature through ``anchor_column``.

    This avoids pretending that the simplified public feature pipeline
    exactly reproduces the original competition anchor.
    """

    n_rows = len(engineered)

    if anchor_probability is not None:

        return validate_prediction(
            prediction=anchor_probability,
            expected_rows=n_rows,
            name="Anchor probability",
        ).astype(np.float32)

    if anchor_column is None:
        raise ValueError(
            "Either 'anchor_probability' or "
            "'anchor_column' must be provided."
        )

    if anchor_column not in engineered.columns:
        raise KeyError(
            f"Anchor feature '{anchor_column}' "
            "was not generated by the feature pipeline."
        )

    anchor = (
        engineered[
            anchor_column
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    return validate_prediction(
        prediction=anchor,
        expected_rows=n_rows,
        name="Anchor probability",
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Submission generation
# ---------------------------------------------------------------------

def build_submission(
    test: pd.DataFrame,
    prediction: np.ndarray,
    sample_submission: pd.DataFrame | None = None,
    prior: float = 0.5,
) -> pd.DataFrame:
    """
    Create a competition-style submission dataframe.

    If ``sample_submission`` is supplied, its row order is preserved.
    Predictions are matched through ``row_id`` rather than assuming that
    test rows and template rows are already in identical order.
    """

    if ID_COLUMN not in test.columns:
        raise KeyError(
            f"Test data must contain "
            f"'{ID_COLUMN}'."
        )

    prediction = validate_prediction(
        prediction=prediction,
        expected_rows=len(test),
        name="Final prediction",
    )

    prediction_map = dict(
        zip(
            test[
                ID_COLUMN
            ].tolist(),
            prediction.tolist(),
        )
    )

    if sample_submission is None:

        return pd.DataFrame(
            {
                ID_COLUMN:
                    test[
                        ID_COLUMN
                    ].values,

                TARGET_COLUMN:
                    prediction,
            }
        )

    if (
        ID_COLUMN
        not in sample_submission.columns
    ):
        raise KeyError(
            f"Sample submission must contain "
            f"'{ID_COLUMN}'."
        )

    submission = (
        sample_submission.copy()
    )

    submission[
        TARGET_COLUMN
    ] = [
        prediction_map.get(
            row_id,
            prior,
        )
        for row_id
        in submission[
            ID_COLUMN
        ]
    ]

    return submission


def save_submission(
    submission: pd.DataFrame,
    path: str = "submission.csv",
) -> None:
    """
    Save a submission dataframe as UTF-8 CSV.
    """

    required_columns = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = (
        required_columns
        - set(
            submission.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Submission is missing columns: "
            f"{sorted(missing_columns)}"
        )

    submission.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# End-to-end inference
# ---------------------------------------------------------------------

def run_inference(
    test: pd.DataFrame,
    feature_transform,
    feature_artifacts,
    feature_order: list[str],
    lightgbm_models,
    nn_members,
    sample_submission: pd.DataFrame | None = None,
    anchor_probability: np.ndarray | None = None,
    anchor_column: str | None = None,
    prior: float = 0.5,
):
    """
    Run the portfolio-level V7 inference pipeline.

    Parameters
    ----------
    test:
        Evaluation dataframe.

    feature_transform:
        Callable that applies train-derived feature engineering.

        Expected signature:

            feature_transform(test, feature_artifacts)

    feature_artifacts:
        Lookup tables / historical statistics created from training data.

    feature_order:
        Exact feature order expected by the trained models.

    lightgbm_models:
        Trained LightGBM ensemble members.

    nn_members:
        Neural-network inference members.

    sample_submission:
        Optional competition submission template.

    anchor_probability:
        Optional precomputed anchor probability.

        This is useful when the original competition anchor was generated
        by a richer private / competition feature pipeline.

    anchor_column:
        Optional name of an engineered feature containing the anchor.

        Used only when ``anchor_probability`` is not supplied.

    prior:
        Fallback probability for template row IDs that cannot be matched.

    Returns
    -------
    tuple
        ``(submission, predictions)``

        where ``predictions`` contains the individual model-family
        predictions and the final blended probability.
    """

    # -------------------------------------------------------------
    # 1. Feature engineering
    # -------------------------------------------------------------

    engineered = feature_transform(
        test,
        feature_artifacts,
    )

    if not isinstance(
        engineered,
        pd.DataFrame,
    ):
        raise TypeError(
            "feature_transform must return "
            "a pandas DataFrame."
        )

    if len(engineered) != len(test):
        raise ValueError(
            "Feature engineering changed the "
            "number of evaluation rows."
        )

    # -------------------------------------------------------------
    # 2. Resolve historical NN anchor
    # -------------------------------------------------------------

    anchor = resolve_anchor_probability(
        engineered=engineered,
        anchor_probability=(
            anchor_probability
        ),
        anchor_column=(
            anchor_column
        ),
    )

    # -------------------------------------------------------------
    # 3. Convert engineered features to model matrix
    #
    # Categorical encoding is handled inside to_model_matrix().
    # -------------------------------------------------------------

    X = to_model_matrix(
        engineered,
        feature_order,
    )

    X = validate_feature_matrix(X)

    # -------------------------------------------------------------
    # 4. Model inference
    # -------------------------------------------------------------

    predictions = predict_v7(
        X=X,
        lightgbm_models=(
            lightgbm_models
        ),
        nn_members=(
            nn_members
        ),
        anchor_probability=(
            anchor
        ),
    )

    # -------------------------------------------------------------
    # 5. Submission construction
    # -------------------------------------------------------------

    submission = build_submission(
        test=test,
        prediction=(
            predictions[
                "final"
            ]
        ),
        sample_submission=(
            sample_submission
        ),
        prior=prior,
    )

    return (
        submission,
        predictions,
    )
