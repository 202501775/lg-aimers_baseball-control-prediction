"""
Inference utilities for the final V7 ensemble.

Pipeline
--------
1. Transform each pitch row using train-derived feature artifacts.
2. Generate LightGBM ensemble probabilities.
3. Generate neural-network farm probabilities.
4. Blend the two model families using the final V7 weights.
5. Build a submission dataframe while preserving row_id order.

Important
---------
Features are computed from:
    evaluation row
    +
    training-derived artifacts

No feature depends on other evaluation rows.

The competition submission used a lightweight NumPy implementation
for neural-network inference to avoid requiring PyTorch at runtime.
This portfolio module focuses on the model-level inference flow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ensemble import blend_v7


ID_COLUMN = "row_id"
TARGET_COLUMN = "control_success"


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------

def validate_feature_matrix(
    X: np.ndarray,
) -> np.ndarray:
    """
    Validate and normalize the model feature matrix.
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

    return np.ascontiguousarray(X)


def validate_prediction(
    prediction: np.ndarray,
    expected_rows: int,
    name: str,
) -> np.ndarray:
    """
    Ensure one valid probability exists for every input row.
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

    Models may be either:
    - sklearn-style LGBMClassifier objects
    - native LightGBM Booster objects
    """
    X = validate_feature_matrix(X)

    predictions = []

    for model in models:

        if hasattr(
            model,
            "predict_proba",
        ):
            prediction = (
                model.predict_proba(X)[:, 1]
            )

        else:
            prediction = model.predict(X)

        prediction = validate_prediction(
            prediction,
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

    return np.mean(
        np.column_stack(
            predictions
        ),
        axis=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------
# NN-farm inference
# ---------------------------------------------------------------------

def predict_nn_farm(
    members,
    X: np.ndarray,
    anchor_probability: np.ndarray,
) -> np.ndarray:
    """
    Average predictions from NN-farm members.

    Each member is expected to expose a predict method with:

        member.predict(X, anchor_logit)

    The original competition submission exported the trained NN
    parameters and reproduced the forward pass using NumPy.
    """
    X = validate_feature_matrix(X)

    anchor_probability = validate_prediction(
        anchor_probability,
        expected_rows=len(X),
        name="Anchor probability",
    )

    anchor_logit = np.log(
        anchor_probability
        / (
            1.0
            - anchor_probability
        )
    )

    predictions = []

    for member in members:

        prediction = member.predict(
            X,
            anchor_logit,
        )

        prediction = validate_prediction(
            prediction,
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

    return np.mean(
        np.column_stack(
            predictions
        ),
        axis=1,
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Final V7 inference
# ---------------------------------------------------------------------

def predict_v7(
    X: np.ndarray,
    lightgbm_models,
    nn_members,
    anchor_probability: np.ndarray,
) -> dict:
    """
    Generate final V7 probabilities.

    Best public-leaderboard blend:

        25% LightGBM
        75% NN farm
    """
    X = validate_feature_matrix(X)

    lgb_prediction = (
        predict_lightgbm_ensemble(
            lightgbm_models,
            X,
        )
    )

    nn_prediction = (
        predict_nn_farm(
            nn_members,
            X,
            anchor_probability,
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

    If a sample submission is provided, its row order is preserved.
    Predictions are joined using row_id rather than assuming the test
    dataframe and template use identical ordering.
    """
    if ID_COLUMN not in test.columns:
        raise KeyError(
            f"Test data must contain '{ID_COLUMN}'."
        )

    prediction = validate_prediction(
        prediction,
        expected_rows=len(test),
        name="Final prediction",
    )

    prediction_map = dict(
        zip(
            test[ID_COLUMN].tolist(),
            prediction.tolist(),
        )
    )

    if sample_submission is None:

        submission = pd.DataFrame(
            {
                ID_COLUMN:
                    test[ID_COLUMN].values,

                TARGET_COLUMN:
                    prediction,
            }
        )

        return submission

    if ID_COLUMN not in sample_submission.columns:
        raise KeyError(
            f"Sample submission must contain '{ID_COLUMN}'."
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
    Save submission using a portable UTF-8 CSV format.
    """
    required = {
        ID_COLUMN,
        TARGET_COLUMN,
    }

    missing = (
        required
        - set(
            submission.columns
        )
    )

    if missing:
        raise ValueError(
            f"Submission is missing columns: {sorted(missing)}"
        )

    submission.to_csv(
        path,
        index=False,
        encoding="utf-8",
    )


# ---------------------------------------------------------------------
# End-to-end helper
# ---------------------------------------------------------------------

def run_inference(
    test: pd.DataFrame,
    feature_transform,
    feature_artifacts,
    feature_order: list[str],
    lightgbm_models,
    nn_members,
    anchor_column: str,
    sample_submission: pd.DataFrame | None = None,
    prior: float = 0.5,
):
    """
    End-to-end inference helper.

    Parameters
    ----------
    feature_transform:
        Callable that applies train-derived feature engineering.

    feature_artifacts:
        Historical lookup tables created only from training data.

    feature_order:
        Exact model feature order.

    anchor_column:
        Smoothed season-level control-success feature used as the
        neural-network logit anchor.
    """
    engineered = feature_transform(
        test,
        feature_artifacts,
    )

    if anchor_column not in engineered.columns:
        raise KeyError(
            f"Anchor feature '{anchor_column}' was not generated."
        )

    anchor_probability = (
        engineered[
            anchor_column
        ]
        .to_numpy(
            dtype=np.float32
        )
    )

    # Categorical variables should already be encoded by the feature
    # pipeline before conversion to the model matrix.
    X = np.ascontiguousarray(
        engineered[
            feature_order
        ].to_numpy(
            dtype=np.float32
        )
    )

    predictions = predict_v7(
        X=X,
        lightgbm_models=lightgbm_models,
        nn_members=nn_members,
        anchor_probability=(
            anchor_probability
        ),
    )

    submission = build_submission(
        test=test,
        prediction=predictions["final"],
        sample_submission=(
            sample_submission
        ),
        prior=prior,
    )

    return (
        submission,
        predictions,
    )
