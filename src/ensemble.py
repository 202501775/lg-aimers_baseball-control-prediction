"""
Ensemble utilities for the final V7 prediction pipeline.

Best public-leaderboard configuration
-------------------------------------
LightGBM ensemble : 25%
Neural-network farm : 75%

Public LB score:
1115.241988

The blend was selected experimentally after comparing multiple
LightGBM / neural-network weighting ratios.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------
# Final V7 blend
# ---------------------------------------------------------------------

LGB_WEIGHT = 0.25
NN_WEIGHT = 0.75


def validate_probabilities(
    prediction: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Validate a probability vector before ensembling.
    """
    prediction = np.asarray(
        prediction,
        dtype=np.float32,
    )

    if prediction.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D probability array."
        )

    if not np.all(
        np.isfinite(prediction)
    ):
        raise ValueError(
            f"{name} contains NaN or infinite values."
        )

    if (
        prediction.min() < 0.0
        or prediction.max() > 1.0
    ):
        raise ValueError(
            f"{name} contains values outside [0, 1]."
        )

    return prediction


def blend_v7(
    lgb_prediction: np.ndarray,
    nn_prediction: np.ndarray,
    lgb_weight: float = LGB_WEIGHT,
    nn_weight: float = NN_WEIGHT,
) -> np.ndarray:
    """
    Blend LightGBM and NN-farm probabilities.

    Final best-performing V7 configuration:

        prediction =
            0.25 * LightGBM
            +
            0.75 * NN farm

    Parameters
    ----------
    lgb_prediction:
        Averaged probability from the LightGBM ensemble.

    nn_prediction:
        Averaged probability from the neural-network farm.

    Returns
    -------
    np.ndarray
        Final control_success probability.
    """

    lgb_prediction = validate_probabilities(
        lgb_prediction,
        "LightGBM prediction",
    )

    nn_prediction = validate_probabilities(
        nn_prediction,
        "NN prediction",
    )

    if len(lgb_prediction) != len(nn_prediction):
        raise ValueError(
            "LightGBM and NN predictions must have "
            "the same number of rows."
        )

    if not np.isclose(
        lgb_weight + nn_weight,
        1.0,
    ):
        raise ValueError(
            "Ensemble weights must sum to 1."
        )

    prediction = (
        lgb_weight * lgb_prediction
        + nn_weight * nn_prediction
    )

    return prediction.astype(
        np.float32
    )


# ---------------------------------------------------------------------
# Blend search
# ---------------------------------------------------------------------

def search_blend_weights(
    y_true: np.ndarray,
    lgb_prediction: np.ndarray,
    nn_prediction: np.ndarray,
    scorer,
    lgb_weights: np.ndarray | None = None,
):
    """
    Search LightGBM / NN blending ratios on validation predictions.

    The competition workflow tested several ratios before selecting
    the final 25% LightGBM + 75% NN configuration.

    Parameters
    ----------
    y_true:
        Ground-truth validation labels.

    lgb_prediction:
        LightGBM validation probabilities.

    nn_prediction:
        NN-farm validation probabilities.

    scorer:
        Callable with signature:

            scorer(y_true, prediction)

        The caller can provide the competition metric or another
        validation metric.

    lgb_weights:
        Candidate LightGBM weights. The NN weight is 1 - LGB weight.

    Returns
    -------
    list[dict]
        Candidate weights and their validation scores.
    """

    if lgb_weights is None:
        lgb_weights = np.arange(
            0.0,
            1.01,
            0.05,
        )

    lgb_prediction = validate_probabilities(
        lgb_prediction,
        "LightGBM prediction",
    )

    nn_prediction = validate_probabilities(
        nn_prediction,
        "NN prediction",
    )

    results = []

    for lgb_weight in lgb_weights:

        nn_weight = (
            1.0
            - float(lgb_weight)
        )

        prediction = (
            float(lgb_weight)
            * lgb_prediction
            +
            nn_weight
            * nn_prediction
        )

        score = scorer(
            y_true,
            prediction,
        )

        results.append(
            {
                "lgb_weight":
                    float(lgb_weight),

                "nn_weight":
                    nn_weight,

                "score":
                    float(score),
            }
        )

    return results


# ---------------------------------------------------------------------
# Prediction diagnostics
# ---------------------------------------------------------------------

def ensemble_diagnostics(
    lgb_prediction: np.ndarray,
    nn_prediction: np.ndarray,
) -> dict:
    """
    Inspect diversity between the two model families.

    Ensemble gains generally depend on both:
    - strong individual models
    - imperfectly correlated prediction errors
    """

    lgb_prediction = validate_probabilities(
        lgb_prediction,
        "LightGBM prediction",
    )

    nn_prediction = validate_probabilities(
        nn_prediction,
        "NN prediction",
    )

    correlation = np.corrcoef(
        lgb_prediction,
        nn_prediction,
    )[0, 1]

    final_prediction = blend_v7(
        lgb_prediction,
        nn_prediction,
    )

    return {
        "lgb_mean":
            float(
                np.mean(
                    lgb_prediction
                )
            ),

        "nn_mean":
            float(
                np.mean(
                    nn_prediction
                )
            ),

        "prediction_correlation":
            float(correlation),

        "mean_absolute_difference":
            float(
                np.mean(
                    np.abs(
                        lgb_prediction
                        - nn_prediction
                    )
                )
            ),

        "final_mean":
            float(
                np.mean(
                    final_prediction
                )
            ),
    }
