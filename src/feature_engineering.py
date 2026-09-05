"""
Feature engineering pipeline for pitch-level control_success prediction.

Design principles
-----------------
1. All historical lookup tables are built only from training data.
2. For temporal validation, each season is transformed using artifacts
   derived from prior seasons only.
3. Player / matchup rates are smoothed to reduce small-sample variance.
4. Current-season form is reconstructed from cumulative as-of statistics.
5. No feature depends on other rows from the evaluation set.

This file is a cleaned portfolio version of the feature pipeline used
during the LG Aimers baseball prediction project.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

PITCHER_RATE_COLS = {
    "asof_pitcher_success_rate": "succ",
    "asof_pitcher_reverse_rate": "rev",
    "asof_pitcher_middle_rate": "mid",
    "asof_pitcher_ball_rate": "ball",
    "asof_pitcher_strike_rate": "strike",
}

PITCH_MIX_COLS = {
    "asof_pitcher_fastball_rate": "fastball",
    "asof_pitcher_breaking_rate": "breaking",
    "asof_pitcher_offspeed_rate": "offspeed",
}

BATTER_RATE_COLS = {
    "asof_batter_success_rate": "batter_succ",
    "asof_batter_middle_rate": "batter_mid",
}


# ---------------------------------------------------------------------
# Historical artifacts
# ---------------------------------------------------------------------

def _aggregate_success(
    df: pd.DataFrame,
    keys: list[str],
) -> pd.DataFrame:
    """
    Aggregate sample count and number of successful pitches.

    Returns
    -------
    pd.DataFrame
        Multi-indexed table with:
        - n: number of observations
        - s: number of control_success == 1
    """
    if len(df) == 0:
        return pd.DataFrame({"n": [], "s": []})

    return (
        df.groupby(keys)["control_success"]
        .agg(n="size", s="sum")
    )


def build_historical_artifacts(
    train: pd.DataFrame,
    upto_season: int,
) -> dict:
    """
    Build train-derived lookup tables available up to a given season.

    This function is intentionally separated from transform_features()
    so that evaluation rows never contribute to historical statistics.

    Parameters
    ----------
    train:
        Full training dataframe.
    upto_season:
        Last season allowed when creating historical aggregates.

    Returns
    -------
    dict
        Lookup tables and prior probabilities.
    """
    history = train[train["season"] <= upto_season].copy()
    latest = train[train["season"] == upto_season].copy()

    artifacts = {"upto_season": int(upto_season)}

    # Player-level historical performance
    artifacts["pitcher_career"] = _aggregate_success(
        history, ["pitcher_id"]
    )
    artifacts["batter_career"] = _aggregate_success(
        history, ["batter_id"]
    )

    # Previous-season performance
    artifacts["pitcher_prev"] = _aggregate_success(
        latest, ["pitcher_id"]
    )
    artifacts["batter_prev"] = _aggregate_success(
        latest, ["batter_id"]
    )

    # Matchup / context interactions
    artifacts["pitcher_batter"] = _aggregate_success(
        history,
        ["pitcher_id", "batter_id"],
    )

    artifacts["pitcher_count"] = _aggregate_success(
        history,
        ["pitcher_id", "balls_before", "strikes_before"],
    )

    artifacts["pitcher_batter_hand"] = _aggregate_success(
        history,
        ["pitcher_id", "batter_hand"],
    )

    artifacts["pitcher_inning"] = _aggregate_success(
        history,
        ["pitcher_id", "inning"],
    )

    artifacts["pitcher_base_state"] = _aggregate_success(
        history,
        ["pitcher_id", "base_state"],
    )

    artifacts["batter_count"] = _aggregate_success(
        history,
        ["batter_id", "balls_before", "strikes_before"],
    )

    artifacts["batter_pitcher_hand"] = _aggregate_success(
        history,
        ["batter_id", "pitcher_hand"],
    )

    # League-level prior
    if len(latest):
        prior = float(latest["control_success"].mean())
    elif len(history):
        prior = float(history["control_success"].mean())
    else:
        prior = 0.50

    artifacts["prior"] = prior

    return artifacts


# ---------------------------------------------------------------------
# Empirical Bayes smoothing
# ---------------------------------------------------------------------

def _add_smoothed_rate(
    frame: pd.DataFrame,
    table: pd.DataFrame,
    keys: list[str],
    name: str,
    prior: float,
    strength: float,
) -> pd.DataFrame:
    """
    Add historical count, raw rate and Empirical-Bayes-smoothed rate.

    Smoothed rate:

        (successes + strength * prior) / (n + strength)

    Small-sample groups are therefore pulled toward the league prior.
    """
    out = frame

    if table is None or len(table) == 0:
        out[f"{name}_n"] = 0.0
        out[f"{name}_success"] = 0.0
    else:
        lookup = table.rename(
            columns={
                "n": f"{name}_n",
                "s": f"{name}_success",
            }
        )

        out = out.merge(
            lookup,
            how="left",
            left_on=keys,
            right_index=True,
        )

        out[f"{name}_n"] = out[f"{name}_n"].fillna(0.0)
        out[f"{name}_success"] = (
            out[f"{name}_success"].fillna(0.0)
        )

    n = out[f"{name}_n"]
    s = out[f"{name}_success"]

    out[f"{name}_rate_raw"] = s / n.replace(0, np.nan)

    out[f"{name}_rate"] = (
        s + strength * prior
    ) / (
        n + strength
    )

    return out


# ---------------------------------------------------------------------
# Main feature transformation
# ---------------------------------------------------------------------

def transform_features(
    df: pd.DataFrame,
    artifacts: dict,
    smoothing_strength: float = 100.0,
) -> pd.DataFrame:
    """
    Generate model features from raw pitch-level rows.

    Every historical feature uses only information stored in `artifacts`.
    """
    x = df.reset_index(drop=True).copy()

    prior = float(artifacts["prior"])


    # ---------------------------------------------------------------
    # 1. Player-level historical ratings
    # ---------------------------------------------------------------

    basic_groups = [
        (
            "pitcher_career",
            ["pitcher_id"],
            "pitcher_career",
        ),
        (
            "batter_career",
            ["batter_id"],
            "batter_career",
        ),
        (
            "pitcher_prev",
            ["pitcher_id"],
            "pitcher_prev",
        ),
        (
            "batter_prev",
            ["batter_id"],
            "batter_prev",
        ),
    ]

    for table_name, keys, feature_name in basic_groups:
        x = _add_smoothed_rate(
            x,
            artifacts[table_name],
            keys,
            feature_name,
            prior,
            smoothing_strength,
        )


    # ---------------------------------------------------------------
    # 2. Current-season decomposition
    # ---------------------------------------------------------------

    pitcher_n = x["asof_pitcher_n"].astype(float)
    batter_n = x["asof_batter_n"].astype(float)

    # Approximate number of pitches accumulated this season
    x["pitcher_season_n"] = (
        pitcher_n - x["pitcher_career_n"]
    ).clip(lower=0)

    x["batter_season_n"] = (
        batter_n - x["batter_career_n"]
    ).clip(lower=0)

    # Recover cumulative successes from as-of rate × sample count
    x["pitcher_season_success"] = (
        x["asof_pitcher_success_rate"].fillna(0.0)
        * pitcher_n
        - x["pitcher_career_success"]
    ).clip(lower=0)

    x["batter_season_success"] = (
        x["asof_batter_success_rate"].fillna(0.0)
        * batter_n
        - x["batter_career_success"]
    ).clip(lower=0)

    # Raw current-season rates
    x["pitcher_season_rate"] = (
        x["pitcher_season_success"]
        / x["pitcher_season_n"].replace(0, np.nan)
    )

    x["batter_season_rate"] = (
        x["batter_season_success"]
        / x["batter_season_n"].replace(0, np.nan)
    )

    # Smoothed current-season rates
    x["pitcher_season_rate_smoothed"] = (
        x["pitcher_season_success"]
        + smoothing_strength * prior
    ) / (
        x["pitcher_season_n"]
        + smoothing_strength
    )

    x["batter_season_rate_smoothed"] = (
        x["batter_season_success"]
        + smoothing_strength * prior
    ) / (
        x["batter_season_n"]
        + smoothing_strength
    )

    # Sample-size features
    x["log_pitcher_season_n"] = np.log1p(
        x["pitcher_season_n"]
    )

    x["log_batter_season_n"] = np.log1p(
        x["batter_season_n"]
    )


    # ---------------------------------------------------------------
    # 3. Context-specific historical ratings
    # ---------------------------------------------------------------

    context_groups = [
        (
            "pitcher_batter",
            ["pitcher_id", "batter_id"],
            "pitcher_batter",
            20.0,
        ),
        (
            "pitcher_count",
            ["pitcher_id", "balls_before", "strikes_before"],
            "pitcher_count",
            50.0,
        ),
        (
            "pitcher_batter_hand",
            ["pitcher_id", "batter_hand"],
            "pitcher_vs_batter_hand",
            50.0,
        ),
        (
            "pitcher_inning",
            ["pitcher_id", "inning"],
            "pitcher_inning",
            50.0,
        ),
        (
            "pitcher_base_state",
            ["pitcher_id", "base_state"],
            "pitcher_base_state",
            50.0,
        ),
        (
            "batter_count",
            ["batter_id", "balls_before", "strikes_before"],
            "batter_count",
            50.0,
        ),
        (
            "batter_pitcher_hand",
            ["batter_id", "pitcher_hand"],
            "batter_vs_pitcher_hand",
            50.0,
        ),
    ]

    for table_name, keys, feature_name, strength in context_groups:
        x = _add_smoothed_rate(
            x,
            artifacts[table_name],
            keys,
            feature_name,
            prior,
            strength,
        )


    # ---------------------------------------------------------------
    # 4. Relative / interaction features
    # ---------------------------------------------------------------

    x["pitcher_season_minus_career"] = (
        x["pitcher_season_rate_smoothed"]
        - x["pitcher_career_rate"]
    )

    x["pitcher_season_minus_prev"] = (
        x["pitcher_season_rate_smoothed"]
        - x["pitcher_prev_rate"]
    )

    x["batter_season_minus_career"] = (
        x["batter_season_rate_smoothed"]
        - x["batter_career_rate"]
    )

    x["pitcher_batter_minus_pitcher"] = (
        x["pitcher_batter_rate"]
        - x["pitcher_career_rate"]
    )

    x["pitcher_count_minus_pitcher"] = (
        x["pitcher_count_rate"]
        - x["pitcher_career_rate"]
    )

    x["pitcher_hand_matchup_minus_pitcher"] = (
        x["pitcher_vs_batter_hand_rate"]
        - x["pitcher_career_rate"]
    )

    x["pitcher_minus_batter"] = (
        x["pitcher_season_rate_smoothed"]
        - x["batter_season_rate_smoothed"]
    )


    # ---------------------------------------------------------------
    # 5. Recent-form features
    # ---------------------------------------------------------------

    x["pitcher_prev1_minus_career"] = (
        x["asof_pitcher_prev1_game_success_rate"]
        - x["pitcher_career_rate"]
    )

    x["pitcher_prev3_minus_career"] = (
        x["asof_pitcher_prev3_game_success_rate"]
        - x["pitcher_career_rate"]
    )

    x["pitcher_prev5_minus_career"] = (
        x["asof_pitcher_prev5_game_success_rate"]
        - x["pitcher_career_rate"]
    )

    x["pitcher_form_3g"] = (
        x["asof_pitcher_prev3_game_success_rate"]
        - x["asof_pitcher_success_rate"]
    )

    x["pitcher_form_5g"] = (
        x["asof_pitcher_prev5_game_success_rate"]
        - x["asof_pitcher_success_rate"]
    )

    x["pitcher_middle_form"] = (
        x["asof_pitcher_prev3_game_middle_rate"]
        - x["asof_pitcher_middle_rate"]
    )


    # ---------------------------------------------------------------
    # 6. Game-state features
    # ---------------------------------------------------------------

    x["hand_matchup"] = (
        x["pitcher_hand"].astype(str)
        + "_"
        + x["batter_hand"].astype(str)
    )

    x["count_state"] = (
        x["balls_before"].astype(str)
        + "-"
        + x["strikes_before"].astype(str)
    )

    x["count_advantage"] = (
        x["strikes_before"]
        - x["balls_before"]
    )

    x["two_strikes"] = (
        x["strikes_before"] == 2
    ).astype(np.int8)

    x["three_balls"] = (
        x["balls_before"] == 3
    ).astype(np.int8)

    x["abs_score_diff"] = (
        x["score_diff_pitcher_team"].abs()
    )

    x["log_leverage_index"] = np.log1p(
        x["li"]
    )

    x["scoring_position"] = (
        (x["runner_on_2b"] == 1)
        | (x["runner_on_3b"] == 1)
    ).astype(np.int8)


    # ---------------------------------------------------------------
    # 7. Pitch-mix uncertainty
    # ---------------------------------------------------------------

    pitch_mix_columns = list(PITCH_MIX_COLS.keys())

    mix = x[pitch_mix_columns].clip(
        lower=1e-9,
        upper=1.0,
    )

    x["pitch_mix_entropy"] = -(
        mix * np.log(mix)
    ).sum(axis=1)


    return x


# ---------------------------------------------------------------------
# Categorical encoding
# ---------------------------------------------------------------------

CATEGORY_MAPS = {
    "top_bottom": {
        "B": 0,
        "T": 1,
    },
    "game_type": {
        "F": 0,
        "R": 1,
    },
    "base_state": {
        "123": 0,
        "12_": 1,
        "1_3": 2,
        "1__": 3,
        "_23": 4,
        "_2_": 5,
        "__3": 6,
        "___": 7,
    },
    "hand_matchup": {
        "1_1": 0,
        "1_2": 1,
        "2_1": 2,
        "2_2": 3,
    },
    "count_state": {
        f"{balls}-{strikes}": idx
        for idx, (balls, strikes) in enumerate(
            [
                (b, s)
                for b in range(4)
                for s in range(3)
            ]
        )
    },
}


def to_model_matrix(
    features: pd.DataFrame,
    feature_order: list[str],
) -> np.ndarray:
    """
    Convert engineered dataframe into a deterministic float32 matrix.

    Unknown categorical values are encoded as -1.
    Missing requested features are filled with NaN.
    """
    x = features.copy()

    for column, mapping in CATEGORY_MAPS.items():
        if column in x.columns:
            x[column] = (
                x[column]
                .astype(str)
                .map(mapping)
                .fillna(-1)
                .astype(np.int16)
            )

    for column in feature_order:
        if column not in x.columns:
            x[column] = np.nan

    return np.ascontiguousarray(
        x[feature_order].to_numpy(dtype=np.float32)
    )
