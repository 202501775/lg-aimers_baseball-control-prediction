"""
Neural-network training pipeline for pitch-level control prediction.

Best-model NN strategy
----------------------
1. Categorical embeddings + numerical features
2. Robust numerical normalization
3. Season-level success estimate used as a logit anchor
4. Multi-task auxiliary learning
5. Multiple random seeds
6. Late-epoch snapshot ensembling ("NN farm")

The final V7 NN farm used:
- 2 auxiliary-task configurations
- 3 random seeds per configuration
- 16 late-epoch snapshots per run
- 96 snapshot members in total

This file is a cleaned portfolio version of the original competition
training code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

EMBEDDING_COLUMNS = [
    "top_bottom",
    "game_type",
    "base_state",
    "hand_matchup",
    "count_state",
    "pitcher_team_id",
    "batter_team_id",
    "pitcher_hand",
    "batter_hand",
    "inning",
]


@dataclass
class NNConfig:
    hidden_dim: int = 64
    dropout: float = 0.30

    learning_rate: float = 1e-3
    weight_decay: float = 1e-2

    batch_size: int = 8192

    epochs: int = 60
    snapshot_start: int = 44

    auxiliary_weight: float = 0.30


DEFAULT_SEEDS = [0, 1, 2]


AUXILIARY_CONFIGS = {
    "aux2": {
        "targets": [
            "reverse",
            "middle",
        ],
        "weight": 0.30,
    },

    "aux4": {
        "targets": [
            "reverse",
            "middle",
            "ball",
            "strike",
        ],
        "weight": 0.30,
    },
}


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def logit(
    probability: np.ndarray,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Convert probability into log-odds.

    The neural network predicts a residual correction on top of an
    existing season-level success estimate rather than learning the
    entire probability independently.
    """
    p = np.clip(
        np.asarray(
            probability,
            dtype=np.float32,
        ),
        eps,
        1.0 - eps,
    )

    return np.log(
        p / (1.0 - p)
    ).astype(np.float32)


def embedding_dimensions(
    cardinalities: list[int],
    cap: int = 8,
    exponent: float = 0.35,
) -> list[int]:
    """
    Determine compact embedding dimensions.

    Equivalent to the dimension rule used in the original V7 NN code.
    """
    return [
        min(
            cap,
            max(
                2,
                int(
                    round(
                        cardinality
                        ** exponent
                    )
                ),
            ),
        )
        for cardinality in cardinalities
    ]


# ---------------------------------------------------------------------
# Numerical preprocessing
# ---------------------------------------------------------------------

class RobustNumericScaler:
    """
    Numerical preprocessing used by the NN pipeline.

    Procedure
    ---------
    1. Estimate statistics from a random training subset.
    2. Replace missing values with the median.
    3. Scale by the 1st-to-99th percentile range.
    4. Clip normalized values to [-5, 5].

    Statistics must be estimated only from training data.
    """

    def __init__(self):
        self.median_: np.ndarray | None = None
        self.range_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        sample_size: int = 200_000,
        seed: int = 0,
    ):
        rng = np.random.RandomState(seed)

        n_rows = len(X)

        if n_rows > sample_size:
            index = rng.choice(
                n_rows,
                sample_size,
                replace=False,
            )

            sample = X[
                np.sort(index)
            ]

        else:
            sample = X

        self.median_ = np.nanmedian(
            sample,
            axis=0,
        ).astype(np.float32)

        q01, q99 = np.nanpercentile(
            sample,
            [1, 99],
            axis=0,
        )

        self.range_ = np.maximum(
            q99 - q01,
            1e-6,
        ).astype(np.float32)

        return self

    def transform(
        self,
        X: np.ndarray,
    ) -> np.ndarray:
        """
        Apply fitted robust normalization.
        """
        if (
            self.median_ is None
            or self.range_ is None
        ):
            raise RuntimeError(
                "Scaler must be fitted first."
            )

        out = np.ascontiguousarray(
            X,
            dtype=np.float32,
        ).copy()

        missing = np.isnan(out)

        if missing.any():
            rows, cols = np.where(
                missing
            )

            out[
                rows,
                cols,
            ] = self.median_[cols]

        out -= self.median_
        out /= self.range_

        np.clip(
            out,
            -5.0,
            5.0,
            out=out,
        )

        return out


# ---------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------

class MultiTaskMLP(nn.Module):
    """
    Multi-task MLP used in the V7 NN farm.

    Input
    -----
    categorical embeddings
    +
    normalized numerical features

    Output
    ------
    primary:
        residual logit for control_success

    auxiliary:
        logits for related pitch outcomes

    Final primary prediction:

        sigmoid(anchor_logit + residual_logit)
    """

    def __init__(
        self,
        cardinalities: list[int],
        n_numeric: int,
        n_auxiliary: int,
        hidden_dim: int = 64,
        dropout: float = 0.30,
        embedding_cap: int = 8,
    ):
        super().__init__()

        dimensions = embedding_dimensions(
            cardinalities,
            cap=embedding_cap,
        )

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    cardinality,
                    dimension,
                )
                for (
                    cardinality,
                    dimension,
                )
                in zip(
                    cardinalities,
                    dimensions,
                )
            ]
        )

        input_dim = (
            sum(dimensions)
            + n_numeric
        )

        self.trunk = nn.Sequential(

            nn.BatchNorm1d(
                input_dim
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                input_dim,
                hidden_dim,
            ),

            nn.SiLU(),

            nn.BatchNorm1d(
                hidden_dim
            ),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),

            nn.SiLU(),

            nn.Dropout(
                dropout / 2,
            ),
        )

        self.primary_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

        self.auxiliary_head = nn.Linear(
            hidden_dim // 2,
            n_auxiliary,
        )

        # Start from the anchor probability.
        nn.init.zeros_(
            self.primary_head.weight
        )

        nn.init.zeros_(
            self.primary_head.bias
        )

    def forward(
        self,
        categorical: torch.Tensor,
        numerical: torch.Tensor,
        return_auxiliary: bool = False,
    ):
        embedded = [
            embedding(
                categorical[:, index]
            )

            for index, embedding
            in enumerate(
                self.embeddings
            )
        ]

        x = torch.cat(
            embedded
            + [numerical],
            dim=1,
        )

        representation = (
            self.trunk(x)
        )

        primary_logit = (
            self.primary_head(
                representation
            )
            .squeeze(1)
        )

        if return_auxiliary:

            auxiliary_logits = (
                self.auxiliary_head(
                    representation
                )
            )

            return (
                primary_logit,
                auxiliary_logits,
            )

        return primary_logit


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_member(
    categorical: np.ndarray,
    numerical: np.ndarray,
    anchor_probability: np.ndarray,
    y_primary: np.ndarray,
    y_auxiliary: np.ndarray,
    cardinalities: list[int],
    seed: int,
    config: NNConfig,
    device: str | None = None,
):
    """
    Train one NN run and collect late-epoch snapshots.

    With the default configuration:

        epochs = 60
        snapshot_start = 44

    snapshots are collected for epochs 44..59,
    giving 16 snapshots per run.
    """

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    torch.manual_seed(seed)

    categorical_t = torch.from_numpy(
        categorical.astype(
            np.int64
        )
    )

    numerical_t = torch.from_numpy(
        numerical.astype(
            np.float32
        )
    )

    anchor_t = torch.from_numpy(
        logit(
            np.clip(
                anchor_probability,
                0.05,
                0.95,
            )
        )
    )

    target_t = torch.from_numpy(
        y_primary.astype(
            np.float32
        )
    )

    auxiliary_t = torch.from_numpy(
        y_auxiliary.astype(
            np.float32
        )
    )

    dataset = TensorDataset(
        categorical_t,
        numerical_t,
        anchor_t,
        target_t,
        auxiliary_t,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )

    model = MultiTaskMLP(
        cardinalities=cardinalities,
        n_numeric=numerical.shape[1],
        n_auxiliary=y_auxiliary.shape[1],
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    binary_cross_entropy = (
        nn.BCEWithLogitsLoss()
    )

    snapshots = []

    for epoch in range(
        config.epochs
    ):

        model.train()

        for (
            categorical_batch,
            numerical_batch,
            anchor_batch,
            target_batch,
            auxiliary_batch,
        ) in loader:

            categorical_batch = (
                categorical_batch.to(
                    device
                )
            )

            numerical_batch = (
                numerical_batch.to(
                    device
                )
            )

            anchor_batch = (
                anchor_batch.to(
                    device
                )
            )

            target_batch = (
                target_batch.to(
                    device
                )
            )

            auxiliary_batch = (
                auxiliary_batch.to(
                    device
                )
            )

            optimizer.zero_grad()

            (
                residual_logit,
                auxiliary_logits,
            ) = model(
                categorical_batch,
                numerical_batch,
                return_auxiliary=True,
            )

            final_logit = (
                residual_logit
                + anchor_batch
            )

            primary_loss = (
                binary_cross_entropy(
                    final_logit,
                    target_batch,
                )
            )

            auxiliary_loss = (
                binary_cross_entropy(
                    auxiliary_logits,
                    auxiliary_batch,
                )
            )

            loss = (
                primary_loss
                + config.auxiliary_weight
                * auxiliary_loss
            )

            loss.backward()

            optimizer.step()

        # ---------------------------------------------------------
        # Late-epoch snapshot ensemble
        # ---------------------------------------------------------

        if epoch >= config.snapshot_start:

            state = {
                key: value
                .detach()
                .cpu()
                .clone()

                for key, value
                in model.state_dict().items()

                # Auxiliary heads are not needed at inference time.
                if not key.startswith(
                    "auxiliary_head."
                )
            }

            snapshots.append(
                state
            )

    return snapshots


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_snapshot(
    model: MultiTaskMLP,
    categorical: np.ndarray,
    numerical: np.ndarray,
    anchor_probability: np.ndarray,
    batch_size: int = 32_768,
    device: str | None = None,
) -> np.ndarray:
    """
    Generate control_success probabilities for one trained snapshot.
    """
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)
    model.eval()

    categorical_t = torch.from_numpy(
        categorical.astype(
            np.int64
        )
    )

    numerical_t = torch.from_numpy(
        numerical.astype(
            np.float32
        )
    )

    anchor_t = torch.from_numpy(
        logit(
            np.clip(
                anchor_probability,
                0.05,
                0.95,
            )
        )
    )

    predictions = []

    for start in range(
        0,
        len(categorical_t),
        batch_size,
    ):

        end = (
            start
            + batch_size
        )

        residual_logit = model(
            categorical_t[
                start:end
            ].to(device),

            numerical_t[
                start:end
            ].to(device),
        )

        probability = torch.sigmoid(
            residual_logit
            + anchor_t[
                start:end
            ].to(device)
        )

        predictions.append(
            probability
            .cpu()
            .numpy()
        )

    return np.concatenate(
        predictions
    )


# ---------------------------------------------------------------------
# NN farm
# ---------------------------------------------------------------------

def train_nn_farm(
    categorical: np.ndarray,
    numerical: np.ndarray,
    anchor_probability: np.ndarray,
    y_primary: np.ndarray,
    auxiliary_targets: dict[str, np.ndarray],
    cardinalities: list[int],
    seeds: list[int] = DEFAULT_SEEDS,
):
    """
    Train the V7-style NN farm.

    Default farm
    ------------

    Configuration 1:
        reverse + middle auxiliary targets

    Configuration 2:
        reverse + middle + ball + strike

    For each configuration:
        3 seeds

    For each run:
        16 late-epoch snapshots

    Total:
        2 × 3 × 16 = 96 snapshot members
    """

    farm = []

    for (
        configuration_name,
        auxiliary_config,
    ) in AUXILIARY_CONFIGS.items():

        target_names = (
            auxiliary_config[
                "targets"
            ]
        )

        auxiliary_matrix = (
            np.column_stack(
                [
                    auxiliary_targets[
                        name
                    ]
                    for name
                    in target_names
                ]
            )
            .astype(
                np.float32
            )
        )

        config = NNConfig(
            auxiliary_weight=(
                auxiliary_config[
                    "weight"
                ]
            )
        )

        for seed in seeds:

            snapshots = (
                train_one_member(
                    categorical=(
                        categorical
                    ),

                    numerical=(
                        numerical
                    ),

                    anchor_probability=(
                        anchor_probability
                    ),

                    y_primary=(
                        y_primary
                    ),

                    y_auxiliary=(
                        auxiliary_matrix
                    ),

                    cardinalities=(
                        cardinalities
                    ),

                    seed=seed,

                    config=config,
                )
            )

            farm.append(
                {
                    "configuration":
                        configuration_name,

                    "seed":
                        seed,

                    "targets":
                        target_names,

                    "snapshots":
                        snapshots,
                }
            )

    return farm


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

def describe_farm(
    farm: list[dict],
) -> dict:
    """
    Return simple metadata describing the trained ensemble.
    """
    n_runs = len(farm)

    n_snapshots = sum(
        len(
            member["snapshots"]
        )
        for member in farm
    )

    return {
        "training_runs":
            n_runs,

        "snapshot_members":
            n_snapshots,

        "expected_default_members":
            96,
    }
