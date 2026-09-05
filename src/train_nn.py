"""
Neural-network training pipeline for pitch-level control prediction.

Main ideas used in the project
------------------------------
1. Categorical embeddings for game/context variables
2. Robust normalization of numerical features
3. A strong probability prior used as a logit anchor
4. Multi-task auxiliary targets to improve representation learning
5. Multi-seed + snapshot ensembling ("NN farm")

This is a cleaned portfolio version of the V7 neural-network pipeline.
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

@dataclass
class NNConfig:
    hidden_dim: int = 64
    dropout: float = 0.30

    learning_rate: float = 1e-3
    weight_decay: float = 1e-2

    batch_size: int = 8192

    epochs: int = 60

    # Late epochs are saved as snapshot ensemble members.
    snapshot_start: int = 44

    auxiliary_weight: float = 0.30


DEFAULT_SEEDS = [0, 1, 2]


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def logit(
    probability: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Convert probabilities into logits.

    The project used an existing season-level success estimate as an
    anchor rather than forcing the neural network to learn the entire
    probability from zero.
    """
    p = np.clip(
        np.asarray(probability, dtype=np.float32),
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
    Choose compact embedding dimensions from category cardinalities.

    This keeps small categorical variables inexpensive while allowing
    larger vocabularies slightly more representation capacity.
    """
    return [
        min(
            cap,
            max(
                2,
                int(round(cardinality ** exponent)),
            ),
        )
        for cardinality in cardinalities
    ]


# ---------------------------------------------------------------------
# Numeric preprocessing
# ---------------------------------------------------------------------

class RobustNumericScaler:
    """
    Robust numerical preprocessing used before neural-network training.

    - median imputation
    - scaling by the 1st–99th percentile range
    - clipping extreme normalized values
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

        n = len(X)

        if n > sample_size:
            idx = rng.choice(
                n,
                sample_size,
                replace=False,
            )
            sample = X[np.sort(idx)]
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

        if self.median_ is None or self.range_ is None:
            raise RuntimeError(
                "Scaler must be fitted before transform()."
            )

        out = np.asarray(
            X,
            dtype=np.float32,
        ).copy()

        missing = np.isnan(out)

        if missing.any():
            rows, cols = np.where(missing)
            out[rows, cols] = self.median_[cols]

        out -= self.median_
        out /= self.range_

        np.clip(
            out,
            -5.0,
            5.0,
            out=out,
        )

        return np.ascontiguousarray(out)


# ---------------------------------------------------------------------
# Multi-task neural network
# ---------------------------------------------------------------------

class MultiTaskMLP(nn.Module):
    """
    Embedding + numerical-feature MLP.

    The primary head predicts control_success.

    Auxiliary heads predict related pitch outcomes such as:
        - reverse miss
        - middle miss
        - ball
        - strike

    Auxiliary targets are used only during training.
    """

    def __init__(
        self,
        cardinalities: list[int],
        n_numeric: int,
        n_auxiliary: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.30,
    ):
        super().__init__()

        dims = embedding_dimensions(
            cardinalities
        )

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(cardinality, dim)
                for cardinality, dim
                in zip(cardinalities, dims)
            ]
        )

        input_dim = sum(dims) + n_numeric

        self.trunk = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Dropout(dropout),

            nn.Linear(
                input_dim,
                hidden_dim,
            ),
            nn.SiLU(),

            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),

            nn.Linear(
                hidden_dim,
                hidden_dim // 2,
            ),
            nn.SiLU(),

            nn.Dropout(dropout / 2),
        )

        self.primary_head = nn.Linear(
            hidden_dim // 2,
            1,
        )

        self.auxiliary_head = nn.Linear(
            hidden_dim // 2,
            n_auxiliary,
        )

        # Begin near the anchor probability.
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
    ):
        embeddings = [
            embedding(
                categorical[:, i]
            )
            for i, embedding
            in enumerate(self.embeddings)
        ]

        x = torch.cat(
            embeddings + [numerical],
            dim=1,
        )

        representation = self.trunk(x)

        primary_logit = (
            self.primary_head(
                representation
            )
            .squeeze(1)
        )

        auxiliary_logits = (
            self.auxiliary_head(
                representation
            )
        )

        return (
            primary_logit,
            auxiliary_logits,
        )


# ---------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------

def train_one_seed(
    categorical: np.ndarray,
    numerical: np.ndarray,
    anchor_probability: np.ndarray,
    y_primary: np.ndarray,
    y_auxiliary: np.ndarray,
    cardinalities: list[int],
    seed: int,
    config: NNConfig | None = None,
    device: str | None = None,
):
    """
    Train one multi-task neural-network member.

    Prediction is modeled as:

        sigmoid(anchor_logit + neural_network_residual)

    rather than learning the full probability independently.
    """
    if config is None:
        config = NNConfig()

    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    torch.manual_seed(seed)

    categorical_t = torch.from_numpy(
        categorical.astype(np.int64)
    )

    numerical_t = torch.from_numpy(
        numerical.astype(np.float32)
    )

    anchor_t = torch.from_numpy(
        logit(anchor_probability)
    )

    primary_t = torch.from_numpy(
        y_primary.astype(np.float32)
    )

    auxiliary_t = torch.from_numpy(
        y_auxiliary.astype(np.float32)
    )

    dataset = TensorDataset(
        categorical_t,
        numerical_t,
        anchor_t,
        primary_t,
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

    primary_loss_fn = (
        nn.BCEWithLogitsLoss()
    )

    auxiliary_loss_fn = (
        nn.BCEWithLogitsLoss()
    )

    snapshots = []

    for epoch in range(config.epochs):

        model.train()

        for (
            cat_batch,
            num_batch,
            anchor_batch,
            target_batch,
            aux_batch,
        ) in loader:

            cat_batch = cat_batch.to(device)
            num_batch = num_batch.to(device)
            anchor_batch = anchor_batch.to(device)
            target_batch = target_batch.to(device)
            aux_batch = aux_batch.to(device)

            optimizer.zero_grad()

            residual_logit, aux_logits = model(
                cat_batch,
                num_batch,
            )

            final_logit = (
                residual_logit
                + anchor_batch
            )

            primary_loss = primary_loss_fn(
                final_logit,
                target_batch,
            )

            auxiliary_loss = auxiliary_loss_fn(
                aux_logits,
                aux_batch,
            )

            loss = (
                primary_loss
                + config.auxiliary_weight
                * auxiliary_loss
            )

            loss.backward()

            optimizer.step()

        # ---------------------------------------------------------
        # Snapshot ensemble
        # ---------------------------------------------------------

        if epoch >= config.snapshot_start:

            snapshot = {
                key: value
                .detach()
                .cpu()
                .clone()

                for key, value
                in model.state_dict().items()
            }

            snapshots.append(snapshot)

    return model, snapshots


# ---------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------

@torch.no_grad()
def predict_model(
    model: MultiTaskMLP,
    categorical: np.ndarray,
    numerical: np.ndarray,
    anchor_probability: np.ndarray,
    batch_size: int = 32_768,
    device: str | None = None,
) -> np.ndarray:
    """
    Predict control_success probabilities for one NN snapshot.
    """
    if device is None:
        device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    model = model.to(device)
    model.eval()

    cat = torch.from_numpy(
        categorical.astype(np.int64)
    )

    num = torch.from_numpy(
        numerical.astype(np.float32)
    )

    anchor = torch.from_numpy(
        logit(anchor_probability)
    )

    predictions = []

    for start in range(
        0,
        len(cat),
        batch_size,
    ):
        end = start + batch_size

        cat_batch = cat[start:end].to(device)
        num_batch = num[start:end].to(device)
        anchor_batch = anchor[start:end].to(device)

        residual_logit, _ = model(
            cat_batch,
            num_batch,
        )

        probability = torch.sigmoid(
            residual_logit
            + anchor_batch
        )

        predictions.append(
            probability.cpu().numpy()
        )

    return np.concatenate(predictions)


# ---------------------------------------------------------------------
# NN farm
# ---------------------------------------------------------------------

def train_nn_farm(
    categorical: np.ndarray,
    numerical: np.ndarray,
    anchor_probability: np.ndarray,
    y_primary: np.ndarray,
    y_auxiliary: np.ndarray,
    cardinalities: list[int],
    seeds: list[int] = DEFAULT_SEEDS,
    config: NNConfig | None = None,
):
    """
    Train multiple seeds and collect late-epoch snapshots.

    The competition model averaged predictions across these independent
    members to reduce variance and improve stability.
    """
    farm = []

    for seed in seeds:

        _, snapshots = train_one_seed(
            categorical=categorical,
            numerical=numerical,
            anchor_probability=anchor_probability,
            y_primary=y_primary,
            y_auxiliary=y_auxiliary,
            cardinalities=cardinalities,
            seed=seed,
            config=config,
        )

        farm.append(
            {
                "seed": seed,
                "snapshots": snapshots,
            }
        )

    return farm
