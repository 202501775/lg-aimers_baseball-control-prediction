# 🏗️ Model Architecture

This document describes the architecture of the final V7 pipeline used
for pitch-level `control_success` probability prediction.

The system was designed around three principles:

1. Respect the temporal structure of baseball data.
2. Combine stable historical priors with pitch-level context.
3. Ensemble models that learn different representations of the same pitch.

---

# 1. System Overview

```text
                  Raw Pitch Data
                 (~1.47M pitches)
                         │
                         ▼
              Temporal Feature Pipeline
                         │
          ┌──────────────┼──────────────┐
          │              │              │
          ▼              ▼              ▼
     Game Context   Player History   Interactions
          │              │              │
          │        Empirical-Bayes      │
          │           Smoothing         │
          └──────────────┼──────────────┘
                         │
                         ▼
                Model Feature Matrix
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       LightGBM Ensemble          NN Farm
                               Anchor + MTL
                                     │
                               96 snapshots
              │                     │
              └──────────┬──────────┘
                         ▼
                 Weighted Ensemble
                  25% LGB + 75% NN
                         │
                         ▼
              control_success
                  probability
```

---

# 2. Temporal Data Architecture

Baseball performance changes over time.

A pitcher's command, a batter's tendencies, pitch usage, and the league
environment can all change between seasons.

For that reason, the project avoids treating observations as
exchangeable samples.

Instead of a random split:

```text
2019 ─┐
2020  │
2021  ├── Training
2022  │
2023 ─┘

2024 ───── Validation / future prediction
```

historical features for a season are generated from information available
before that season.

Conceptually:

\[
\text{Features}_{t}
=
f
\left(
x_t,
\mathcal{H}_{<t}
\right)
\]

where:

- \(x_t\) is the current pitch context
- \(\mathcal{H}_{<t}\) is historical information available before the
  prediction period

This design reduces temporal leakage and better approximates the actual
evaluation setting.

---

# 3. Feature Architecture

The feature pipeline converts each pitch into several groups of signals.

## 3.1 Game context

Examples include:

- inning
- top / bottom
- ball-strike count
- outs
- base state
- score difference
- leverage / pressure context
- pitcher and batter handedness
- home / away context

These variables describe the immediate situation in which the pitch was
thrown.

---

## 3.2 Historical player features

Historical information is aggregated for both pitchers and batters.

Examples include:

```text
Pitcher
├── historical control success
├── reverse / middle miss rates
├── ball / strike rates
├── pitch-mix tendencies
└── contextual performance

Batter
├── historical success environment
├── middle-rate tendencies
└── contextual matchup history
```

Multiple aggregation levels allow the model to represent both general
player ability and context-specific behavior.

---

# 4. Empirical-Bayes Smoothing

Raw historical rates become unstable when sample sizes are small.

For example:

```text
Pitcher A
success = 8 / 10

Pitcher B
success = 800 / 1000
```

Both are observed rates, but they do not have the same statistical
reliability.

Historical rates are therefore shrunk toward a prior:

\[
\hat{p}
=
\frac{s + kp_0}
     {n+k}
\]

where:

- \(s\): observed successes
- \(n\): number of historical observations
- \(p_0\): prior probability
- \(k\): prior strength

As \(n\) increases:

\[
\hat{p}
\rightarrow
\frac{s}{n}
\]

while small samples remain closer to the prior.

This creates more stable representations for sparse players and
high-dimensional interaction groups.

---

# 5. Contextual Interactions

Pitch control is not purely an individual pitcher property.

The same pitcher can behave differently depending on:

```text
pitcher × batter
pitcher × count
pitcher × batter handedness
pitcher × inning
pitcher × base state

batter × count
batter × pitcher handedness
```

The feature pipeline therefore contains multiple interaction-level
historical aggregates.

Derived variables also compare related signals, for example:

```text
pitcher historical ability
        −
batter historical environment
```

These relative features allow the model to reason about a matchup rather
than treating pitcher and batter statistics independently.

---

# 6. LightGBM Branch

The first model family is a LightGBM ensemble.

```text
Feature Matrix
      │
      ├── LightGBM A
      ├── LightGBM B
      ├── LightGBM C
      └── LightGBM D
             │
             ▼
       Average / Ensemble
             │
             ▼
          P_LGB
```

Tree-based models are well suited to the engineered feature space because
they can efficiently capture:

- nonlinear thresholds
- count-state interactions
- player-history interactions
- game-state effects

Multiple model variants reduce dependence on one fitted estimator.

---

# 7. Neural-Network Branch

The second branch learns a different representation.

Its input combines:

```text
Categorical embeddings
        +
Normalized numerical features
```

The main categorical embedding variables include:

```text
top_bottom
game_type
base_state
hand_matchup
count_state
pitcher_team_id
batter_team_id
pitcher_hand
batter_hand
inning
```

The resulting embeddings are concatenated with normalized numerical
features before entering the MLP.

---

# 8. Anchor-Based Residual Learning

Instead of asking the neural network to learn the entire probability from
scratch, the model starts from a smoothed historical estimate.

Let:

\[
p_a
=
\text{historical anchor probability}
\]

The model predicts a residual logit:

\[
r_\theta = f_\theta(x)
\]

and the final probability becomes:

\[
p
=
\sigma
\left(
\operatorname{logit}(p_a)
+
r_\theta
\right)
\]

Architecture:

```text
Historical estimate
        │
        ▼
   logit(anchor)
        │
        │         Pitch features
        │              │
        │              ▼
        │             MLP
        │              │
        │       residual logit
        │              │
        └─────── + ─────┘
                 │
                 ▼
              sigmoid
                 │
                 ▼
       control_success probability
```

This allows the network to focus on learning contextual corrections to a
strong baseball-informed baseline.

---

# 9. Multi-Task Learning

The neural network was trained with related pitch-outcome targets in
addition to the primary binary target.

Two configurations were used.

## Configuration A

```text
Shared Representation
       │
       ├── control_success
       ├── reverse
       └── middle
```

## Configuration B

```text
Shared Representation
       │
       ├── control_success
       ├── reverse
       ├── middle
       ├── ball
       └── strike
```

Training objective:

\[
L
=
L_{\text{primary}}
+
\lambda L_{\text{aux}}
\]

with auxiliary weight:

\[
\lambda = 0.3
\]

The auxiliary heads are used during training to improve representation
learning.

They are not required for final inference.

---

# 10. NN Farm

Model variance was reduced through both seed ensembling and snapshot
averaging.

The final V7 NN farm used:

```text
2 auxiliary configurations
        ×
3 random seeds
        ×
16 late-epoch snapshots
        =
96 snapshot members
```

Importantly, this represents **96 snapshots from 6 training runs**, not
96 independently trained neural networks.

The final neural prediction is obtained by averaging across the farm:

\[
P_{\text{NN}}
=
\frac{1}{M}
\sum_{m=1}^{M}
P_m
\]

where \(M\) is the number of snapshot members.

---

# 11. Final Ensemble

The two model families capture the feature space differently.

```text
                 Feature Matrix
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
      LightGBM                 NN Farm
          │                       │
          ▼                       ▼
        P_LGB                   P_NN
          │                       │
          └───────────┬───────────┘
                      │
                      ▼
            0.25 × P_LGB
                  +
            0.75 × P_NN
                      │
                      ▼
                Final Probability
```

The best observed public-leaderboard configuration was:

\[
P_{\text{final}}
=
0.25P_{\text{LGB}}
+
0.75P_{\text{NN}}
\]

with a Public LB score of:

## **1115.241988**

A nearby 30% / 70% blend produced **1113.9323**, supporting the decision
to give the stronger neural component slightly more weight.

---

# 12. Inference Isolation

A critical constraint of the feature architecture is:

> A prediction for one evaluation row must not depend on other evaluation
> rows.

For pitch \(i\):

\[
\text{features}_i
=
f
\left(
x_i,
\text{training artifacts}
\right)
\]

and never:

\[
f
\left(
x_i,
x_{test,1},
\dots,
x_{test,n}
\right)
\]

This prevents accidental leakage through hidden-test distributions or
future observations.

The same principle is applied to:

- historical player rates
- categorical mappings
- priors
- scaling statistics
- feature ordering

All reusable artifacts are derived from training data.

---

# 13. Inference Architecture

The competition inference pipeline was designed conceptually as:

```text
Evaluation Row
      │
      ▼
Train-derived
Feature Artifacts
      │
      ▼
Feature Engineering
      │
      ▼
Model Matrix
      │
      ├──────────────┐
      ▼              ▼
  LightGBM        NN Farm
      │              │
      └──────┬───────┘
             ▼
       25 / 75 Blend
             │
             ▼
       Probability
             │
             ▼
      submission.csv
```

A later deployment experiment also explored replacing serialized Python
model objects with portable arrays and NumPy inference.

Local numerical equivalence was verified for key exported components.

However, the final hidden-evaluation deployment did not complete
successfully, so the project does not claim successful production-scale
deployment of that portable implementation.

---

# 14. Why Two Model Families?

The final architecture deliberately combines a tree model and a neural
model.

### LightGBM

Strong at:

- threshold-based relationships
- nonlinear tabular interactions
- engineered statistical features

### Neural Network

Strong at:

- categorical embeddings
- distributed representations
- residual corrections
- multi-task representation learning

The goal was not simply to maximize the number of models.

Instead:

\[
\text{Ensemble value}
\approx
\text{individual strength}
+
\text{error diversity}
\]

This was supported by experiments such as DART, where a model with highly
correlated predictions provided limited ensemble value.

---

# 15. Architecture Summary

The final modeling strategy can be summarized as:

```text
Temporal validation
        +
Leakage-safe historical features
        +
Empirical-Bayes smoothing
        +
Context / matchup interactions
        +
LightGBM ensemble
        +
Anchor-based multi-task NN farm
        +
Snapshot / seed averaging
        +
Validation-driven model blending
```

The largest lesson from the project was that strong tabular ML performance
did not come from model complexity alone.

The final improvements came from coordinating:

**validation design, domain-aware features, stable probability estimates,
model diversity, and careful ensembling.**
