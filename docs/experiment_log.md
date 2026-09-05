# 🧪 Experiment Log

This document summarizes the major experiments conducted during the
LG Aimers baseball pitch-control prediction project.

Rather than treating leaderboard improvement as a pure hyperparameter
search problem, experiments focused on three questions:

1. How should temporal distribution shift be handled?
2. Which model families provide genuinely complementary predictions?
3. How can pitch-level context and historical player information be
   converted into stable probability estimates?

---

## 1. Experimental Setup

### Task

Predict the probability of `control_success` for each pitch.

The training dataset contained approximately **1.47 million pitch-level
observations** across multiple seasons.

A major challenge was that baseball data is inherently temporal:
player ability, pitch usage, league environment, and game context can
change from season to season.

Because of this, random train/validation splits were avoided for the
main experiments.

### Validation strategy

The primary validation setup followed a chronological structure:

```text
Past seasons
     │
     ▼
Training
     │
     ▼
Future season
     │
     ▼
Validation
```

Historical lookup features were also constructed only from information
available before the validation season.

This was designed to reduce temporal leakage and better approximate
future-season inference.

---

# 2. Baseline — V4

The early pipeline relied mainly on:

- game-state variables
- pitcher / batter historical statistics
- count information
- handedness
- basic historical-rate features
- gradient-boosting models

### Public LB

**≈ 1087**

### Observation

The baseline confirmed that historical pitcher performance was highly
predictive, but raw historical rates had two important weaknesses:

1. small-sample instability
2. season-to-season distribution shift

This motivated stronger historical smoothing and model diversification.

---

# 3. Empirical-Bayes Historical Features

Player and contextual success rates can become unreliable when the
number of observations is small.

To stabilize them, historical rates were shrunk toward a prior:

\[
\hat{p}
=
\frac{s + k p_0}{n+k}
\]

where:

- \(s\): historical successes
- \(n\): number of observations
- \(p_0\): prior success probability
- \(k\): smoothing strength

This was applied to several levels of baseball context, including:

- pitcher
- batter
- count
- pitcher × count
- pitcher × batter handedness
- batter × pitcher handedness
- pitcher × batter matchup
- inning / base-state context

### Why it helped

A raw rate based on a small number of pitches can be extremely noisy.

Empirical-Bayes smoothing allowed:

```text
small sample
    ↓
stronger shrinkage toward prior

large sample
    ↓
greater trust in player-specific history
```

This produced more stable player/context representations.

---

# 4. V6 — Model Simplification

An intermediate version tested whether additional model families were
actually improving the ensemble.

Some components added complexity without providing enough independent
signal, so the pipeline was simplified around stronger components.

### Public LB

**≈ 1091**

### Lesson

Adding another model does not automatically improve an ensemble.

A useful ensemble member needs both:

- strong standalone performance
- sufficiently different prediction errors

This became an important criterion for later experiments.

---

# 5. Stronger Neural-Network Farm — V7

The largest improvement came from strengthening the neural-network
component.

The NN used:

- categorical embeddings
- numerical context features
- a smoothed season-level probability as a logit anchor
- auxiliary pitch-outcome targets
- multiple random seeds
- late-epoch snapshot averaging

Instead of predicting the full probability from scratch, the network
learned a correction:

\[
p =
\sigma
\left(
\operatorname{logit}(p_{\text{anchor}})
+
f_\theta(x)
\right)
\]

where \(p_{\text{anchor}}\) represents an existing smoothed estimate.

---

## Multi-task learning

Two auxiliary configurations were explored.

### Auxiliary configuration A

```text
control_success
├── reverse
└── middle
```

### Auxiliary configuration B

```text
control_success
├── reverse
├── middle
├── ball
└── strike
```

These related outcomes encouraged the network to learn a richer
representation of pitch-control behavior.

---

## NN Farm

The final V7-style NN farm combined:

```text
2 auxiliary configurations
×
3 random seeds
×
16 late-epoch snapshots
=
96 snapshot members
```

Averaging these members reduced prediction variance and made the neural
component more stable.

### Result

The stronger NN farm produced a substantial leaderboard improvement over
earlier versions.

---

# 6. Ensemble Weight Search

LightGBM and the NN farm captured different parts of the problem.

Several blend ratios were tested.

Two strong configurations were:

| LightGBM | NN Farm | Public LB |
|---:|---:|---:|
| 30% | 70% | 1113.9323 |
| **25%** | **75%** | **1115.2420** |

The best observed configuration was therefore:

\[
P_{\text{final}}
=
0.25P_{\text{LGB}}
+
0.75P_{\text{NN}}
\]

### Best Public LB

# **1115.241988**

This became the best-performing V7 configuration.

---

# 7. DART Experiment

A DART-based gradient-boosting model was tested as a possible ensemble
member.

### Result

Standalone validation performance was weaker than the strongest existing
models.

More importantly, its predictions were highly correlated with the
existing LightGBM models:

```text
correlation ≈ 0.96–0.97
```

### Decision

**Rejected from the final ensemble.**

### Lesson

The experiment reinforced that ensemble diversity matters.

Even a reasonable model may contribute little when its predictions are
too similar to those of an existing model.

---

# 8. Residual Neural Network

Another experiment attempted to train a neural network on the residual
errors of a LightGBM anchor.

Conceptually:

\[
p =
p_{\text{LGB}}
+
f_\theta(x)
\]

or an equivalent residual correction in probability/logit space.

### Validation result

Approximate internal validation scores:

```text
LightGBM anchor : 878.64
Residual NN     : 840.80
V7 end-to-end NN: 932.77
```

The residual model did not provide sufficient improvement.

### Decision

**Rejected.**

### Lesson

A strong base model does not guarantee that its residuals contain an
easy-to-learn structure.

The end-to-end NN representation was more effective.

---

# 9. Recency Weighting

Because baseball performance changes over time, older seasons were
down-weighted to test whether recent observations should receive more
importance.

Several decay strengths were evaluated.

### Result

The strongest result occurred around:

```text
decay = 1.0
```

More aggressive recency weighting degraded validation performance.

### Decision

No aggressive time decay was used in the best V7 configuration.

### Interpretation

Older observations still contained useful information.

Simply assuming that newer data is always more valuable caused useful
historical signal to be discarded.

---

# 10. Probability Sharpening

A post-processing experiment attempted to increase prediction confidence
by scaling logits.

Conceptually:

\[
p'
=
\sigma
\left(
\alpha \cdot
\operatorname{logit}(p)
\right)
\]

with:

```text
α = 1.12
```

### Result

Public LB decreased:

```text
before sharpening ≈ 1091
after sharpening  ≈ 1089
```

### Decision

**Rejected.**

### Lesson

More confident predictions are not necessarily better calibrated
predictions.

Probability post-processing must be validated rather than assumed to
help.

---

# 11. Four-Class Auxiliary Learning

A later experiment reconstructed pitch outcomes into four classes:

```text
0 → control success
1 → middle miss
2 → reverse miss
3 → far miss
```

Approximately **1.47 million usable rows** were available for this
reconstructed target.

The goal was to determine whether learning *how* a pitch missed could
improve the binary success probability.

### Result

The experiment showed useful signal but did not replace the best V7
pipeline.

### Decision

Kept as an experimental direction rather than part of the
best leaderboard model.

---

# 12. Factorization-Machine Interaction Experiment

A later-stage experiment explored explicit interaction modeling using a
Factorization Machine style neural network.

The motivation was that baseball outcomes depend heavily on interactions
such as:

```text
pitcher × batter
pitcher × count
pitcher × handedness
context × player ability
```

The FM interaction term was:

\[
\frac{1}{2}
\left[
\left(\sum_i v_i\right)^2
-
\sum_i v_i^2
\right]
\]

A dedicated pitcher × batter Hadamard interaction was also tested.

### Decision

This was an experimental extension and **was not part of the
1115.24 best leaderboard model**.

It is documented because it influenced later work on interaction-aware
architectures.

---

# 13. Deployment Experiment

Competition inference introduced a separate engineering challenge.

The initial deployment pipeline relied on serialized Python model
objects, which created environment-compatibility issues.

A later portable inference design therefore explored:

- NumPy model parameters
- JSON / array-based artifacts
- direct tree traversal
- NumPy neural-network forward passes
- reduced runtime dependencies

Local numerical checks showed that exported models could closely
reproduce the original model outputs.

However, the final hidden-evaluation deployment did **not** complete
successfully.

The exact failure cause was not conclusively identified.

Therefore, this project does **not** claim successful production-scale
deployment of the portable inference implementation.

### Deployment lesson

Local numerical equivalence is not sufficient.

A competition or production inference pipeline should also be tested
under realistic scale and environment constraints:

```text
correctness
+
serialization compatibility
+
memory usage
+
runtime
+
full-scale inference
```

This became one of the most important engineering lessons from the
project.

---

# 14. Leaderboard Progression

| Version | Main change | Public LB |
|---|---|---:|
| V4 | Early temporal / historical pipeline | ~1087 |
| V6 | Simplified ensemble | ~1091 |
| V7 | Stronger NN farm | ~1110 |
| V7 Attack | 30% LGB + 70% NN | 1113.9323 |
| **V7 Attack** | **25% LGB + 75% NN** | **1115.2420** |

The progression was not driven by one single hyperparameter.

The largest improvements came from:

- better temporal validation
- more stable historical features
- stronger NN representation learning
- multi-seed / snapshot averaging
- selecting complementary model families
- empirical ensemble-weight search

---

# 15. Key Takeaways

### 1. Validation design can matter more than model complexity

For temporal sports data, a random split can provide misleading
feedback.

The validation scheme must resemble the actual prediction setting.

### 2. Historical rates require uncertainty modeling

A player's observed success rate is not equally reliable at every sample
size.

Empirical-Bayes shrinkage provided a simple way to encode this
uncertainty.

### 3. Diversity is essential for ensembling

DART demonstrated that a new model family is not useful simply because
its algorithm is different.

Prediction diversity must be measured.

### 4. Neural networks benefited from a strong prior

Using a smoothed historical estimate as a logit anchor allowed the NN to
focus on learning contextual corrections.

### 5. Negative experiments are valuable

Residual learning, aggressive recency weighting, DART, and probability
sharpening did not become part of the best model.

They still helped narrow the search space and clarified what the data
supported.

### 6. Deployment requires its own validation

Model accuracy and inference reliability are separate problems.

Full-scale execution testing should be treated as part of the ML
pipeline rather than as a final packaging step.
