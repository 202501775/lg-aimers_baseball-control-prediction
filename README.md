# ⚾ LG Aimers 9th — Baseball Pitch Control Prediction

> 투구 단위 데이터를 활용하여 투수의 **제구 성공 확률(`control_success`)**을 예측한 머신러닝 프로젝트입니다.

LG Aimers 9기 데이터 사이언스 프로젝트에서 약 **147만 건의 pitch-level 데이터**를 활용하여  
각 투구의 제구 성공 확률을 예측하는 모델을 개발했습니다.

단순한 모델 성능 향상뿐만 아니라 **시간에 따른 분포 변화(Temporal Distribution Shift)**,
야구 도메인 기반 Feature Engineering, Empirical Bayes Rating, 모델 앙상블,
확률 보정 및 제한된 실행 환경에서의 추론 파이프라인까지 전체 ML workflow를 경험했습니다.

---

## 📌 Project Overview

| Item | Description |
|---|---|
| Task | Pitch-level `control_success` probability prediction |
| Domain | Baseball / Sports Analytics |
| Training Data | 약 1.47M pitches |
| Main Metric | Probability prediction score |
| Main Models | LightGBM, Logistic Regression, Neural Network |
| Key Methods | Temporal Validation, Empirical Bayes, Trackman Features, Multiclass Learning, Ensemble |
| Best Public LB | **1115.24** |

---

## 🎯 Problem

목표는 각 투구가 주어진 경기 상황에서 **의도한 위치에 성공적으로 제구될 확률**을 예측하는 것입니다.

단순한 이진 분류 문제처럼 보이지만 실제 데이터에서는 다음과 같은 문제가 존재했습니다.

- 투수와 타자의 실력이 시즌에 따라 변화
- 시즌별 데이터 분포 차이
- 투수별 표본 수의 큰 차이
- 경기 상황과 볼카운트에 따른 제구 난이도 변화
- Trackman 데이터의 부분적인 결측
- 동일한 실패라도 실패 방향에 따라 서로 다른 패턴 존재

따라서 random split 위주의 검증보다는 **시간 구조와 야구 도메인을 반영한 모델링**이 중요하다고 판단했습니다.

---

## 🔍 Validation Strategy

### Temporal Validation

Random K-Fold 대신 시즌을 기준으로 validation을 구성했습니다.

과거 시즌으로 학습하고 미래 시즌을 검증하는 방식으로 실제 leaderboard 환경과 유사한
**temporal distribution shift**를 반영하고자 했습니다.

이를 통해 단순한 validation score뿐 아니라

- 연도별 일반화 성능
- 모델 간 prediction correlation
- 새로운 feature의 안정성
- calibration 변화

를 함께 확인했습니다.

---

## 🛠 Feature Engineering

### 1. Game Context Features

투구 당시의 경기 상황을 표현하는 변수를 생성했습니다.

- Ball / Strike Count
- Outs
- Inning
- Score Difference
- Base State
- Pitcher / Batter Handedness
- Handedness Matchup
- Count State
- Pressure / Leverage State

---

### 2. Empirical Bayes Ratings

단순 평균은 표본이 적은 선수에게 매우 불안정하다는 문제가 있었습니다.

이를 완화하기 위해 선수 및 경기 상황별 성공률에 **Empirical Bayes Shrinkage**를 적용했습니다.

주요 rating 단위:

- Pitcher
- Batter
- Count
- Handedness Matchup
- League
- Pressure State
- Pitcher × Count
- Pitcher × Batter Hand
- Batter × Pitcher Hand
- Count × Matchup
- Count × Pressure
- Higher-order interactions

또한 과거 시즌의 영향력을 점차 감소시키기 위해 **time-decay**를 적용했습니다.

---

### 3. Trackman Features

투수의 구종 및 투구 특성을 반영하기 위해 Trackman 기반 정보를 추가했습니다.

예시:

- Fastball usage
- Breaking ball usage
- Offspeed usage
- Pitch-type distribution
- Pitcher-level pitch characteristics

Trackman 정보가 존재하지 않는 선수도 안정적으로 처리할 수 있도록 missing indicator와
fallback logic을 함께 구성했습니다.

---

## 🤖 Modeling

서로 다른 inductive bias를 가진 모델을 결합하여 prediction diversity를 확보했습니다.

### Gradient Boosting

Binary target을 직접 예측하는 GBDT ensemble을 구성했습니다.

여러 seed의 모델을 학습하여 variance를 감소시켰습니다.

### Multiclass Auxiliary Learning

단순히 성공/실패만 학습하지 않고 투구 결과를

```text
Success
Middle-side Miss
Reverse-side Miss
Far-side Miss
