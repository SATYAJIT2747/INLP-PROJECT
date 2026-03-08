# AmbiStory Experiments (Trials 1-3)

This repository tracks model development across three experiment stages for the AmbiStory-style plausibility scoring task.

## Overview

- `trial1`: Baseline architecture progression (single encoder -> dual encoder, with/without GloVe).
- `trial2`: Loss-function experiments on the dual-encoder setup.
- `trial3`: Task-aligned modeling (ranking-aware training, richer interactions, meta features, KL head).

## Trial 1: Baselines and Embedding Strategy

1. `trial1/Model 1 Single-Encoder BiSimpleRNN, No GloVe.ipynb`
- Single-encoder BiSimpleRNN regression baseline.
- Uses concatenated text fields (`precontext`, `sentence`, `ending`, `judged_meaning`).
- Keras Tuner search over embedding size, RNN units/layers, dropout, L2, LR.

2. `trial1/Model 2 Dual-EncoderBiSimpleRNN,GloVeFrozen (trainable=False).ipynb`
- Dual encoder: separate story and judged-meaning branches.
- Uses GloVe embeddings (`glove.6B.100d.txt`) with frozen weights.
- Includes attention over encoded sequences.

3. `trial1/Model 3 Dual-EncoderBiSimpleRNN,GloVeTrain able(trainable=True).ipynb`
- PyTorch dual-encoder variant with trainable GloVe initialization.
- Manual vocabulary/tokenization pipeline.
- Continues regression objective while enabling embedding adaptation.

## Trial 2: Loss Function Variants

1. `trial2/Model4–Dual-EncoderBiSimpleRNN,GaussianWeighted MSE.ipynb`
- Keeps dual-encoder structure.
- Applies Gaussian/variance-style weighting to MSE using sample uncertainty (`stdev`).

2. `trial2/Model5– Dual-Encoder BiSimpleRNN, Huber Loss (δ = 1.0).ipynb`
- Switches objective to Huber loss for robust regression.
- Intended to reduce sensitivity to outliers.

3. `trial2/Model6–Dual-EncoderBiSimpleRNN,PairwiseRank ing Loss.ipynb`
- Adds pairwise ranking signal to encourage correct relative ordering.
- Aims to improve rank-based metrics (for example Spearman).

## Trial 3: Task-Aligned Breakthrough Models

1. `trial3/Model7-Breakthrough  BiGRU + Rich Interac tions + Meta Features + Setup Ranking.ipynb`
- Includes a detailed problem analysis section (metric/task mismatch, leakage risk, unused signals).
- Introduces grouped splitting by setup and ranking-aware training.
- Uses richer interactions and extra meta features.

2. `trial3/Model8– BiGRU + Annotator-Distribution KL Head.ipynb`
- Multi-objective design: regression + distribution learning via KL divergence.
- Uses annotator choice distributions as soft targets.
- Incorporates pairwise ranking and setup-aware batching.

## Data and Expected Files

Most notebooks expect dataset files in Colab-like paths:

- `/content/train.json`
- `/content/test.json` (for inference notebooks)
- `/content/glove.6B.100d.txt`

Some notebooks include helper cells to download GloVe.

## Typical Dependencies

- `torch`
- `tensorflow` / `keras`
- `keras-tuner`
- `optuna`
- `scikit-learn`
- `scipy`
- `numpy`

## Suggested Reading Order

1. Trial 1 for baseline setup and architecture evolution.
2. Trial 2 for objective/loss ablations.
3. Trial 3 for final task-aware formulations.
