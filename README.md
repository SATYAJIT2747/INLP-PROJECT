# AmbiStory Experiments (Task 5)

This repository tracks model development across recurrent baseline trials (Phase 1) and transformer-based architectures (Phase 2) for the SemEval 2026 AmbiStory plausibility scoring task.

## Overview

- `trial1`: Baseline architecture progression (single encoder -> dual encoder, with/without GloVe).
- `trial2`: Loss-function experiments on the dual-encoder setup.
- `trial3`: Task-aligned modeling (ranking-aware training, richer interactions, meta features, KL head).
- `phase2`: Transformer progression addressing Phase 1 failure modes (variance compression, selection bias) via grouped CV and geometric feature engineering.

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

1. `trial3/Model7-Breakthrough BiGRU + Rich Interactions + Meta Features + Setup Ranking.ipynb`
- Includes a detailed problem analysis section (metric/task mismatch, leakage risk, unused signals).
- Introduces grouped splitting by setup and ranking-aware training.
- Uses richer interactions and extra meta features.

2. `trial3/Model8– BiGRU + Annotator-Distribution KL Head.ipynb`
- Multi-objective design: regression + distribution learning via KL divergence.
- Uses annotator choice distributions as soft targets.
- Incorporates pairwise ranking and setup-aware batching.

3. `trial3/Model9– Siamese BiGRU + Cross-Attention + Contrastive Loss.ipynb`
- Replaces simple mean-pooling with a Siamese cross-attention stack.
- Jointly optimizes KL (label-smoothed), pairwise setup ranking, Huber regression, and contrastive cross-sense ranking.
- Achieved highest internal validation score, but highlighted vulnerabilities to prediction-variance compression and selection bias.

## Phase 2: Transformer Cross-Encoders and Bi-Encoders

1. `phase2/Attempt1– DeBERTa Cross-Encoder + Grouped 5-Fold CV.ipynb`
- Replaces recurrent stack with `DeBERTa-v3-base` cross-encoder.
- Replaces fixed single-split with homonym-grouped 5-fold cross-validation to explicitly resolve Phase 1 selection bias.
- Uses Huber loss with a pairwise ranking auxiliary term.

2. `phase2/Attempt2– DeBERTa NLI Warm-Start + PCA Ablation.ipynb`
- Initializes DeBERTa with an NLI-fine-tuned checkpoint (`cross-encoder/nli-deberta-v3-base`) to leverage prior textual entailment knowledge.
- Ablates PCA compression of the `[CLS]` feature space.

3. `phase2/Attempt3– MPNet Bi-Encoder + Geometric Features (Final System).ipynb`
- Final submitted system achieving ρ=0.535.
- Encodes story and meaning separately using `all-mpnet-base-v2`.
- Assembles an explicit geometric feature vector (cosine similarity, Euclidean, Manhattan, token Jaccard, ending flag) alongside a frozen NLI entailment score.
- Implements PCA compression and Stochastic Weight Averaging (SWA).

## Data and Expected Files

Most notebooks expect dataset files in Colab-like paths:
- `/content/train.json`
- `/content/test.json` (for inference notebooks)
- `/content/glove.6B.100d.txt`

Some notebooks include helper cells to download GloVe or specific transformer checkpoints.

## Typical Dependencies

- `torch`
- `tensorflow` / `keras`
- `keras-tuner`
- `transformers` (Hugging Face)
- `sentence-transformers`
- `optuna`
- `scikit-learn`
- `scipy`
- `numpy`

## Suggested Reading Order

1. **Trial 1:** Baseline setup and architecture evolution.
2. **Trial 2:** Objective/loss ablations.
3. **Trial 3:** Task-aware formulations and the diagnosis of single-split/variance collapse.
4. **Phase 2:** The final, winning transformer progressions and geometric feature engineering.

## Presentation Link 
https://docs.google.com/presentation/d/17reGxkRE9YjeljC0eA7Q9DnhiR1RKlh8_ta041p825A/edit?usp=sharing
