# Benchmark and Leakage Audit

## Summary

The public benchmark runner in `main.py` is chronological, train-only
normalized, and train-only clustered. Under the released benchmark workflow, we
do not find a direct information leakage path from test labels into training or
checkpoint selection.

## Checked Points

### 1. Chronological split

Each market is split by time into:

- training: 1162 days
- validation: 294 days
- test: 728 days

Temporal order is preserved throughout the benchmark.

### 2. Feature scaling

Exogenous features are standardized with a scaler fitted on the training split
only. The fitted scaler is then reused on validation and test.

This is benchmark-safe.

### 3. Clustering

The Gaussian mixture models used for clustering are fitted on the training split
only:

- point-wise GMM
- sampled point-wise GMM
- segment-wise GMM

Validation and test splits only consume clustering features derived from the
training-fitted models.

This is benchmark-safe.

### 4. Checkpoint selection

Checkpoint selection uses validation MAE only.

This is benchmark-safe.

### 5. Final reporting

The test split is used only for final evaluation after the model checkpoint has
already been selected.

This is benchmark-safe.

## Publication Note

This repository is intended as a fixed-configuration benchmark release. The
public code exposes a single finalized configuration rather than the internal
development workflow used to arrive at it.
