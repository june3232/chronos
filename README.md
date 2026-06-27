# Chronos-MCANN Benchmark

This repository contains the public benchmark code for our Chronos-based
mixture-of-experts forecasting experiments across five electricity markets:
`BE`, `DE`, `FR`, `NP`, and `PJM`.

## Repository Layout

- `main.py`
  - benchmark entry point
- `src/config.py`
  - market definitions, chronological splits, and fixed model settings
- `src/core.py`
  - data processing, clustering, router, model construction, training, and evaluation
- `data/`
  - benchmark CSV files
- `models/chronos-2-local/`
  - local Chronos-2 checkpoint used by the benchmark
- `notes/benchmark_audit.md`
  - benchmark protocol and leakage audit

## Benchmark Protocol

- Chronological split for each market:
  - training: 1162 days
  - validation: 294 days
  - test: 728 days
- Forecast task:
  - day-ahead 24-hour forecasting
- Input information:
  - market price history
  - market-specific known exogenous forecasts
- Model family:
  - Chronos-based mixture of experts with a clustering-guided attention router

## Running the Benchmark

```bash
python main.py
```

Outputs are written to:

- `artifacts/runs/<timestamp>/run.log`
- `artifacts/runs/<timestamp>/summary.csv`
- `artifacts/runs/<timestamp>/checkpoints/<market>_best.pt`

Each market-specific checkpoint stores the best validation-selected fine-tuned
weights for that run.

## Fixed Release Configuration

The public release uses a single fixed configuration that was finalized during
development and kept unchanged for the benchmark run in this repository.

Key settings:

- context length: 672
- expert learning rate: 2e-4
- router learning rate: 3e-5
- batch size: 96
- LoRA scope: `attn_plus_head_ffn`
- LoRA rank: 16
- LoRA dropout: 0.0
- cluster prior weight: 0.03
- sequence weight: 0.5
- warm expert epochs: 5
- warm expert weight: 0.3
- post-warm expert weight: 0.05
- router hidden dimension: 64
- router heads: 2
- router layers: 1

## Notes

- The benchmark and leakage review is documented in `notes/benchmark_audit.md`.
- The local Chronos-2 checkpoint is included under
  `models/chronos-2-local/`. The `model.safetensors` file is tracked with Git
  LFS.
