# Mamba from scratch — HDFS log anomaly detection

Mamba implemented from scratch in NumPy and PyTorch, applied to anomaly detection on HDFS logs.

## Results

Trained on HDFS log sequences, evaluated on 10,000 held-out samples. ~3% positive class.

| Metric | Score |
|---|---|
| F1 | 95.0% |
| ROC-AUC | 98.3% |
| PR-AUC | 96.2% |

## Correctness check

NumPy and PyTorch implementations agree to **6×10⁻⁶ error** across the forward pass. Gradient flow verified through all 4 blocks.

## Architecture

4 Mamba blocks, each:
- Input projection → SiLU → selective state space (Δ, B, C projections from input)
- Discretization (continuous → discrete time)
- Causal convolution for local context
- Selective scan (prefix-scan on GPU)
- Gated residual connection

## Compile

Scan reformulated as associative prefix-scan and compiled for NVIDIA T4.

## Stack

Python, PyTorch, NumPy, CUDA.

## What's in the notebook

- `Untitled0.ipynb` — model implementation, NumPy ↔ PyTorch verification, training loop, evaluation.
