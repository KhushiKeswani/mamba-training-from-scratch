# Mamba Training from Scratch

This project is a compact, hands-on implementation of the Mamba architecture built from first principles in PyTorch. It focuses on the core idea behind Mamba: replacing expensive attention with a selective state-space model that updates a hidden state over time.

The repository is intentionally educational: the notebook walks through the architecture step by step, from the basic recurrence to the final Mamba block.

## Why this matters

Transformers are powerful, but attention scales poorly with long sequences because it compares every token to every other token. Mamba addresses that by using a recurrent state-space formulation:

- the model keeps a latent state,
- the state updates over time,
- the transition depends on the current input,
- the result is a more efficient sequence model for long contexts.

## Architecture at a glance

```text
Input tokens
      |
      v
  Input projection
      |
      +---------------------------+
      |                           |
      v                           v
  SSM branch                Gate branch
      |                           |
      v                           v
  Causal depthwise conv      SiLU gate
      |
      v
 Generate delta, B, C
      |
      v
 Discretize A and B
      |
      v
 Selective scan / recurrence
      |
      +---------------------------+
      |                           |
      v                           v
   SSM output                 Skip path
      |                           |
      +-------------+-------------+
                    |
                    v
             Output projection
                    |
                    v
              Residual connection
                    |
                    v
                  Output
```

## Core idea

The central recurrence is:

```python
h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
y_t = C_t * h_t + D * x_t
```

This means each token updates a hidden state using a token-dependent transition. That is the selective behavior that makes Mamba different from a standard fixed SSM.

## What is implemented here

This repo includes:

- input projection into SSM and gate branches,
- causal depthwise convolution,
- generation of input-dependent delta, B, and C parameters,
- discretization of the continuous-time state-space system,
- a selective scan of the hidden state,
- output projection and residual connection,
- validation that the sequential recurrence and scan-based version agree.

## Important technical details

### 1. Selectivity
The model generates different parameters for each token, so the update behavior changes dynamically based on input content. This gives the model contextual control over how much information to keep, forget, or emit.

### 2. Convolution before recurrence
The input is first passed through a causal depthwise convolution. This adds local temporal mixing before the state-space dynamics operate.

### 3. Scan-based computation
The recurrence can be executed in a sequential loop or in a prefix-scan style. The scan version is more efficient on GPU hardware and is the practical path used in large-scale implementations.

### 4. Stability
The implementation uses careful parameter initialization and stability guards such as softplus on the timestep parameter and clamping to avoid exploding or invalid state values.

## Performance note

The notebook includes a comparison between a naive sequential scan and a parallel-style scan. On the tested setup, the scan-based version was measurably faster.

Example result:

```text
Sequential: 0.000288s
Parallel-style scan: 0.000112s
Speedup: ~2.57x
```

This shows the practical value of the Mamba-style recurrence and scan formulation.

## Quick usage

```python
import torch
from torch import nn

# Example: initialize a Mamba block
model = MambaBlock(d_model=256, d_state=16, d_conv=4, expand=2)

x = torch.randn(2, 128, 256)
y = model(x)

print(x.shape)
print(y.shape)
```

## Repository structure

```text
mamba-training-from-scratch/
├── README.md
├── Untitled0.ipynb
└── project notebook containing the full implementation
```

## Summary

This repository is a practical, readable implementation of Mamba that makes the architecture understandable without hiding the math. It is best used as a learning reference for how selective state-space models work, how they are implemented, and why they are efficient for long sequences.

If you want, I can next make the README even more polished for GitHub by adding a cleaner hero section, sharper project positioning, and a more presentation-ready diagram.