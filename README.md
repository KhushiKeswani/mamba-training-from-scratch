# Mamba Training from Scratch

This repository is a compact, from-first-principles implementation of the Mamba state-space model. The goal is to make the architecture understandable at the code level: how the input is split, how the state-space parameters are generated, how the recurrence evolves, and how the block is finally gated and projected back to the original embedding size.

This is not a large production training pipeline. It is a focused implementation of the core Mamba block, built to explain the architecture clearly and correctly.

## What Mamba is doing

The main idea is simple:

- attention computes pairwise interactions and scales poorly with long sequences,
- Mamba instead keeps a hidden state and updates it sequentially,
- the update is controlled by input-dependent parameters,
- this gives a more efficient long-sequence model while preserving expressiveness.

At the block level, the computation is:

```
h_t = A_bar_t * h_{t-1} + B_bar_t * x_t

y_t = C_t * h_t + D * x_t
```

This is a selective state-space recurrence. The key difference from standard SSMs is that the transition parameters depend on the current input, so the model can adapt its behavior dynamically.

## Correct architecture flow

```text
Input: [B, L, d_model]
        |
        v
┌────────────────────────────┐
│ Input projection           │
│ xz = in_proj(x)            │
│ xz: [B, L, 2*d_inner]      │
│ split -> x_ssm, x_gate     │
└────────────────────────────┘
        |
        +---------------------------+
        |                           |
        v                           v
   x_ssm branch                x_gate branch
   [B, L, d_inner]           [B, L, d_inner]
        |                           |
        v                           |
┌────────────────────────────┐     |
│ Causal depthwise conv     │     |
│ Conv1d, groups=d_inner    │     |
│ causal temporal mixing    │     |
│ output: [B, L, d_inner]   │     |
└────────────────────────────┘     |
        |                           |
        v                           |
┌────────────────────────────┐     |
│ SiLU activation            │     |
│ x_conv = SiLU(conv_output) │     |
│ output: [B, L, d_inner]   │     |
└────────────────────────────┘     |
        |                           |
        v                           |
┌────────────────────────────┐     |
│ Parameter generation       │     |
│ x_proj(x_conv)             │     |
│ splits into dt, B, C      │     |
│ dt -> dt_proj -> delta     │     |
└────────────────────────────┘     |
        |                           |
        v                           |
┌────────────────────────────┐     |
│ Discretization             │     |
│ A_bar = exp(delta * A)     │     |
│ B_bar = delta * B          │     |
│ b = B_bar * x_ssm          │     |
└────────────────────────────┘     |
        |                           |
        v                           |
┌────────────────────────────┐     |
│ Selective scan             │     |
│ h_t = A_bar_t * h_{t-1}    │     |
│      + b_t                │     |
│ states: [B, L, d_inner, d_state] │
└────────────────────────────┘     |
        |                           |
        v                           |
┌────────────────────────────┐     |
│ SSM output readout         │     |
│ y = sum(C * states)        │     |
│ y = y + D * x_ssm          │     |
│ output: [B, L, d_inner]   │     |
└────────────────────────────┘     |
        |                           |
        +---------------------------+
                            |
                            v
                    ┌──────────────────────┐
                    │ SiLU gate            │
                    │ gate = SiLU(x_gate)  │
                    │ output: [B, L, d_inner]
                    └──────────────────────┘
                            |
                            v
                    ┌──────────────────────┐
                    │ Combine              │
                    │ y = y * gate         │
                    └──────────────────────┘
                            |
                            v
                    ┌──────────────────────┐
                    │ Output projection    │
                    │ d_inner -> d_model   │
                    │ out_proj(y)          │
                    └──────────────────────┘
                            |
                            v
                    ┌──────────────────────┐
                    │ Residual connection │
                    │ output = x + y       │
                    └──────────────────────┘
                            |
                            v
                     Output: [B, L, d_model]
```

## Stage-by-stage explanation

### 1. Input projection

The first operation is a linear projection:

```python
xz = self.in_proj(x)

x_ssm, x_gate = xz.chunk(2, dim=-1)
```

This splits the hidden representation into two branches:

- `x_ssm`: used for the state-space recurrence
- `x_gate`: used for the multiplicative gate

Shape intuition:

- input: `[B, L, d_model]`
- projection output: `[B, L, 2 * d_inner]`
- split result:
  - `x_ssm`: `[B, L, d_inner]`
  - `x_gate`: `[B, L, d_inner]`

### 2. Causal depthwise convolution

Before the state-space recurrence, the SSM branch is processed with a causal depthwise 1D convolution:

```python
x_ssm = x_ssm.transpose(1, 2)
x_ssm = self.conv1d(x_ssm)
x_ssm = x_ssm[:, :, :x.shape[1]]
x_ssm = x_ssm.transpose(1, 2)
```

This is important because it mixes nearby tokens while preserving causality. The model does not see future tokens.

### 3. SiLU on the SSM path

The SSM branch is activated with SiLU after the causal convolution:

```python
x_ssm = torch.nn.functional.silu(x_ssm)
```

This is the first SiLU in the block. It is applied to the transformed SSM signal before the model generates the state-space parameters.

### 4. Generate selective parameters

The model then projects the activated SSM branch into input-dependent state-space parameters:

```python
x_dbl = self.x_proj(x_ssm)
dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

dt = self.dt_proj(dt)
delta = torch.nn.functional.softplus(dt)
```

This is the main selective mechanism.

- `delta` controls the step size per timestep
- `B` determines how the current token affects the hidden state
- `C` determines how the hidden state is read out

These are not fixed for the whole sequence; they depend on the current token representation.

### 5. Discretization of the state-space model

The continuous-time dynamics are discretized using a diagonal decay matrix `A` and a time-varying input term:

```python
A = -torch.exp(self.A_log)
A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0))
B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2)
b = B_bar * x_ssm.unsqueeze(-1)
```

This gives:

- `A_bar`: discretized decay / transition factor
- `B_bar`: input influence term
- `b`: token-dependent input contribution to the hidden state

### 6. Selective scan / state recurrence

The hidden state is updated sequentially over time:

```python
states = compiled_mamba_scan(A_bar, b, h_init)
```

The underlying recurrence is:

```python
h_t = A_bar_t * h_{t-1} + b_t
```

This is the central Mamba operation. It is a selective recurrent state update.

### 7. Output readout and skip connection

After the state sequence is computed, the model reads out information via `C` and adds a skip connection:

```python
y = torch.sum(C.unsqueeze(-2) * states, dim=-1)
y = y + self.D * x_ssm
```

This is the SSM output branch. `D` acts like a residual path from the current transformed input to the output signal.

### 8. Gate on the second branch

The second branch is also activated with SiLU:

```python
gate = torch.nn.functional.silu(x_gate)
y = y * gate
```

This is the second SiLU in the block and it is used as a gating mechanism. The gate decides how much of the SSM output should pass through.

### 9. Output projection and residual connection

Finally the output is projected back to the model width and combined with the original input:

```python
y = self.out_proj(y)
return y + x
```

This residual connection helps stabilize the block and preserves the original token representation while adding transformed sequence context.

## Where SiLU is used

The notebook uses SiLU in exactly two places:

1. After causal convolution in the SSM branch

```python
x_ssm = torch.nn.functional.silu(x_ssm)
```

2. On the gate branch before the final multiply

```python
gate = torch.nn.functional.silu(x_gate)
y = y * gate
```

So the implementation is consistent with the Mamba block design:

- one SiLU for the signal branch after local temporal mixing,
- one SiLU for the gating branch that decides how much information passes through.

## Core recurrence in one line

The model is best summarized as:

```python
h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
y_t = C_t * h_t + D * x_t
```

This is the basis of the selective state-space layer.

## What this repo includes

This repository contains:

- a toy state-space recurrence in NumPy,
- a basic selective SSM example,
- a prefix-scan formulation for efficient computation,
- a direct comparison between sequential and scan-based state updates,
- a full Mamba block in PyTorch,
- a forward-pass validation showing the model outputs finite values.

## Performance note

The notebook compares sequential state updates against a parallel-style scan. The measured values are:

```text
Sequential: 0.000288s
Parallel-style scan: 0.000112s
```

This demonstrates the value of a scan-based formulation for the same recurrence.

## Summary

The correct architecture flow is:

```
input
-> in_proj
-> split into SSM branch and gate branch
-> causal conv on SSM branch
-> SiLU on SSM branch
-> generate dt, B, C
-> discretize A and B
-> selective scan recurrence
-> output readout + skip connection
-> SiLU gate
-> multiply output by gate
-> out_proj
-> residual add
-> output
```

That is the precise structure implemented in this repository.
