# Mamba Training from Scratch

This repository is a from-first-principles implementation of the Mamba state space model architecture, built as a compact, educational, and technically grounded notebook. It is not a production model package; it is a deep-dive implementation focused on understanding how Mamba works internally, how its selective state-space dynamics are derived, and how the recurrence can be computed efficiently.

The project is designed to answer a practical question: how can we implement a Mamba block with the same structural ideas as the original architecture, without hidden framework magic? The answer lives in the notebook: explicit parameter generation, selective scan logic, discretization of continuous dynamics, causal depthwise convolution, gating, residual flow, and performance comparisons between naive sequential recurrence and a parallel-style scan.

---

## Project purpose

This repository exists to make the architecture transparent.

Mamba is important because it addresses one of the core limitations of Transformers: quadratic attention cost in sequence length. Instead of computing pairwise token interactions through attention scores, Mamba models sequences with a recurrent state-space formulation where state updates are governed by parameterized linear dynamics.

The implementation in this repo demonstrates the following:

- how a selective state-space model is parameterized from input features,
- how a state update is discretized for sequence modeling,
- how hidden states are propagated over time,
- how the recurrence can be expressed as a scan operation,
- how the block is assembled into a Mamba-like layer,
- how the design maps to real throughput gains.

The key goal is not just to reproduce the paper, but to build intuition about the architecture and show that the mechanism is computationally coherent and implementable from scratch.

---

## Why Mamba matters

Language models and long-context systems run into severe cost and scaling constraints as sequence length grows. Attention scales quadratically with the length of the context, which becomes a bottleneck for long documents, codebases, and multimodal sequences.

Mamba shifts the design emphasis from pairwise interaction to state evolution:

- the model keeps a compressed internal state,
- state transitions are controlled by input-dependent parameters,
- only the current token and previous state are needed for each step,
- sequence processing can be organized as structured recurrent updates or a prefix scan,
- the result is a model that is much more efficient for long context processing.

This is the central value proposition: Mamba preserves a high-capacity sequence model while reducing the dependence on quadratic attention patterns.

---

## Repository scope

At the moment, the repository is intentionally small and self-contained.

File layout:

- README.md — project overview and architecture explanation
- Untitled0.ipynb — main implementation notebook containing the actual Mamba logic

The notebook includes the entire architecture flow in executable Python cells rather than a formal package split. This makes the code unusually readable and useful for educational understanding.

---

## Architectural summary

The implemented model follows the Mamba block structure used in the paper, with the following sequence of operations:

1. Input projection
2. Causal depthwise convolution
3. Input-dependent parameter generation
4. Discretization of the continuous state-space system
5. Selective scan over sequence positions
6. Output computation from state values
7. Gating and residual connection
8. Output projection back to model dimension

At a high level:

Input
  -> project to SSM branch + gate branch
  -> causal conv on SSM branch
  -> generate delta, B, C from the transformed signal
  -> discretize A and B
  -> scan the state over time
  -> combine state output with skip path
  -> gate and project back
  -> residual add

This is the functional blueprint of a Mamba block.

---

## Core technical architecture

### 1. Input projection

The model takes an input tensor of shape:

- [batch, sequence_length, d_model]

The input is projected into two branches:

- x_ssm: the branch used for the state-space recurrence
- x_gate: the branch used for gating after the state output is computed

The projection is:

- xz = in_proj(x)
- x_ssm, x_gate = chunk(xz, 2)

This splits the expanded hidden representation into the sequence-modeling pathway and the nonlinear gating pathway.

### 2. Causal depthwise convolution

Before the SSM recurrence, the hidden signal passes through a causal depthwise 1D convolution.

This matters because it gives the sequence dynamics some local temporal mixing before the state-space recurrence is applied. Mamba uses a causal convolution so that each token only sees information from current and previous positions, never future ones.

The implementation does:

- transpose from [B, L, d_inner] to [B, d_inner, L]
- apply depthwise conv1d with kernel size d_conv
- crop back to original sequence length
- transpose back
- apply SiLU activation

This creates a locally filtered, nonlinear representation that serves as the input to the state-space parameter generation.

### 3. Parameter generation

The model generates input-specific parameters for the selective SSM:

- delta (time step scale)
- B (state input influence)
- C (output readout from state)

The implementation uses an x_proj layer to generate a combined feature map:

- dt, B, C = split(x_proj(x_ssm), [dt_rank, d_state, d_state])

Where:

- dt_rank is roughly d_model / 16
- B and C each have dimension d_state
- dt is then projected through dt_proj to produce the actual time-step parameter used in the discretization of the state model

The delta term is critical because it controls how aggressively the hidden state changes at each time step. This is the selective aspect of the architecture: the transition dynamics are not fixed for all tokens; they depend on the current input content.

### 4. Discretization

The underlying SSM is continuous-time, but the implementation needs a discrete-time recurrence for sequence modeling.

The model defines a stable transition matrix A as:

- A = -exp(log(arange(1, d_state + 1)))

This is a structured diagonal initialization that stabilizes the state transitions. The step-size parameter delta modulates the update magnitude.

Then the discrete transition and input terms are computed as:

- A_bar = exp(delta.unsqueeze(-1) * A.unsqueeze(0))
- B_bar = delta.unsqueeze(-1) * B.unsqueeze(-2)
- b = B_bar * x_ssm.unsqueeze(-1)

This makes the system operate in discrete time while retaining expressive state dynamics. In practical terms, each token updates the current hidden state according to a token-dependent transition.

### 5. Selective scan recurrence

The core recurrence is a state update over time:

- h_t = A_bar_t * h_{t-1} + b_t
- y_t = C_t * h_t + D * x_t

This is the central mechanism of Mamba. It makes the model a state-space sequence model rather than an attention-only sequence model.

The notebook implements this recurrence both in a simple sequential form and in a parallel-style scan abstraction. The sequential version is conceptually clear:

```
for t in range(L):
    h = A_bar[t] * h + b[t]
    states.append(h)
```

This is mathematically correct but not the fastest to execute when the sequence is long.

The project also includes a prefix-scan formulation, where the recurrence is transformed into repeated pairwise combine operations:

```
def combine(pair2, pair1):
    a2, b2 = pair2
    a1, b1 = pair1
    a_combined = a2 * a1
    b_combined = a2 * b1 + b2
    return a_combined, b_combined
```

This is the conceptual bridge between the sequential recurrence and the hardware-efficient implementation strategy. Prefix scan is what makes the model practical on GPUs at scale.

### 6. Output readout and gating

The state output is computed by projecting the hidden state with C:

- y = sum(C.unsqueeze(-2) * states, dim=-1)

Then the skip connection is added:

- y = y + D * x_ssm

This is the SSM output path. It is then combined with a gating branch:

- gate = silu(x_gate)
- y = y * gate

Finally the model projects back to the original model width and adds residual input:

- y = out_proj(y)
- output = x + y

The residual connection keeps the block stable and preserves the original token representation while adding transformed sequence information.

---

## Mathematical structure

The model is built around a selective state-space recurrence.

A standard continuous-time SSM has the form:

```
 dx/dt = A x + B u
 y = C x + D u
```

In the Mamba formulation, the system becomes input-conditioned:

```
A_t = A(delta_t)
B_t = B(delta_t, x_t)
C_t = C(x_t)
```

which means each time step generates its own transition and observation parameters based on the current token. This selective dependence is the key reason the model can adapt to context in a more deliberate way than a fixed linear system.

The recurrence used in the notebook is effectively:

```
h_t = A_bar_t h_{t-1} + B_bar_t x_t

y_t = C_t h_t + D x_t
```

This is the discrete state-space recurrence that makes the model behave like a time-aware memory mechanism.

---

## Key implementation details

### State shape

The state-space evolution is computed over a hidden state of shape:

- [d_inner, d_state]

where:

- d_inner is the expanded hidden dimension,
- d_state is the internal state dimension, typically 16.

This gives the model a richer state representation than a single scalar recurrent memory, while still staying much cheaper than dense attention over all tokens.

### Parameter initialization philosophy

The notebook uses careful initialization patterns for stability:

- input projection weights are scaled by sqrt(d_model)
- convolution weights are scaled by sqrt(d_conv)
- x_proj weights are scaled by sqrt(d_inner)
- output projection is scaled conservatively
- dt_proj weights use a bounded uniform distribution
- dt bias is initialized to a log-space value that places delta in a stable region
- A_log uses a logarithmic spacing to initialize a well-structured decaying state transition
- D is initialized to ones for stronger residual flow

These choices matter because the state-space recurrence is sensitive to parameter scale. Without careful initialization, the recurrent dynamics can become numerically unstable or collapse.

### Numerical stability

The notebook includes several protection measures:

- dt is clamped to a bounded range
- A_bar is clamped to a safe upper limit
- softplus is used on dt so the step parameter remains positive
- outputs are checked for finite values

This becomes especially important when implementing SSMs from scratch, where hidden-state blowups and unstable dynamics can quietly appear even if the architecture is conceptually sound.

---

## Performance analysis

The notebook includes a direct comparison between:

- a sequential selective scan,
- a parallel-style prefix scan implementation.

The measured values from the notebook are:

- Sequential: 0.000288s
- Parallel-style scan: 0.000112s

This is a speedup of approximately 2.57x for the tested configuration.

The actual numbers are not meant to be universal benchmarks; they are representative of the execution environment used in the notebook (Tesla T4 GPU, sequence length 128, model dimension 256). But they clearly demonstrate the architectural value: the scan formulation avoids a purely serial dependency chain and better exploits parallel compute hardware.

### Complexity perspective

The selective scan has a structural recurrence, which means the work is linear in sequence length in the sense of the state update chain, but the scan formulation allows better hardware utilization. In practice, the parallel-style scan reduces the effective serial depth and can significantly improve runtime on GPU hardware.

This is one of the most important engineering lessons in the notebook: the recurrence itself may be conceptually simple, but choosing the correct computational formulation changes throughput dramatically.

---

## Why this repository is valuable

This project is not just a toy model tutorial. It is a meaningful implementation of a highly important architecture because it demonstrates:

- the actual state-space recurrence behind Mamba,
- the selective mechanism that makes it different from standard SSMs,
- the interaction between convolution, gating, and recurrence,
- the practical transformation from mathematical recurrence to efficient scan-based execution,
- the ability to validate correctness by comparing sequential and parallel versions.

This creates direct learning value for anyone who wants to understand long-context modeling beyond the level of high-level abstraction.

---

## Notebook-driven architecture flow

The notebook is organized into progressive stages. Each section adds one architectural concept at a time:

1. Basic state-space recurrence in NumPy
2. Selective SSM with discrete-time dynamics
3. Prefix-scan formulation for sequence processing
4. Correctness comparison between sequential and scan-based updates
5. Mamba-like forward pass in NumPy
6. PyTorch implementation of MambaBlock
7. Validation of shape and finite behavior
8. Parameter transfer from NumPy-defined values to the PyTorch block

This step-by-step structure is useful because it teaches architecture as a narrative: start with a toy linear recurrence, then turn it into a selective state-space model, then turn that into an efficient block-level neural network primitive.

---

## What is implemented in the current repo

The current repository implements a Mamba-like block with:

- input projection,
- causal depthwise convolution,
- selective parameter generation,
- state-space discretization,
- selective scan recurrence,
- gating,
- output projection,
- residual connection,
- optional torch.compile optimization layer for the scan primitive.

It does not currently include:

- a full training pipeline,
- tokenizer or dataset preparation,
- a full language-model head,
- multi-block stacking,
- a large pretrained checkpoint,
- distributed training orchestration,
- a monolithic production library.

That is fine for this repo's purpose. The strength is conceptual depth, not scale beyond the notebook.

---

## Suggested next steps

If this repository were expanded, the next meaningful milestones would be:

1. Build a complete Mamba model stack with multiple blocks.
2. Add a language modeling head for next-token prediction.
3. Benchmark on synthetic long-sequence tasks and real text corpora.
4. Compare to Transformer and LSTM baselines under the same setup.
5. Optimize with fused kernels or more structure-aware scan implementations.
6. Package it as a reusable model module with training utilities.

This would transform the current educational prototype into a much more comprehensive architectural project.

---

## Usage

The notebook is designed to be run in a Jupyter environment. It imports PyTorch, NumPy, and standard scientific Python utilities, and validates the architecture directly in code.

Example flow:

```python
model = MambaBlock().to(device)
model.eval()

x = torch.randn(2, 128, 256, device=device)
with torch.no_grad():
    y = model(x)

print(x.shape)
print(y.shape)
print(torch.isfinite(y).all().item())
```

This emits output similar to:

- Input: torch.Size([2, 128, 256])
- Output: torch.Size([2, 128, 256])
- Finite: True

That is exactly the signal you want from a correctly implemented block: dimensional consistency, stable hidden-state evolution, and no NaN/Inf values.

---

## Real technical significance

The most important point is this:

Mamba is not just a different attention mechanism; it is a different computational worldview for sequence modeling.

Instead of building sequences from pairwise interactions, it builds them from evolving latent state. This creates a more scalable sequence architecture with better long-context behavior and a stronger focus on stateful memory.

The notebook captures that idea with explicit code rather than abstract diagrams alone. That is where its technical value lies.

---

## Citation

If this implementation is used for research or educational purposes, the original architecture paper is the appropriate reference:

```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Goel, Karan and Re, Christopher},
  journal={arXiv preprint arXiv:2312.08956},
  year={2023}
}
```

---

## Bottom line

This repository is a compact but technically serious implementation of Mamba from scratch. It does not hide the math behind abstractions, and it does not treat the architecture as a black box. Instead, it exposes the actual mechanisms that make Mamba work:

- selective state updates,
- causal convolution,
- discretized recurrence,
- structured scan execution,
- gated residual output,
- long-context scalability.

That makes it an excellent technical reference for understanding Mamba at a depth beyond surface-level summaries.

Last updated: October 2026
