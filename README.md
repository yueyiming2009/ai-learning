# ai-learning

Personal notes on the math, algorithms, and systems behind modern ML — worked
derivations, algorithm walkthroughs, and engineering plumbing for training and
running large models.

**Site:** [yueyiming2009.github.io/ai-learning](https://yueyiming2009.github.io/ai-learning)

## Contents

```
docs/        Rendered HTML pages (published via GitHub Pages)
triton/      Triton GPU kernel notebooks
rl/          Reinforcement-learning systems notes
```

The site index is organized into three top-level domains.

### 01 · Math — [docs/](docs/)
Worked derivations from scratch: gradients, probabilistic principles, and the
algebra behind common losses, layers, and RL objectives.

| Page | Topic |
|---|---|
| [Math notation](https://yueyiming2009.github.io/ai-learning/math-notation.html) | Symbol-by-symbol reference for ML papers |
| [NLL loss](https://yueyiming2009.github.io/ai-learning/nll-loss.html) | The maximum-likelihood principle behind CE, MSE, BCE, DPO |
| [Cross entropy](https://yueyiming2009.github.io/ai-learning/cross-entropy.html) | Softmax + CE forward/backward; the `p − y` gradient |
| [LayerNorm gradient](https://yueyiming2009.github.io/ai-learning/layernorm-gradient.html) | Full backward through mean, variance, normalized input |
| [RMSNorm gradient](https://yueyiming2009.github.io/ai-learning/rmsnorm-gradient.html) | Two-term `dx` derivation; side-by-side with LayerNorm |
| [BatchNorm gradient](https://yueyiming2009.github.io/ai-learning/batchnorm-gradient.html) | Backward across the batch dim; train vs. inference |
| [SwiGLU](https://yueyiming2009.github.io/ai-learning/swiglu.html) | Gated FFN forward/backward; weight grads for all three projections |
| [RoPE gradient](https://yueyiming2009.github.io/ai-learning/rope-gradient.html) | Per-pair rotation forward + symmetric backward |
| [RLHF](https://yueyiming2009.github.io/ai-learning/rlhf.html) | Bradley–Terry RM, KL-regularized RL, policy gradient, GAE, PPO |
| [GRPO](https://yueyiming2009.github.io/ai-learning/grpo.html) | DeepSeek-R1's critic-free PPO with group-relative advantages |
| [DPO](https://yueyiming2009.github.io/ai-learning/dpo-loss.html) | KL-constrained RLHF → closed-form policy → logistic loss |

### 02 · Algorithms — [docs/](docs/)
Procedures and data structures — how core ML algorithms are actually
implemented when correctness and speed both matter.

| Page | Topic |
|---|---|
| [Production BPE](https://yueyiming2009.github.io/ai-learning/bpe-production.html) | Incremental pair counts, reverse index, linked-list + heap encoder |

### 03 · Systems — [triton/](triton/), [rl/](rl/)
Engineering notes — GPU kernels, distributed training topology, and the
plumbing that turns the math into a working training run.

| Page | Topic |
|---|---|
| [Triton — Introduction](triton/intro.ipynb) | Program IDs, block pointers, masking, the launch grid |
| [Fused Softmax kernel](triton/fused_softmax.ipynb) | Row-wise online softmax in one fused Triton kernel |
| [PPO training walkthrough (verl)](rl/ppo_training_walkthrough.md) | One PPO step on LLaMA-7B/8 GPUs: every shape, every NCCL call, TP/DP/PP |

## Publishing

The HTML pages under `docs/` are published via GitHub Pages
(Settings → Pages → `main` / `/docs`). Notebooks and markdown files render
directly on GitHub.
