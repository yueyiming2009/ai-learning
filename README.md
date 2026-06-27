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

The site index ([`docs/index.html`](https://yueyiming2009.github.io/ai-learning)) is
organized into four top-level domains and is the canonical, complete list of pages. The
tables below are a high-level map of each domain, not an exhaustive index.

### 01 · Math — [docs/](docs/)
Worked derivations from scratch: gradients, probabilistic principles, and the
algebra behind common losses, layers, and RL objectives.

| Page | Topic |
|---|---|
| [Math notation](https://yueyiming2009.github.io/ai-learning/math-notation.html) | Symbol-by-symbol reference for ML papers |
| [Backpropagation](https://yueyiming2009.github.io/ai-learning/backprop.html) | Chain rule on a graph as reverse-mode vector–Jacobian products; why a scalar loss forces reverse mode; the per-op VJP rules behind every gradient page |
| [NLL loss](https://yueyiming2009.github.io/ai-learning/nll-loss.html) | The maximum-likelihood principle behind CE, MSE, BCE, DPO |
| [Cross entropy](https://yueyiming2009.github.io/ai-learning/cross-entropy.html) | Softmax + CE forward/backward; the `p − y` gradient |
| [LayerNorm gradient](https://yueyiming2009.github.io/ai-learning/layernorm-gradient.html) | Full backward through mean, variance, normalized input |
| [RMSNorm gradient](https://yueyiming2009.github.io/ai-learning/rmsnorm-gradient.html) | Two-term `dx` derivation; side-by-side with LayerNorm |
| [BatchNorm gradient](https://yueyiming2009.github.io/ai-learning/batchnorm-gradient.html) | Backward across the batch dim; train vs. inference |
| [SwiGLU](https://yueyiming2009.github.io/ai-learning/swiglu.html) | Gated FFN forward/backward; weight grads for all three projections |
| [RoPE gradient](https://yueyiming2009.github.io/ai-learning/rope-gradient.html) | Per-pair rotation forward + symmetric backward |
| [Optimizers — working math](https://yueyiming2009.github.io/ai-learning/optimizers.html) | SGD→AdamW update rules: noise, momentum/Nesterov, Adam moments, clipping, schedules, decoupled decay |
| [Optimizer norms](https://yueyiming2009.github.io/ai-learning/optimizer-norms.html) | SGD/sign-SGD/Adam/Muon as steepest descent in different norms |
| [Importance sampling](https://yueyiming2009.github.io/ai-learning/importance-sampling.html) | The off-policy reweighting `E_p[f]=E_q[(p/q)f]` behind the PPO/GRPO ratio; unbiasedness, the variance blow-up, why clipping exists |
| [RLHF](https://yueyiming2009.github.io/ai-learning/rlhf.html) | Bradley–Terry RM, KL-regularized RL, policy gradient, GAE, PPO |
| [GRPO](https://yueyiming2009.github.io/ai-learning/grpo.html) | DeepSeek-R1's critic-free PPO with group-relative advantages |
| [DPO](https://yueyiming2009.github.io/ai-learning/dpo-loss.html) | KL-constrained RLHF → closed-form policy → logistic loss |

### 02 · Algorithms & Methods — [docs/](docs/)
Procedures and training methods — tokenization, architecture mechanisms, the RL
theory stack (MDP → value functions → MC/TD → policy gradient → MCTS/AlphaZero →
tree search for LLMs), and the alignment methods (RLHF, GRPO, DPO, distillation)
that apply them to post-training.

| Page | Topic |
|---|---|
| [Production BPE](https://yueyiming2009.github.io/ai-learning/bpe-production.html) | Incremental pair counts, reverse index, linked-list + heap encoder |

### 03 · Systems — [triton/](triton/), [rl/](rl/)
Engineering notes — GPU kernels, distributed training topology, and the
plumbing that turns the math into a working training run.

| Page | Topic |
|---|---|
| [GPU Hardware for ML](https://yueyiming2009.github.io/ai-learning/gpu-hardware.html) | SIMT execution model, inside an SM (Tensor Cores, scratchpad, TMA), the roofline, NVSwitch crossbar, the network cliff |
| [MFU](https://yueyiming2009.github.io/ai-learning/mfu.html) | Model FLOPs Utilization: the 6N rule + attention term, MFU vs HFU, worked example, calculator |
| [TPU Hardware for ML](https://yueyiming2009.github.io/ai-learning/tpu-hardware.html) | MXU systolic array, VMEM/HBM, the ICI nearest-neighbor torus, GPU-vs-TPU contrast |
| [Sharding & Collectives](https://yueyiming2009.github.io/ai-learning/collectives.html) | Sharding notation, `{Uₓ}` unreduced state, the cost model, the 4 collectives derived |
| [NCCL](https://yueyiming2009.github.io/ai-learning/nccl.html) | The 5-layer stack: topology, ring/tree/NVLS, LL/LL128/Simple protocols, channels, selection |
| [Tensor Parallelism](https://yueyiming2009.github.io/ai-learning/tensor-parallelism.html) | Megatron column/row split → one AllReduce per block; why TP caps at one NVLink node |
| [Data Parallelism](https://yueyiming2009.github.io/ai-learning/torch-distributed.html) | `torch.distributed` foundation → DDP's 16 B/param wall, FSDP's all-gather/free lifecycle (animated), the ZeRO ladder's 1.5×-comm-for-1/N-state trade, DeviceMesh |
| [Triton — Introduction](triton/intro.ipynb) | Program IDs, block pointers, masking, the launch grid |
| [Fused Softmax kernel](triton/fused_softmax.ipynb) | Row-wise online softmax in one fused Triton kernel |
| [PPO training walkthrough (verl)](https://yueyiming2009.github.io/ai-learning/ppo-training-walkthrough.html) | One PPO step on LLaMA-7B/8 GPUs: every shape, every NCCL call, TP/DP/PP |
| [Async RL — policy lag, IS correction, TIM](https://yueyiming2009.github.io/ai-learning/async-rl.html) | Decoupled generation/training, token vs sequence IS, eliminate/correct/precision TIM fix families |

### 04 · Practices — [rl/](rl/), [docs/](docs/)
Running real jobs — end-to-end training walkthroughs of production RL systems,
efficiency and diagnosis techniques, and inference serving workloads.

| Page | Topic |
|---|---|
| [PPO walkthrough (VeRL)](https://yueyiming2009.github.io/ai-learning/ppo-training-walkthrough.html) | One PPO step on LLaMA-7B/8 GPUs: every shape, every NCCL call, TP/DP/PP |
| [GRPO walkthrough (VeRL)](https://yueyiming2009.github.io/ai-learning/grpo-training-walkthrough.html) | Group rollout batching, z-score advantage, k3 KL, what disappears vs PPO |
| [slime / Miles walkthrough](https://yueyiming2009.github.io/ai-learning/slime-miles-walkthrough.html) | One PPO step on a 30B MoE: SGLang rollout server, Megatron EP, TIM elimination |
| [Resource pool placement](https://yueyiming2009.github.io/ai-learning/resource-pool-placement.html) | Actor/rollout/critic GPU allocation; hybrid-engine FSDP↔vLLM transitions |
| [Async RL](https://yueyiming2009.github.io/ai-learning/async-rl.html) | Policy lag, token vs sequence IS correction, eliminate/correct/precision TIM fixes |
| [Reward pipelining](https://yueyiming2009.github.io/ai-learning/reward-pipelining.html) | Hiding slow-reward latency behind a decoupled queue; how reward lag composes with policy lag |

## Publishing

The HTML pages under `docs/` are published via GitHub Pages
(Settings → Pages → `main` / `/docs`). Notebooks and markdown files render
directly on GitHub.
