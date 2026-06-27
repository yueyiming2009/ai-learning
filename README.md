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

The site index ([`docs/index.html`](https://yueyiming2009.github.io/ai-learning)) is the
canonical, complete list of pages, organized into four top-level domains. The summaries
below are a high-level map, not an exhaustive index.

### 01 · Deep Learning Foundations — [docs/](docs/)
Worked derivations from scratch — the notation, the probabilistic principle behind the
losses, the gradients through each architecture block, the optimizer update rules, and the
tokenizer/routing math around the model. Representative pages: math notation, backpropagation
(vector–Jacobian products), NLL & cross-entropy, LayerNorm/RMSNorm/BatchNorm gradients,
SwiGLU, RoPE, optimizers, importance sampling, production BPE, MoE routing.

### 02 · Reinforcement Learning — [docs/](docs/), [rl/](rl/)
From first-principles RL theory to LLM post-training and the systems that run it.
**Theory:** MDP → value functions → Monte Carlo → TD → DQN → policy gradient → MCTS →
AlphaZero → tree search for LLMs. **Alignment:** RLHF, GRPO, DPO, DeepSeek-R1, cascade RL,
agentic RL, distillation. **At scale:** PPO/GRPO/slime–Miles training walkthroughs,
resource-pool placement, async RL and its monitoring, reward pipelining.

### 03 · Systems & Hardware — [docs/](docs/), [triton/](triton/)
The engineering substrate, built bottom-up — GPU/TPU hardware and MFU, Triton kernels,
sharding & collectives (NCCL), tensor/pipeline/sequence/MoE parallelism, and the ZeRO/FSDP
distributed-training ladder.

### 04 · Inference — [docs/](docs/)
Serving a trained model fast — the prefill/decode compute model, KV cache, and batching;
techniques (FlashAttention, PagedAttention, speculative decoding, quantization, prefill–decode
disaggregation); and workloads (offline batch, online SLO serving, RL rollout/reward serving).

## Publishing

The HTML pages under `docs/` are published via GitHub Pages
(Settings → Pages → `main` / `/docs`). Notebooks and markdown files render
directly on GitHub.
