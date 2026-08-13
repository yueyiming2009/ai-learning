# ai-learning

Personal notes on the math, algorithms, and systems behind modern ML — worked
derivations, algorithm walkthroughs, and engineering plumbing for training and
running large models.

**Site:** [yueyiming2009.github.io/ai-learning](https://yueyiming2009.github.io/ai-learning)

## Contents

```
docs/                Rendered HTML pages (published via GitHub Pages)
  index.html         Site front page — the canonical, complete list of pages
  foundations/       01 · Deep Learning Foundations
  hardware/          02 · Hardware
  transformer/       03 · Transformer & Pretraining
  systems/           04 · Systems & Scaling
  rl/                05 · Reinforcement Learning
  inference/         06 · Inference
triton/              Triton GPU kernel notebooks
rl/                  Reinforcement-learning systems notes (markdown)
```

The `docs/` subfolders mirror the six domains of the site index one-for-one; a page's
folder *is* its domain.

The site index ([`docs/index.html`](https://yueyiming2009.github.io/ai-learning)) is the
canonical, complete list of pages, organized into six top-level domains. The summaries
below are a high-level map, not an exhaustive index.

### 01 · Deep Learning Foundations — [docs/foundations/](docs/foundations/)
Worked derivations from scratch, each standing alone from the transformer — the notation, the
probabilistic principle behind the losses, the gradients through each component, the optimizer
update rules, and the tokenizer math. Representative pages: math notation, backpropagation
(vector–Jacobian products), NLL & cross-entropy & perplexity, LayerNorm/RMSNorm/BatchNorm
gradients, SwiGLU, RoPE, optimizers, importance sampling, log-probabilities from logits,
production BPE, test-time compute scaling, the Jacobian lens.

### 02 · Hardware — [docs/hardware/](docs/hardware/)
The physical accelerator — GPU and TPU architecture: SM and tensor-core layout, the memory
hierarchy, and interconnect topology.

### 03 · Transformer & Pretraining — [docs/transformer/](docs/transformer/)
Assembling the Foundations pieces into a working model and training it. **Architecture:**
self-attention (forward & backward), the pre-norm block, linear attention, Mamba & SSMs
(discretization → selection → the Mamba-2 duality), Gated DeltaNet, KDA,
sparse attention (the lightning indexer), hybrid attention stacks, MoE routing, and the 2026
residual-stream redesigns (hyper-connections / attention residuals). **Training:** the next-token objective and single-device loop, multi-token prediction
(parallel heads vs. DeepSeek-V3's sequential MTP modules, and the free draft head that falls out),
and the neural scaling laws that size the run (Kaplan vs. Chinchilla, the compute-optimal frontier).
**Model Architectures:** case studies reading complete 2026 frontier designs through those
components — DeepSeek-V4 (sequence-compressed sparse attention, mHC residual lanes), Qwen3.5
(the 3:1 Gated DeltaNet hybrid scaled to 397B, native early-fusion vision), Kimi K3
(KDA + NoPE gated MLA, Block AttnRes, latent-space MoE, MXFP4 QAT), GLM-5 (MLA-256, IndexShare
indexer sharing, Muon Split, shared-weight MTP), MiniMax-M2.5 (the full-attention counterpoint
and its published ablation), and Nemotron 3 & 3.5 (Mamba-2 hybrid, LatentMoE, NVFP4 pretraining,
and Lightning — the 30B whose generation bump is pure training).

### 04 · Systems & Scaling — [docs/systems/](docs/systems/), [triton/](triton/)
The engineering substrate that trains it across many devices, built bottom-up — MFU, the PyTorch
execution model (operators, dispatch, eager vs. graph), Triton kernels and the chunkwise
linear-attention kernels (FLA), sharding & collectives
(NCCL), tensor/pipeline/sequence/MoE parallelism, the ZeRO/FSDP distributed-training ladder, and
the training-framework landscape (Megatron-LM vs. DeepSpeed vs. FSDP/TorchTitan).

### 05 · Reinforcement Learning — [docs/rl/](docs/rl/), [rl/](rl/)
From first-principles RL theory to LLM post-training and the systems that run it.
**Theory:** MDP → value functions → Monte Carlo → TD → DQN → policy gradient → PPO → MCTS →
AlphaZero → tree search for LLMs. **Alignment:** RLHF, GRPO, DPO, DeepSeek-R1, cascade RL,
distillation. **Agentic:** ReAct, agentic RL over long horizons, multi-turn tool-use training in
verl. **Environments:** the training-environment stack — anatomy + ecosystem landscape,
OpenEnv, verifiers &amp; the Environments Hub, verifier/reward design, sandboxing at scale,
task generation &amp; curriculum. **Frameworks:** the framework landscape, verl (HybridFlow) architecture, resource-pool
placement, TRL, the slime–Miles walkthrough. **At scale:** PPO/GRPO training walkthroughs, async
RL and its monitoring, reward pipelining.

### 06 · Inference — [docs/inference/](docs/inference/)
Serving a trained model fast — the prefill/decode compute model, KV cache, and batching;
techniques (FlashAttention, PagedAttention, speculative decoding, quantization, prefill–decode
disaggregation, MoE serving and the expert coverage law); and workloads (offline batch, online SLO serving, RL rollout/reward serving).

## Publishing

The HTML pages under `docs/` are published via GitHub Pages
(Settings → Pages → `main` / `/docs`). Notebooks and markdown files render
directly on GitHub.
