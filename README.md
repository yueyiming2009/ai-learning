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
canonical, complete list of pages, organized into six top-level domains. The summaries
below are a high-level map, not an exhaustive index.

### 01 · Deep Learning Foundations — [docs/](docs/)
Worked derivations from scratch, each standing alone from the transformer — the notation, the
probabilistic principle behind the losses, the gradients through each component, the optimizer
update rules, and the tokenizer math. Representative pages: math notation, backpropagation
(vector–Jacobian products), NLL & cross-entropy & perplexity, LayerNorm/RMSNorm/BatchNorm
gradients, SwiGLU, RoPE, optimizers, importance sampling, log-probabilities from logits,
production BPE, test-time compute scaling, the Jacobian lens.

### 02 · Hardware — [docs/](docs/)
The physical accelerator — GPU and TPU architecture: SM and tensor-core layout, the memory
hierarchy, and interconnect topology.

### 03 · Transformer & Pretraining — [docs/](docs/)
Assembling the Foundations pieces into a working model and training it. **Architecture:**
self-attention (forward & backward), the pre-norm block, linear attention, Gated DeltaNet, MoE
routing. **Training:** the next-token objective and single-device loop, and the neural scaling
laws that size the run (Kaplan vs. Chinchilla, the compute-optimal frontier).

### 04 · Systems & Scaling — [docs/](docs/), [triton/](triton/)
The engineering substrate that trains it across many devices, built bottom-up — MFU, the PyTorch
execution model (operators, dispatch, eager vs. graph) and Triton kernels, sharding & collectives
(NCCL), tensor/pipeline/sequence/MoE parallelism, the ZeRO/FSDP distributed-training ladder, and
the training-framework landscape (Megatron-LM vs. DeepSpeed vs. FSDP/TorchTitan).

### 05 · Reinforcement Learning — [docs/](docs/), [rl/](rl/)
From first-principles RL theory to LLM post-training and the systems that run it.
**Theory:** MDP → value functions → Monte Carlo → TD → DQN → policy gradient → PPO → MCTS →
AlphaZero → tree search for LLMs. **Alignment:** RLHF, GRPO, DPO, DeepSeek-R1, cascade RL,
distillation. **Agentic:** ReAct, agentic RL over long horizons, multi-turn tool-use training in
verl. **Frameworks:** the framework landscape, verl (HybridFlow) architecture, resource-pool
placement, TRL, the slime–Miles walkthrough. **At scale:** PPO/GRPO training walkthroughs, async
RL and its monitoring, reward pipelining.

### 06 · Inference — [docs/](docs/)
Serving a trained model fast — the prefill/decode compute model, KV cache, and batching;
techniques (FlashAttention, PagedAttention, speculative decoding, quantization, prefill–decode
disaggregation, MoE serving and the expert coverage law); and workloads (offline batch, online SLO serving, RL rollout/reward serving).

## Publishing

The HTML pages under `docs/` are published via GitHub Pages
(Settings → Pages → `main` / `/docs`). Notebooks and markdown files render
directly on GitHub.
