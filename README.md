# ai-learning

Personal notes on the math, algorithms, and systems behind modern ML — worked
derivations, algorithm walkthroughs, and engineering plumbing for training and
running large models.

**Site:** [yueyiming2009.github.io/ai-learning](https://yueyiming2009.github.io/ai-learning)

## Start here

**Follow the whole process:** [Model Training, End to End](docs/transformer/model-training-lifecycle.html)
maps data, objectives, checkpoints, and evaluation gates across the lifecycle.
Then [MAI-Thinking-1](docs/models/mai-thinking-1.html) follows one reported recipe from
MAI-Base-1 development through independent specialist climbs, self-distillation restarts,
consolidation, and final evaluation.

**Explore a model family:** [Models & Families](https://yueyiming2009.github.io/ai-learning/#models)
groups architecture and training coverage into one canonical model/release-series page, with
explicit lineage labels and links to related models.

**Explore a component:** use the reference domains below. Pre-training is a stage,
distillation is a technique used at several stages, and async RL is an execution strategy.
The guides link to those deep references rather than duplicate their derivations.
The generic lifecycle stays under Transformer & Pretraining; named-model architecture and
development recipes live together in Models.

## Contents

```
docs/                Rendered HTML pages (published via GitHub Pages)
  index.html         Site front page — the canonical, complete list of pages
  foundations/       01 · Deep Learning Foundations
  hardware/          02 · Hardware
  transformer/       03 · Transformer & Pretraining
  systems/           04 · Systems & Scaling
  rl/                05 · Reinforcement Learning
  models/            06 · Models & Families
  inference/         07 · Inference
triton/              Triton GPU kernel notebooks
rl/                  Reinforcement-learning systems notes (markdown)
```

The `docs/` subfolders mirror the seven domains of the site index one-for-one; a canonical
page's folder *is* its domain. Old model URLs remain as compatibility redirects, not duplicate pages.

The site index ([`docs/index.html`](https://yueyiming2009.github.io/ai-learning)) is the
canonical, complete list of pages, organized into seven top-level domains. The summaries
below are a high-level map, not an exhaustive index.

Use **Search pages** on the site index to filter cards by title, description, tag, or
category. Searches are case-insensitive and match all entered words. Clear the search
(or press Escape) to restore the full index and its previous expanded domains.

### 01 · Deep Learning Foundations — [docs/foundations/](docs/foundations/)
Worked derivations from scratch, each standing alone from the transformer — the notation, the
probabilistic principle behind the losses, the gradients through each component, the optimizer
update rules, and the tokenizer math. Representative pages: math notation, backpropagation
(vector–Jacobian products), NLL & cross-entropy & perplexity, LayerNorm/RMSNorm/BatchNorm
gradients, SwiGLU, RoPE, optimizers, importance sampling, log-probabilities from logits,
production BPE, test-time compute scaling, the Jacobian lens.
[Diffusion models](docs/foundations/diffusion-models.html) derive Gaussian corruption,
the noise-prediction objective, and reverse sampling, with an interactive density plot.

### 02 · Hardware — [docs/hardware/](docs/hardware/)
The physical accelerator, in four pages: inside the GPU chip (SIMT, SMs and Tensor Cores, the
memory hierarchy, TMA tensor maps/barriers/multicast, the roofline), the GPU interconnect (NVSwitch crossbar, rail-optimized
fat-tree, the bandwidth cliff), GPU generations (Blackwell, the H100→B300 lineup,
B300 vs. GB300), and the TPU's systolic array and torus.

### 03 · Transformer & Pretraining — [docs/transformer/](docs/transformer/)
Assembling the Foundations pieces into a working model and training it. **Core Architecture:**
self-attention (forward & backward), the pre-norm block, recurrent depth / loop transformers
plus their open-model and serving landscape, MoE routing, and the 2026 residual-stream redesigns
(hyper-connections / attention residuals).
[Latent MoE](docs/transformer/latent-moe.html) owns the shared compressed-expert mechanism:
tensor/gradient flow, parameter/FLOP/logical-traffic budgets, and variant links to Kimi,
Nemotron, MAI-Base-1, and Muse Spark Open.
**Sequence Mixing & Memory:** linear attention, Mamba & SSMs
(discretization → selection → the Mamba-2 duality), Gated DeltaNet, KDA,
sparse attention (the lightning indexer), hybrid attention stacks, and conditional memory.
**Training:** the next-token objective and single-device loop, multi-token prediction
(parallel heads vs. DeepSeek-V3's sequential MTP modules, and the free draft head that falls out),
and the neural scaling laws that size the run (Kaplan vs. Chinchilla, the compute-optimal frontier).
[Diffusion language models](docs/transformer/diffusion-language-models.html) add a different
generative objective: masked/uniform-token corruption, same-position denoising, weighted
cross-entropy, and blockwise sampling, with an interactive generation-order comparison.
The [training lifecycle](docs/transformer/model-training-lifecycle.html) connects these
mechanisms across stages; named-model case studies are grouped in Models & Families below.

### 04 · Systems & Scaling — [docs/systems/](docs/systems/), [triton/](triton/)
The engineering substrate that trains it across many devices, built bottom-up — MFU, the PyTorch
execution model (operators, dispatch, eager vs. graph), CUDA streams and events (ordering,
cross-stream dependencies, copy/compute overlap, default-stream semantics, and PyTorch stream
lifetimes), CUDA Graphs (capture, instantiation, static-address replay, PyTorch APIs, and shape
bucketing), the kernel-programming ladder —
the CUDA SIMT model (threads/warps/coalescing/occupancy plus barriers, fences, and mbarrier),
cuBLAS (BLAS levels, cublasLt,
batched GEMM, determinism), Triton kernels, CUTLASS (the GEMM
tiling hierarchy, collectives, stream-K, epilogue fusion), CuTe DSL (CUTLASS 4's
layout algebra and thread-level Python kernels), cuTile (CUDA 13.1's tile-level model and the
Tile IR virtual ISA), the Blackwell kernel model (tcgen05, TMEM, CTA pairs, block-scaled
NVFP4/MX MMA), and the chunkwise
linear-attention kernels (FLA) — sharding & collectives
(NCCL), tensor/pipeline/sequence/MoE parallelism, the ZeRO/FSDP distributed-training ladder, and
the training-framework landscape (the cross-framework comparison and crossover) with per-framework
deep dives into Megatron-LM (the vertically integrated NVIDIA stack: fusion, overlap, FP8, EP),
DeepSpeed (the ZeRO ladder, Offload/Infinity, ZeRO++), TorchTitan (the DTensor/DeviceMesh
substrate, the per-block wrapping order, async TP, float8, reshardable checkpoints), and
[YOLO](docs/systems/yolo.html) (MAI's in-house stack: descriptive sharding with explicit
communication, custom kernels, deterministic training, MoE memory bounds, and goodput-driven
checkpoint/recovery). The comparison distinguishes public adoption choices from YOLO's
report-only design evidence; no controlled head-to-head YOLO benchmark is disclosed.
[TVM & FFI](docs/systems/tvm-ffi.html) separates model/kernel compilation from the
cross-language runtime boundary, with Python/C++ CPU lessons, DLPack tensor ownership,
GPU stream contracts, and an interactive call-overhead model.

### 05 · Reinforcement Learning — [docs/rl/](docs/rl/), [rl/](rl/)
From first-principles RL theory to LLM post-training and the systems that run it.
**Theory:** MDP → imitation learning (behavior cloning, expressive policies, DAgger/HG-DAgger) →
value functions → Monte Carlo → TD → DQN → policy gradient → PPO → MCTS →
AlphaZero → tree search for LLMs. Imitation separates expert supervision from reward optimization
and connects learner-state coverage to LLM distillation.
[PPO/GRPO backward derivation](docs/rl/ppo-grpo-backward.html) follows the clipped loss through
the ratio, KL term, log-probabilities, logits, and model weights, with worked examples.
**Alignment:** RLHF, GRPO, DPO, cascade RL,
distillation, and generative recommendation (retrieval vs ranking, slate rewards, and a dated
research/production landscape), paired with a worked ranking-RL training loop, interactive
policy updates, a comparison with general/LLM RL, and a separate-project GenRank–verl integration
guide (ownership, batch/probability contracts, and packaging). **Agentic:** ReAct, the agent-architecture landscape (Deep Agents, graphs,
delegation, handoffs, reflection, and search), agentic RL over long horizons, multi-turn tool-use
training in verl, harness/trainer decoupling, and gated learning from real-world production experience.
Start an experiment with [Practical Agentic RL Training](docs/rl/practical-agentic-rl.html):
a source-checked MiMo code-task path through pinned inputs, environment/verifier controls,
baseline measurement, trajectory audits, small updates, and held-out checkpoint evaluation.
The guide is not yet execution-validated; the existing concept, VeRL-internals, harness-contract,
and GRPO pages remain the deep references rather than being merged into the recipe.
**Environments:** the training-environment stack — anatomy + ecosystem landscape,
OpenEnv, verifiers &amp; the Environments Hub, verifier/reward design, sandboxing at scale,
task generation &amp; curriculum. **Frameworks:** the framework landscape (self-operated runtimes
versus Tinker's managed LoRA training API, separating algorithm ownership from runtime control),
verl (HybridFlow) architecture, resource-pool
placement, TRL, the slime–Miles walkthrough. **Training:** start with
[RL Training Basics & Metrics](docs/rl/rl-training-metrics.html) — one prompt and three answers
through a GRPO update, introducing dashboard fields as they arise. Equations, PPO/critic
details, and the source-pinned metric glossary are expandable references.
Continue to PPO/GRPO training walkthroughs, then async RL and
its monitoring, and reward pipelining.

### 06 · Models & Families — [docs/models/](docs/models/)

Architecture and training walkthroughs share one canonical page per model or closely related
release series. Each page has a family/lineage map, links to its architecture and training
coverage, and explicit evidence limits. Generic algorithms and framework walkthroughs remain
in their own domains.

| Family | Model-centered pages and relationships |
|---|---|
| DeepSeek | [R1](docs/models/deepseek-r1.html) joins V3-Base's architecture with the R1-Zero/R1 branches and dense-student distillation; [V4](docs/models/deepseek-v4.html) is a later architecture generation, not an asserted R1 fine-tune. |
| Microsoft MAI | [Thinking-1](docs/models/mai-thinking-1.html) covers Base-1 through specialist consolidation; [Base-2](docs/models/mai-base-2.html) is a later pre-training/co-design case, not Thinking-1's base. |
| Mistral | [Mistral Large 4 / Le Chonk](docs/models/mistral-large-4.html) connects a source-pinned block diagram, indexed MLA and KV/index reuse, full-width experts, analytical cache budgets, and the preview → additional RL → release story. |
| Meta Muse | [Muse Spark Open](docs/models/muse-spark-open.html) combines media-bank fusion, local/global gated GQA, residual gates, compressed experts, memory budgets, and a pinned public SFT customization path; it keeps original-training unknowns separate. |
| Qwen | [Qwen3.5](docs/models/qwen-3-5.html) combines its design-testbed progression, dense/MoE variants, and SFT engineering walkthrough. |
| Kimi | [K3](docs/models/kimi-k3.html) connects its hybrid architecture to multimodal pre-training, nine domain/effort teachers, token-reward MOPD, and persistent long-horizon RL state. |
| GLM | [GLM-5 / 5.2](docs/models/glm-5.html) shares a release-series page with architectural changes and training/serving notes. |
| MiniMax | [M2.5](docs/models/minimax-m2-5.html) connects the full-attention decision to its agentic-training emphasis. |
| Nemotron | [3 / 3.5](docs/models/nemotron-3.html) covers size variants and the Nano/Lightning architecture-versus-training comparison. |
| Gemma | [DiffusionGemma](docs/models/diffusion-gemma.html) is an adaptation of a Gemma 4 backbone, not an unchanged-model inference switch. |
| OpenAI GPT | [GPT-6 Astra](docs/models/gpt-6-astra.html) distinguishes public behavior/evaluations from undisclosed architecture and training. |
| TypeSafe Jev | [Jev](docs/models/jev.html) connects its decision interface, calibration, serving, and unconfirmed implementation hypotheses. |

Checkpoint inheritance, teacher-data transfer, design reuse, and release succession are
different edges. Family grouping never establishes an otherwise undisclosed parent checkpoint.
Legacy model URLs and their section fragments remain supported through redirects.

### 07 · Inference — [docs/inference/](docs/inference/)
Serving a trained model fast — the prefill/decode compute model, KV cache, and batching;
techniques (FlashAttention, PagedAttention, speculative decoding, quantization, prefill–decode
disaggregation, MoE serving and the expert coverage law); and workloads (offline batch, online SLO serving, RL rollout/reward serving).

## Publishing

The HTML pages under `docs/` are published via GitHub Pages
(Settings → Pages → `main` / `/docs`). Notebooks and markdown files render
directly on GitHub.
