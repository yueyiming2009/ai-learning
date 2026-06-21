# PPO Training Walkthrough in slime/Miles: SGLang, Megatron, and MoE Parallelism

This document traces one PPO step through the slime/Miles stack: SGLang as the rollout server,
Megatron-LM as the training engine, and expert parallelism for MoE models. It is designed to
be read alongside the [verl walkthrough](ppo_training_walkthrough.md) — sections that are
identical in both stacks are noted and not repeated in full.

---

## 1. Concrete Configuration

### 1.1 Model — Qwen3-MoE style (dense attention, sparse FFN)

| Symbol | Name | Value |
|--------|------|-------|
| H | hidden size | 2048 |
| A | attention heads | 16 |
| d | head dim (H/A) | 128 |
| L | transformer layers | 24 |
| E | total experts per MoE layer | 64 |
| K | top-k experts active per token | 2 |
| F | FFN intermediate dim per expert | 1024 |
| V | vocabulary size | 32000 |

MoE replaces the dense FFN in each transformer layer with E=64 independent expert FFNs plus a
learned router. Only K=2 experts are activated per token; the other 62 are skipped entirely.

### 1.2 Hardware and Parallelism

```
Total GPUs: 8  (2 nodes × 4 GPUs, NVLink intra-node, InfiniBand cross-node)

TP = 2   tensor parallel (attention weights split across heads)
EP = 2   expert parallel (experts sharded across GPUs)
DP = 2   data parallel (ZeRO-3 / Megatron distributed optimizer)
PP = 1   pipeline parallel (disabled)

TP × EP × DP × PP = 2 × 2 × 2 × 1 = 8  ✓
```

GPU layout (DP × EP × TP):

```
Node 0:
  GPU 0  →  DP=0, EP=0, TP=0
  GPU 1  →  DP=0, EP=0, TP=1
  GPU 2  →  DP=0, EP=1, TP=0
  GPU 3  →  DP=0, EP=1, TP=1
Node 1:
  GPU 4  →  DP=1, EP=0, TP=0
  GPU 5  →  DP=1, EP=0, TP=1
  GPU 6  →  DP=1, EP=1, TP=0
  GPU 7  →  DP=1, EP=1, TP=1
```

Each EP rank (a TP-parallel pair within one DP rank) holds E/EP = 32 experts.

### 1.3 Architecture: Server Boundary vs. HybridEngine

This is the defining structural difference from verl.

**verl — HybridEngine (co-located monolith)**:
Training and rollout workers share the same GPU memory space and the same NCCL process group.
Weight updates flow in-process via NCCL AllGather — no network hop, no process boundary.

**slime/Miles — server boundary**:
SGLang runs as a standalone HTTP inference server. The training cluster (Megatron) and the rollout
server (SGLang) are separate OS processes, possibly on separate GPU pools. Weight updates are
serialized and pushed over HTTP.

```
verl:
  [FSDP training workers] ──NCCL AllGather──> [vLLM rollout workers]
   same process group, shared GPU memory

slime/Miles:
  [Megatron training workers] ──HTTP POST /update_weights──> [SGLang server]
   separate processes; may share or split GPUs across nodes
```

The server boundary is the source of both slime/Miles's flexibility (rollout and training can scale
independently) and its engineering complexity (weight serialization, format conversion, HTTP latency).

### 1.4 Batch Sizes

| Symbol | Name | Value |
|--------|------|-------|
| B | prompts per step | 16 |
| n | rollouts per prompt | 4 |
| B_seq | total sequences = B×n | 64 |
| P | prompt length | 512 |
| R | response length | 512 |
| S | full sequence length = P+R | 1024 |
| M | PPO mini-batch size (global) | 16 |
| E_PPO | PPO epochs | 1 |

Per-DP-rank sizes:

| Quantity | Formula | Value |
|----------|---------|-------|
| sequences per DP rank (rollout) | B_seq / DP | 32 |
| mini-batch per DP rank (training) | M / DP | 8 |

---

## 2. Parallelism Strategies

### 2.1 Tensor Parallelism (TP=2)

Identical to verl. Column-parallel Q/K/V, row-parallel O and W_down, one AllReduce per sub-layer.
See the verl walkthrough §2.1 for the full derivation with explicit tensor shapes.

Full weight table with TP=2 (H=2048, attention only — expert weights handled by EP below):

| Weight | Shape (full) | Shape per TP rank | Parallel type |
|--------|-------------|-------------------|---------------|
| W_Q | (2048, 2048) | (2048, 1024) | column |
| W_K | (2048, 2048) | (2048, 1024) | column |
| W_V | (2048, 2048) | (2048, 1024) | column |
| W_O | (2048, 2048) | (1024, 2048) | row → AllReduce |

### 2.2 Expert Parallelism (EP=2) — new vs. verl

In MoE models the dense FFN is replaced by E=64 parallel expert FFNs plus a learned router.
With EP=2, each EP rank (a TP-parallel pair of GPUs) holds E/EP = 32 experts.

**Expert weight memory (per EP rank)**:
```
32 experts × 2 matrices (W_up, W_down) × (2048 × 1024) × 2 bytes
= 32 × 2 × 2048 × 1024 × 2  ≈  256 MB per EP rank  (vs 512 MB without EP)
```

**Token routing and the two AlltoAll operations**:

For a training mini-batch (8 sequences × 1024 tokens = 8192 tokens per DP rank):

```
# Step 1 — Router (local, no communication)
scores  = x @ W_router           →  (8192, E=64)   bfloat16
top_idx, top_weight = topk(k=2)  →  (8192, 2)  each

# Step 2 — AlltoAll dispatch: send tokens to the GPU holding their assigned experts
# EP rank 0 holds experts 0–31; EP rank 1 holds experts 32–63
dispatch_in  : (8192×K=16384, H=2048)  bfloat16   tokens reordered by expert assignment
NCCL — AlltoAll across EP group (intra-node, NVLink):
  each GPU sends tokens targeting experts on the other EP rank
  each GPU receives tokens targeting its own 32 experts
  volume per GPU (cross-EP): 8192 × H × 2 bytes ≈ 32 MB

# Step 3 — Local expert FFN (no communication)
# Each of the 32 local experts processes the tokens it received
expert_out : (tokens_received, H=2048)   result from up/down projections

# Step 4 — AlltoAll combine: send expert outputs back to token origin GPUs
NCCL — AlltoAll across EP group:
  volume per GPU (cross-EP): ≈ 32 MB

# Step 5 — Weighted sum at token positions (local)
out : (8192, H=2048)   bfloat16   K=2 expert outputs combined by router weights
```

Compared to verl's dense FFN (one AllReduce of ~16 MB for the same H), the MoE layer pays
**two AlltoAll calls of ~32 MB each**. AlltoAll is more communication-efficient than AllReduce
per byte because it avoids the full ring-reduction; but its volume scales with the number of
dispatched tokens × hidden dim, independent of model width.

**Expert load imbalance**: the router may send many tokens to a few popular experts and few to
the rest. Unlike dense attention (all tokens touch all weights), unbalanced MoE routing causes
some GPUs to do far more compute than others, stalling the pipeline. Miles applies an auxiliary
load-balancing loss to keep per-expert token counts close to uniform.

### 2.3 Data Parallelism (DP=2, Megatron ZeRO)

Functionally identical to verl's FSDP (ZeRO-3): each DP rank holds 1/DP of each parameter.
AllGather before compute; ReduceScatter after backward. Implementation is Megatron's distributed
optimizer rather than PyTorch FSDP, but the NCCL primitives are the same.

Memory per GPU (attention weights only, TP+DP sharded):
```
Attention params per TP rank: 4 × (2048 × 1024) × 2 bytes = 16 MB
DP shard: 16 MB / DP=2 ≈ 8 MB per GPU
```

Expert params per EP rank: ~256 MB (not DP-sharded in simple EP; sharded in EP+DP variants).

### 2.4 SGLang as Rollout Server

SGLang replaces vLLM and runs as an HTTP inference server rather than an in-process library.

```
Training controller
    │  HTTP POST /generate
    │  { "prompts": [...], "n": 4, "max_new_tokens": 512, "return_logprobs": true }
    ▼
SGLang server  (continuous batching engine, PagedAttention KV cache)
    │  — prefill: parallel forward pass over prompt tokens
    │  — decode:  autoregressive, K=2 experts activated per step
    │  — stores:  log_prob_behavior for each generated token
    ▼
    │  HTTP response
    │  { "sequences": [...], "log_probs": [...] }
    ▼
Training controller
```

Multiple generation requests are batched continuously — new prompts enter as GPU slots free,
without waiting for prior sequences to finish. This is standard continuous batching (PagedAttention).

**TIM elimination** (Training-Inference Mismatch): slime/Miles's core claim is that SGLang and
Megatron compute numerically identical log probs for the same token sequence under the same weights.
The required conditions:
- Same floating-point precision (both bfloat16, or both FP8 with identical quantization schedules)
- Same TP degree
- Same attention kernel (FlashAttention implementation must match)
- Same expert routing (same router weights, same top-k tie-breaking rule)

When these hold, `log_prob_behavior ≡ log_prob_training`, so the IS ratio ρ = 1 by construction.
No IS correction is needed. See the async-rl page for the contrast with IS-correction approaches.

---

## 3. Phase-by-Phase Walkthrough

---

### Phase 1 — Data Loading

**Where**: Training controller (CPU)

The controller prepares the global prompt batch:
```
prompts_global : (B=16, P=512)  int64   on controller CPU
```

Unlike verl (Ray object-store scatter to co-located workers), prompts are HTTP-posted to the SGLang
server. The controller may pipeline requests — sending the next batch before the current one
finishes — to keep SGLang's continuous batcher saturated.

---

### Phase 2 — Rollout (SGLang Server)

**Where**: SGLang server

#### 2a. Prefill

For each of L=24 layers, the attention sub-layer is identical to verl Phase 2a (TP AllReduce
after W_O). The FFN sub-layer changes:

```
Attention AllReduce (SUM) across TP group per layer:
  tensor: (B_seq/DP=32, P=512, H=2048)  bfloat16
  volume: 32 × 512 × 2048 × 2 = 64 MB  (×2 per layer for attn + dummy FFN AllReduce if dense head)
```

MoE FFN per layer:
```
# Router
scores = x @ W_router  →  (32, 512, E=64)

# AlltoAll dispatch (EP group, intra-node NVLink)
NCCL — AlltoAll:
  input:  (32×512×K=32768, H=2048)  bfloat16
  cross-EP volume per GPU: ≈ 64 MB

# Local expert FFN (32 experts on this EP rank)
# AlltoAll combine
NCCL — AlltoAll:
  cross-EP volume per GPU: ≈ 64 MB
```

**Per-layer total (prefill)**: 2 TP AllReduce × 64 MB + 2 EP AlltoAll × 64 MB = **256 MB**.
**Prefill total**: 24 layers × 256 MB = **6 GB**.

#### 2b. Decode (R=512 steps)

Each step processes one new token per sequence `(32, 1, H=2048)`:

```
Attention AllReduce: (32, 1, 2048) bf16 = 128 KB each
MoE AlltoAll dispatch: (32×K=64, 2048) bf16 ≈ 256 KB each
MoE AlltoAll combine:  ≈ 256 KB
```

**Per-decode-step total**: 24 layers × (2 × 128 KB + 2 × 256 KB) = **18 MB per step**.
**Decode total for R=512**: 512 × 18 MB ≈ **9 GB**.

#### 2c. Log-Prob Collection

SGLang records `log_prob_behavior` for each sampled token during decode. These are the log probs
of the actual generated tokens under the current router+expert forward pass.

```
log_probs_behavior : (B_seq=64, R=512)  float32   stored on SGLang server
```

The AllGather across TP for the logit vocab slice is identical to verl Phase 2c.

#### 2d. Output

SGLang returns via HTTP response (JSON or binary protocol):
```
sequences        : (64, S=1024)  int64
log_probs        : (64, R=512)   float32
```

Controller receives and deserializes. No Ray object-store overhead.

---

### Phase 3 — Reward Computation

Identical to verl Phase 3.
```
token_level_rewards : (64, 512)  float32
```

---

### Phase 4 — Old Log-Probs

**Where**: SGLang server (already computed during rollout)

**Key difference from verl**: verl recomputes old log probs with a separate non-autoregressive
forward pass for numerical stability. In slime/Miles, `log_probs_behavior` was already recorded
during Phase 2c and is returned in the HTTP response. When TIM elimination holds, no extra forward
pass is needed.

```
old_log_probs = log_probs_behavior   # (64, 512)  float32   zero extra compute
```

If TIM cannot be guaranteed (e.g., TP degree changes between rollout and training), a Megatron
recompute pass is run with matched precision and kernel settings. In practice, the TIM elimination
design means this fallback is rarely needed.

Communication: none (data already in controller memory from Phase 2 HTTP response).

---

### Phase 5 — Reference Policy Log-Probs

**Where**: Megatron reference policy workers (frozen weights)

Full non-autoregressive Megatron forward pass over all 64 sequences. MoE layers add 2 AlltoAll
per layer in addition to the TP AllReduce seen in verl.

Mini-batch per DP rank: 32 sequences.

```
Per layer:
  TP AllReduce (attention): (32, 1024, 2048) bf16 × 2 = 2 × 128 MB = 256 MB
  EP AlltoAll (FFN):        ≈ 2 × 64 MB = 128 MB
  Megatron ZeRO AllGather:  ~28 MB (attention weights only)
```

Output:
```
ref_log_probs : (64, 512)  float32
```

---

### Phase 6 — Value Estimation (Critic)

**Where**: Megatron critic workers

Same structure as Phase 5. Value head projects the last hidden state to a scalar per token:
```
values : (64, 1024)  float32
```

---

### Phase 7 — Advantage Estimation (GAE)

**Where**: Controller CPU

Identical to verl Phase 7. KL penalty, GAE backward pass, whitening. No communication.

```
advantages : (64, 512)  float32
returns    : (64, 512)  float32
```

---

### Phase 8 — Actor Update (PPO)

**Where**: Megatron actor workers (all 8 GPUs)

Mini-batch per DP rank: 8 sequences.

#### 8a. Megatron ZeRO AllGather

Before each layer, Megatron AllGathers the full TP-sharded parameters from their ZeRO shards.
Identical in primitive to verl's FSDP AllGather:

```
NCCL — AllGather across DP group (2 GPUs, cross-node InfiniBand):
  per layer: ~28 MB (attention; expert weights handled separately per EP rank)
```

#### 8b. Forward Pass

For each of L=24 layers:

**Attention** (TP AllReduce, same as verl):
```
NCCL — AllReduce (SUM) per layer ×2 (TP group, intra-node):
  tensor: (8, 1024, 2048)  bfloat16
  volume: 8 × 1024 × 2048 × 2 = 32 MB each
```

**MoE FFN** (EP AlltoAll, new vs. verl):
```
# Router (local)
scores = x @ W_router  →  (8, 1024, 64)

# AlltoAll dispatch (intra-node NVLink, EP group)
NCCL — AlltoAll:
  input:  (8×1024×K=16384, H=2048)  bfloat16
  cross-EP volume per GPU: ≈ 32 MB

# 32 local experts process their assigned tokens
# AlltoAll combine
NCCL — AlltoAll:
  cross-EP volume per GPU: ≈ 32 MB
```

#### 8c. PPO Loss

Because TIM is eliminated, `old_log_probs ≡ log_probs_behavior` at step 0. The surrogate reduces
to standard PPO — no IS weighting overhead:

```
log_ratio = new_log_probs - old_log_probs    # (8, 512)  float32
ratio     = exp(log_ratio)                   # ≈ 1.0 at start of each PPO epoch
actor_loss = -mean(min(ratio * adv, clamp(ratio, 1-eps, 1+eps) * adv) * mask)
```

#### 8d. Backward Pass

Mirror of forward: TP AllReduce for attention gradient, EP AlltoAll for expert gradient,
Megatron ZeRO ReduceScatter to re-shard gradients across DP ranks.

```
NCCL — ReduceScatter across DP group (cross-node InfiniBand) per layer:
  volume: ~28 MB (attention weights; expert gradients handled within EP group)

NCCL — AlltoAll ×2 across EP group (intra-node) per layer:
  volume: ≈ 32 MB each (expert input gradient dispatch and combine)
```

#### 8e. Optimizer Step

Megatron distributed optimizer: local AdamW on each GPU's ZeRO shard. No communication.

---

### Phase 9 — Critic Update

Same structure as Phase 8, using value regression loss:
```
value_clipped = clip(value_pred, old_values - eps_v, old_values + eps_v)
value_loss    = 0.5 * mean((value_clipped - returns)^2 * mask)
```

All NCCL operations identical in type and volume to Phase 8.

---

### Phase 10 — Weight Synchronization

**Where**: Megatron training workers → SGLang server

**Key difference from verl**: verl AllGathers weights in-process and hands them to the co-located
rollout engine via shared GPU memory. Miles HTTP-pushes serialized weights across the process boundary.

```
Step 1 — Megatron AllGather (reconstruct full TP-sharded model from ZeRO shards):
  NCCL — AllGather across DP group (2 GPUs, cross-node):
    input:  ~875 MB ZeRO shard per GPU   (attention + expert weights, 24 layers)
    output: ~1.75 GB full TP=2 copy per GPU

Step 2 — HTTP weight push to SGLang:
  Controller serializes from Megatron's TP-sharded layout
  HTTP POST /update_weights  { "state_dict": <binary blob ~1.75 GB> }
  Runs every K training steps (async mode) or every step (sync mode)

Step 3 — SGLang hot-swap:
  Drains or pauses in-flight requests
  Loads new parameter tensors into existing memory
  Resumes accepting generation requests
```

The HTTP boundary adds ~100–500 ms of serialization + network latency per sync, but enables
independent process lifetimes and separate GPU allocation for training vs. rollout.

---

## 4. Communication Summary Table

| Phase | Primitive | Group | Tensor | Volume | Direction |
|-------|-----------|-------|--------|--------|-----------|
| Rollout prefill — attention (per layer) | AllReduce ×2 | TP (intra-node) | (32,512,2048) bf16 | 64 MB ×2 | intra-node |
| Rollout prefill — MoE FFN (per layer) | AlltoAll ×2 | EP (intra-node) | (32768, 2048) bf16 | ~64 MB ×2 | intra-node |
| Rollout decode — attention (per layer, per step) | AllReduce ×2 | TP | (32,1,2048) bf16 | 128 KB ×2 | intra-node |
| Rollout decode — MoE FFN (per layer, per step) | AlltoAll ×2 | EP | (64, 2048) bf16 | ~256 KB ×2 | intra-node |
| Ref/value fwd (per layer) | AllReduce ×2 + AlltoAll ×2 | TP + EP | (32,1024,2048) bf16 | ~32 MB each | intra-node |
| Megatron ZeRO fwd (per layer) | AllGather | DP (cross-node) | attention weights | ~28 MB | cross-node |
| Megatron ZeRO bwd (per layer) | ReduceScatter | DP (cross-node) | attention grads | ~28 MB | cross-node |
| TP backward (per layer) | AllReduce ×2 | TP (intra-node) | (8,1024,2048) bf16 | 32 MB ×2 | intra-node |
| EP backward (per layer) | AlltoAll ×2 | EP (intra-node) | (16384, 2048) bf16 | ~32 MB ×2 | intra-node |
| Weight sync | HTTP push | — | ~1.75 GB state dict | 1.75 GB | cross-process |

**What is new vs. verl**: the EP AlltoAll pair at each MoE FFN layer. In verl (dense model) there
is no AlltoAll — only TP AllReduce after W_down. In Miles (MoE), the two AlltoAll calls replace
the single dense FFN AllReduce and add cross-EP communication that is absent from the verl walkthrough.

---

## 5. verl vs. slime/Miles: Side-by-Side

| Dimension | verl | slime / Miles |
|-----------|------|---------------|
| Rollout engine | vLLM (in-process, SPMD) | SGLang (HTTP server, continuous batching) |
| Training engine | PyTorch FSDP (ZeRO-3) | Megatron-LM (ZeRO + PP + EP) |
| Rollout–training boundary | In-process NCCL (same process group) | HTTP (separate processes) |
| MoE / expert parallelism | Dense-first; EP not in core | MoE-native; EP AlltoAll built in |
| Weight sync primitive | NCCL AllGather in-memory (~3.5 GB) | HTTP push (~1.75 GB for this config) |
| Phase 4 (old log-probs) | Separate recompute forward pass | Returned by SGLang in Phase 2 (free) |
| Training-inference mismatch | IS correction (ρ = exp(log_new − log_old)) | Elimination (bitwise-identical SGLang ↔ Megatron) |
| Rollout scheduling | Co-located only (shared GPUs) | Separate GPU pools possible |
| Battle-tested scale | Dense models up to ~70B | MoE models up to 355B+ (GLM-4.6, DeepSeek-V4) |

**Choose verl** when: dense models, FSDP-based training stack, simpler ops, tight in-memory NCCL
coupling is preferred.

**Choose slime/Miles** when: MoE models, Megatron training stack, need separate scaling of rollout
vs. training capacity, or want SGLang-native features (prefix caching, structured output,
speculative decoding).

---

## 6. Key Code Pointers

| Concept | File | Notes |
|---------|------|-------|
| PPO training loop | `slime/trainer/ppo_trainer.py` | Main controller |
| SGLang rollout worker | `slime/rollout/sglang_rollout.py` | HTTP client to SGLang |
| Weight push | `slime/rollout/sglang_rollout.py` | `update_weights()` HTTP call |
| Megatron actor | `slime/worker/actor_worker.py` | Megatron fwd/bwd |
| Expert parallelism config | `slime/config/` | `ep_size`, `num_experts`, `topk` |
| Miles enterprise layer | `miles/` (radixark/miles) | Adds ops tooling on top of slime |
| TIM elimination details | slime/MILES blog post | Bitwise-identical SGLang ↔ Megatron |
