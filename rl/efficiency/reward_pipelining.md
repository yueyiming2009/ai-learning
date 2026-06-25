# Reward Pipelining in RL Post-Training

Decoupling reward computation from trajectory generation to hide reward latency on the training critical path.

---

## 1. The Reward Bottleneck

In synchronous RL, reward computation happens serially between rollout and training:

```
rollout → [wait for all rewards] → train → repeat
```

For simple rule-based rewards (format check, exact-match, regex) this cost is negligible — a few milliseconds per batch. For richer reward sources it is not:

| Reward source | Latency per trajectory | Blocking? |
|---|---|---|
| Rule-based verifier (math, code) | 1–50 ms | Mild |
| Process reward model (PRM) | 0.5–2 s (N forward passes) | Yes |
| LLM-as-judge | 2–15 s (separate model call) | Yes |
| Sandboxed execution (code eval) | 5–60 s (spawn + run) | Severe |

When generation takes 3 s and the reward call takes 8 s, rollout workers sit idle for 8 s per step. The pipeline stall is independent of generation latency — you can speed up decoding all you want and the bottleneck remains.

---

## 2. Reward Pipelining: Decoupling Reward from Generation

Reward pipelining overlaps reward computation with the next generation batch:

```
rollout pool:   [gen batch 1] → [gen batch 2] → [gen batch 3] → ...
reward workers:              [reward batch 1] → [reward batch 2] → ...
trainer:                                      [train on batch 1] → ...
```

Trajectories are submitted to a reward queue immediately upon completion. Generation starts on the next prompt batch before any rewards return. Rewards trickle back asynchronously and are matched with their trajectories in a training buffer. The trainer drains the buffer whenever a trajectory has both a trajectory and a reward attached.

The reward workers form an independent service: one or more reward models (or verifiers, or judge LLMs) running continuously, processing trajectories from the queue and writing results back.

**What this requires:**
- A queue or buffer that holds (trajectory, prompt_id, reward_placeholder) tuples
- A matching step that joins trajectory + reward before the trainer sees the sample
- A staleness policy for rewards that take too long (drop or use a fallback)

---

## 3. Reward Sources and Batching

The pipelining benefit depends on how reward calls can be structured:

### Batched verifier calls

Rule-based verifiers and lightweight discriminators are embarrassingly parallel. Accumulate $B$ trajectories and evaluate them in one call — the per-trajectory overhead amortizes across the batch. The reward workers simply run the verifier in a loop; the main loop never blocks.

### Batched reward model inference

A PRM or outcome reward model is a standard LLM inference job. Treat it identically to the rollout model: batch trajectories, run prefill + decode, return scalars. With prefix caching the shared-prompt prefix is computed once per batch. The reward worker is a vLLM/SGLang instance, not a bespoke service.

Key sizing constraint: the reward model's batch size $B_r$ and throughput $T_r$ must satisfy:

$$T_r \geq T_{rollout}$$

where $T_{rollout}$ is the rate at which rollout generates trajectories. If $T_r < T_{rollout}$, the queue grows without bound — the reward worker is the new bottleneck regardless of pipelining.

### LLM-as-judge

The judge is a separate (often larger) model. Reward pipelining is most valuable here — judge latency (2–15 s) far exceeds generation latency for short responses. The judge call goes into the queue; generation continues. The main constraint is that judge throughput must keep up with rollout throughput, which usually requires either a smaller judge model or multiple judge replicas.

### Multi-source rewards

Production systems often mix reward sources: a fast rule-based component (grammar, format) plus a slow judge component. Pipeline only the slow component. The fast reward is computed inline (negligible cost), attached immediately to the trajectory, and the slow reward is added asynchronously when it arrives. The training buffer waits for the union.

---

## 4. Queue Mechanics and Back-Pressure

The reward queue has a natural depth — the number of in-flight reward requests at any time. Depth is set by the ratio of reward latency to reward throughput:

$$\text{queue depth} \approx \frac{L_{reward}}{1 / T_r} = L_{reward} \times T_r$$

where $L_{reward}$ is the end-to-end latency of a single reward call and $T_r$ is the reward throughput (trajectories/s).

**Too shallow:** reward computation is still on the critical path — the trainer drains the buffer faster than rewards arrive and stalls waiting for rewards.

**Too deep:** high memory pressure (all in-flight trajectories must be held in the buffer), and rewards that arrive late are stale relative to the training policy that consumed earlier batches.

**Back-pressure:** if the trainer is slower than reward computation, the buffer fills. Bound the buffer size and pause submission of new trajectories to the reward queue when the buffer is full — this propagates back-pressure to the rollout pool and prevents unbounded memory growth.

---

## 5. Off-Policy Implications

For most reward formulations, reward pipelining does not introduce off-policy bias: the reward is a function of the trajectory alone (correctness, format, code execution output) and does not depend on the current policy $\pi_\theta$. A trajectory generated at step $t$ gets the same reward whether it is evaluated at step $t$ or step $t+K$.

The exception is a **policy-conditioned reward** — e.g., a value-based PRM that uses $V^\pi$ to score intermediate states, or a relative reward that compares the trajectory against the current policy's distribution. If the reward explicitly depends on $\pi_{\theta_{current}}$ and the policy has advanced $K$ steps since generation, the reward is computed under a stale policy. This is a genuine off-policy problem distinct from the generation-to-training lag covered by IS correction.

**In practice:** most deployed reward sources (outcome RM, LLM judge, code verifier) are trajectory-only. The off-policy concern applies mainly to PRM designs that incorporate a learnable value function.

### Interaction with IS correction

Reward pipelining introduces a second dimension of staleness on top of generation-to-training lag. A trajectory can be:

- Generated under $\pi_k$ (generation lag = $K_{gen}$)
- Rewarded under a stale judge or RM (reward lag = $K_{rew}$)

These are independent. IS correction (token-level or sequence-level) addresses only $K_{gen}$. $K_{rew}$ requires bounding the reward queue depth and enforcing a reward freshness budget.

---

## 6. Composing Reward Pipelining with Async RL

[Async RL](async_rl.md) decouples generation from training. Reward pipelining decouples reward from generation. The two compose naturally into a fully decoupled three-stage pipeline:

```
generation workers:   [gen] → [gen] → [gen] → ...    (continuous)
reward workers:       [rew] → [rew] → [rew] → ...    (continuous)
trainer:              [step] → [step] → [step] → ...  (continuous)
```

Each stage drains from the stage before it and pushes to a buffer. There are now two independent lag parameters:

- $K_{gen}$: policy lag from generation to training (managed by IS correction and `staleness_threshold`)
- $K_{rew}$: reward lag from reward evaluation to training (managed by queue depth and reward freshness budget)

The total staleness budget is the sum. If $K_{gen} + K_{rew}$ is large, IS correction must handle a wider distribution shift — the IS ratio $\pi_{new}(a_t|s_t)/\pi_{old}(a_t|s_t)$ is now more extreme because $\pi_{old}$ is further back in training history.

**Tuning priority:** minimize $K_{rew}$ first (it adds staleness for free, with no IS correction benefit), then tune $K_{gen}$ to the maximum value stable under IS correction.

---

## 7. Key References

| | |
|---|---|
| [Inference in RL — Rollout and Reward Model Serving](rl-inference.html) | Rollout model vs. reward model throughput matching; GRPO prefix sharing; placement tradeoffs |
| [Async RL — Policy Lag, IS Correction, and TIM](async_rl.md) | Generation-to-training decoupling; IS correction math; TIM |
| [RL Systems: Mind the Gap](https://newsletter.semianalysis.com/p/rl-systems-mind-the-gap-matching) | Throughput-matching framework; reward model as a pipeline stage |
| [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) | Open-source async reward scheduling in production RLHF |
| [VeRL fully-async docs](https://verl.readthedocs.io/en/latest/advance/fully_async.html) | `trigger_parameter_sync_step`, `staleness_threshold`, `partial_rollout` — extends to reward scheduling |
