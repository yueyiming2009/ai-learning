# Asynchronous RL for LLM Post-Training

Concise source notes for the rendered [Async RL](../../docs/rl/async-rl.html) page.
State of the ecosystem: September 2026.

---

## 1. What Async RL Removes

Synchronous RL alternates between full-batch rollout and optimization:

```text
rollout → wait for the slowest trajectory → reward → update → publish weights
```

The bottleneck is the barrier, not a universal claim that training is always cheaper than
generation. Long trajectories, tools, sandboxes, graders, phase switches, and weight publication
all create idle capacity. Async systems use streaming, buffering, admission control, and overlap
to keep useful work available while another request or phase is late.

---

## 2. Three Scheduling Regimes

| Regime | Placement | Behavior |
|---|---|---|
| Sync | Usually one pool | Complete the rollout batch, then update |
| Collocated async | One time-shared pool | Train once enough samples finish; pause/abort and later resume remaining trajectories |
| Separate async | Dedicated rollout and trainer pools | Generate and optimize concurrently through a bounded buffer |

Collocated async belongs under the scheduling taxonomy. Agent Lightning waits for current model
requests to finish before switching phases; verl's V1 path can abort inference requests and
resubmit the retained token prefix. In both cases, an agent trajectory may span the phase boundary
even though inference and training do not execute simultaneously on the same GPUs.

Do not confuse request-level concurrency with async training. For example, verl
`rollout.mode: async` selects server/agent-loop execution but does not itself permit stale data;
`trainer.v1.trainer_mode: colocate_async|separate_async` does.

---

## 3. Staleness Is a Distribution

Stamp each token or trajectory segment with its generating weight version:

$$
K_{i,t}=v_{\mathrm{train}}-v_{i,t}.
$$

`K=0` means same-version data, not necessarily a synchronous schedule. A long trajectory can
contain several versions, so record oldest, newest, and mean version or retain the version per
segment.

Age comes from:

- generation and tool latency;
- reward/grader latency;
- completed-buffer residence;
- the interval between weight publications.

A useful operational approximation is:

$$
K \approx K_{\mathrm{publication}}
+u\,(T_{\mathrm{generation}}+T_{\mathrm{reward}}+W_{\mathrm{queue}}),
$$

where $K_{\mathrm{publication}}$ is the rollout engine's lag when generation begins and `u` is
trainer updates per second. This describes an age distribution, not a universal steady-state
formula.

Queue regimes:

- rollout production below trainer demand → trainer starvation;
- approximately balanced rates → a small buffer absorbs jitter;
- rollout production above trainer demand → queue age grows unless backpressure, drop, or retry
  rules intervene.

---

## 4. Three Policies

Use explicit roles:

- $\mu$: behavior policy that sampled the token;
- $\pi_{\mathrm{prox}}$: frozen proximal policy for the update;
- $\pi_\theta$: actor being optimized.

Decoupled PPO separates:

$$
b_t=\frac{\pi_{\mathrm{prox}}(a_t|s_t)}{\mu(a_t|s_t)}
\qquad\text{and}\qquad
q_t=\frac{\pi_\theta(a_t|s_t)}{\pi_{\mathrm{prox}}(a_t|s_t)}.
$$

`b_t` corrects rollout-to-trainer mismatch. `q_t` is PPO's within-update trust-region ratio.
Bypass mode sets $\pi_{\mathrm{prox}}=\mu$ and uses rollout log-probs directly as the PPO anchor.

### Correction units

**Exact trajectory IS**

$$
W(\tau)=\prod_{t=1}^{T} b_t.
$$

This is an exact change of measure under support assumptions but has extreme variance over long
autoregressive sequences.

**Token-level rollout correction**

$$
\hat g_{\mathrm{token}}
=\sum_t \tilde b_t A_t\nabla_\theta\log\pi_\theta(a_t|s_t).
$$

Each token ratio is used independently. This is lower variance and practical, but it is not the
exact trajectory change of measure for a sequence-level reward.

**GSPO geometric sequence score**

$$
s_i(\theta)=
\exp\left(
\frac{1}{|y_i|}
\sum_t\log
\frac{\pi_\theta(y_{i,t}|x,y_{i,<t})}
     {\pi_{\mathrm{old}}(y_{i,t}|x,y_{i,<t})}
\right).
$$

GSPO clips a response as one unit using this length-normalized geometric mean. It is a deliberate
surrogate, not exact unbiased trajectory IS. A framework option named `sequence` may instead mean
the raw product; inspect the formula rather than the label.

---

## 5. Stabilization

| Mechanism | Acts on | Trade-off |
|---|---|---|
| PPO clipping | $q_t$ | Bounds optimizer movement; biased surrogate when active |
| Token TIS | Each $b_t$ | Lower variance; not exact sequence correction |
| Sequence TIS | $\prod_t b_t$ | Sequence-consistent; extreme variance and length sensitivity |
| Ratio / KL masking | Tokens or sequences | Removes toxic tails; discards experience |
| GSPO-style clipping | Geometric sequence score | Length-comparable sequence trust signal; not exact IS |

TIS truncates continuous weights. IcePop-style ratio bounds or rejection sampling zero selected
weights or mask tokens/sequences. Current verl keeps continuous IS weighting and binary rejection
masking as independent components and supports token-, product-sequence-, and KL-statistic
variants.

No correction repairs missing support or a changed token trajectory. Retokenization, rewritten
context, or altered tool traces require token preservation or sample forking.

---

## 6. Three Mismatches

| Mismatch | Difference | IS sufficient? |
|---|---|---|
| Temporal lag | Older weight version | Sometimes |
| Numerical TIM | Same nominal weights/tokens, different engine probabilities | Can reweight/filter, with bias/variance cost |
| Reconstruction mismatch | Different token IDs, context, or action sequence | No |

Numerical TIM sources include precision and quantization contracts, batch-dependent reductions,
TP topology, fused kernels, MoE routing, and sparse top-k selection.

Diagnosis:

- controlled same-version replay on exact sampled tokens;
- absolute log-prob differences, Pearson correlation, and $k_1/k_2/k_3$ statistics;
- TIS clip fractions and rejection rates;
- dataset/harness slices rather than only global averages.

Mitigation families:

- **Eliminate/align:** batch-invariant kernels, topology alignment, routing replay, supported
  Zero-KL recipes.
- **Reduce:** validated precision changes, shorter publication cadence, less aggressive
  quantization.
- **Correct/reject:** decoupled PPO, bypass mode, TIS, ratio/KL rejection, sequence trust-region
  masking.
- **Preserve identity:** TITO, sampling-time token/log-prob capture, strict prefix checks.

---

## 7. Operational Controls

- placement mode;
- in-flight admission;
- completed-buffer capacity;
- weight-sync cadence;
- maximum version span;
- drop/retry/wait policy;
- partial-rollout pause mode;
- evaluation placement and checkpoint attribution.

Current naming differs by implementation:

- verl V1: `trainer.v1.trainer_mode`, replay-buffer `max_off_policy_threshold`;
- verl experimental `fully_async_policy`: `trigger_parameter_sync_step`,
  fractional `staleness_threshold`, `partial_rollout`;
- Miles: `--async-max-concurrent-samples`, `--async-data-buffer-capacity-factor`,
  `--update-weights-interval`, `--max-weight-staleness`, `--pause-generation-mode`.

Monitor the causal chain:

```text
sample age
  → rollout/train mismatch
    → TIS or rejection activation
      → PPO clipping
        → reward, entropy, gradient, and eval outcomes
```

See [Monitoring & Diagnosis](../../docs/rl/async-rl-monitoring.html).

---

## 8. Current Systems

| System | Current pattern |
|---|---|
| verl | Sync, collocated async, and separate async; decoupled/bypass correction; replay-buffer freshness gate |
| AReaL | Separate streaming pools, mixed-version batches, Decoupled PPO, staleness filtering |
| Miles | Persistent rollout worker, bounded buffer, sample-granularity scheduling, TITO/R3, IS/masking, supported Zero-KL recipes, P2P/delta weights |
| Agent Lightning | Harness-facing collocated async through an API gateway |
| MiMo-V2.6 | Live fully async mixed-task/mixed-harness run with public staleness, TIM, TIS, sampler, grader, and restart telemetry; recipe details pending |

Reported speedups are not directly comparable: AReaL reports up to 2.57x, verl's separate
fully-async recipe 2.35–2.67x on its 128-GPU setup, and Agent Lightning roughly 2x for its
collocated setup. Hardware, task horizon, pool sizing, batch semantics, and quality constraints
differ.

---

## References

- [verl Fully Async Policy Trainer](https://verl.readthedocs.io/en/latest/advance/fully_async.html)
- [verl Rollout Correction](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) and [mathematical formulations](https://verl.readthedocs.io/en/latest/algo/rollout_corr_math.html)
- [AReaL](https://arxiv.org/abs/2505.24298)
- [Agent Lightning v1.0](https://arxiv.org/abs/2608.17528)
- [GSPO](https://arxiv.org/abs/2507.18071)
- [Trust Region Masking](https://arxiv.org/abs/2512.23075)
- [Diagnosing Training-Inference Mismatch](https://arxiv.org/abs/2605.14220)
- [Defeating TIM via FP16](https://arxiv.org/abs/2510.26788)
- [Miles v0.1](https://www.lmsys.org/blog/2026-08-18-miles-v0-1/) and [Fully Async docs](https://miles.radixark.com/docs/user-guide/fully-async)
- [GLM-5](https://arxiv.org/abs/2602.15763)
- [MiMo-V2.6 live RL metrics](https://mimo.xiaomi.com/rl/#overview)
