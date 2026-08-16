---
name: add-page
description: Place a new (or moved) learning page into the right docs/ domain folder and into the index correctly. Use whenever you create a new explainer page, rename or move one, or are deciding where a topic belongs — it gives the canonical domain/subsection taxonomy (which is also the docs/ folder layout), the relative-link rules, the exact card markup to add, and the triggers that mean the taxonomy itself needs to change (which is a question for the user, not a silent edit).
---

# Place a new page into the index

This repo is a self-contained ML/RL learning site. Every topic is a standalone HTML file
(inline CSS + MathJax, no build step) living in a **domain folder** under `docs/`, and
`docs/index.html` is the front page that links to all of them. A new page isn't "done" until
it sits in the right folder *and* has a **card** in the matching place on the index. This
skill is the single source of truth for *where* both of those are.

**The folder structure mirrors the taxonomy one-for-one.** `docs/` holds exactly
`index.html` plus six domain directories: `foundations/ hardware/ transformer/ systems/
rl/ inference/`. A page's directory *is* its domain — pick the domain once (table below)
and it determines both the file's path and the section its card goes in. Never add a
loose `.html` at the top of `docs/`.

Related: [`add-plot`](../add-plot/SKILL.md) governs the figures *inside* a page; this skill
governs the page's *entry on the index*.

## The taxonomy (6 collapsible domains)

`docs/index.html` is organized into six top-level **domains**, each a
`<details class="domain" id="…">` that is **collapsed by default** (no `open` attribute). Its
`<summary class="domain-header">` carries the numbered badge, the `<h1>`, and a `▸` chevron;
inside the `<details>` come a `<p class="domain-desc">` and the **subsections** (`<h2>`,
restyled as small uppercase labels), each holding a `card-grid` of cards. The domains run in
reading order — isolated math, then the chips, then the model, then the cluster, then RL, then
serving. Within Systems & Scaling and within Inference the subsections run bottom-up
(kernels → up; foundations → workloads).

| # | Domain (folder / `id` / class / accent) | Subsections (in page order) | A page belongs here if it's about… |
|---|---|---|---|
| 01 | **Deep Learning Foundations** — `docs/foundations/` / `#foundations` / `domain-foundations` / indigo | Foundations · Loss Functions · Normalization · Activations & FFN · Positional Encoding · Optimization · Sampling & Estimation · Tokenization · Test-Time Scaling · Interpretability | the math of how a model computes & trains, component by component: a gradient, a loss's probabilistic basis, an optimizer rule, the tokenizer algorithm — **derivations that stand alone from the transformer** |
| 02 | **Hardware** — `docs/hardware/` / `#hardware` / `domain-hardware` / lime | Accelerators | the physical chip: SM/tensor-core layout, memory hierarchy, interconnect topology |
| 03 | **Transformer & Pretraining** — `docs/transformer/` / `#transformer` / `domain-transformer` / violet | Architecture · Training · Model Architectures | assembling the pieces into a working model and training it: attention, the block, the layer variants (linear attn, DeltaNet, MoE routing), the next-token objective, how to **size** the run (scaling laws) — and the **model case studies** that read a complete frontier architecture (DeepSeek-V4, Qwen3.5, Kimi K3) through those components |
| 04 | **Systems & Scaling** — `docs/systems/` / `#systems` / `domain-systems` / orange | Performance · Kernel Programming · Communication & Parallelism · Distributed Training | the substrate that trains it across many devices: kernels, MFU, collectives, parallelism, distributed plumbing. Kernel Programming has a read-in-order intro (CUDA → cuBLAS → Triton → CUTLASS → CuTe DSL → cuTile, then the Blackwell kernel model and FLA as deep dives) — keep its card order matched to it |
| 05 | **Reinforcement Learning** — `docs/rl/` / `#rl` / `domain-rl` / teal | RL Theory · Alignment & RLHF · Agentic RL · RL Frameworks · Training Walkthroughs · RL Efficiency | anything RL — the theory (MDP → tree search), the alignment methods (RLHF/GRPO/DPO/…), the agentic vertical (ReAct, long-horizon tool use, multi-turn rollout training), the frameworks that run RL jobs (verl/TRL/landscape/placement), and the systems pages (walkthroughs, async RL, efficiency) |
| 06 | **Inference** — `docs/inference/` / `#inference` / `domain-inference` / blue | Inference Foundations · Inference Techniques · Inference Workloads | serving a trained model fast: the prefill/decode compute model, its optimizations, and serving workloads |

Rule of thumb: **Foundations = "why this formula is true, in isolation," Hardware = "the chip,"
Transformer & Pretraining = "the model, assembled and trained," Systems & Scaling = "the cluster
that trains it," RL = "anything reinforcement learning, theory through production,"
Inference = "serving it fast."** RL and Inference each own their *whole* vertical — an RL
training walkthrough goes under RL (not a generic "practices" bucket), and an inference serving
recipe goes under Inference.

**The Foundations ↔ Transformer seam is the one that gets mis-drawn.** The test is whether the
page is a *standalone derivation* or a *transformer component*. SwiGLU and RoPE are gradient
derivations that happen to be used in transformers → Foundations. MoE routing describes a layer
of the model → Transformer & Pretraining. If a page's subject only exists inside a transformer,
it is not a Foundations page. Note both domains have a subsection that has at some point been
called "Architecture" — do not let the name alone route you.

## How to add the card

1. **Pick the domain, then the subsection** using the table above. If two subsections in the
   same domain both fit, pick the more specific one. The page file itself must live in that
   domain's folder — `docs/<domain>/PAGE.html`. If you are *moving* an existing page, use
   `git mv` and then fix every link that points at it (see step 3).
2. **Find the insertion point**: the `<div class="card-grid">` immediately under that
   subsection's `<h2>` in `docs/index.html`. Append the card as the last `<a class="card">`
   in that grid (or order it by the learning sequence if the subsection has a "read in order"
   intro paragraph — match the order stated there).
3. **Add the card** using this exact shape (copy a sibling card and adapt — don't invent new
   structure):
   ```html
   <a class="card" href="PAGE.html">
     <span class="card-tag TAGCLASS">Short Tag</span>
     <div class="card-title">Full Title — Optional Subtitle</div>
     <p class="card-desc">One–two sentences naming what is actually derived/covered — the
       specific quantities, the key result, the interactive bit if any. Not a vague blurb.</p>
     <div class="card-arrow">Read guide →</div>
   </a>
   ```
   - `href` is **domain-relative from `docs/`**: `inference/kv-cache.html`, `foundations/swiglu.html`.
     The index lives at `docs/index.html`, so it addresses every page through its domain folder.
   - `card-arrow` text matches the page type: `Read derivation →` (Foundations), `Read algorithm →`,
     `Read guide →`, `Read walkthrough →` (RL walkthroughs), `Read reference →` (diagnostic pages).
   - **Tag class**: reuse an existing one — don't add a new CSS class for a one-off. Available:
     `tag-found tag-loss tag-norm tag-pos tag-act tag-opt tag-algo tag-rl-theory tag-rl tag-grpo`
     `tag-kern tag-hw tag-par tag-nccl tag-dist tag-infer`. If none fit the color you want, use
     an inline `style="background:#…;color:#…;"` on the `<span>` (as the Cascade/Agentic/
     Distillation cards do) rather than defining a new class.
4. If the page is also listed in `README.md`, add a row there too (see "Keep sources in sync").

## Links from inside a page (relative, one level down)

A content page lives at `docs/<domain>/PAGE.html`, so every internal link is written from
inside that folder:

| Link target | `href` from a content page |
|---|---|
| the index | `../index.html` (plus `#domain` for a section, e.g. `../index.html#rl`) |
| a page in the **same** domain | bare filename — `ppo.html` |
| a page in **another** domain | `../<domain>/PAGE.html` — e.g. `../foundations/swiglu.html` |

Getting this wrong is the single easiest mistake here. After editing links, run the checker
in "Verify before finishing".

## The page's own top-left home link

Every content page's sticky `<nav>` opens with a `nav-brand` that is the **home icon + the page
title, both linking to `../index.html`** (not `#`) — the index is one level up from the page's
domain folder:

```html
<a class="nav-brand" href="../index.html" aria-label="Home — all notes">
  <svg class="nav-home" viewBox="0 0 24 24" width="16" height="16" fill="none"
       stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
       aria-hidden="true"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/></svg>
  Page Title
</a>
```

The `.nav-brand` CSS rule must include `display:inline-flex; align-items:center; gap:7px;` so the
icon (which inherits the page accent via `currentColor`) sits beside the title. Easiest path:
copy the whole `<nav>` from an existing page (e.g. `docs/foundations/swiglu.html`) when
scaffolding a new one.

## Reminder — when the *structure* needs to change (ask, don't silently restructure)

The 6-domain taxonomy is deliberate, and since the folders mirror it, changing the taxonomy now
also **moves files on disk**. Adding a card is routine; changing the taxonomy is a **user
decision** because it touches the folder layout, the hero nav pills, the accent-color CSS, the
README, and this file. After placing a card, check these triggers and **surface any that fire to the user**
instead of acting on them:

- **No subsection fits.** The page belongs to a domain but none of its `<h2>` subsections cover
  it → propose a *new subsection* (cheap, in-domain) and confirm the label with the user.
- **It fits no domain, or fits two.** A topic that spans domains (or none) is a sign the seams
  are off → flag it; don't force it into the least-bad bucket.
- **A new top-level domain seems needed.** STOP. Adding a 7th domain changes the hero pills
  (`.hero-domains`), needs a new `domain-*` accent block in the CSS, a numbered badge, a new
  collapsed `<details id="…">` whose `id` the open-on-hash script will match, a new `docs/`
  folder, and a README domain. Confirm scope with the user first.
- **A subsection exceeds ~6–8 cards, or a domain exceeds ~6 subsections.** The page is getting
  dense again — the exact problem this structure was built to fix. Flag it as a split candidate.

When the taxonomy *does* change (with user sign-off), update all of them in the same commit so
they never disagree: the `docs/` folder layout (`git mv` the pages, then re-fix their links),
`docs/index.html` (the `<details>` sections + `.hero-domains` pills + `.domain-*` accent CSS +
the open-on-hash JS), `README.md`, and **this skill's taxonomy table**.

## Verify before finishing

No Node in this env. Sanity-check with `python3` / `grep`:
- the page file is under `docs/<domain>/`, and no stray `.html` (other than `index.html`) sits
  directly in `docs/`:
  ```bash
  ls docs/*.html   # must print only docs/index.html
  ```
- **every internal link on every page resolves** — the one check that catches relative-path
  mistakes:
  ```bash
  python3 - <<'EOF'
  import re
  from pathlib import Path
  docs = Path("docs"); bad = 0
  for f in sorted(docs.rglob("*.html")):
      for t in re.findall(r'href="([^"]+)"', f.read_text()):
          if "://" in t or t.startswith(("#", "mailto:")): continue
          path = t.split("#")[0]
          if path and not (f.parent / path).exists():
              print("BROKEN", f, "->", t); bad += 1
  print("broken:", bad)
  EOF
  ```
- the card sits inside the intended subsection's `card-grid` (not orphaned between sections);
- the `card-tag` class exists in the `<style>` block (or uses an inline `style=`);
- tags/divs balance (the card is one `<a>…</a>` with the four inner elements).

Then commit and push per repo convention (auto-commit+push is expected in this repo).
