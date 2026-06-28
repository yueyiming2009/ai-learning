---
name: add-page
description: Place a new (or moved) docs/*.html learning page into the index correctly. Use whenever you create a new explainer page, rename one, or are deciding where a topic belongs — it gives the canonical domain/subsection taxonomy of docs/index.html, the exact card markup to add, and the triggers that mean the taxonomy itself needs to change (which is a question for the user, not a silent edit).
---

# Place a new page into the index

This repo is a self-contained ML/RL learning site. Every topic is a standalone
`docs/*.html` file (inline CSS + MathJax, no build step), and `docs/index.html` is the
front page that links to all of them. A new page isn't "done" until it has a **card** in
the right place on the index. This skill is the single source of truth for *where* that is.

Related: [`add-plot`](../add-plot/SKILL.md) governs the figures *inside* a page; this skill
governs the page's *entry on the index*.

## The taxonomy (4 collapsible domains)

`docs/index.html` is organized into four top-level **domains**, each a
`<details class="domain" id="…">` that is **collapsed by default** (no `open` attribute). Its
`<summary class="domain-header">` carries the numbered badge, the `<h1>`, and a `▸` chevron;
inside the `<details>` come a `<p class="domain-desc">` and the **subsections** (`<h2>`,
restyled as small uppercase labels), each holding a `card-grid` of cards. Within Systems &
Hardware and within Inference the subsections run bottom-up (hardware → up; foundations →
workloads).

| # | Domain (`id` / class / accent) | Subsections (in page order) | A page belongs here if it's about… |
|---|---|---|---|
| 01 | **Deep Learning Foundations** — `#foundations` / `domain-foundations` / indigo | Foundations · Loss Functions · Normalization · Activations & FFN · Positional Encoding · Optimization · Sampling & Estimation · Tokenization · Architecture | the math of how a model computes & trains: a gradient, a loss's probabilistic basis, an optimizer rule, the tokenizer/routing math |
| 02 | **Systems & Hardware** — `#systems` / `domain-systems` / orange | Hardware · GPU Kernels · Communication & Parallelism · Distributed Training | the training substrate: chips, kernels, collectives, parallelism, distributed plumbing |
| 03 | **Reinforcement Learning** — `#rl` / `domain-rl` / teal | RL Theory · Alignment & RLHF · Training Walkthroughs · RL Efficiency | anything RL — the theory (MDP → tree search), the alignment methods (RLHF/GRPO/DPO/…), and the systems that run RL jobs (walkthroughs, async RL, efficiency) |
| 04 | **Inference** — `#inference` / `domain-inference` / blue | Inference Foundations · Inference Techniques · Inference Workloads | serving a trained model fast: the prefill/decode compute model, its optimizations, and serving workloads |

Rule of thumb: **Foundations = "why the formula is true," Systems & Hardware = "the machine
that trains it," RL = "anything reinforcement learning, theory through production,"
Inference = "serving it fast."** RL and Inference each own their *whole* vertical — an RL
training walkthrough goes under RL (not a generic "practices" bucket), and an inference serving
recipe goes under Inference.

## How to add the card

1. **Pick the domain, then the subsection** using the table above. If two subsections in the
   same domain both fit, pick the more specific one.
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
   - `href` is the bare filename (`kv-cache.html`), no `docs/` prefix — the index lives in `docs/`.
   - `card-arrow` text matches the page type: `Read derivation →` (Foundations), `Read algorithm →`,
     `Read guide →`, `Read walkthrough →` (RL walkthroughs), `Read reference →` (diagnostic pages).
   - **Tag class**: reuse an existing one — don't add a new CSS class for a one-off. Available:
     `tag-found tag-loss tag-norm tag-pos tag-act tag-opt tag-algo tag-rl-theory tag-rl tag-grpo`
     `tag-kern tag-hw tag-par tag-nccl tag-dist tag-infer`. If none fit the color you want, use
     an inline `style="background:#…;color:#…;"` on the `<span>` (as the Cascade/Agentic/
     Distillation cards do) rather than defining a new class.
4. If the page is also listed in `README.md`, add a row there too (see "Keep sources in sync").

## The page's own top-left home link

Every content page's sticky `<nav>` opens with a `nav-brand` that is the **home icon + the page
title, both linking to `index.html`** (not `#`):

```html
<a class="nav-brand" href="index.html" aria-label="Home — all notes">
  <svg class="nav-home" viewBox="0 0 24 24" width="16" height="16" fill="none"
       stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
       aria-hidden="true"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/></svg>
  Page Title
</a>
```

The `.nav-brand` CSS rule must include `display:inline-flex; align-items:center; gap:7px;` so the
icon (which inherits the page accent via `currentColor`) sits beside the title. Easiest path:
copy the whole `<nav>` from an existing page (e.g. `swiglu.html`) when scaffolding a new one.

## Reminder — when the *structure* needs to change (ask, don't silently restructure)

The 4-domain taxonomy is deliberate. Adding a card is routine; changing the taxonomy is a
**user decision** because it touches the hero nav pills, the accent-color CSS, the README, and
this file. After placing a card, check these triggers and **surface any that fire to the user**
instead of acting on them:

- **No subsection fits.** The page belongs to a domain but none of its `<h2>` subsections cover
  it → propose a *new subsection* (cheap, in-domain) and confirm the label with the user.
- **It fits no domain, or fits two.** A topic that spans domains (or none) is a sign the seams
  are off → flag it; don't force it into the least-bad bucket.
- **A new top-level domain seems needed.** STOP. Adding a 5th domain changes the hero pills
  (`.hero-domains`), needs a new `domain-*` accent block in the CSS, a numbered badge, and a new
  collapsed `<details id="…">` whose `id` the open-on-hash script will match — plus a README
  domain. Confirm scope with the user first.
- **A subsection exceeds ~6–8 cards, or a domain exceeds ~6 subsections.** The page is getting
  dense again — the exact problem this structure was built to fix. Flag it as a split candidate.

When the taxonomy *does* change (with user sign-off), update all of them in the same commit so
they never disagree: `docs/index.html` (the `<details>` sections + `.hero-domains` pills +
`.domain-*` accent CSS + the open-on-hash JS), `README.md`, and **this skill's taxonomy table**.

## Verify before finishing

No Node in this env. Sanity-check with `python3` / `grep`:
- the `href` points to a file that exists in `docs/`;
- the card sits inside the intended subsection's `card-grid` (not orphaned between sections);
- the `card-tag` class exists in the `<style>` block (or uses an inline `style=`);
- tags/divs balance (the card is one `<a>…</a>` with the four inner elements).

Then commit and push per repo convention (auto-commit+push is expected in this repo).
