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

## The taxonomy (4 domains — keep it)

`docs/index.html` is organized into four top-level **domains** (`<section class="domain">`,
each with a numbered badge, an `<h1>`, and a one-line description). Inside each domain are
**subsections** (`<h2>`, restyled as small uppercase labels) that hold a `card-grid` of cards.
Domains are ordered by abstraction; within Systems the subsections run bottom-up (hardware → up).

| # | Domain (`id` / class / accent) | Subsections (in page order) | A page belongs here if it's about… |
|---|---|---|---|
| 01 | **Math** — `#math` / `domain-math` / indigo | Foundations · Loss Functions · Normalization · Activations & FFN · Positional Encoding · Optimization · Sampling & Estimation | a worked derivation from scratch: a gradient, a loss's probabilistic basis, an update rule |
| 02 | **Algorithms & Methods** — `#algorithms` / `domain-algo` / teal | Tokenization · Architecture · RL Theory · Alignment & RLHF | a procedure or training method: a data-structure algorithm, an architecture mechanism, an RL algorithm, a post-training method |
| 03 | **Systems** — `#systems` / `domain-sys` / orange | Hardware · GPU Kernels (Triton) · Communication & Parallelism · Distributed Training · Inference Foundations · Inference Techniques | the engineering substrate: chips, kernels, collectives, parallelism, the inference compute model and its optimizations |
| 04 | **Practices** — `#practices` / `domain-prac` / green | Training Walkthroughs · RL Efficiency · Inference Workloads | running a *real* job: an end-to-end walkthrough, an efficiency/diagnosis technique, a serving workload |

Rule of thumb: **Math = "why the formula is true," Algorithms = "the method/procedure,"
Systems = "the machine that runs it," Practices = "operating it in production."** If a page
explains a concept, it's Math/Algorithms; if it explains making that concept fast or running
it at scale, it's Systems/Practices.

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
   - `card-arrow` text matches the page type: `Read derivation →` (Math), `Read algorithm →`,
     `Read guide →`, `Read walkthrough →` (Practices), `Read reference →` (diagnostic pages).
   - **Tag class**: reuse an existing one — don't add a new CSS class for a one-off. Available:
     `tag-found tag-loss tag-norm tag-pos tag-act tag-opt tag-algo tag-rl-theory tag-rl tag-grpo`
     `tag-kern tag-hw tag-par tag-nccl tag-dist tag-infer`. If none fit the color you want, use
     an inline `style="background:#…;color:#…;"` on the `<span>` (as the Cascade/Agentic/
     Distillation cards do) rather than defining a new class.
4. If the page is also listed in `README.md`, add a row there too (see "Keep sources in sync").

## Reminder — when the *structure* needs to change (ask, don't silently restructure)

The 4-domain taxonomy is deliberate. Adding a card is routine; changing the taxonomy is a
**user decision** because it touches the hero nav pills, the accent-color CSS, the README, and
this file. After placing a card, check these triggers and **surface any that fire to the user**
instead of acting on them:

- **No subsection fits.** The page belongs to a domain but none of its `<h2>` subsections cover
  it → propose a *new subsection* (cheap, in-domain) and confirm the label with the user.
- **It fits no domain, or fits two.** A topic that spans domains (or none) is a sign the seams
  are off → flag it; don't force it into the least-bad bucket.
- **A new top-level domain seems needed.** STOP. Adding a 5th domain changes hero pills
  (`.hero-domains`), needs a new `domain-*` accent block in the CSS, a numbered badge, and a
  README domain — confirm scope with the user first. (A previously-discussed candidate: promoting
  Inference out of Systems into its own domain once it outgrows two subsections.)
- **A subsection exceeds ~6–8 cards, or a domain exceeds ~6 subsections.** The page is getting
  dense again — the exact problem this structure was built to fix. Flag it as a split candidate.

When the taxonomy *does* change (with user sign-off), update all three in the same commit:
`docs/index.html` (sections + `.hero-domains` pills + `.domain-*` accent CSS), `README.md`,
and **this skill's taxonomy table**, so they never disagree.

## Verify before finishing

No Node in this env. Sanity-check with `python3` / `grep`:
- the `href` points to a file that exists in `docs/`;
- the card sits inside the intended subsection's `card-grid` (not orphaned between sections);
- the `card-tag` class exists in the `<style>` block (or uses an inline `style=`);
- tags/divs balance (the card is one `<a>…</a>` with the four inner elements).

Then commit and push per repo convention (auto-commit+push is expected in this repo).
