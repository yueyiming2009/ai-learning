---
name: add-plot
description: Add an interactive visualization to a learning page in this repo whenever it aids understanding. Use when creating or editing any docs/<domain>/*.html explainer page that describes a function, activation, distribution, curve, geometric relationship, comparison between methods, or training/optimization dynamic — anything where seeing the shape beats reading the formula. Follows the self-contained canvas plotting pattern established on docs/foundations/swiglu.html (no external chart libraries).
---

# Add a plot when it aids understanding

This repo is a self-contained ML/RL learning site (`docs/<domain>/*.html` across six domain
folders, plus `rl/` and `triton/`).
Each page is a single standalone HTML file with inline CSS + MathJax, **no build step and no
external JS libraries**. When a page explains something with a *shape* — an activation, a
loss curve, a probability distribution, a gating mechanism, a comparison of two methods, an
optimization trajectory — add a small interactive `<canvas>` plot. A reader who can see the
curve grasps it far faster than one parsing the LaTeX.

## When to add a plot (and when not to)

Add one when the concept has a natural visual:
- A function of one variable (activations, kernels, schedules, decay curves).
- A family of curves under a swept parameter (e.g. temperature, β, learning rate).
- A distribution / density, or how it shifts under an operation.
- A geometric or comparative relationship (two estimators, bias/variance, before/after).

Skip it when the idea is purely algebraic/structural (matrix shapes, index bookkeeping,
control flow) where a plot would be decoration, not insight. Prefer **one well-captioned
plot that makes a single point** over several busy ones.

## The pattern (copy from docs/foundations/swiglu.html)

`docs/foundations/swiglu.html` is the reference implementation. Reuse its three pieces verbatim and adapt:

1. **CSS** — the `.figure`, `.figure canvas` (`width:100%; height:300px`), `.legend`,
   `.legend-item`, `.lg` / `.lg.dash`, and `.figure-caption` rules. Match the page's existing
   `--accent`, `--border`, `--muted` CSS variables so the figure looks native to the page.

2. **HTML** — place the figure right after the prose/equation it illustrates:
   ```html
   <div class="figure">
     <canvas id="plot-NAME" aria-label="describe what the plot shows"></canvas>
     <div class="legend">
       <span class="legend-item"><span class="lg" style="--c:#6d28d9"></span> label</span>
       <!-- one item per series; use class "lg dash" for a dashed reference series -->
     </div>
     <p class="figure-caption">
       <strong>One-line takeaway.</strong> Then 1–2 sentences naming the key features the
       reader should notice (the dip, the crossover, the asymptote). MathJax \(inline\) is fine.
     </p>
   </div>
   ```

3. **JS** — a single vanilla-canvas `drawPlot(canvas, cfg)` helper in a `<script>` before
   `</body>`. It must:
   - handle high-DPI with `devicePixelRatio` (set `canvas.width/height = cssSize * dpr`,
     then `ctx.setTransform(dpr,0,0,dpr,0,0)`);
   - draw light gridlines + tick labels, then emphasized zero axes;
   - **clip to the plot rect** before stroking curves so out-of-range values don't bleed;
   - take a config of `{xMin,xMax,yMin,yMax, xTicks, yTicks, series:[{fn,color,width,dash}]}`
     and sample each `fn` densely (~260 points);
   - re-render on `resize` (debounced with `requestAnimationFrame`).

   Define the math (`sigmoid`, `swish`, etc.) as small pure functions and choose `yMin/yMax`
   so every series stays inside the frame — verify the min/max of each curve over the domain.

## Style conventions

- Accent / primary series: `#6d28d9`. Lighter sibling: `#a78bfa`. Secondary/derivative:
  `#0d9488` (teal). Contrast/negative family: `#f59e0b` / `#d97706` (amber). Reference series
  (e.g. ReLU baseline): gray `#9ca3af`, dashed.
- Always add a legend and a caption — a plot without a stated takeaway is half-finished.
- Keep it dependency-free and offline-capable: no CDN chart libs, no `<img>` to external hosts.

## Verify before finishing

There's no Node in this env. Sanity-check with `python3`: confirm braces/parens balance in the
script, that each canvas `id` is both present in the HTML and referenced via `getElementById`,
and that every series' min/max over the domain falls within `[yMin, yMax]`. Then commit and push
per repo convention.
