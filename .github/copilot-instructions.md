# Copilot Instructions for `ai-learning`

## Commands

This repository does **not** have a package-based build, lint, or test suite. The repo-native validation is simple filesystem and link checking for the static site under `docs/`.

### Whole-site validation

```bash
ls docs/*.html
# should print only docs/index.html

python3 - <<'EOF'
import re
from pathlib import Path

docs = Path("docs")
bad = 0
for f in sorted(docs.rglob("*.html")):
    for t in re.findall(r'href="([^"]+)"', f.read_text()):
        if "://" in t or t.startswith(("#", "mailto:")):
            continue
        path = t.split("#")[0]
        if path and not (f.parent / path).exists():
            print("BROKEN", f, "->", t)
            bad += 1
print("broken:", bad)
EOF
```

### Single-page link check

```bash
python3 - <<'EOF'
import re
from pathlib import Path

f = Path("docs/foundations/swiglu.html")  # replace with the page you edited
bad = 0
for t in re.findall(r'href="([^"]+)"', f.read_text()):
    if "://" in t or t.startswith(("#", "mailto:")):
        continue
    path = t.split("#")[0]
    if path and not (f.parent / path).exists():
        print("BROKEN", f, "->", t)
        bad += 1
print("broken:", bad)
EOF
```

## High-level architecture

- This repo is a **static learning site**, not an application with a build pipeline. The published artifact is the hand-authored HTML in `docs/`, served by GitHub Pages.
- `docs/index.html` is the **canonical site map**. It is both the homepage and the complete inventory of learning pages, organized into six top-level domains: `foundations`, `hardware`, `transformer`, `systems`, `rl`, and `inference`.
- Each page under `docs/<domain>/` is a **standalone HTML document** with inline CSS and inline JavaScript. There is no shared templating system, component framework, or asset pipeline. Math rendering is done with MathJax loaded directly in each page.
- The homepage uses collapsible `<details class="domain">` sections with card grids for each subsection, plus a small script that opens the correct domain when the URL hash targets a section or card anchor.
- `README.md` is the high-level map of the site and explains the domain taxonomy. Keep it aligned with meaningful taxonomy or page-organization changes.
- `rl/` contains markdown notes and walkthroughs that live outside the published HTML site, and `triton/` contains notebooks. They are supplemental content, not generated sources for `docs/`.

## Key conventions

- The **folder under `docs/` is the page's domain**. Do not place content HTML directly in `docs/`; only `docs/index.html` should live there at the top level.
- A new or moved page is not complete until **both** are updated:
  1. the page file in the correct `docs/<domain>/` folder
  2. the matching card in the correct subsection of `docs/index.html`
- Relative links are important and easy to break:
  - from a content page to the homepage: `../index.html`
  - to another page in the same domain: bare filename such as `ppo.html`
  - across domains: `../<domain>/<page>.html`
- Content pages should follow the established structure visible in existing pages such as `docs/foundations/swiglu.html`: sticky top nav, `nav-brand` home link to `../index.html`, hero block, prose sections, and inline styling/scripts kept inside the page.
- Reuse existing card markup and tag styles from `docs/index.html`. Prefer existing `card-tag` classes; if a new tag color is needed, use inline `style` on the tag rather than introducing one-off CSS classes.
- When a concept benefits from a visual, use the existing **self-contained canvas plot** pattern from `docs/foundations/swiglu.html`: inline CSS, inline vanilla JS, CSS-variable-driven colors, and no external charting library.
- The six-domain taxonomy is deliberate. If a topic does not clearly fit an existing domain or subsection, treat that as a repo-structure decision and surface it instead of silently inventing a new top-level bucket.
- The most detailed repo-specific guidance for these workflows already lives in:
  - `.claude/skills/add-page/SKILL.md`
  - `.claude/skills/add-plot/SKILL.md`
  Follow those when adding pages, moving pages, updating index cards, or adding interactive figures.
