# LLM Inference & GPU Architecture

An interactive, self-contained course for a ~1-hour session on what happens when an LLM runs inference on a GPU:
prefill and decode, the KV cache, GPU memory and compute, compute-bound vs memory-bandwidth-bound workloads,
precision and quantization, GPU generations, multi-GPU scaling and serving optimizations.

## View it

Open **`LLM-Inference-GPU-Course.html`** in any modern browser (Chrome, Edge, Firefox, Safari). No server, install or build step is needed.

```
git clone https://github.com/tagmeagain/inference-engineering.git
open inference-engineering/LLM-Inference-GPU-Course.html     # macOS; or double-click the file
```

Notes:
- Everything (styles, scripts, visualizations) is inside the one HTML file. The only external request is for Google Fonts; offline, the page falls back to system fonts and still works.
- Progress checkmarks are stored in the browser's localStorage, per browser.
- Navigation: part tabs at the top, topic list on the left, Previous / Complete & continue at the bottom, `Alt + ←/→` to move between topics, and `#t=N` in the URL to jump to topic N.

## Project layout

| Path | What it is |
|---|---|
| `LLM-Inference-GPU-Course.html` | The built course: open this |
| `src/` | Source: `styles.css`, `core.js` (navigation, helpers, players), `part1.js` … `part7.js`, `capstone.js` (All Together) |
| `build.sh` | Bundles `src/` into the single HTML file |
| `dist/artifact.html` | Same page without `<html>/<head>/<body>` wrappers, for hosted publishing |
| `backup/` | Earlier snapshots of the course |

## Edit and rebuild

Edit files in `src/`, then run:

```
./build.sh
```

All performance numbers in the course use an imaginary example GPU and simplified formulas, and are labelled as illustrative.
