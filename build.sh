#!/bin/sh
# Bundles src/ into one self-contained HTML file.
#   LLM-Inference-GPU-Course.html  – full document, open directly in a browser
#   dist/artifact.html             – same page without <html>/<head>/<body> wrappers (for hosted publishing)
set -e
cd "$(dirname "$0")"
mkdir -p dist
FILES="src/core.js src/part1.js src/part2.js src/part3.js src/part4.js src/part5.js src/part6.js src/part7.js src/capstone.js"
{
  echo '<title>GPU Inference Course</title>'
  echo '<meta name="description" content="Interactive course: LLM inference and the GPU concepts behind it.">'
  echo '<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
  echo '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,600;12..96,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">'
  echo '<style>'; cat src/styles.css; echo '</style>'
  echo '<div id="app" class="app"></div>'
  echo '<script>'
  for f in $FILES; do echo "/* ---- $f ---- */"; cat "$f"; echo; done
  echo 'boot();'
  echo '</script>'
} > dist/artifact.html
{
  echo '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">'
  cat dist/artifact.html
  echo '</html>'
} > LLM-Inference-GPU-Course.html
echo "Built LLM-Inference-GPU-Course.html ($(wc -c < LLM-Inference-GPU-Course.html) bytes)"
