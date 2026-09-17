/* ============================================================
   PART 4 — Precision & Quantization (topics 19–23)
   ============================================================ */
part({
  num: 4, title: "Precision & Quantization",
  blurb: "Fewer bits per number means fewer bytes to store and to move. Ends with how to estimate the GPU memory a deployment needs.",
  story: "Precision / quantization changes <b>both</b> compute and memory needs: smaller numbers → less HBM used → less data to move per token.",
});

/* ---------- tiny floating-point simulator ---------- */
function fpSpec(E, M, fn = false) {
  const bias = Math.pow(2, E - 1) - 1;
  const topExp = fn ? Math.pow(2, E) - 1 : Math.pow(2, E) - 2;
  const topMant = fn ? Math.pow(2, M) - 2 : Math.pow(2, M) - 1;
  return { E, M, fn, bias, topExp, topMant, max: Math.pow(2, topExp - bias) * (1 + topMant / Math.pow(2, M)), minSub: Math.pow(2, 1 - bias - M) };
}
const FP = {
  FP32: fpSpec(8, 23), FP16: fpSpec(5, 10), BF16: fpSpec(8, 7), E4M3: fpSpec(4, 3, true), E5M2: fpSpec(5, 2),
};
function encodeFP(x, f) {
  const sign = x < 0 || Object.is(x, -0) ? 1 : 0, a = Math.abs(x);
  if (a === 0) return { v: 0, status: "exact", s: sign, e: 0, m: 0 };
  let e = Math.floor(Math.log2(a));
  if (e < 1 - f.bias) e = 1 - f.bias;
  const step = Math.pow(2, e - f.M);
  const q = Math.round(a / step) * step;
  if (q > f.max) return { v: sign ? -Infinity : Infinity, status: "overflow", s: sign, e: f.fn ? f.topExp : Math.pow(2, f.E) - 1, m: f.fn ? f.topMant : 0 };
  if (q === 0) return { v: 0, status: "underflow", s: sign, e: 0, m: 0 };
  const e2 = Math.floor(Math.log2(q));
  let eField, mField;
  if (e2 < 1 - f.bias) { eField = 0; mField = Math.round(q / f.minSub); }
  else { eField = e2 + f.bias; mField = Math.round((q / Math.pow(2, e2) - 1) * Math.pow(2, f.M)); }
  return { v: sign ? -q : q, status: q === a ? "exact" : "rounded", s: sign, e: eField, m: mField };
}
function fpValues(f) {
  const out = [];
  for (let e = 0; e <= f.topExp; e++) {
    const mm = e === f.topExp ? f.topMant : Math.pow(2, f.M) - 1;
    for (let m = 0; m <= mm; m++) out.push(e === 0 ? m * f.minSub : Math.pow(2, e - f.bias) * (1 + m / Math.pow(2, f.M)));
  }
  return out.filter(v => v > 0);
}
function bitStrip(name, f, enc) {
  const bin = (v, n) => v.toString(2).padStart(n, "0");
  const s = enc ? String(enc.s) : "±", e = enc ? bin(enc.e, f.E) : "e".repeat(f.E), m = enc ? bin(enc.m, f.M) : "m".repeat(f.M);
  const cell = (ch, bg, fg) => `<span style="display:inline-grid;place-items:center;min-width:${f.M > 12 ? 12 : 18}px;height:22px;background:${bg};color:${fg};font-family:var(--f-mono);font-size:.7rem;border-radius:2px">${ch}</span>`;
  return `<div style="display:flex;gap:2px;flex-wrap:nowrap;align-items:center"><span class="mono" style="width:52px;font-size:.8rem;font-weight:600">${name}</span>
    ${cell(s, "var(--kv)", "#fff")}${e.split("").map(c => cell(c, "var(--memory)", "#fff")).join("")}${m.split("").map(c => cell(c, "var(--compute-soft)", "var(--compute)")).join("")}</div>`;
}
function showNum(v) {
  if (!isFinite(v)) return v > 0 ? "+∞ (overflow)" : "−∞ (overflow)";
  if (v === 0) return "0";
  const a = Math.abs(v);
  return a >= 1e5 || a < 1e-3 ? v.toExponential(4) : String(+v.toPrecision(7));
}

/* ---------- Topic 19 ---------- */
const FORMATS = [
  { k: "FP32", bits: [1, 8, 23], bytes: 4, use: "Full precision. Used for training master weights; rarely needed for serving.", kind: "float" },
  { k: "TF32", bits: [1, 8, 15], bytes: 4, use: "Not a storage format: numbers are stored as FP32 (4 bytes), but matrix multiplies on newer GPUs use 8 exponent + 15 mantissa bits internally for speed.", kind: "float", tf: true },
  { k: "FP16", bits: [1, 5, 10], bytes: 2, use: "Half precision. A common serving format; small range can overflow.", kind: "float" },
  { k: "BF16", bits: [1, 8, 7], bytes: 2, use: "Brain float. Same range as FP32 with 2 bytes: the default for many modern LLM checkpoints.", kind: "float" },
  { k: "FP8", bits: [1, 4, 3], bytes: 1, use: "8-bit float (E4M3 or E5M2). Needs scale factors and recent hardware; common for fast serving.", kind: "float" },
  { k: "INT8", bits: [8], bytes: 1, use: "8-bit integer plus a scale. Weights (and sometimes activations) quantized.", kind: "int" },
  { k: "INT4", bits: [4], bytes: 0.5, use: "4-bit integer plus scales, usually per group of weights. Big memory savings; care needed for accuracy.", kind: "int" },
];
topic({
  num: 19, title: "Numeric formats",
  question: "How many bytes does each weight take?",
  render: () => `
    ${say(
      "Every weight is a number, and every number takes some bits to store. <b>Fewer bits = less memory</b>, but also fewer possible values, so numbers get rounded.",
      "The whole game of precision is choosing <b>how few bits you can use</b> before the model's answers get worse."
    )}
    ${see("What does each format look like in bits? Click one.", `
      <div class="grid3" id="t19-cards"></div>
      <div class="stage-panel" id="t19-detail"></div>
      <div class="btn-row" style="gap:16px;font-size:.8rem"><span><span class="pill" style="background:var(--kv);color:#fff">s</span> sign</span><span><span class="pill" style="background:var(--memory);color:#fff">e</span> exponent (range)</span><span><span class="pill compute">m</span> mantissa (detail)</span><span><span class="pill" style="background:var(--accent);color:#fff">i</span> integer</span></div>`)}
    ${see("How big is my model in each format?", `
      <div class="controls">${slider("t19-p", "Model parameters (billions)", 1, 405, 1, 70)}</div>
      <div class="bars-h" id="t19-bars"></div>
      <p class="c-muted" style="font-size:.82rem">Red line = 80 GB. Weights only.</p>`, "Try it")}
    ${ex(`<div class="calc">70B parameters<br>FP32: 70B × 4 = 280 GB · FP16/BF16: 140 GB · FP8/INT8: 70 GB · INT4: 35 GB</div>`)}
    ${tech(`<ul>
      <li><b>Floating point</b> = sign × 2<sup>exponent</sup> × (1.mantissa). Exponent bits give <b>range</b> (how big or small), mantissa bits give <b>precision</b> (how finely spaced values are).</li>
      <li><b>Integer</b> formats store whole numbers in a small range (INT8: −128…127, INT4: −8…7). A separate <b>scale</b> maps them back to real values.</li>
      <li>Real memory use is <b>higher</b> than parameters × bytes: quantization scales and zero-points, layers kept at higher precision (embeddings, norms), metadata, runtime buffers, and the KV cache.</li>
    </ul>`)}
    ${fx(["Weight memory ≈ parameters × bytes per parameter (+ overhead)"], "FP32 = 4 · TF32 = 4 (stored as FP32) · FP16 = BF16 = 2 · FP8 = INT8 = 1 · INT4 = 0.5 bytes")}
    ${inf(`<p>Bytes per parameter changes two things at once: how much <b class="c-memory">HBM capacity</b> the weights take, and how many bytes must be <b class="c-memory">moved per decode step</b>.</p>`)}
    ${check({
      q: "How much weight memory does a 13B-parameter model need in FP16?",
      opts: ["13 GB", "26 GB", "52 GB", "6.5 GB"],
      a: 1, why: "13B × 2 bytes = 26 GB (before overhead).",
    })}`,
  mount(root, ctx) {
    const bar = (f) => {
      if (f.kind === "int") return `<div style="display:flex;gap:2px">${"<span style='width:14px;height:16px;background:var(--accent);border-radius:2px'></span>".repeat(f.bits[0])}</div>`;
      const [s, e, m] = f.bits, cells = [];
      cells.push(`<span style="width:6px;height:16px;background:var(--kv);border-radius:1px"></span>`);
      for (let i = 0; i < e; i++) cells.push(`<span style="width:6px;height:16px;background:var(--memory);border-radius:1px"></span>`);
      for (let i = 0; i < m; i++) cells.push(`<span style="width:6px;height:16px;background:var(--compute);border-radius:1px;opacity:.7"></span>`);
      if (f.tf) for (let i = 0; i < 8; i++) cells.push(`<span style="width:6px;height:16px;background:var(--surface-3);border-radius:1px"></span>`);
      return `<div style="display:flex;gap:1px;flex-wrap:wrap">${cells.join("")}</div>`;
    };
    $("#t19-cards", root).innerHTML = FORMATS.map((f, i) => `<button type="button" class="gen-card" data-i="${i}"><span class="gname" style="font-size:1.1rem">${f.k}</span>${bar(f)}<span class="mono" style="font-size:.8rem">${f.bytes} byte${f.bytes === 1 ? "" : "s"} / param</span></button>`).join("");
    const pick = (i) => {
      $$("#t19-cards .gen-card", root).forEach((c, j) => c.classList.toggle("sel", i === j));
      const f = FORMATS[i];
      $("#t19-detail", root).innerHTML = `<h4>${f.k}</h4>
        <p class="mono" style="font-size:.85rem">${f.kind === "int" ? `${f.bits[0]}-bit integer` : `1 sign + ${f.bits[1]} exponent + ${f.bits[2]} mantissa bits${f.tf ? " (used inside matrix multiplies; stored as 32 bits)" : ""}`}</p>
        <p>${f.use}</p><p><b>70B model:</b> ${fmt(70 * f.bytes, 1)} GB of weights</p>`;
    };
    $("#t19-cards", root).onclick = e => { const c = e.target.closest(".gen-card"); if (c) pick(+c.dataset.i); };
    pick(3);
    bindSliders(root, { "t19-p": x => x + "B" }, v => {
      const P = v["t19-p"], max = P * 4, scale = Math.max(max, 80) * 1.05;
      $("#t19-bars", root).innerHTML = FORMATS.filter(f => f.k !== "TF32").map(f => {
        const g = P * f.bytes;
        return `<div class="row" style="grid-template-columns:60px 1fr 80px"><span class="mono">${f.k}</span>
          <div style="position:relative;height:18px;background:var(--surface-2);border-radius:3px">
            <div class="b" style="width:${g / scale * 100}%;height:100%;background:${g > 80 ? "var(--bad)" : "var(--memory)"}"></div>
            <div style="position:absolute;top:-3px;bottom:-3px;left:${80 / scale * 100}%;width:2px;background:var(--bad)"></div></div>
          <span class="v">${fmt(g, 1)} GB</span></div>`;
      }).join("");
    });
  },
});

/* ---------- Topic 20 ---------- */
topic({
  num: 20, title: "FP16 vs BF16",
  question: "Both use 16 bits. Why do they behave differently?",
  render: () => `
    ${say(
      "Both formats spend 16 bits, but split them differently.",
      "<b>FP16</b> spends more bits on <b class='c-compute'>detail</b> (mantissa), so it can tell apart numbers that are very close, but it can't store numbers above 65,504.",
      "<b>BF16</b> spends more bits on <b class='c-memory'>range</b> (exponent), exactly like FP32, so huge and tiny values are safe, but nearby values get rounded together more."
    )}
    ${see("What actually gets stored? Type a number or pick one.", `
      <div class="btn-row">
        <input type="number" id="t20-in" value="70000" step="any" style="width:170px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface)">
        ${[["1.003", "1.003"], ["3.14159265", "π"], ["70000", "70,000"], ["0.00000001", "0.00000001"], ["0.1", "0.1"]].map(([v, t]) => `<button type="button" class="btn small" data-v="${v}">${t}</button>`).join("")}
      </div>
      <div class="scroll-x"><div style="display:grid;gap:8px;min-width:560px" id="t20-rows"></div></div>
      <div class="scroll-x"><table class="cmp-table" id="t20-table"></table></div>`)}
    ${ex(`<div class="calc">70,000 → FP16: <span class="c-bad">overflow (∞)</span> · BF16: 70,144 (close)<br>
      1.003 → FP16: 1.00293 (close) · BF16: <span class="c-bad">1.0</span> (the 0.003 is lost)<br>
      0.00000001 → FP16: <span class="c-bad">0 (underflow)</span> · BF16: ≈ 0.00000001</div>`)}
    ${tech(`<div class="scroll-x"><table class="cmp-table">
      <tr><th></th><th>FP16</th><th>BF16</th><th>FP32</th></tr>
      <tr><td>Bits (sign / exponent / mantissa)</td><td>1 / 5 / 10</td><td>1 / 8 / 7</td><td>1 / 8 / 23</td></tr>
      <tr><td>Largest value</td><td>65,504</td><td>≈ 3.4 × 10³⁸</td><td>≈ 3.4 × 10³⁸</td></tr>
      <tr><td>Step size near 1.0</td><td>2⁻¹⁰ ≈ 0.001</td><td>2⁻⁷ ≈ 0.008</td><td>2⁻²³ ≈ 0.0000001</td></tr>
    </table></div>
    <p>BF16's 8 exponent bits match FP32, so converting FP32 → BF16 never overflows. That's why BF16 became the default for training and storing many LLMs, while FP16 may need loss scaling or careful handling of outlier activations.</p>`)}
    ${inf(`<p>For memory and bandwidth they are identical: <b>2 bytes per parameter</b>. The choice is about numerical safety: BF16 avoids overflow in large activations; FP16 keeps finer detail when values are well-behaved. Older GPUs may lack fast BF16 support.</p>`)}
    ${check({
      q: "An activation reaches 100,000. What happens in FP16 vs BF16?",
      opts: ["Both store it fine", "FP16 overflows to infinity; BF16 stores it (approximately)", "BF16 overflows; FP16 is fine", "Both round it to 65,504"],
      a: 1, why: "FP16's largest value is 65,504. BF16 has FP32's exponent range, so 100,000 fits easily (with coarser detail).",
    })}`,
  mount(root, ctx) {
    const run = () => {
      const x = parseFloat($("#t20-in", root).value);
      if (!isFinite(x)) return;
      const fp32 = Math.fround(x);
      const rows = [["FP32", FP.FP32], ["FP16", FP.FP16], ["BF16", FP.BF16]];
      $("#t20-rows", root).innerHTML = rows.map(([n, f]) => bitStrip(n, f, encodeFP(n === "FP32" ? fp32 : x, f))).join("");
      $("#t20-table", root).innerHTML = `<tr><th>Format</th><th>Stored value</th><th>Result</th><th>Error</th></tr>` +
        rows.map(([n, f]) => {
          const r = encodeFP(x, f), err = isFinite(r.v) ? Math.abs(r.v - x) : Infinity;
          const tag = r.status === "overflow" ? `<span class="pill bad">overflow</span>` : r.status === "underflow" ? `<span class="pill bad">underflow → 0</span>` : r.status === "exact" ? `<span class="pill ok">exact</span>` : `<span class="pill mixed">rounded</span>`;
          return `<tr><td class="mono"><b>${n}</b></td><td class="mono">${showNum(r.v)}</td><td>${tag}</td><td class="mono">${isFinite(err) ? (err === 0 ? "0" : err.toExponential(2)) : "∞"}</td></tr>`;
        }).join("");
    };
    $("#t20-in", root).addEventListener("input", run);
    $$("button[data-v]", root).forEach(b => b.onclick = () => { $("#t20-in", root).value = b.dataset.v; run(); });
    run();
  },
});

/* ---------- Topic 21 ---------- */
topic({
  num: 21, title: "FP8",
  question: "How can a float fit in one byte, and which split should it use?",
  render: () => `
    ${say(
      "With only 8 bits there's a hard choice: spend them on <b class='c-memory'>range</b> or on <b class='c-compute'>detail</b>.",
      "<b>E4M3</b> (4 exponent, 3 mantissa bits) has more detail but a small range (up to 448). <b>E5M2</b> (5 exponent, 2 mantissa bits) reaches 57,344 but its values are further apart. A <b>scale factor</b> shifts each tensor's values into the range the format covers best."
    )}
    ${see("Where are the representable values of each format?", `
      <svg class="svg-block" viewBox="0 0 720 190" id="t21-line"></svg>
      <div class="btn-row">
        <input type="number" id="t21-in" value="3.3" step="any" style="width:140px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:var(--surface)">
        ${["0.3", "3.3", "100", "1000", "0.01"].map(v => `<button type="button" class="btn small" data-v="${v}">${v}</button>`).join("")}
      </div>
      <div class="scroll-x"><div style="display:grid;gap:6px;min-width:380px" id="t21-bits"></div></div>
      <div class="scroll-x"><table class="cmp-table" id="t21-table"></table></div>`)}
    ${ex(`<div class="calc">3.3 → E4M3: 3.25 (off by 0.05) · E5M2: 3.5 (off by 0.2)<br>
      1,000 → E4M3: <span class="c-bad">out of range</span> without scaling · E5M2: 1,024<br>
      With a scale of 4: store 1,000 ÷ 4 = 250 in E4M3 → 256 × 4 = 1,024</div>`)}
    ${tech(`<ul>
      <li><b>E4M3</b> (1 + 4 + 3 bits): max 448, finer steps. No infinity (the “FN” variant uses those bit patterns for values). Commonly used for weights and activations in inference.</li>
      <li><b>E5M2</b> (1 + 5 + 2 bits): max 57,344, coarser steps, IEEE-style infinities. More common for gradients in training, where range matters more.</li>
      <li>FP8 always comes with <b>scaling factors</b> (per tensor, per channel or per block) chosen from the data's range, so the numbers land where the format has good resolution.</li>
    </ul>`)}
    ${inf(`<p>FP8 halves memory and bytes moved compared with FP16/BF16, <b>and</b> on GPUs with native FP8 support the matrix multiplies themselves can run in FP8, reducing compute too. That's why it helps both prefill (compute) and decode (bandwidth). It needs hardware support (recent generations) and calibration to keep accuracy.</p>`)}
    ${check({
      q: "Which FP8 format gives finer detail but a smaller range?",
      opts: ["E5M2", "E4M3", "Both are identical", "Neither: FP8 is an integer format"],
      a: 1, why: "E4M3 gives one more bit to the mantissa (detail) and one fewer to the exponent (range), so it maxes out at 448 but has finer steps.",
    })}`,
  mount(root, ctx) {
    const X = v => 40 + (Math.log10(v) + 3) / 8 * 660;  // 1e-3 .. 1e5
    const drawLine = (x) => {
      let s = "";
      for (let e = -3; e <= 5; e++) s += `<line x1="${X(Math.pow(10, e))}" x2="${X(Math.pow(10, e))}" y1="20" y2="160" stroke="var(--line)" stroke-dasharray="2 4"/><text x="${X(Math.pow(10, e))}" y="178" text-anchor="middle" font-size="11" class="muted">${e >= 0 ? Math.pow(10, e).toLocaleString("en-US") : Math.pow(10, e)}</text>`;
      [["E4M3", FP.E4M3, 55, "var(--compute)"], ["E5M2", FP.E5M2, 120, "var(--memory)"]].forEach(([n, f, y, c]) => {
        s += `<text x="40" y="${y - 22}" font-size="12" font-weight="700">${n} <tspan class="muted" font-weight="400">max ${f.max.toLocaleString("en-US")}</tspan></text>`;
        fpValues(f).filter(v => v >= 1e-3).forEach(v => s += `<line x1="${X(v)}" x2="${X(v)}" y1="${y - 12}" y2="${y + 12}" stroke="${c}" stroke-width="1.3"/>`);
        s += `<line x1="${X(f.max)}" x2="${X(f.max)}" y1="${y - 18}" y2="${y + 18}" stroke="var(--bad)" stroke-width="2"/>`;
      });
      if (x > 0) s += `<line x1="${X(clamp(x, 1e-3, 1e5))}" x2="${X(clamp(x, 1e-3, 1e5))}" y1="18" y2="160" stroke="var(--ink)" stroke-width="2"/><text x="${X(clamp(x, 1e-3, 1e5)) + 4}" y="16" font-size="11" font-weight="700">${x}</text>`;
      $("#t21-line", root).innerHTML = s;
    };
    const run = () => {
      const x = parseFloat($("#t21-in", root).value); if (!isFinite(x)) return;
      drawLine(Math.abs(x));
      $("#t21-bits", root).innerHTML = bitStrip("E4M3", FP.E4M3, encodeFP(x, FP.E4M3)) + bitStrip("E5M2", FP.E5M2, encodeFP(x, FP.E5M2));
      $("#t21-table", root).innerHTML = `<tr><th>Format</th><th>Stored value</th><th>Error</th></tr>` + [["E4M3", FP.E4M3], ["E5M2", FP.E5M2]].map(([n, f]) => {
        const r = encodeFP(x, f);
        const val = r.status === "overflow" ? (f.fn ? `<span class="c-bad">out of range (no ∞ in E4M3; frameworks clamp to ±448)</span>` : `<span class="c-bad">overflow → ∞</span>`) : showNum(r.v);
        return `<tr><td class="mono"><b>${n}</b></td><td class="mono">${val}</td><td class="mono">${isFinite(r.v) ? fmt(Math.abs(r.v - x), 3) : "–"}</td></tr>`;
      }).join("");
    };
    $("#t21-in", root).addEventListener("input", run);
    $$("button[data-v]", root).forEach(b => b.onclick = () => { $("#t21-in", root).value = b.dataset.v; run(); });
    run();
  },
});

/* ---------- Topic 22 ---------- */
topic({
  num: 22, title: "INT8 / INT4 quantization",
  question: "How do you squeeze a decimal weight into a small integer?",
  render: () => `
    ${say(
      "Find the biggest weight in a group. Stretch the integer range to cover it: that stretch is the <b>scale</b>. Divide each weight by the scale and round to a whole number. Store the whole numbers plus the one scale.",
      "To use the weight, multiply back: <b>integer × scale</b>. You get something close to the original, but not exact. That gap is the quantization error."
    )}
    ${see("What happens to real weights when we quantize them?", `
      ${flow([{ t: "FP value (e.g. 0.42)" }, { t: "÷ scale", c: "memory" }, { t: "round & clip" }, { t: "integer stored (e.g. 49)", c: "kv" }, { t: "× scale when used", c: "compute" }])}
      <div class="btn-row">
        ${seg("t22-bits", [["8", "INT8 (−127…127)"], ["4", "INT4 (−7…7)"]], "8")}
        ${seg("t22-gran", [["tensor", "One scale for all"], ["group", "One scale per group of 4"]], "tensor")}
        <label style="display:flex;gap:6px;align-items:center;font-size:.88rem"><input type="checkbox" id="t22-out"> Add an outlier weight</label>
      </div>
      <div class="scroll-x"><table class="cmp-table" id="t22-table"></table></div>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t22-scale">–</span><span class="k">scale(s)</span></div>
        <div class="stat"><span class="v" id="t22-err">–</span><span class="k">average error</span></div>
        <div class="stat"><span class="v" id="t22-mem">–</span><span class="k">memory vs FP16</span></div>
      </div>`)}
    ${see("Where does accuracy start to drop?", `
      <svg class="svg-block" viewBox="0 0 700 220" id="t22-cliff"></svg>
      <p class="c-muted" style="font-size:.82rem">${illus("Illustrative trend, not a benchmark")} The exact cliff depends on the model, the method (e.g. GPTQ, AWQ, SmoothQuant), group size and task.</p>`, "Accuracy trade-off")}
    ${ex(`<div class="calc">Weights [0.42, −1.10, 0.05, 0.88], INT8<br>
      scale = 1.10 ÷ 127 = 0.00866<br>
      0.42 ÷ 0.00866 = 48.5 → store <b>48</b> → use 48 × 0.00866 = <b>0.4157</b> (error 0.004)</div>`)}
    ${tech(`<ul>
      <li><b>Weight-only quantization</b> (e.g. <b>W4A16</b>): weights stored in 4 bits, dequantized to 16-bit on the fly for the matrix multiply. Cuts memory and bytes moved; compute stays about FP16-level (with some dequantization cost).</li>
      <li><b>Weight + activation quantization</b> (e.g. <b>W8A8</b>): both weights and activations are 8-bit, so the multiply itself can run in INT8 on supported kernels. Saves memory <b>and</b> compute, but activations have outliers that make it harder.</li>
      <li><b>Granularity</b>: per-tensor, per-channel or per-group scales. Smaller groups = better accuracy, slightly more overhead.</li>
      <li><b>Accuracy cliff</b>: quality usually holds well at 8 bits, often at 4 bits with good methods, then drops sharply around 3 bits and below.</li>
    </ul>`)}
    ${fx(["scale = max |w| ÷ q_max      (q_max = 127 for INT8, 7 for INT4)", "q = round(w ÷ scale)        w ≈ q × scale"], "Symmetric quantization shown. Asymmetric schemes add a zero-point.")}
    ${inf(`<p>One outlier inflates the scale and crushes every small weight toward zero. That's why real methods use small groups, keep sensitive layers in higher precision, or smooth out outliers before quantizing.</p>`)}
    ${check({
      q: "W4A16 means…",
      opts: ["4-bit activations, 16-bit weights", "4-bit weights, 16-bit activations", "4 GPUs and 16 layers", "4-bit KV cache and 16-bit weights"],
      a: 1, why: "W = weights, A = activations. W4A16 stores weights in 4 bits and does the maths with 16-bit activations.",
    })}`,
  mount(root, ctx) {
    const base = [0.42, -1.10, 0.05, 0.88, -0.31, 0.67, -0.92, 0.18];
    const run = () => {
      const bits = +segVal(root, "t22-bits"), gran = segVal(root, "t22-gran");
      const w = base.slice(); if ($("#t22-out", root).checked) w[3] = 6.5;
      const qmax = bits === 8 ? 127 : 7;
      const groups = gran === "tensor" ? [w.map((_, i) => i)] : [[0, 1, 2, 3], [4, 5, 6, 7]];
      const scales = [], rows = [];
      groups.forEach((g, gi) => {
        const s = Math.max(...g.map(i => Math.abs(w[i]))) / qmax; scales.push(s);
        g.forEach(i => { const q = clamp(Math.round(w[i] / s), -qmax, qmax); rows[i] = { w: w[i], q, d: q * s, g: gi }; });
      });
      const maxAbs = Math.max(...w.map(Math.abs));
      const barW = (v) => `<div style="position:relative;height:10px;width:140px;background:var(--surface-2);border-radius:3px"><div style="position:absolute;top:0;bottom:0;left:50%;width:1px;background:var(--line)"></div><div style="position:absolute;top:0;bottom:0;${v >= 0 ? "left:50%" : `right:50%`};width:${Math.abs(v) / maxAbs * 50}%;background:var(--accent);border-radius:2px"></div></div>`;
      $("#t22-table", root).innerHTML = `<tr><th>Original (FP16)</th><th></th><th>Stored int</th><th>Recovered</th><th>Error</th></tr>` + rows.map(r => {
        const err = Math.abs(r.d - r.w), bad = err > 0.08;
        return `<tr><td class="mono">${r.w.toFixed(2)}</td><td>${barW(r.w)}</td><td class="mono c-kv">${r.q}</td><td class="mono">${r.d.toFixed(3)}</td><td class="mono ${bad ? "c-bad" : ""}">${err.toFixed(3)}</td></tr>`;
      }).join("");
      $("#t22-scale", root).textContent = scales.map(s => s.toFixed(4)).join(" / ");
      $("#t22-err", root).textContent = (rows.reduce((a, r) => a + Math.abs(r.d - r.w), 0) / rows.length).toFixed(4);
      $("#t22-mem", root).textContent = bits === 8 ? "½ (plus scales)" : "¼ (plus scales)";
    };
    bindSeg(root, "t22-bits", run); bindSeg(root, "t22-gran", run);
    $("#t22-out", root).addEventListener("change", run);
    run();
    const pts = [[16, 100], [8, 99.7], [6, 99.2], [5, 98.4], [4, 96.8], [3, 88], [2, 45]];
    const X = b => 70 + (16 - b) / 14 * 580, Y = q => 190 - (q - 40) / 60 * 170;
    let s = `<line x1="70" y1="190" x2="660" y2="190" stroke="var(--line)"/><line x1="70" y1="20" x2="70" y2="190" stroke="var(--line)"/>`;
    [40, 70, 100].forEach(q => s += `<text x="64" y="${Y(q) + 4}" text-anchor="end" font-size="11" class="muted">${q}%</text>`);
    pts.forEach(([b]) => s += `<text x="${X(b)}" y="206" text-anchor="middle" font-size="11" class="muted">${b}-bit</text>`);
    s += `<rect x="${X(3.5)}" y="20" width="${X(2) - X(3.5) + 10}" height="170" fill="var(--bad-soft)" opacity=".7"/><text x="${X(2.6)}" y="36" text-anchor="middle" font-size="12" font-weight="700" style="fill:var(--bad)">accuracy cliff</text>`;
    s += `<polyline fill="none" stroke="var(--accent)" stroke-width="3" points="${pts.map(([b, q]) => `${X(b)},${Y(q)}`).join(" ")}"/>`;
    pts.forEach(([b, q]) => s += `<circle cx="${X(b)}" cy="${Y(q)}" r="5" fill="var(--accent)"/>`);
    s += `<text x="80" y="16" font-size="11" class="muted">relative quality</text>`;
    $("#t22-cliff", root).innerHTML = s;
  },
});

/* ---------- Topic 23 ---------- */
topic({
  num: 23, title: "Why quantization helps memory-bound inference",
  question: "Why does quantization make decode faster, not just smaller?",
  render: () => `
    ${say(
      "Remember: at low batch sizes, decode speed is mostly limited by <b>how many bytes must travel from HBM to the chip for each token</b>.",
      "Quantized weights are fewer bytes. Fewer bytes to move per token means <b>less waiting</b>, so more tokens per second, on the very same GPU."
    )}
    ${see("Same GPU, same model: FP16 vs INT4. Which finishes more tokens?", `
      <div style="display:grid;gap:14px">
        <div style="display:grid;grid-template-columns:90px 1fr 120px;gap:10px;align-items:center">
          <b class="mono">FP16</b>
          <div style="display:grid;gap:4px">${flow([{ t: "140 GB", c: "memory" }, { t: "HBM", c: "memory" }, { t: "compute", c: "compute" }])}
            <div class="meter memory"><div class="track"><i id="t23-a" style="width:0%;transition:none"></i></div></div></div>
          <div class="stat"><span class="v" id="t23-at">0</span><span class="k">tokens</span></div>
        </div>
        <div style="display:grid;grid-template-columns:90px 1fr 120px;gap:10px;align-items:center">
          <b class="mono">INT4</b>
          <div style="display:grid;gap:4px">${flow([{ t: "35 GB", c: "memory" }, { t: "HBM", c: "memory" }, { t: "compute", c: "compute" }])}
            <div class="meter memory"><div class="track"><i id="t23-b" style="width:0%;transition:none"></i></div></div></div>
          <div class="stat"><span class="v" id="t23-bt">0</span><span class="k">tokens</span></div>
        </div>
      </div>
      <div class="grid2">
        <div><div class="mini-label" style="margin-bottom:4px">Memory capacity: 70B weights vs one 80 GB GPU</div>
          <div class="stack"><span class="s-w" style="width:100%">FP16 140 GB (1.75× the GPU)</span></div>
          <div class="stack" style="margin-top:6px"><span class="s-w" style="width:43.75%">INT4 35 GB</span><span class="s-free" style="width:56.25%">45 GB left for KV cache</span></div></div>
        <div class="stat-row">
          <div class="stat"><span class="v c-memory">~7 tok/s</span><span class="k">FP16 upper bound (1,000 GB/s)</span></div>
          <div class="stat"><span class="v c-ok">~28 tok/s</span><span class="k">INT4 upper bound</span></div>
        </div>
      </div>
      <p class="c-muted" style="font-size:.82rem">Batch 1, weights only, slow-motion animation. ${illus()}</p>`)}
    ${ex(`<div class="calc">70B × 2 bytes (FP16) = <b>140 GB</b> → 1,000 GB/s ÷ 140 GB ≈ 7 tokens/s ceiling<br>
      70B × 0.5 bytes (INT4) = <b>35 GB</b> → 1,000 GB/s ÷ 35 GB ≈ 28 tokens/s ceiling</div>`)}
    ${tech(`<ul>
      <li>Quantization reduces: <b>1.</b> memory capacity needed, <b>2.</b> bytes moved per decode step, <b>3.</b> pressure on HBM bandwidth.</li>
      <li>Real speedups are smaller than the byte ratio: dequantization kernels cost compute, some layers stay in higher precision, and the KV cache isn't shrunk by weight-only quantization (KV cache quantization is a separate technique).</li>
      <li>Accuracy must be checked on your own tasks, especially at 4 bits and below.</li>
    </ul>`)}
    ${inf(`<p>Quantization is one of the most direct fixes for memory-bound decode: it moves the workload <b>right</b> on the roofline (more FLOPs per byte), and it can make a model fit on fewer GPUs.</p>`)}
    ${check({
      q: "My model is 140 GB in FP16 and I quantize the weights to INT4. Approximately how much weight memory do I need?",
      opts: ["70 GB", "35 GB", "17.5 GB", "280 GB"],
      a: 1, why: "FP16 is 2 bytes per parameter; INT4 is 0.5 bytes: 4× smaller. 140 GB ÷ 4 = 35 GB (plus a little for scales).",
    })}`,
  mount(root, ctx) {
    let pa = 0, pb = 0, ta = 0, tb = 0;
    const slow = 10;
    ctx.loop(dt => {
      pa += dt / (0.14 * slow); pb += dt / (0.035 * slow);
      if (pa >= 1) { pa -= 1; ta++; } if (pb >= 1) { pb -= 1; tb++; }
      const a = $("#t23-a", root), b = $("#t23-b", root); if (!a) return;
      a.style.width = pa * 100 + "%"; b.style.width = pb * 100 + "%";
      $("#t23-at", root).textContent = ta; $("#t23-bt", root).textContent = tb;
    });
  },
});

/* ---------- Topic: Estimating GPU memory for inference ---------- */
const MEM_MODELS = {
  "8B": { name: "Llama-3-8B-style", N: 8e9, L: 32, kvH: 8, hd: 128, attn: "GQA" },
  "13B": { name: "Llama-2-13B-style", N: 13e9, L: 40, kvH: 40, hd: 128, attn: "MHA" },
  "70B": { name: "Llama-3-70B-style", N: 70e9, L: 80, kvH: 8, hd: 128, attn: "GQA" },
  "405B": { name: "Llama-3.1-405B-style", N: 405e9, L: 126, kvH: 8, hd: 128, attn: "GQA" },
};
topic({
  num: 0, title: "Estimating GPU memory for inference",
  question: "How much GPU memory does it take to serve a model, and how many GPUs do I need?",
  render: () => `
    ${say(
      "Before you deploy a model, you need to answer two questions: <b>how many GPUs</b> do I need, and <b>how many users</b> can they serve at once? Both come down to one sum.",
      "While serving, GPU memory has to hold four things at the same time: the <b class='c-memory'>weights</b>, the <b class='c-kv'>KV cache</b>, <b class='c-compute'>activations</b> (temporary working space) and <b>runtime overhead</b>. Estimate each one, add them up, and compare with the memory you have."
    )}
    ${see("How do the four parts add up for my deployment?", `
      <div class="controls">
        <div class="ctl"><label>Model</label>${seg("me-model", [["8B", "8B"], ["13B", "13B (MHA)"], ["70B", "70B"], ["405B", "405B"]], "70B")}</div>
        <div class="ctl"><label>Weight precision</label>${seg("me-wp", [["2", "BF16"], ["1", "FP8 / INT8"], ["0.5", "INT4"]], "2")}</div>
        <div class="ctl"><label>KV cache precision</label>${seg("me-kp", [["2", "BF16"], ["1", "FP8"]], "2")}</div>
        <div class="ctl"><label>GPU memory</label>${seg("me-gm", [["24", "24 GB"], ["48", "48 GB"], ["80", "80 GB"], ["141", "141 GB"], ["192", "192 GB"]], "80")}</div>
        <div class="ctl"><label>GPUs (tensor parallel)</label>${seg("me-g", [["1", "1"], ["2", "2"], ["4", "4"], ["8", "8"]], "4")}</div>
        ${slider("me-ctx", "Context per request (prompt + output tokens)", 0, 7, 1, 3)}
        ${slider("me-users", "Concurrent requests", 1, 128, 1, 16)}
      </div>
      ${player("me-pl")}
      <div class="caption" id="me-cap"></div>
      <div class="grid2" style="grid-template-columns:minmax(0,1.3fr) minmax(0,1fr);align-items:start">
        <div style="display:grid;gap:8px">
          <div class="calc" id="me-calc" style="background:var(--surface-2);border-radius:8px;padding:10px 14px"></div>
        </div>
        <div style="display:grid;gap:8px">
          <div class="mini-label">Memory on each GPU</div>
          <div class="stack" id="me-bar" style="height:44px"></div>
          <div class="btn-row" style="gap:12px;font-size:.78rem">
            <span><span class="pill" style="background:var(--memory);color:#fff">■</span> weights</span>
            <span><span class="pill" style="background:var(--kv);color:#fff">■</span> KV cache</span>
            <span><span class="pill" style="background:var(--compute);color:#fff">■</span> activations + overhead</span>
            <span style="color:var(--bad)">| usable limit (90%)</span></div>
          <div class="stat-row">
            <div class="stat"><span class="v" id="me-fit">–</span><span class="k">verdict</span></div>
            <div class="stat"><span class="v" id="me-maxu">–</span><span class="k">max concurrent requests at this context</span></div>
            <div class="stat"><span class="v" id="me-maxc">–</span><span class="k">max context at this many requests</span></div>
          </div>
        </div>
      </div>
      <p class="c-muted" style="font-size:.82rem">Rules of thumb: activations + temporary buffers ≈ 2 GB per GPU, CUDA/framework overhead ≈ 1 GB per GPU, and keep 10% of memory free as headroom (serving engines like vLLM reserve about 90% by default). Real engines measure activation memory at startup. ${illus("Estimate, not a guarantee")}</p>`)}
    ${ex(`<div class="calc">Llama-3-70B-style (80 layers, 8 KV heads, head size 128), BF16 weights and KV, 8,192-token context, 16 requests, 4 × 80 GB GPUs<br>
      <b>1. Weights</b>   = 70B × 2 bytes = 140 GB → ÷ 4 GPUs = <b>35 GB</b> each<br>
      <b>2. KV per token</b> = 2 × 80 × 8 × 128 × 2 bytes = 327,680 bytes ≈ 0.33 MB<br>
      <b>3. KV total</b>  = 0.33 MB × 8,192 × 16 ≈ 42.9 GB → ÷ 4 = <b>10.7 GB</b> each<br>
      <b>4. Activations + overhead</b> ≈ 2 + 1 = <b>3 GB</b> each<br>
      <b>5. Total</b> ≈ 48.7 GB per GPU vs 72 GB usable → <span class="c-ok">fits</span>, with room for more requests</div>`)}
    ${fx([
      "Weights (GB) ≈ parameters (billions) × bytes per parameter",
      "KV cache per token (bytes) = 2 × layers × KV heads × head size × bytes per number",
      "KV cache total = per token × context length × concurrent requests",
      "Per GPU (tensor parallel over N) ≈ (weights + KV total) ÷ N + activations + overhead  ≤  90% of GPU memory",
    ], "Find layers, KV heads (num_key_value_heads) and head size (hidden_size ÷ num_attention_heads) in the model's config.json.")}
    ${tech(`<ul>
      <li><b>Weights</b> are fixed once loaded. Quantized formats add a little for scales (per group), and some layers (embeddings, norms, output head) often stay in 16-bit.</li>
      <li><b>KV cache</b> is the part that changes with traffic: it scales with context length × concurrent sequences, and with the number of <b>KV heads</b> (GQA models need far less than MHA models).</li>
      <li><b>Activations</b> are temporary tensors of the current forward pass. They peak during large prefills and scale with the number of tokens processed in one step (chunked prefill caps this).</li>
      <li><b>Overhead</b>: CUDA context, kernels' workspace, communication buffers for multi-GPU, memory fragmentation.</li>
      <li>With <b>tensor parallelism</b>, weights and KV cache are split across GPUs; activations and overhead are paid on every GPU.</li>
    </ul>`)}
    ${inf(`<p>The weights decide the <b>minimum</b> number of GPUs. The KV cache decides <b>how many users and how much context</b> fit on top. Every extra GB freed (by quantizing weights, FP8 KV cache, GQA models or paged KV cache) turns directly into more concurrent requests, which means larger batches and better throughput for memory-bound decode.</p>`)}
    ${check({
      q: "A Llama-3-8B-style model has 32 layers, 8 KV heads and head size 128, with an FP16 KV cache. Roughly how much KV cache does one 8,000-token conversation need?",
      opts: ["About 100 MB", "About 1 GB", "About 10 GB", "About 16 GB"],
      a: 1, why: "Per token: 2 × 32 × 8 × 128 × 2 bytes = 131,072 bytes (≈ 0.13 MB). × 8,000 tokens ≈ 1.05 GB.",
    })}
    ${check({
      q: "Your 70B deployment fits, but only 6 users at 32k context. Which change lets you serve the most extra users without adding GPUs?",
      opts: ["Use a larger vocabulary", "Switch the KV cache from FP16 to FP8", "Increase the context length", "Use BF16 instead of FP16 weights"],
      a: 1, why: "An FP8 KV cache halves the KV memory per token, roughly doubling the users that fit (if the freed memory was the limit). BF16 and FP16 are the same size.",
    })}`,
  mount(root, ctx) {
    const ctxs = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072];
    const S = { model: "70B", wp: 2, kp: 2, gm: 80, g: 4 };
    let frame = 0, vals = { "me-ctx": 3, "me-users": 16 };
    const GBv = b => b / 1e9;
    const compute = () => {
      const m = MEM_MODELS[S.model], ctxLen = ctxs[vals["me-ctx"]], users = vals["me-users"], G = S.g;
      const weights = m.N * S.wp, kvTok = 2 * m.L * m.kvH * m.hd * S.kp, kvTot = kvTok * ctxLen * users;
      const actOv = 3e9, usable = S.gm * 1e9 * 0.9;
      const perGpu = (weights + kvTot) / G + actOv;
      const free = usable - weights / G - actOv;
      const maxU = free > 0 ? Math.floor(free * G / (kvTok * ctxLen)) : 0;
      const maxC = free > 0 ? Math.floor(free * G / (kvTok * users)) : 0;
      return { m, ctxLen, users, G, weights, kvTok, kvTot, actOv, usable, perGpu, maxU, maxC };
    };
    const render = (i) => {
      frame = i;
      const r = compute(), cap = S.gm * 1e9;
      const lines = [
        `<b>1. Weights</b> = ${r.m.N / 1e9}B params × ${S.wp} bytes = <b>${gb(r.weights)}</b>${r.G > 1 ? ` → ÷ ${r.G} GPUs = <b>${gb(r.weights / r.G)}</b> each` : ""}`,
        `<b>2. KV per token</b> = 2 × ${r.m.L} layers × ${r.m.kvH} KV heads × ${r.m.hd} × ${S.kp} bytes = <b>${r.kvTok.toLocaleString("en-US")} bytes</b> (${gb(r.kvTok)})`,
        `<b>3. KV total</b> = ${gb(r.kvTok)} × ${r.ctxLen.toLocaleString("en-US")} tokens × ${r.users} requests = <b>${gb(r.kvTot)}</b>${r.G > 1 ? ` → ÷ ${r.G} = <b>${gb(r.kvTot / r.G)}</b> each` : ""}`,
        `<b>4. Activations + overhead</b> ≈ 2 GB + 1 GB = <b>3 GB</b> per GPU`,
        `<b>5. Per GPU</b> ≈ <b>${gb(r.perGpu)}</b> vs usable ${gb(r.usable)} (90% of ${S.gm} GB) → ${r.perGpu <= r.usable ? `<span class="c-ok">fits</span>` : `<span class="c-bad">does not fit</span>`}`,
      ];
      $("#me-calc", root).innerHTML = lines.map((l, j) => `<div style="opacity:${j <= i ? 1 : .25};transition:opacity .3s">${l}</div>`).join("");
      const w = i >= 0 ? r.weights / r.G : 0, kv = i >= 2 ? r.kvTot / r.G : 0, ao = i >= 3 ? r.actOv : 0;
      const scale = Math.max(cap, w + kv + ao) * 1.02;
      $("#me-bar", root).innerHTML = `<span class="s-w" style="width:${w / scale * 100}%">${w / scale > .12 ? gb(w) : ""}</span><span class="s-kv" style="width:${kv / scale * 100}%">${kv / scale > .12 ? gb(kv) : ""}</span><span class="s-act" style="width:${ao / scale * 100}%"></span><span class="s-free" style="flex:1"></span>
        <div class="cap-line" style="left:${r.usable / scale * 100}%"></div>${scale > cap * 1.03 ? `<div class="cap-line" style="left:${cap / scale * 100}%;background:var(--ink)"></div>` : ""}`;
      const fits = r.perGpu <= r.usable;
      $("#me-fit", root).innerHTML = i < 4 ? "–" : fits ? `<span class="c-ok">fits</span>` : `<span class="c-bad">does not fit</span>`;
      $("#me-maxu", root).textContent = i < 4 ? "–" : r.maxU.toLocaleString("en-US");
      $("#me-maxc", root).textContent = i < 4 ? "–" : r.maxC > 0 ? `${r.maxC.toLocaleString("en-US")} tokens` : "0 (weights don't fit)";
      const caps = [
        ["Step 1 · weights", `The weights are a fixed cost: every parameter takes a number of bytes set by the precision. ${r.G > 1 ? `With ${r.G} GPUs in tensor parallel, each GPU holds about 1/${r.G} of them.` : ""}`],
        ["Step 2 · KV cache per token", `Every token in every request stores one Key and one Value vector in every layer. ${r.m.name} uses ${r.m.attn} with ${r.m.kvH} KV heads, which sets the cost per token.`],
        ["Step 3 · KV cache total", "Multiply by how many tokens each request keeps (prompt + output) and by how many requests run at the same time. This is the part that grows with traffic."],
        ["Step 4 · activations and overhead", "Add working space for the current forward pass plus CUDA and framework overhead, paid on every GPU. It's small next to weights and KV cache, but not zero."],
        ["Step 5 · compare with the GPU", fits
          ? `Fits with ${gb(r.usable - r.perGpu)} spare per GPU. At this context you could run up to ${r.maxU.toLocaleString("en-US")} requests at once.`
          : r.maxC > 0 ? `Too much. Options: fewer requests (max ${r.maxU.toLocaleString("en-US")} here), shorter context, an FP8 KV cache, lower-precision weights, or more GPUs.` : "The weights alone don't fit. Use more GPUs, bigger GPUs, or lower-precision weights."],
      ];
      $("#me-cap", root).innerHTML = `<span class="cap-step">${caps[i][0]}</span><span>${caps[i][1]}</span>`;
    };
    const pl = bindPlayer(root, "me-pl", ctx, { frames: 5, render, interval: 3200 });
    pl.show(4);
    const rerender = () => render(frame);
    [["me-model", "model", v => v], ["me-wp", "wp", Number], ["me-kp", "kp", Number], ["me-gm", "gm", Number], ["me-g", "g", Number]]
      .forEach(([id, k, conv]) => bindSeg(root, id, v => { S[k] = conv(v); rerender(); }));
    bindSliders(root, { "me-ctx": v => ctxs[v].toLocaleString("en-US"), "me-users": v => v }, v => { vals = v; rerender(); });
  },
});
