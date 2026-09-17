/* ============================================================
   PART 3 — Inference Performance (topics 10–18)
   ============================================================ */
part({
  num: 3, title: "Inference Performance",
  blurb: "FLOPs measure computation, bandwidth measures data movement, and arithmetic intensity connects them. This is where prefill and decode part ways.",
  story: "FLOPs measure computation → bandwidth measures data movement → <b>arithmetic intensity</b> connects them → prefill tends toward compute-bound → <b>low-batch decode tends toward memory-bandwidth-bound</b>.",
});

/* ---------- shared simplified performance model ----------
   Bottleneck model: a pass takes as long as the slower of
   (a) computing its FLOPs and (b) moving its bytes.
   Deliberately simple, for teaching only. */
const flopsPerToken = (m, ctxLen) => 2 * m.N + 2 * m.L * ctxLen * m.d;
function decodeStep(m, bpp, B, ctxLen, gpu = DEMO, kvb = 2) {
  const weights = m.N * bpp, kv = kvBytesPerToken(m, kvb) * ctxLen * B;
  const bytes = weights + kv, flops = B * flopsPerToken(m, ctxLen);
  const tm = bytes / (gpu.bwGBs * 1e9), tc = flops / gpu.flops, T = Math.max(tm, tc);
  return { weights, kv, bytes, flops, tm, tc, T, tps: B / T, perUser: 1 / T, ai: flops / bytes, mfu: tc / T, mbu: tm / T };
}
function prefillPass(m, bpp, P, B, gpu = DEMO) {
  const weights = m.N * bpp;
  const flops = B * (2 * m.N * P + m.L * m.d * P * P);
  const bytes = weights;
  const tm = bytes / (gpu.bwGBs * 1e9), tc = flops / gpu.flops, T = Math.max(tm, tc);
  return { weights, bytes, flops, tm, tc, T, ai: flops / bytes, mfu: tc / T, mbu: tm / T };
}
function boundPill(tc, tm) {
  if (tm > 1.5 * tc) return `<span class="pill memory">memory-bandwidth-bound</span>`;
  if (tc > 1.5 * tm) return `<span class="pill compute">compute-bound</span>`;
  return `<span class="pill mixed">near the crossover</span>`;
}

/* ---------- Topic 10 ---------- */
topic({
  num: 10, title: "FLOPs and FLOPS",
  question: "How do we count “how much maths” the model needs?",
  render: () => `
    ${say(
      "A <b class='c-compute'>FLOP</b> is one floating-point operation: one multiply or one add of two decimal numbers. Count them and you know <b>how much maths</b> a job needs.",
      "<b class='c-compute'>FLOPS</b> (capital S) is a <b>speed</b>: floating-point operations <b>per second</b>. It says how fast hardware can do that maths."
    )}
    ${see("How do multiplies and adds pile up in C = A × B?", `
      <div class="controls">${slider("t10-n", "Matrix size n (n × n)", 2, 5, 1, 3)}</div>
      <div class="scroll-x" id="t10-mats"></div>
      <div class="stat-row">
        <div class="stat"><span class="v c-compute" id="t10-mul">0</span><span class="k">multiplies</span></div>
        <div class="stat"><span class="v c-compute" id="t10-add">0</span><span class="k">adds</span></div>
        <div class="stat"><span class="v" id="t10-tot">0</span><span class="k">total FLOPs so far</span></div>
        <div class="stat"><span class="v" id="t10-formula">–</span><span class="k">formula for full result</span></div>
      </div>
      <div class="btn-row"><button type="button" class="btn primary" id="t10-play">▶ Compute C</button><button type="button" class="btn" id="t10-fast">Finish instantly</button></div>`)}
    ${ex(`<div class="calc">3 × 3 matrices: each output number = 3 multiplies + 2 adds = 5 FLOPs<br>
      9 output numbers × 5 = <b>45 FLOPs</b> (≈ 2n³ = 54 for large n)<br>
      One token through one 8,192 × 8,192 projection: 8,192 × 8,192 × 2 ≈ <b>134 million FLOPs</b></div>`)}
    ${tech(`<ul>
      <li><b>FLOPs</b> (lower-case s = plural): a <i>count</i> of floating-point operations for a job. Not a speed.</li>
      <li><b>FLOPS</b> (upper-case S = per second): a <i>rate</i>. Hardware is rated by how many FLOPs per second it can perform at best (its peak compute).</li>
      <li>Multiplying a vector of size d<sub>in</sub> by a d<sub>in</sub> × d<sub>out</sub> weight matrix takes about <b>2 × d<sub>in</sub> × d<sub>out</sub></b> FLOPs, which is <b>2 FLOPs per weight</b>.</li>
    </ul>`)}
    ${fx(["FLOPs for (vector × weight matrix) ≈ 2 × number of weights in the matrix", "Time for a job ≥ FLOPs needed ÷ FLOPS available"])}
    ${inf(`<p>A Transformer is mostly weight matrices, and every token goes through all of them. So a model with <b>N</b> parameters needs roughly <b class="c-compute">2N FLOPs for every token</b> it processes. For a 70B-class model that's about 140 billion FLOPs per token.</p>`)}
    ${check({
      q: "Which one is a speed?",
      opts: ["FLOPs: the number of operations in a forward pass", "FLOPS: operations per second", "Parameters", "Tokens"],
      a: 1, why: "FLOPs (lower-case s) is a count of work. FLOPS (per second) is a rate, used to describe how fast hardware can compute.",
    })}`,
  mount(root, ctx) {
    let n = 3, A, B, C, done = 0, timer = null;
    const rnd = () => Math.floor(Math.random() * 5) + 1;
    const reset = () => {
      A = Array.from({ length: n }, () => Array.from({ length: n }, rnd));
      B = Array.from({ length: n }, () => Array.from({ length: n }, rnd));
      C = A.map(r => B[0].map((_, j) => r.reduce((s, x, k) => s + x * B[k][j], 0)));
      done = 0; if (timer) { ctx.stop(timer); timer = null; } draw(-1);
    };
    const m = (M, name, hiR = -1, hiC = -1, showUpTo = Infinity) => `<div style="display:inline-grid;gap:3px;vertical-align:middle"><div class="mini-label" style="text-align:center">${name}</div>
      <div style="display:grid;grid-template-columns:repeat(${n},38px);gap:3px">${M.flat().map((v, i) => {
        const r = Math.floor(i / n), c = i % n, on = r === hiR || c === hiC;
        const hidden = name === "C" && i >= showUpTo;
        return `<span class="tok ${on ? "lit" : name === "C" && !hidden ? "gen" : ""}" style="text-align:center;padding:3px 0">${hidden ? "·" : v}</span>`;
      }).join("")}</div></div>`;
    const draw = (cell) => {
      const r = cell >= 0 ? Math.floor(cell / n) : -1, c = cell >= 0 ? cell % n : -1;
      $("#t10-mats", root).innerHTML = `<div style="display:flex;gap:10px;align-items:center;font-family:var(--f-mono);white-space:nowrap">
        ${m(A, "A", r, -1)}<span>×</span>${m(B, "B", -1, c)}<span>=</span>${m(C, "C", -1, -1, done)}</div>
        ${cell >= 0 ? `<p class="mono" style="font-size:.82rem;margin-top:6px">C[${r}][${c}] = ${A[r].map((x, k) => `${x}×${B[k][c]}`).join(" + ")} = ${C[r][c]}</p>` : ""}`;
      $("#t10-mul", root).textContent = done * n;
      $("#t10-add", root).textContent = done * (n - 1);
      $("#t10-tot", root).textContent = done * (2 * n - 1);
      $("#t10-formula", root).textContent = `${n * n}×(${n}+${n - 1}) = ${n * n * (2 * n - 1)}`;
    };
    bindSliders(root, { "t10-n": x => `${x} × ${x}` }, v => { n = v["t10-n"]; reset(); });
    $("#t10-play", root).onclick = () => {
      if (timer) return; if (done >= n * n) { done = 0; }
      timer = ctx.every(Math.max(180, 900 / n), () => {
        done++; draw(done - 1);
        if (done >= n * n) { ctx.stop(timer); timer = null; ctx.after(700, () => draw(-1)); }
      });
    };
    $("#t10-fast", root).onclick = () => { if (timer) { ctx.stop(timer); timer = null; } done = n * n; draw(-1); };
  },
});

/* ---------- Topic 11 ---------- */
topic({
  num: 11, title: "FLOP counting for a Transformer forward pass",
  question: "Where does “about 2N FLOPs per token” come from?",
  render: () => `
    ${say("We don't memorise the formula. We build it: count the FLOPs of one weight matrix, add up the matrices in one layer, add the attention comparisons, then multiply by the number of layers.")}
    ${see("Can we derive the FLOPs per token step by step?", `
      <div class="btn-row" id="t11-steps"></div>
      <div class="stage-panel" id="t11-panel"></div>`, "Derive it")}
    ${see("How does compute change with model shape and context?", `
      <div class="btn-row">${seg("t11-preset", [["8B", "8B-class"], ["70B", "70B-class"], ["405B", "405B-class"]], "70B")}<span class="c-muted" style="font-size:.82rem">presets set layers and hidden size</span></div>
      <div class="controls">
        ${slider("t11-L", "Layers (L)", 1, 126, 1, 80)}
        ${slider("t11-d", "Hidden size (d)", 0, 7, 1, 5)}
        ${slider("t11-n", "Sequence length (n)", 0, 7, 1, 3)}
      </div>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t11-N">–</span><span class="k">parameters ≈ 12·L·d²</span></div>
        <div class="stat"><span class="v c-compute" id="t11-tok">–</span><span class="k">FLOPs per token</span></div>
        <div class="stat"><span class="v c-compute" id="t11-pre">–</span><span class="k">prefill FLOPs for all n tokens</span></div>
        <div class="stat"><span class="v" id="t11-att">–</span><span class="k">share from attention comparisons</span></div>
      </div>
      <div><div class="mini-label" style="margin-bottom:4px">FLOPs per token, split by where they come from</div>
        <div class="stack" id="t11-bar"></div></div>
      <p class="c-muted" style="font-size:.82rem">Ignores embeddings, norms, biases and the final vocabulary projection. Real models (with GQA or gated MLPs) differ a little. ${illus("Approximate")}</p>`, "Try it")}
    ${ex(`<div class="calc">70B-class: L = 80, d = 8,192<br>
      weights per layer: 12 × 8,192² ≈ 805M → × 80 ≈ <b>64B parameters</b><br>
      FLOPs per token ≈ 24 × 80 × 8,192² ≈ <b>129 billion</b> (+ attention term)<br>
      2,000-token prompt ≈ <b>260 trillion FLOPs</b> of prefill</div>`)}
    ${fx(["FLOPs per token ≈ 24·L·d² + 2·L·n·d", "               ≈ 2N + 2·L·n·d      (because N ≈ 12·L·d²)"], "The first term is weight multiplies. The second is attention comparing the token with n earlier tokens.")}
    ${inf(`<p>Prefill runs this for <b>every prompt token in one pass</b>: lots of FLOPs at once. Decode runs it for <b>one token per pass</b>: far fewer FLOPs per pass, but the pass still touches <b>all N weights</b>. Keep that difference in mind for the next topics.</p>`)}
    ${check({
      q: "Using the rule “about 2 FLOPs per parameter per token”, roughly how many FLOPs does an 8B model need for a 1,000-token prefill (ignoring attention)?",
      opts: ["16 billion", "16 trillion", "8 trillion", "2 thousand"],
      a: 1, why: "2 × 8B = 16 billion FLOPs per token, × 1,000 tokens = 16 trillion FLOPs.",
    })}`,
  mount(root, ctx) {
    const steps = [
      ["One matrix", `A token vector (size d) × weight matrix (d × d): each output needs d multiplies and about d adds.<div class="calc">FLOPs ≈ 2 × d × d = <b>2d²</b>  (2 FLOPs per weight)</div>`],
      ["Attention projections", `Each layer has four d × d matrices: W<sub>q</sub>, W<sub>k</sub>, W<sub>v</sub>, W<sub>o</sub>.<div class="calc">4 × 2d² = <b>8d²</b>   (weights: 4d²)</div>`],
      ["MLP", `The MLP expands d → 4d, then shrinks 4d → d. That's two matrices of 4d² weights each.<div class="calc">2 × (2 × 4d²) = <b>16d²</b>   (weights: 8d²)</div>`],
      ["Attention scores", `The token's Query is compared with the Keys of all n tokens in context (and their Values are mixed). This doesn't use weights, but it grows with context.<div class="calc">≈ <b>2·n·d</b> per layer</div>`],
      ["One layer", `Add it up.<div class="calc">8d² + 16d² + 2nd = <b>24d² + 2nd</b>   (weights per layer: 12d²)</div>`],
      ["All layers", `Multiply by the number of layers L. Since parameters N ≈ 12·L·d²:<div class="calc">FLOPs per token ≈ 24·L·d² + 2·L·n·d ≈ <b>2N + 2·L·n·d</b></div>`],
    ];
    let cur = 0;
    const show = (i) => {
      cur = i;
      $("#t11-steps", root).innerHTML = steps.map((s, j) => `<button type="button" class="btn small ${j === cur ? "on" : ""}" data-i="${j}">${j + 1}. ${s[0]}</button>`).join("") +
        (cur < steps.length - 1 ? `<button type="button" class="btn small primary" data-i="${cur + 1}">Next →</button>` : "");
      $("#t11-panel", root).innerHTML = `<div class="mini-label">Step ${i + 1} of ${steps.length}</div><h4>${steps[i][0]}</h4><div>${steps[i][1]}</div>`;
    };
    $("#t11-steps", root).onclick = e => { const b = e.target.closest("button"); if (b) show(+b.dataset.i); };
    show(0);

    const ds = [512, 1024, 2048, 4096, 5120, 8192, 12288, 16384];
    const ns = [128, 512, 1024, 2048, 4096, 8192, 32768, 131072];
    const run = bindSliders(root, {
      "t11-L": x => x, "t11-d": x => ds[x].toLocaleString("en-US"), "t11-n": x => ns[x].toLocaleString("en-US"),
    }, v => {
      const L = v["t11-L"], d = ds[v["t11-d"]], n = ns[v["t11-n"]];
      const w = 24 * L * d * d, att = 2 * L * n * d, tok = w + att;
      const pre = n * w + L * d * n * n;
      $("#t11-N", root).textContent = words(12 * L * d * d);
      $("#t11-tok", root).textContent = words(tok);
      $("#t11-pre", root).textContent = words(pre);
      $("#t11-att", root).textContent = `${fmt(att / tok * 100, 1)}%`;
      const wp = w / tok * 100;
      $("#t11-bar", root).innerHTML = `<span class="s-act" style="width:${wp}%">${wp > 25 ? "weight multiplies" : ""}</span><span class="s-kv" style="width:${100 - wp}%">${100 - wp > 18 ? "attention scores" : ""}</span>`;
    });
    bindSeg(root, "t11-preset", k => {
      const m = MODELS[k];
      $("#t11-L", root).value = m.L; $("#t11-d", root).value = ds.indexOf(m.d); run();
    });
  },
});

/* ---------- Topic 12 ---------- */
topic({
  num: 12, title: "Memory bandwidth",
  question: "How fast can weights get from HBM to the compute?",
  render: () => `
    ${say(
      "To compute a layer, the chip needs that layer's weights. For one decode step it needs the weights of <b>every</b> layer, so in effect <b class='c-memory'>the whole model is streamed from HBM to the chip for each new token</b>.",
      "The computation for one token is small. The data it needs is huge. So the pipe between HBM and the chip can become the thing everyone waits on."
    )}
    ${see("How long does the GPU spend moving weights vs computing, per token?", `
      <div class="controls">
        <div class="ctl"><label>Model (FP16)</label>${seg("t12-model", [["8B", "8B (16 GB)"], ["70B", "70B (140 GB)"]], "70B")}</div>
        ${slider("t12-bw", "HBM bandwidth (GB/s)", 200, 2000, 100, 1000)}
      </div>
      <div style="display:grid;grid-template-columns:auto 1fr auto;gap:10px;align-items:center">
        <div class="part-box" style="border-color:var(--memory);color:var(--memory);cursor:default">HBM<br><small id="t12-wlabel">140 GB weights</small></div>
        <div style="display:grid;gap:6px">
          <div class="meter memory"><div class="mh"><span>Streaming weights for the next token</span><span id="t12-pct">0%</span></div><div class="track"><i id="t12-bar" style="width:0%;transition:none"></i></div></div>
          <div class="meter compute"><div class="mh"><span>Computing (Tensor Cores busy)</span><span id="t12-cpct"></span></div><div class="track"><i id="t12-cbar" style="width:0%;transition:none"></i></div></div>
        </div>
        <div class="part-box" style="border-color:var(--compute);color:var(--compute);cursor:default">SMs<br><small>compute</small></div>
      </div>
      <div class="tokens" id="t12-out" style="min-height:32px"></div>
      <div class="stat-row">
        <div class="stat"><span class="v c-memory" id="t12-tm">–</span><span class="k">time moving weights / token</span></div>
        <div class="stat"><span class="v c-compute" id="t12-tc">–</span><span class="k">time computing / token</span></div>
        <div class="stat"><span class="v" id="t12-tps">–</span><span class="k">tokens / sec (batch 1)</span></div>
      </div>
      <p class="c-muted" style="font-size:.82rem">Animation runs in slow motion. ${illus()} GPU peak compute ${words(DEMO.flops)} FLOPS.</p>`)}
    ${ex(`<div class="calc">70B FP16, batch 1, bandwidth 1,000 GB/s ${illus()}<br>
      moving: 140 GB ÷ 1,000 GB/s = <b class="c-memory">0.14 s</b> per token<br>
      computing: 140 billion FLOPs ÷ 200 trillion FLOPS = <b class="c-compute">0.0007 s</b> per token<br>
      → the chip spends almost all its time <b>waiting for data</b> (~7 tokens/sec)</div>`)}
    ${tech(`<p><b>Memory bandwidth</b> is the rate at which data can be transferred between memory and the processor, in GB/s. For GPU memory it depends on the memory technology and how many parallel signal lines connect it to the chip; HBM exists precisely to make this number large.</p>`)}
    ${fx(["time per decode step ≳ (weight bytes + KV cache bytes) ÷ bandwidth"])}
    ${inf(`<p>During low-batch decode a huge model streams a huge amount of weight data to do a relatively small amount of maths for the next token. Faster HBM or fewer bytes (quantization) shortens the wait; more FLOPS does not.</p>`)}
    ${check({
      q: "A 70B FP16 model runs at batch 1. You double the GPU's peak FLOPS, but keep bandwidth the same. What happens to decode speed?",
      opts: ["About doubles", "Barely changes", "Halves", "The model stops fitting"],
      a: 1, why: "The chip was already mostly waiting for weights to arrive. Moving 140 GB per token takes the same time, so speed barely changes.",
    })}`,
  mount(root, ctx) {
    let model = "70B", bw = 1000, prog = 0, tokens = 0;
    const words8 = [" The", " answer", " is", " that", " memory", " moves", " data", " while", " cores", " compute", " …"];
    const recompute = () => {
      const m = MODELS[model], wGB = m.N * 2 / 1e9;
      const tm = wGB / bw, tc = 2 * m.N / DEMO.flops;
      $("#t12-wlabel", root).textContent = `${fmt(wGB, 0)} GB weights`;
      $("#t12-tm", root).textContent = `${fmt(tm * 1000, 1)} ms`;
      $("#t12-tc", root).textContent = `${fmt(tc * 1000, 2)} ms`;
      $("#t12-tps", root).textContent = `${fmt(1 / (tm + tc), 1)} tok/s`;
      $("#t12-cpct", root).textContent = `${fmt(tc / (tm + tc) * 100, 1)}% of the time`;
      return { tm, tc };
    };
    bindSliders(root, { "t12-bw": x => x.toLocaleString("en-US") }, v => { bw = v["t12-bw"]; recompute(); });
    bindSeg(root, "t12-model", k => { model = k; recompute(); });
    let flash = 0;
    ctx.loop(dt => {
      const { tm, tc } = recompute();
      const slow = 12; // slow motion factor
      prog += dt / (tm * slow);
      if (flash > 0) flash -= dt;
      if (prog >= 1) {
        prog = 0; flash = Math.max(.08, tc * slow * 30);
        const o = $("#t12-out", root);
        if (o) { if (tokens % words8.length === 0) o.innerHTML = ""; o.insertAdjacentHTML("beforeend", `<span class="tok gen new">${words8[tokens % words8.length]}</span>`); }
        tokens++;
      }
      const b = $("#t12-bar", root); if (b) b.style.width = (prog * 100) + "%";
      const p = $("#t12-pct", root); if (p) p.textContent = Math.round(prog * 100) + "%";
      const cb = $("#t12-cbar", root); if (cb) cb.style.width = flash > 0 ? "100%" : "0%";
    });
  },
});

/* ---------- Topic 13 ---------- */
topic({
  num: 13, title: "Arithmetic intensity",
  question: "Is this job mostly maths, or mostly moving data?",
  render: () => `
    ${say(
      "Some jobs do a <b>lot of maths on a little data</b>. Others do <b>a little maths on a lot of data</b>.",
      "During decode we only generate one new token, but the model still needs everything stored in its weights. So the GPU can spend more time <b class='c-memory'>moving data</b> than <b class='c-compute'>doing calculations</b>. <b>Arithmetic intensity</b> puts a number on this: how much maths you get per byte you move."
    )}
    ${see("What happens to the ratio when more tokens share one read of the weights?", `
      <div class="controls">${slider("t13-t", "Tokens processed in one pass (prompt length, or batch size in decode)", 0, 13, 1, 0)}</div>
      <div class="grid2">
        <div style="display:grid;gap:8px">
          ${meter("FLOPs done (log scale)", "compute", "t13-f")}
          ${meter("Bytes moved (log scale)", "memory", "t13-b")}
        </div>
        <div class="stat-row">
          <div class="stat"><span class="v" id="t13-ai">–</span><span class="k">arithmetic intensity (FLOPs per byte)</span></div>
          <div class="stat"><span class="v" id="t13-kind">–</span><span class="k">this workload looks like</span></div>
        </div>
      </div>
      <svg class="svg-block" viewBox="0 0 700 70" id="t13-gauge"></svg>
      <p class="c-muted" style="font-size:.82rem">70B-class model, FP16 weights, ignoring the KV cache. The dashed line is where the example GPU switches from waiting on memory to waiting on compute (200 FLOPs/byte). ${illus()}</p>`)}
    ${ex(`<div class="calc">Decode, batch 1: ~140 billion FLOPs ÷ ~140 GB moved ≈ <b class="c-memory">1 FLOP per byte</b><br>
      Prefill, 2,000 tokens: ~280 trillion FLOPs ÷ ~140 GB ≈ <b class="c-compute">2,000 FLOPs per byte</b><br>
      Toy Y = X·W from topic 8: 12 FLOPs ÷ 16 bytes = 0.75</div>`)}
    ${tech(`<p><b>Arithmetic intensity</b> (also called operational intensity) is the ratio of computation to data movement for a workload. It's a property of the <i>workload</i>, not the hardware. Comparing it with the hardware's ratio of compute to bandwidth tells you which will run out first.</p>`)}
    ${fx(["Arithmetic intensity = FLOPs ÷ bytes moved", "Hardware crossover ≈ peak FLOPS ÷ bandwidth (bytes/s)"], "Below the crossover: memory runs out first. Above it: compute runs out first.")}
    ${inf(`<ul><li><b>High intensity</b> (prefill of long prompts, large batches): lots of maths per byte → the chip's compute is the busy part.</li>
      <li><b>Low intensity</b> (decode at batch 1): lots of data movement per unit of maths → the memory pipe is the busy part.</li></ul>`)}
    ${check({
      q: "Decode at batch 1 does roughly 2N FLOPs and moves roughly 2N bytes (FP16). What is its arithmetic intensity?",
      opts: ["About 1 FLOP per byte (low)", "About 1,000 FLOPs per byte (high)", "Zero", "It depends only on the GPU"],
      a: 0, why: "2N FLOPs ÷ 2N bytes ≈ 1. That's very low: almost no maths per byte moved, which is why low-batch decode waits on memory.",
    })}`,
  mount(root, ctx) {
    const m = MODELS["70B"], bytes = m.N * 2, knee = DEMO.flops / (DEMO.bwGBs * 1e9);
    bindSliders(root, { "t13-t": x => Math.pow(2, x).toLocaleString("en-US") }, v => {
      const T = Math.pow(2, v["t13-t"]), flops = 2 * m.N * T, ai = flops / bytes;
      const lf = Math.log10(flops), lb = Math.log10(bytes);
      setMeter(root, "t13-f", (lf - 10) / 6, words(flops));
      setMeter(root, "t13-b", (lb - 10) / 6, gb(bytes));
      $("#t13-ai", root).textContent = fmt(ai, 1);
      $("#t13-kind", root).innerHTML = ai < knee / 2 ? `<span class="pill memory">data-movement heavy</span>` : ai > knee * 2 ? `<span class="pill compute">compute heavy</span>` : `<span class="pill mixed">balanced</span>`;
      const x = (a) => 40 + (Math.log10(a) + 1) / 5 * 620;
      $("#t13-gauge", root).innerHTML = `
        <defs><linearGradient id="t13g" x1="0" x2="1"><stop offset="0" stop-color="var(--memory)"/><stop offset="1" stop-color="var(--compute)"/></linearGradient></defs>
        <rect x="40" y="22" width="620" height="12" rx="6" fill="url(#t13g)" opacity=".8"/>
        <line x1="${x(knee)}" x2="${x(knee)}" y1="14" y2="42" stroke="var(--ink)" stroke-dasharray="3 3"/>
        ${[0.1, 1, 10, 100, 1000, 10000].map(a => `<text x="${x(a)}" y="58" text-anchor="middle" font-size="11" class="muted">${a}</text>`).join("")}
        <text x="40" y="12" font-size="11" class="muted">← moving data</text><text x="660" y="12" font-size="11" text-anchor="end" class="muted">doing maths →</text>
        <circle cx="${x(clamp(ai, .1, 1e4))}" cy="28" r="9" fill="var(--surface)" stroke="var(--ink)" stroke-width="3"/>`;
    });
  },
});

/* ---------- Topic 14 ---------- */
topic({
  num: 14, title: "Compute-bound vs memory-bound",
  question: "When the GPU is slow, what is it waiting for?",
  render: () => `
    ${say(
      "Every job needs two things: <b class='c-compute'>work capacity</b> (something that processes) and a <b class='c-memory'>supply of inputs</b> (the data to process). Whichever runs out first sets the speed.",
      "Picture a production line. <b class='c-compute'>Machines</b> process parts (compute). A <b class='c-memory'>conveyor</b> brings parts from the warehouse (memory movement). If the machines are busy non-stop, the line is <b class='c-compute'>compute-bound</b>. If machines stand idle waiting for parts, it is <b class='c-memory'>memory-bound</b>, and faster machines won't help."
    )}
    ${see("Is the line waiting on machines or on parts?", `
      <div class="controls">
        ${slider("t14-chef", "Machine capacity (units/min)", 1, 20, 1, 12)}
        ${slider("t14-del", "Parts delivered (enough for units/min)", 1, 20, 1, 4)}
      </div>
      <svg class="svg-block" viewBox="0 0 700 170" id="t14-svg"></svg>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t14-out">–</span><span class="k">units produced / min</span></div>
        <div class="stat"><span class="v" id="t14-util">–</span><span class="k">machine capacity in use</span></div>
        <div class="stat"><span class="v" id="t14-bound">–</span><span class="k">the line is</span></div>
      </div>
      <div class="btn-row">${seg("t14-map", [["kitchen", "Factory words"], ["gpu", "GPU words"]], "kitchen")}</div>
      <div class="scroll-x"><table class="cmp-table">
        <tr><th>Production line</th><th>GPU</th></tr>
        <tr><td>Machines processing parts</td><td><span class="c-compute">Tensor Cores / CUDA cores doing FLOPs</span></td></tr>
        <tr><td>Conveyor bringing parts from the warehouse</td><td><span class="c-memory">Weights and KV cache moving from HBM (bandwidth)</span></td></tr>
        <tr><td>How much processing each crate of parts needs</td><td>The workload's arithmetic intensity</td></tr>
        <tr><td>Units produced</td><td>Tokens produced</td></tr>
      </table></div>`)}
    ${ex(`<div class="calc">Machines can process 12 units/min, but the conveyor only brings parts for 4 units/min<br>
      → 4 units/min produced, machines busy 33% of the time → <b class="c-memory">memory-bound</b><br>
      Double the machines → still 4 units/min. Double the conveyor → 8 units/min.</div>`)}
    ${tech(`<ul>
      <li><b>Compute-bound:</b> performance is limited mainly by the processor's compute capability (FLOPS). Faster compute → faster job.</li>
      <li><b>Memory-bound</b> (memory-bandwidth-bound): performance is limited mainly by how fast data can be moved to and from the processor. Faster memory access → faster job; faster compute does little.</li>
    </ul>`)}
    ${fx(["throughput ≈ min( compute capacity , data-delivery capacity )"], "Simplified bottleneck model: the slower side sets the pace.")}
    ${inf(`<p>The same GPU can be compute-bound for one phase and memory-bound for another. Next: why prefill leans compute-bound and low-batch decode leans memory-bandwidth-bound.</p>`)}
    ${check({
      q: "A GPU's Tensor Cores are idle half the time, waiting for weights to arrive from HBM. What speeds it up?",
      opts: ["A GPU with more Tensor Cores", "Faster memory (more bandwidth) or fewer bytes to move", "A faster CPU", "More disk space"],
      a: 1, why: "The limit is data delivery, not computation. In GPU terms: a memory-bound workload improves with more bandwidth or fewer bytes to move, not more FLOPS.",
    })}`,
  mount(root, ctx) {
    let chef = 12, del = 4, gpuWords = false, phase = 0;
    const svg = $("#t14-svg", root);
    const draw = () => {
      const out = Math.min(chef, del), util = out / chef;
      const nChefs = 8, busy = Math.round(util * nChefs);
      const L = gpuWords ? { d: "HBM → chip (bandwidth)", c: "Tensor Cores", s: "tokens out" } : { d: "Parts on the conveyor", c: "Machines", s: "units out" };
      const crates = Math.round(del / 20 * 10);
      let s = `<text x="20" y="20" font-size="12" font-weight="600">${L.d}</text>`;
      for (let i = 0; i < 10; i++) {
        const x = 20 + ((i * 18 + phase * 40 * del / 10) % 180);
        if (i < crates) s += `<rect x="${x}" y="40" width="14" height="14" rx="2" fill="var(--memory)"/>`;
      }
      s += `<rect x="20" y="58" width="190" height="4" fill="var(--line)"/>
        <text x="240" y="20" font-size="12" font-weight="600">${L.c} (${busy} of ${nChefs} busy)</text>`;
      for (let i = 0; i < nChefs; i++) {
        const cx = 255 + (i % 4) * 58, cy = 55 + Math.floor(i / 4) * 62, on = i < busy;
        const bob = on ? Math.sin(phase * 8 + i) * 3 : 0;
        s += `<circle cx="${cx}" cy="${cy + bob}" r="18" fill="${on ? "var(--compute)" : "var(--surface-3)"}"/>
          <text x="${cx}" y="${cy + 4 + bob}" text-anchor="middle" font-size="10" font-weight="700" style="fill:${on ? "#fff" : "var(--muted)"}">${on ? (gpuWords ? "FLOP" : "work") : "wait"}</text>`;
      }
      s += `<text x="520" y="20" font-size="12" font-weight="600">${L.s}</text>
        <rect x="520" y="40" width="160" height="16" rx="8" fill="var(--surface-3)"/><rect x="520" y="40" width="${out / 20 * 160}" height="16" rx="8" fill="var(--ok)"/>
        <text x="600" y="80" text-anchor="middle" font-size="20" font-weight="700">${out} / min</text>`;
      svg.innerHTML = s;
      $("#t14-out", root).textContent = out;
      $("#t14-util", root).textContent = `${Math.round(util * 100)}%`;
      $("#t14-bound", root).innerHTML = chef < del ? `<span class="pill compute">compute-bound</span>` : chef > del ? `<span class="pill memory">memory-bound</span>` : `<span class="pill mixed">balanced</span>`;
    };
    bindSliders(root, { "t14-chef": x => x, "t14-del": x => x }, v => { chef = v["t14-chef"]; del = v["t14-del"]; draw(); });
    bindSeg(root, "t14-map", k => { gpuWords = k === "gpu"; draw(); });
    ctx.loop(dt => { phase += dt; draw(); });
  },
});

/* ---------- Topic 15 ---------- */
topic({
  num: 15, title: "Why prefill is generally compute-bound",
  question: "Why does a long prompt keep the Tensor Cores busy?",
  render: () => `
    ${say("In prefill, the weights are read once for the pass, but <b>every prompt token</b> gets multiplied by them. A 2,000-token prompt does 2,000 tokens' worth of maths for about one model's worth of data movement. Lots of maths per byte: the chefs are the busy ones.")}
    ${see("What happens when many tokens enter the GPU at once?", `
      ${flow([{ t: "Many input tokens" }, { t: "Large matrix operations", c: "compute" }, { t: "Lots of computation", c: "compute" }, { t: "Tensor Cores heavily used", c: "compute" }, { t: "High arithmetic intensity", c: "compute" }, { t: "Compute-bound tendency", c: "compute" }], { id: "t15-flow" })}
      <div class="controls">${slider("t15-p", "Prompt length (tokens)", 0, 13, 1, 11)}</div>
      <div style="display:grid;grid-template-columns:minmax(0,1fr) 160px;gap:14px;align-items:center">
        <div id="t15-tokens" style="display:flex;flex-wrap:wrap;gap:2px;min-height:60px;align-content:flex-start"></div>
        <div class="model-box busy" id="t15-gpu">GPU<br><small>one prefill pass</small></div>
      </div>
      <div class="grid2">
        <div style="display:grid;gap:8px">${meter("Tensor Core utilization", "compute", "t15-cu")}${meter("HBM bandwidth utilization", "memory", "t15-mu")}</div>
        <div class="stat-row">
          <div class="stat"><span class="v" id="t15-ai">–</span><span class="k">FLOPs per byte</span></div>
          <div class="stat"><span class="v" id="t15-t">–</span><span class="k">prefill time (TTFT part)</span></div>
          <div class="stat"><span class="v" id="t15-b">–</span><span class="k">diagnosis</span></div>
        </div>
      </div>
      <p class="c-muted" style="font-size:.82rem">8B-class model, FP16, on the example GPU (peak ${words(DEMO.flops)} FLOPS, ${DEMO.bwGBs.toLocaleString("en-US")} GB/s). Simplified bottleneck model. ${illus()}</p>`)}
    ${ex(`<div class="calc">8B FP16, 2,048-token prompt ${illus()}<br>
      FLOPs ≈ 2 × 8B × 2,048 ≈ 33 trillion (+ attention) → ≈ 0.17 s of compute<br>
      Bytes ≈ 16 GB of weights → 0.016 s of data movement<br>
      Compute takes ~10× longer → <b class="c-compute">compute-bound</b></div>`)}
    ${tech(`<p>Prefill processes all prompt tokens as one batched matrix operation per layer (plus attention over the prompt). FLOPs grow with prompt length, while the weight bytes read stay roughly fixed, so arithmetic intensity grows roughly in proportion to prompt length. Once it passes the hardware's crossover point, peak FLOPS becomes the limit.</p>`)}
    ${inf(`<p>That's why TTFT for long prompts improves with <b class="c-compute">more compute</b> (a faster GPU, FP8 kernels, splitting the work across GPUs). Very <b>short</b> prompts are an exception: with only a handful of tokens they can still sit on the memory side.</p>`)}
    ${check({
      q: "You make a prompt 4× longer. For prefill, what mostly grows?",
      opts: ["Bytes of weights read", "FLOPs (computation)", "Model size", "Nothing changes"],
      a: 1, why: "The same weights are used for every token, so the weight bytes stay about the same. The number of token × weight multiplies grows about 4× (more with attention).",
    })}`,
  mount(root, ctx) {
    const m = MODELS["8B"]; let k = 0;
    ctx.every(700, () => setFlow($("#t15-flow", root), k++ % 6));
    const draw = (P) => {
      const r = prefillPass(m, 2, P, 1);
      setMeter(root, "t15-cu", r.mfu); setMeter(root, "t15-mu", r.mbu);
      $("#t15-ai", root).textContent = fmt(r.ai, 1);
      $("#t15-t", root).textContent = r.T < 1 ? `${fmt(r.T * 1000, 1)} ms` : `${fmt(r.T, 2)} s`;
      $("#t15-b", root).innerHTML = boundPill(r.tc, r.tm);
      const shown = Math.min(P, 400);
      $("#t15-tokens", root).innerHTML = Array.from({ length: shown }, (_, i) =>
        `<i style="width:9px;height:9px;border-radius:2px;background:var(--accent);opacity:.85;animation:pop .5s ${i * 0.8 / shown}s both"></i>`).join("") +
        (P > shown ? `<span class="c-muted" style="font-size:.78rem;align-self:center"> +${(P - shown).toLocaleString("en-US")} more</span>` : "");
    };
    bindSliders(root, { "t15-p": x => Math.pow(2, x).toLocaleString("en-US") }, v => draw(Math.pow(2, v["t15-p"])));
  },
});

/* ---------- Topic 16 ---------- */
topic({
  num: 16, title: "Why low-batch decode is memory-bandwidth-bound",
  question: "Why does decode wait on HBM, and why does batching help?",
  render: () => `
    ${say(
      "To write <b>one</b> new token, the GPU must run it through <b>every</b> layer, which means reading essentially <b>all the weights</b> (plus that request's KV cache) from HBM.",
      "The maths for that one token is tiny compared with the data it needs. So the Tensor Cores finish quickly and then <b>wait</b> for the next load of weights. The memory pipe sets the pace."
    )}
    ${see("Why does decode wait on HBM?", `
      <div class="grid2" style="grid-template-columns:minmax(220px,300px) 1fr">
        ${flow([{ t: "One new token", c: "kv" }, { t: "Needs model weights (all layers)", c: "memory" }, { t: "Weights reside in HBM", c: "memory" }, { t: "Large amount of data movement", c: "memory" }, { t: "Little computation per byte", c: "compute" }, { t: "Tensor Cores may be underutilized", c: "compute" }, { t: "Memory bandwidth becomes the bottleneck", c: "memory" }], { id: "t16-flow", vertical: true })}
        <div style="display:grid;gap:12px;align-content:start">
          <div class="stage-panel" id="t16-cap" style="min-height:90px"></div>
          ${big("Decode throughput at batch 1 is <b>not</b> mainly set by how many FLOPs the GPU could theoretically perform. It is often limited by <b>how quickly the GPU can move the required model data from memory to the compute hardware</b>.", "Most important idea")}
          ${note("This is a tendency, not a law: it applies most strongly to <b>large dense models at low batch size</b>. Small models, huge batches, or very long contexts shift the balance.", "Accuracy")}
        </div>
      </div>`)}
    ${see("Why does batching help?", `
      <div class="controls">
        <div class="ctl"><label>Model (FP16)</label>${seg("t16-model", [["8B", "8B-class"], ["70B", "70B-class*"]], "8B")}</div>
        ${slider("t16-b", "Batch size (users decoding together)", 0, 8, 1, 0)}
        ${slider("t16-ctx", "Context per user (tokens)", 0, 4, 1, 2)}
      </div>
      <div class="grid2">
        <div style="display:grid;gap:8px">
          ${meter("Tensor Core utilization (MFU-like)", "compute", "t16-cu")}
          ${meter("Bandwidth utilization (MBU-like)", "memory", "t16-mu")}
          <div class="mini-label">Bytes read per step</div>
          <div class="stack" id="t16-stack"></div>
        </div>
        <div class="stat-row">
          <div class="stat"><span class="v" id="t16-tps">–</span><span class="k">total tokens / sec</span></div>
          <div class="stat"><span class="v" id="t16-pu">–</span><span class="k">tokens / sec per user</span></div>
          <div class="stat"><span class="v" id="t16-ai">–</span><span class="k">FLOPs per byte</span></div>
          <div class="stat"><span class="v" id="t16-bd">–</span><span class="k">diagnosis</span></div>
        </div>
      </div>
      <svg class="svg-block" viewBox="0 0 700 200" id="t16-chart"></svg>
      <p class="c-muted" style="font-size:.82rem">Example GPU (peak ${words(DEMO.flops)} FLOPS, ${DEMO.bwGBs.toLocaleString("en-US")} GB/s). *We pretend the 70B-class model fits, to show the maths. Red area = KV cache no longer fits in 80 GB. ${illus("Simplified model · illustrative")}</p>`, "Try batching")}
    ${ex(`<div class="calc">8B FP16, 2,048-token context ${illus()}<br>
      batch 1: read ≈ 16.3 GB/step → ≈ <b>61 tok/s</b>, Tensor Cores under 1% busy<br>
      batch 32: read ≈ 16 GB weights (shared!) + 8.6 GB KV → ≈ <b>1,300 tok/s total</b><br>
      The weights are read <b>once per step for everyone</b>, so batching spreads that cost across users.</div>`)}
    ${tech(`<ul>
      <li>Per decode step: bytes moved ≈ weight bytes + (KV cache bytes per token × context × batch); FLOPs ≈ batch × (2N + 2·L·context·d).</li>
      <li>At batch 1, FLOPs/byte ≈ 1, far below the hardware crossover (peak FLOPS ÷ bandwidth), so bandwidth limits speed.</li>
      <li>Batching multiplies FLOPs by B while weight bytes stay fixed, raising arithmetic intensity and throughput. Per-user speed slowly drops as KV reads and compute grow.</li>
    </ul>`)}
    ${inf(`<p>This single idea explains a lot of real serving practice: <b>batching</b> (and continuous batching), <b>quantizing weights</b> (fewer bytes to move), <b>GQA/MQA</b> (smaller KV cache), and why GPUs with <b>faster HBM</b> speed up decode even when compute is unchanged.</p>`)}
    ${check({
      q: "Why does batching 32 users together raise total decode throughput so much?",
      opts: ["Each user's weights are read separately, but faster", "The weights are read once per step and shared by all 32 tokens, so compute does more work per byte moved", "Batching reduces the model size", "Batching skips the KV cache"],
      a: 1, why: "All requests in a batch use the same weights. One read of the weights now feeds 32 tokens of maths, so arithmetic intensity and throughput go up.",
    })}`,
  mount(root, ctx) {
    const caps = [
      "Decode needs <b>one</b> new token.",
      "…but to produce it, the token passes through <b>every layer</b>, and each layer needs its weight matrices.",
      "Those weights live in <b class='c-memory'>HBM</b>, not on the chip.",
      "So for each token, a model-sized amount of data (140 GB for 70B FP16) has to flow toward the chip.",
      "The maths for one token is small: about 1 FLOP for every byte moved.",
      "The Tensor Cores finish their share quickly and <b>sit idle</b> between deliveries.",
      "Result: the speed of the <b class='c-memory'>HBM → compute</b> path sets tokens per second.",
    ];
    let k = 0;
    const tick = () => { const i = k++ % caps.length; setFlow($("#t16-flow", root), i); $("#t16-cap", root).innerHTML = `<div class="mini-label">Step ${i + 1} of ${caps.length}</div><p>${caps[i]}</p>`; };
    tick(); ctx.every(2300, tick);

    const ctxs = [512, 1024, 2048, 4096, 8192];
    let model = "8B";
    const chart = (m, cl, Bcur) => {
      const pts = [];
      for (let e = 0; e <= 8; e += .25) { const B = Math.pow(2, e); const r = decodeStep(m, 2, B, cl); pts.push({ e, tps: r.tps, fits: r.bytes <= DEMO.hbmGB * 1e9 }); }
      const maxT = Math.max(...pts.map(p => p.tps)) * 1.1;
      const X = e => 60 + e / 8 * 610, Y = t => 170 - t / maxT * 150;
      const firstBad = pts.find(p => !p.fits);
      let s = `<line x1="60" y1="170" x2="670" y2="170" stroke="var(--line)"/><line x1="60" y1="20" x2="60" y2="170" stroke="var(--line)"/>`;
      if (firstBad && model === "8B") s += `<rect x="${X(firstBad.e)}" y="20" width="${670 - X(firstBad.e)}" height="150" fill="var(--bad-soft)"/><text x="664" y="34" text-anchor="end" font-size="11" style="fill:var(--bad)">KV cache doesn't fit →</text>`;
      for (let e = 0; e <= 8; e++) s += `<text x="${X(e)}" y="186" text-anchor="middle" font-size="11" class="muted">${Math.pow(2, e)}</text>`;
      [0, .5, 1].forEach(f => s += `<text x="54" y="${Y(maxT * f / 1.1) + 4}" text-anchor="end" font-size="10" class="muted">${fmt(maxT * f / 1.1, 0)}</text>`);
      s += `<text x="365" y="200" text-anchor="middle" font-size="11" class="muted">batch size</text><text x="64" y="14" font-size="11" class="muted">total tokens/sec</text>`;
      s += `<polyline fill="none" stroke="var(--accent)" stroke-width="2.5" points="${pts.map(p => `${X(p.e)},${Y(p.tps)}`).join(" ")}"/>`;
      const rc = decodeStep(m, 2, Bcur, cl);
      s += `<circle cx="${X(Math.log2(Bcur))}" cy="${Y(rc.tps)}" r="7" fill="var(--compute)" stroke="var(--surface)" stroke-width="2"/>`;
      $("#t16-chart", root).innerHTML = s;
    };
    const run = bindSliders(root, { "t16-b": x => Math.pow(2, x), "t16-ctx": x => ctxs[x].toLocaleString("en-US") }, v => {
      const m = MODELS[model], B = Math.pow(2, v["t16-b"]), cl = ctxs[v["t16-ctx"]];
      const r = decodeStep(m, 2, B, cl);
      setMeter(root, "t16-cu", r.mfu, `${fmt(r.mfu * 100, 1)}%`); setMeter(root, "t16-mu", r.mbu);
      $("#t16-tps", root).textContent = fmt(r.tps, 0);
      $("#t16-pu", root).textContent = fmt(r.perUser, 1);
      $("#t16-ai", root).textContent = fmt(r.ai, 1);
      $("#t16-bd", root).innerHTML = boundPill(r.tc, r.tm);
      const wp = r.weights / r.bytes * 100;
      $("#t16-stack", root).innerHTML = `<span class="s-w" style="width:${wp}%">${wp > 22 ? `weights ${gb(r.weights)}` : ""}</span><span class="s-kv" style="width:${100 - wp}%">${100 - wp > 22 ? `KV ${gb(r.kv)}` : ""}</span>`;
      chart(m, cl, B);
    });
    bindSeg(root, "t16-model", k2 => { model = k2; run(); });
  },
});

/* ---------- Topic 17 ---------- */
topic({
  num: 17, title: "The roofline model",
  question: "Can one picture show both limits at once?",
  render: () => `
    ${say(
      "Draw a house roof. The <b class='c-memory'>slanted part</b> is the memory limit: the more maths a workload does per byte, the higher it can reach. The <b class='c-compute'>flat part</b> is the compute limit: no workload can go above the chip's peak FLOPS.",
      "Put a workload on the x-axis by its arithmetic intensity. Wherever it lands, the roof above it is its best possible performance. Under the slant: memory-bound. Under the flat part: compute-bound."
    )}
    ${see("Where do prefill and decode sit under the roof?", `
      <svg class="svg-block" viewBox="0 0 720 380" id="t17-svg"></svg>
      <div class="controls">
        ${slider("t17-bw", "Memory bandwidth (GB/s): tilts the slanted roof", 200, 3000, 100, 1000)}
        ${slider("t17-f", "Peak compute (trillion FLOPS): raises the flat roof", 50, 1000, 50, 200)}
      </div>
      <div class="stat-row" id="t17-stats"></div>
      <p class="c-muted" style="font-size:.82rem">Log–log axes. Workload positions use an 8B-class FP16 model with the simplified formulas from earlier topics. ${illus("Illustrative values: not a measurement of a real GPU")}</p>`)}
    ${ex(`<div class="calc">Example GPU: bandwidth 1,000 GB/s, peak 200 trillion FLOPS ${illus()}<br>
      crossover (the roof's corner) = 200 trillion ÷ 1 trillion bytes/s = <b>200 FLOPs/byte</b><br>
      decode batch 1 ≈ 1 → far left, under the slant → memory-bound<br>
      prefill 2,048 tokens ≈ 2,000+ → right of the corner → compute-bound</div>`)}
    ${tech(`<p>The roofline model plots attainable performance (FLOPS) against arithmetic intensity (FLOPs/byte) on log–log axes. Attainable performance = min(peak FLOPS, bandwidth × arithmetic intensity). The corner sits at arithmetic intensity = peak FLOPS ÷ bandwidth.</p>`)}
    ${fx(["attainable FLOPS = min( peak FLOPS , bandwidth (bytes/s) × arithmetic intensity )"])}
    ${inf(`<p>Raising the flat roof (more compute) helps prefill. It does <b>nothing</b> for a workload sitting under the slant. Raising the slant (more bandwidth) or moving the workload right (batching, quantization) is what helps low-batch decode.</p>`)}
    ${check({
      q: "A workload sits well to the left of the roof's corner. Which upgrade helps it most?",
      opts: ["Higher peak FLOPS", "Higher memory bandwidth", "More disk space", "A faster CPU"],
      a: 1, why: "Left of the corner the roof is the slanted memory line. Only more bandwidth (or raising the workload's intensity) lifts performance there.",
    })}`,
  mount(root, ctx) {
    const m = MODELS["8B"];
    const wl = [
      { n: "Decode, batch 1", r: () => decodeStep(m, 2, 1, 2048), c: "var(--memory)" },
      { n: "Decode, batch 64", r: () => decodeStep(m, 2, 64, 2048), c: "var(--kv)" },
      { n: "Prefill, 2,048 tokens", r: () => prefillPass(m, 2, 2048, 1), c: "var(--compute)" },
    ];
    bindSliders(root, { "t17-bw": x => x.toLocaleString("en-US"), "t17-f": x => x }, v => {
      const bw = v["t17-bw"] * 1e9, F = v["t17-f"] * 1e12;
      const X = a => 70 + (Math.log10(a) + 1) / 5 * 610;      // 0.1 .. 10^4
      const Y = p => 330 - (Math.log10(p) - 10) / 6 * 300;    // 1e10 .. 1e16
      const knee = F / bw;
      let s = "";
      for (let e = -1; e <= 4; e++) s += `<line x1="${X(Math.pow(10, e))}" x2="${X(Math.pow(10, e))}" y1="30" y2="330" stroke="var(--line)" stroke-dasharray="2 4"/><text x="${X(Math.pow(10, e))}" y="348" text-anchor="middle" font-size="11" class="muted">${Math.pow(10, e)}</text>`;
      for (let e = 10; e <= 16; e++) s += `<line x1="70" x2="680" y1="${Y(Math.pow(10, e))}" y2="${Y(Math.pow(10, e))}" stroke="var(--line)" stroke-dasharray="2 4"/><text x="64" y="${Y(Math.pow(10, e)) + 4}" text-anchor="end" font-size="11" class="muted">10${"⁰¹²³⁴⁵⁶⁷⁸⁹".split("")[Math.floor(e / 10)]}${"⁰¹²³⁴⁵⁶⁷⁸⁹".split("")[e % 10]}</text>`;
      s += `<text x="375" y="372" text-anchor="middle" font-size="12">Arithmetic intensity (FLOPs per byte) →</text>
        <text x="16" y="180" font-size="12" transform="rotate(-90 16 180)" text-anchor="middle">Performance (FLOPs per second) →</text>`;
      const a0 = 0.1, pa0 = bw * a0;
      const kx = X(clamp(knee, .1, 1e4)), ky = Y(F);
      s += `<polygon points="${X(a0)},${Y(Math.max(pa0, 1e10))} ${kx},${ky} ${X(1e4)},${ky} ${X(1e4)},330 ${X(a0)},330" fill="var(--accent-soft)" opacity=".6"/>`;
      s += `<line x1="${X(a0)}" y1="${Y(Math.max(pa0, 1e10))}" x2="${kx}" y2="${ky}" stroke="var(--memory)" stroke-width="4" stroke-linecap="round"/>
        <line x1="${kx}" y1="${ky}" x2="${X(1e4)}" y2="${ky}" stroke="var(--compute)" stroke-width="4" stroke-linecap="round"/>
        <text x="${(X(a0) + kx) / 2 - 10}" y="${(Y(Math.max(pa0, 1e10)) + ky) / 2 - 12}" font-size="12" font-weight="700" style="fill:var(--memory)" transform="rotate(-${Math.atan2(Y(Math.max(pa0, 1e10)) - ky, kx - X(a0)) * 57.3} ${(X(a0) + kx) / 2 - 10} ${(Y(Math.max(pa0, 1e10)) + ky) / 2 - 12})" text-anchor="middle">memory-bandwidth ceiling</text>
        <text x="${Math.min(kx + 10, 560)}" y="${ky - 10}" font-size="12" font-weight="700" style="fill:var(--compute)">compute ceiling</text>
        <line x1="${kx}" x2="${kx}" y1="${ky}" y2="330" stroke="var(--ink)" stroke-dasharray="4 3" opacity=".5"/>
        <text x="${kx + 4}" y="322" font-size="11" class="muted">corner ≈ ${fmt(knee, 0)}</text>`;
      let stats = "";
      wl.forEach((w, i) => {
        const r = w.r(), a = clamp(r.ai, .1, 1e4), perf = Math.min(F, bw * r.ai);
        const px = X(a), py = Y(perf);
        s += `<circle cx="${px}" cy="${py}" r="8" fill="${w.c}" stroke="var(--surface)" stroke-width="2"/>
          <text x="${px + (i === 2 ? -12 : 12)}" y="${py + (i === 1 ? 20 : -12)}" font-size="12" font-weight="600" text-anchor="${i === 2 ? "end" : "start"}">${w.n}</text>`;
        const memB = bw * r.ai < F;
        stats += `<div class="stat"><span class="v" style="color:${w.c}">${w.n}</span><span class="k">intensity ≈ ${fmt(r.ai, 1)} FLOPs/byte · ${memB ? "under the memory roof" : "under the compute roof"}</span></div>`;
      });
      $("#t17-svg", root).innerHTML = s;
      $("#t17-stats", root).innerHTML = stats;
    });
  },
});

/* ---------- Topic 18 ---------- */
topic({
  num: 18, title: "MFU and MBU",
  question: "Is my GPU busy doing the right thing?",
  render: () => `
    ${say(
      "<b class='c-compute'>MFU</b> asks: “Of all the maths this GPU <i>could</i> do per second, how much is my model actually doing?”",
      "<b class='c-memory'>MBU</b> asks: “Of all the data this GPU's memory <i>could</i> move per second, how much is my model actually moving?”",
      "A low MFU sounds bad, but in a memory-bound workload the Tensor Cores are <b>supposed</b> to be waiting: the real question is whether MBU is high."
    )}
    ${see("What do MFU and MBU look like for decode vs prefill?", `
      <div class="btn-row">${seg("t18-mode", [["decode", "Decode"], ["prefill", "Prefill"]], "decode")}</div>
      <div class="controls" id="t18-ctl"></div>
      <div class="grid2">
        <div style="display:grid;gap:10px">${meter("MFU: Model FLOPs Utilization", "compute", "t18-mfu")}${meter("MBU: Model Bandwidth Utilization", "memory", "t18-mbu")}</div>
        <div class="stage-panel" id="t18-diag" style="min-height:90px"></div>
      </div>
      <p class="c-muted" style="font-size:.82rem">8B-class FP16 model, 2,048-token context, example GPU (peak ${words(DEMO.flops)} FLOPS, ${DEMO.bwGBs.toLocaleString("en-US")} GB/s). Enter an observed speed and see the utilization. ${illus()}</p>`)}
    ${ex(`<div class="calc">8B FP16 decode, batch 1, observed 55 tok/s ${illus()}<br>
      MFU = 55 × ~16 billion FLOPs ÷ 200 trillion FLOPS ≈ <b class="c-compute">0.4%</b><br>
      MBU = 55 × ~16.3 GB ÷ 1,000 GB/s ≈ <b class="c-memory">90%</b><br>
      → Tensor Cores nearly idle, memory nearly maxed out: this is <b>normal and efficient</b> for this workload.</div>`)}
    ${fx(["MFU = (tokens/sec × FLOPs per token) ÷ peak FLOPS", "MBU = (decode steps/sec × bytes read per step) ÷ memory bandwidth", "bytes read per step ≈ weight bytes + KV cache bytes (whole batch)"])}
    ${tech(`<p>MFU became popular for measuring training efficiency, where workloads are compute-heavy. MBU was proposed for inference, where decode is often memory-bound. Look at both: high MBU with low MFU means the software is using the memory path well; low values for <b>both</b> suggest overhead (small batches, scheduling gaps, CPU bottlenecks, slow kernels).</p>`)}
    ${inf(`<ul><li>Prefill: expect <b>high MFU</b>, low MBU.</li><li>Low-batch decode: expect <b>low MFU</b>, high MBU.</li><li>Large-batch decode: both rise toward the middle.</li></ul>`)}
    ${check({
      q: "A batch-1 decode server shows 1% MFU and 85% MBU. What's the best reading?",
      opts: ["The server is badly broken", "It's memory-bound and using its bandwidth well; low MFU is expected", "It's compute-bound", "It needs more peak FLOPS"],
      a: 1, why: "Low-batch decode is often memory-bandwidth-bound. High MBU shows the memory path is well used. To go faster, use more bandwidth, fewer bytes, or larger batches, not more FLOPS.",
    })}`,
  mount(root, ctx) {
    const m = MODELS["8B"], cl = 2048;
    const drawControls = (mode) => {
      $("#t18-ctl", root).innerHTML = mode === "decode"
        ? slider("t18-b", "Batch size", 0, 7, 1, 0) + slider("t18-s", "Observed speed (% of the best possible)", 10, 100, 5, 90)
        : slider("t18-p", "Prompt length (tokens)", 6, 13, 1, 11) + slider("t18-s", "Observed speed (% of the best possible)", 10, 100, 5, 90);
      const spec = mode === "decode"
        ? { "t18-b": x => Math.pow(2, x), "t18-s": x => x + "%" }
        : { "t18-p": x => Math.pow(2, x).toLocaleString("en-US"), "t18-s": x => x + "%" };
      bindSliders(root, spec, v => {
        let mfu, mbu, txt;
        if (mode === "decode") {
          const B = Math.pow(2, v["t18-b"]), r = decodeStep(m, 2, B, cl), obs = r.tps * v["t18-s"] / 100;
          mfu = obs * flopsPerToken(m, cl) / DEMO.flops;
          mbu = (obs / B) * r.bytes / (DEMO.bwGBs * 1e9);
          txt = `Observed <b>${fmt(obs, 0)} tokens/sec</b> total at batch ${B}.<br>`;
        } else {
          const P = Math.pow(2, v["t18-p"]), r = prefillPass(m, 2, P, 1), obsT = r.T / (v["t18-s"] / 100);
          mfu = (r.flops / obsT) / DEMO.flops;
          mbu = (r.bytes / obsT) / (DEMO.bwGBs * 1e9);
          txt = `Observed prefill time <b>${fmt(obsT * 1000, 0)} ms</b> for ${P.toLocaleString("en-US")} tokens.<br>`;
        }
        setMeter(root, "t18-mfu", mfu, `${fmt(mfu * 100, 1)}%`);
        setMeter(root, "t18-mbu", mbu, `${fmt(mbu * 100, 1)}%`);
        const reading = mbu > mfu * 1.5 ? "Memory is the busy side → <span class='pill memory'>memory-bound</span>. Low MFU is expected here."
          : mfu > mbu * 1.5 ? "Compute is the busy side → <span class='pill compute'>compute-bound</span>. More FLOPS would help."
            : "Both are similar → <span class='pill mixed'>near the crossover</span>.";
        const slack = Math.max(mfu, mbu) < .6 ? "<br><span class='c-bad'>Neither is near 100%: time is being lost to overhead (scheduling, CPU, small batches).</span>" : "";
        $("#t18-diag", root).innerHTML = `<p>${txt}${reading}${slack}</p>`;
      });
    };
    bindSeg(root, "t18-mode", drawControls);
    drawControls("decode");
  },
});
