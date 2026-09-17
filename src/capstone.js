/* ============================================================
   ALL TOGETHER — Inference Simulator + the whole story
   ============================================================ */
part({
  num: null, title: "All Together",
  blurb: "Set up a deployment and see where the weights and KV cache go, and whether each phase waits on compute or on memory.",
  story: "The full chain, from a prompt to a token, through GPU memory and compute, and every optimization in between.",
});

topic({
  num: null, title: "Inference simulator",
  question: "For my setup, what fills the GPU, and what is it waiting on?",
  render: () => `
    ${note("A <b>simplified educational model</b>, not a production performance predictor. It uses an imaginary GPU (80 GB HBM, 1,000 GB/s bandwidth, 200 trillion FLOPS peak), the rough formulas from Part 3, and assumes tensor parallelism across GPUs with a small communication penalty. Use it to reason about <b>directions and bottlenecks</b>, not exact numbers.", "Read me first")}
    <div class="viz">
      <div class="controls">
        <div class="ctl"><label>Model size</label>${seg("cs-model", [["8B", "8B"], ["70B", "70B"], ["405B", "405B"]], "70B")}</div>
        <div class="ctl"><label>Weight precision</label>${seg("cs-prec", [["BF16", "BF16"], ["FP8", "FP8"], ["INT8", "INT8"], ["INT4", "INT4"]], "BF16")}</div>
        <div class="ctl"><label>Number of GPUs</label>${seg("cs-gpus", [["1", "1"], ["2", "2"], ["4", "4"], ["8", "8"]], "2")}</div>
        ${slider("cs-prompt", "Prompt length (tokens)", 5, 15, 1, 11)}
        ${slider("cs-out", "Output length (tokens)", 4, 12, 1, 9)}
        ${slider("cs-batch", "Batch size (concurrent requests)", 0, 8, 1, 0)}
      </div>
    </div>
    <div class="grid2">
      <div class="viz">
        <div class="viz-q">Where do the weights and KV cache live?</div>
        <div id="cs-fit"></div>
        <div class="stack" id="cs-stack" style="height:40px"></div>
        <div id="cs-gpu-row" style="display:flex;gap:6px;flex-wrap:wrap"></div>
      </div>
      <div class="viz">
        <div class="viz-q">What happens for each request?</div>
        <div class="btn-row">${seg("cs-phase", [["prefill", "Prefill"], ["decode", "Decode"]], "prefill")}</div>
        <div id="cs-flow"></div>
      </div>
    </div>
    <div class="grid2">
      <div class="viz">
        <div class="viz-q">Prefill: is it waiting on compute or memory?</div>
        <div id="cs-pre"></div>
      </div>
      <div class="viz">
        <div class="viz-q">Decode: is it waiting on compute or memory?</div>
        <div id="cs-dec"></div>
      </div>
    </div>
    <div class="grid2">
      <div class="viz"><div class="viz-q">Where do both phases sit on the roofline?</div><svg class="svg-block" viewBox="0 0 460 260" id="cs-roof"></svg></div>
      <div class="viz"><div class="viz-q">What could I try next?</div><div id="cs-tips" style="display:grid;gap:8px"></div></div>
    </div>
    ${check({
      q: "In the simulator, set 70B, BF16, 2 GPUs, batch 1. Then switch to INT4. Which phase speeds up the most, and why?",
      opts: ["Prefill, because INT4 adds compute", "Decode, because there are 4× fewer weight bytes to move per token", "Neither changes", "Both slow down"],
      a: 1, why: "Batch-1 decode is memory-bandwidth-bound, so fewer bytes per step speeds it up directly. Prefill is compute-bound and weight-only INT4 doesn't reduce its FLOPs.",
    })}`,
  mount(root, ctx) {
    const BPP = { BF16: 2, FP8: 1, INT8: 1, INT4: 0.5 };
    const EFF = { 1: 1, 2: 0.9, 4: 0.85, 8: 0.8 };
    let phase = "prefill", flowStep = 0;
    const flows = {
      prefill: [{ t: "Model weights", c: "memory" }, { t: "GPU HBM", c: "memory" }, { t: "Prompt tokens" }, { t: "Prefill (all tokens at once)", c: "compute" }, { t: "KV cache written", c: "kv" }, { t: "First token" }],
      decode: [{ t: "Last token" }, { t: "Read weights + KV cache from HBM", c: "memory" }, { t: "Tensor Cores", c: "compute" }, { t: "Next token" }, { t: "Append K/V to cache", c: "kv" }, { t: "Repeat" }],
    };
    const drawFlow = () => { $("#cs-flow", root).innerHTML = flow(flows[phase], { vertical: true, id: "cs-flow-chain" }); };
    bindSeg(root, "cs-phase", p => { phase = p; flowStep = 0; drawFlow(); });
    drawFlow();
    ctx.every(750, () => setFlow($("#cs-flow-chain", root), flowStep++ % flows[phase].length));

    let sel = { model: "70B", prec: "BF16", gpus: 2 };
    const pill = (tc, tm) => boundPill(tc, tm);
    const ms = s => s < 1 ? `${fmt(s * 1000, 0)} ms` : `${fmt(s, 2)} s`;
    const run = bindSliders(root, {
      "cs-prompt": x => Math.pow(2, x).toLocaleString("en-US"),
      "cs-out": x => Math.pow(2, x).toLocaleString("en-US"),
      "cs-batch": x => Math.pow(2, x),
    }, v => {
      const m = MODELS[sel.model], bpp = BPP[sel.prec], G = +sel.gpus;
      const P = Math.pow(2, v["cs-prompt"]), O = Math.pow(2, v["cs-out"]), B = Math.pow(2, v["cs-batch"]);
      const kvb = sel.prec === "FP8" ? 1 : 2;
      const gpu = { bwGBs: DEMO.bwGBs * G * EFF[G], flops: DEMO.flops * G * EFF[G], hbmGB: DEMO.hbmGB * G };

      // memory
      const weights = m.N * bpp * 1.05, kv = kvBytesPerToken(m, kvb) * (P + O) * B, act = 2e9 * G;
      const cap = gpu.hbmGB * 1e9, used = weights + kv + act, fits = used <= cap;
      const scale = Math.max(cap, used);
      $("#cs-stack", root).innerHTML = `<span class="s-w" style="width:${weights / scale * 100}%">${weights / scale > .15 ? `weights ${gb(weights)}` : ""}</span>
        <span class="s-kv" style="width:${kv / scale * 100}%">${kv / scale > .15 ? `KV ${gb(kv)}` : ""}</span>
        <span class="s-act" style="width:${act / scale * 100}%"></span>
        <span class="s-free" style="width:${Math.max(0, cap - used) / scale * 100}%">${(cap - used) / scale > .15 ? "free" : ""}</span>
        <div class="cap-line" style="left:calc(${cap / scale * 100}% - 2px)"></div>`;
      $("#cs-fit", root).innerHTML = `<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">${fits ? `<span class="pill ok">fits</span>` : `<span class="pill bad">does not fit</span>`}
        <span class="c-muted" style="font-size:.85rem">${gb(used)} needed of ${gb(cap)} (${G} × 80 GB). Weights ${gb(weights)} · KV cache ${gb(kv)} · activations/buffers ${gb(act)}</span></div>`;
      $("#cs-gpu-row", root).innerHTML = Array.from({ length: G }, (_, i) => `<div style="flex:1;min-width:60px;border:2px solid var(--ink);border-radius:6px;padding:4px;font-size:.7rem;text-align:center">
        GPU ${i + 1}<div style="height:6px;border-radius:3px;background:var(--surface-3);margin-top:3px;overflow:hidden"><div style="height:100%;width:${Math.min(100, used / cap * 100)}%;background:${fits ? "var(--memory)" : "var(--bad)"}"></div></div></div>`).join("");

      // prefill for the whole batch, decode at average context
      const pre = prefillPass(m, bpp, P, B, gpu);
      const dec = decodeStep(m, bpp, B, P + O / 2, gpu, kvb);
      const total = pre.T + O * dec.T;
      const warn = fits ? "" : `<p class="c-bad" style="font-size:.85rem">Doesn't fit: numbers below are hypothetical.</p>`;
      $("#cs-pre", root).innerHTML = `${warn}<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">${pill(pre.tc, pre.tm)}<span class="c-muted" style="font-size:.82rem">arithmetic intensity ≈ ${fmt(pre.ai, 1)} FLOPs/byte</span></div>
        ${meter("Compute used (MFU-like)", "compute", "cs-p-mfu")}${meter("Bandwidth used (MBU-like)", "memory", "cs-p-mbu")}
        <div class="stat-row"><div class="stat"><span class="v">${ms(pre.T)}</span><span class="k">TTFT (prefill of ${B > 1 ? `${B} prompts` : "the prompt"})</span></div>
        <div class="stat"><span class="v">${words(pre.flops)}</span><span class="k">FLOPs</span></div></div>`;
      setMeter(root, "cs-p-mfu", pre.mfu); setMeter(root, "cs-p-mbu", pre.mbu);
      $("#cs-dec", root).innerHTML = `${warn}<div style="display:flex;gap:8px;align-items:center;flex-wrap:wrap">${pill(dec.tc, dec.tm)}<span class="c-muted" style="font-size:.82rem">arithmetic intensity ≈ ${fmt(dec.ai, 1)} FLOPs/byte</span></div>
        ${meter("Compute used (MFU-like)", "compute", "cs-d-mfu")}${meter("Bandwidth used (MBU-like)", "memory", "cs-d-mbu")}
        <div class="stat-row"><div class="stat"><span class="v">${fmt(dec.perUser, 1)}</span><span class="k">tokens/sec per request</span></div>
        <div class="stat"><span class="v">${fmt(dec.tps, 0)}</span><span class="k">tokens/sec total</span></div>
        <div class="stat"><span class="v">${ms(total)}</span><span class="k">full response time</span></div></div>
        <p class="c-muted" style="font-size:.8rem">Bytes read per step: ${gb(dec.weights)} weights + ${gb(dec.kv)} KV cache.</p>`;
      setMeter(root, "cs-d-mfu", dec.mfu, `${fmt(dec.mfu * 100, 1)}%`); setMeter(root, "cs-d-mbu", dec.mbu, `${fmt(dec.mbu * 100, 1)}%`);

      // mini roofline
      const bw = gpu.bwGBs * 1e9, F = gpu.flops;
      const X = a => 50 + (Math.log10(a) + 1) / 6 * 390, Y = p => 220 - (Math.log10(p) - 10) / 7 * 200;
      const knee = F / bw;
      let s = `<line x1="50" y1="220" x2="440" y2="220" stroke="var(--line)"/><line x1="50" y1="20" x2="50" y2="220" stroke="var(--line)"/>
        <text x="245" y="250" text-anchor="middle" font-size="11" class="muted">arithmetic intensity (FLOPs/byte, log)</text>
        <text x="14" y="120" font-size="11" class="muted" transform="rotate(-90 14 120)" text-anchor="middle">FLOPS (log)</text>
        <line x1="${X(.1)}" y1="${Y(bw * .1)}" x2="${X(knee)}" y2="${Y(F)}" stroke="var(--memory)" stroke-width="3"/>
        <line x1="${X(knee)}" y1="${Y(F)}" x2="${X(1e5)}" y2="${Y(F)}" stroke="var(--compute)" stroke-width="3"/>`;
      [[pre, "prefill", "var(--compute)"], [dec, "decode", "var(--memory)"]].forEach(([r, n, c], i) => {
        const a = clamp(r.ai, .1, 1e5), p = Math.min(F, bw * r.ai);
        s += `<circle cx="${X(a)}" cy="${Y(p)}" r="7" fill="${c}" stroke="var(--surface)" stroke-width="2"/><text x="${X(a) + (a > 1e3 ? -10 : 10)}" y="${Y(p) + (i ? 18 : -10)}" font-size="11" font-weight="700" text-anchor="${a > 1e3 ? "end" : "start"}">${n}</text>`;
      });
      $("#cs-roof", root).innerHTML = s;

      // suggestions
      const tips = [];
      if (!fits) tips.push(["bad", "Doesn't fit", "Add GPUs (tensor parallelism), lower the precision, shorten context, or reduce batch size. Models with GQA and paged KV cache also help."]);
      if (dec.tm > 1.5 * dec.tc) tips.push(["memory", "Decode is memory-bandwidth-bound", "Increase batch size (continuous batching), quantize weights, shrink the KV cache (GQA, KV quantization), or use GPUs with faster HBM. More FLOPS won't help much."]);
      if (dec.tc > 1.5 * dec.tm) tips.push(["compute", "Decode is compute-bound", "The batch is large enough that compute is the limit: more or faster compute, FP8 kernels, or more GPUs."]);
      if (pre.tc > 1.5 * pre.tm) tips.push(["compute", "Prefill is compute-bound", "For lower TTFT: more compute, FP8 compute, prefix caching for shared prompts, and chunked prefill so long prompts don't stall other users."]);
      if (pre.tm > 1.5 * pre.tc) tips.push(["memory", "Prefill is on the memory side", "The prompt is short enough that reading weights dominates; prefill is already cheap."]);
      if (kv > weights && fits) tips.push(["kv", "KV cache is bigger than the weights", "Long contexts × many requests: paged KV cache, prefix caching, KV quantization, and GQA models matter a lot here."]);
      $("#cs-tips", root).innerHTML = tips.map(([c, h, t]) => `<div class="card flat"><div style="display:flex;gap:8px;align-items:center"><span class="pill ${c === "kv" ? "mixed" : c}">${h}</span></div><p>${t}</p></div>`).join("");
    });
    bindSeg(root, "cs-model", k => { sel.model = k; run(); });
    bindSeg(root, "cs-prec", k => { sel.prec = k; run(); });
    bindSeg(root, "cs-gpus", k => { sel.gpus = k; run(); });
  },
});

topic({
  num: null, title: "The whole story",
  question: "How does it all connect, from prompt to token?",
  render: () => `
    ${say("Every topic in this course is one link in a single chain. Press play to walk through it, or click any link to revisit that topic.")}
    ${see("How does it all fit together?", `
      <div class="btn-row"><button type="button" class="btn primary" id="st-play">▶ Walk the chain</button></div>
      <div class="chain-big" id="st-chain"></div>`, "The chain")}
    ${check({
      q: "Why does low-batch decode of a large dense model often run far below the GPU's peak FLOPS?",
      opts: ["The Tensor Cores are broken", "Each token needs the whole model's weights moved from HBM, and bandwidth limits that", "Decode uses the CPU", "The KV cache blocks computation"],
      a: 1, why: "This is the core idea: decode does little maths per byte, so memory bandwidth, not compute, sets the pace.",
    })}
    ${check({
      q: "Which pair of techniques mainly targets memory-bound decode?",
      opts: ["Chunked prefill and more FLOPS", "Weight quantization and larger batches", "Pipeline parallelism and bigger disks", "Tokenizer changes and sampling"],
      a: 1, why: "Quantization reduces bytes moved per step; batching shares one weight read across many tokens. Both raise tokens per byte moved.",
    })}
    ${check({
      q: "A model doesn't fit on one GPU and your servers have NVLink. What is the usual first choice to split it within a server?",
      opts: ["Data parallelism", "Tensor parallelism", "Prefix caching", "MQA"],
      a: 1, why: "Tensor parallelism splits each layer's matrices across GPUs and relies on fast links like NVLink for per-layer communication.",
    })}`,
  mount(root, ctx) {
    const find = (num) => COURSE.topics.findIndex(t => t.num === num);
    const chain = [
      ["LLM inference", 1], ["Prefill + decode", 3], ["Different computational characteristics", 3], ["GPU executes matrix operations", 8],
      ["Weights + KV cache live primarily in GPU memory", 7], ["SMs execute work", 10], ["Tensor Cores accelerate matrix operations", 10],
      ["Memory hierarchy determines how efficiently data reaches compute", 11], ["FLOPs measure computation", 13], ["Memory bandwidth measures data movement capacity", 15],
      ["Arithmetic intensity connects the two", 16], ["Prefill tends toward compute-bound", 18], ["Low-batch decode tends toward memory-bandwidth-bound", 19],
      ["Precision / quantization changes compute + memory requirements", 26], ["Newer GPUs improve compute + HBM + bandwidth + precision + interconnect", 28],
      ["DP / TP / PP / EP allow scaling across GPUs", 30], ["Advanced serving optimizations improve utilization and reduce repeated work", 36],
    ];
    $("#st-chain", root).innerHTML = chain.map(([t, n], i) => (i ? `<div class="cdown">↓</div>` : "") +
      `<div class="cstep" data-i="${i}"><span class="ci">${String(i + 1).padStart(2, "0")}</span><span>${t}</span><button type="button" class="btn small" data-go="${n}">Topic ${n} →</button></div>`).join("");
    $("#st-chain", root).onclick = e => { const b = e.target.closest("[data-go]"); if (b) ctx.go(find(+b.dataset.go)); };
    let timer = null;
    $("#st-play", root).onclick = () => {
      if (timer) ctx.stop(timer); let k = 0;
      const steps = $$(".cstep", root); steps.forEach(s => s.classList.remove("lit"));
      timer = ctx.every(650, () => { if (k >= steps.length) { ctx.stop(timer); timer = null; return; } steps[k].classList.add("lit"); k++; });
    };
  },
});
