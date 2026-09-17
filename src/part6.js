/* ============================================================
   PART 6 — Scaling LLM Inference (topics 26–30)
   ============================================================ */
part({
  num: 6, title: "Scaling LLM Inference",
  blurb: "When one GPU isn't enough: copy the model, split its matrices, split its layers, or spread its experts.",
  story: "<b>DP / TP / PP / EP</b> let inference scale across GPUs, each trading communication for memory, compute or throughput.",
});

const REQ_COLORS = ["#3559d1", "#cf6a0b", "#8e44c2", "#0a7470", "#b83280", "#2b8543", "#a8750a", "#5a67d8", "#c13535", "#2a7fa8"];

/* ---------- Topic 26 ---------- */
topic({
  num: 26, title: "Why multiple GPUs?",
  question: "When do I need more than one GPU?",
  render: () => `
    ${say("There are two different reasons to add GPUs. Either the job <b>doesn't fit</b> (the weights plus KV cache need more memory than one GPU has), or the job <b>fits but is too slow</b> (you need more compute or more throughput for more users).")}
    ${see("How many 80 GB GPUs does my deployment need?", `
      <div class="controls">
        <div class="ctl"><label>Model</label>${seg("t26-m", [["8B", "8B"], ["70B", "70B"], ["405B", "405B"]], "70B")}</div>
        <div class="ctl"><label>Precision</label>${seg("t26-p", [["2", "BF16"], ["1", "FP8"], ["0.5", "INT4"]], "2")}</div>
        ${slider("t26-u", "Concurrent users", 1, 128, 1, 16)}
        ${slider("t26-c", "Context per user (tokens)", 0, 5, 1, 2)}
      </div>
      <div class="stack" id="t26-stack"></div>
      <div id="t26-gpus" style="display:flex;flex-wrap:wrap;gap:6px"></div>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t26-w">–</span><span class="k">weights</span></div>
        <div class="stat"><span class="v c-kv" id="t26-kv">–</span><span class="k">KV cache (all users)</span></div>
        <div class="stat"><span class="v" id="t26-n">–</span><span class="k">GPUs needed just for memory</span></div>
      </div>
      <p class="c-muted" style="font-size:.82rem">KV cache kept in 16-bit; 10% memory held back for activations and buffers. Throughput needs may require even more GPUs. ${illus("Approximate")}</p>`)}
    ${see("What are the four reasons?", `<div class="grid2">
      <div class="card"><h4>1. The model does not fit</h4><p>405B in BF16 is ~810 GB of weights. No single GPU holds that.</p></div>
      <div class="card"><h4>2. The KV cache does not fit</h4><p>Many users × long contexts can need more memory than the weights.</p></div>
      <div class="card"><h4>3. Need more throughput</h4><p>More users than one GPU can serve at acceptable speed.</p></div>
      <div class="card"><h4>4. Need more compute or bandwidth per token</h4><p>Splitting a model lets several GPUs share each token's work, lowering latency.</p></div></div>`, "Reasons")}
    ${inf(`<p>How you split depends on the reason: copy the whole model (<b>data parallel</b>) for throughput, split matrices (<b>tensor parallel</b>) or layers (<b>pipeline parallel</b>) when it doesn't fit, and spread experts (<b>expert parallel</b>) for MoE models.</p>`)}
    ${check({
      q: "An 8B model fits comfortably on one GPU, but you have 10× more users than it can serve. What is the simplest scaling approach?",
      opts: ["Split its layers across GPUs", "Run several full copies of the model on separate GPUs", "Quantize to 2 bits", "Use a bigger CPU"],
      a: 1, why: "The model already fits, so the need is throughput. Independent copies (data parallelism) scale throughput with almost no communication.",
    })}`,
  mount(root, ctx) {
    const ctxs = [1024, 2048, 4096, 8192, 32768, 131072];
    let model = "70B", bpp = 2;
    const run = bindSliders(root, { "t26-u": x => x, "t26-c": x => ctxs[x].toLocaleString("en-US") }, v => {
      const m = MODELS[model];
      const w = m.N * bpp, kv = kvBytesPerToken(m, 2) * ctxs[v["t26-c"]] * v["t26-u"], total = w + kv;
      const usable = DEMO.hbmGB * 1e9 * 0.9, n = Math.ceil(total / usable);
      $("#t26-w", root).textContent = gb(w); $("#t26-kv", root).textContent = gb(kv);
      $("#t26-n", root).innerHTML = `<span class="${n > 1 ? "c-bad" : "c-ok"}">${n}</span>`;
      const wp = w / total * 100;
      $("#t26-stack", root).innerHTML = `<span class="s-w" style="width:${wp}%">${wp > 18 ? "weights" : ""}</span><span class="s-kv" style="width:${100 - wp}%">${100 - wp > 18 ? "KV cache" : ""}</span>`;
      const shown = Math.min(n, 48);
      $("#t26-gpus", root).innerHTML = Array.from({ length: shown }, (_, i) => {
        const fill = Math.min(1, (total - i * usable) / usable);
        return `<div title="GPU ${i + 1}" style="width:46px;height:56px;border:2px solid var(--ink);border-radius:6px;position:relative;overflow:hidden;background:var(--surface)">
          <div style="position:absolute;left:0;right:0;bottom:0;height:${fill * 100}%;background:var(--memory);opacity:.6"></div>
          <span style="position:absolute;inset:0;display:grid;place-items:center;font-size:.66rem;font-weight:700">GPU ${i + 1}</span></div>`;
      }).join("") + (n > shown ? `<span class="c-muted">+${n - shown} more</span>` : "");
    });
    bindSeg(root, "t26-m", k => { model = k; run(); });
    bindSeg(root, "t26-p", k => { bpp = +k; run(); });
  },
});

/* ---------- Topic 27 ---------- */
topic({
  num: 27, title: "Data parallelism",
  question: "What if I just run more copies of the model?",
  render: () => `
    ${say("<b>Data parallelism</b> = put a <b>full copy of the model on each GPU</b> and send each request to one of the copies. The copies never need to talk to each other while generating. More copies, more requests served.")}
    ${see("What happens to the queue as I add replicas?", `
      <div class="controls">
        ${slider("t27-r", "Model replicas (GPUs)", 1, 4, 1, 1)}
        ${slider("t27-a", "Incoming requests per second", 1, 12, 1, 6)}
      </div>
      <svg class="svg-block" viewBox="0 0 720 260" id="t27-svg"></svg>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t27-cap">–</span><span class="k">capacity (requests/sec)</span></div>
        <div class="stat"><span class="v" id="t27-q">–</span><span class="k">waiting in queue</span></div>
        <div class="stat"><span class="v" id="t27-st">–</span><span class="k">status</span></div>
      </div>
      <p class="c-muted" style="font-size:.82rem">Each replica handles 3 requests/sec in this toy. ${illus()}</p>`)}
    ${ex(`<div class="calc">One GPU serves 3 requests/sec. Traffic is 9 requests/sec.<br>
      1 replica → queue grows forever · 3 replicas → keeps up · each replica still needs the <b>whole model</b> in its own memory</div>`)}
    ${tech(`<ul>
      <li>Each replica holds full weights and its own KV cache. A load balancer routes requests.</li>
      <li>Throughput scales almost linearly with replicas; <b>per-token latency does not improve</b>, because each token is still produced by one GPU.</li>
      <li>Only works if the model fits on one GPU (or one group of GPUs, when combined with other methods).</li>
    </ul>`)}
    ${inf(`<p>Useful when the model fits and you need more users served. It does nothing for a model that doesn't fit, and nothing for batch-1 latency.</p>`)}
    ${check({
      q: "What does data parallelism NOT improve?",
      opts: ["Total requests served per second", "The time to generate each token for a single request", "Ability to handle traffic spikes", "Fault tolerance"],
      a: 1, why: "Each request still runs on a single copy, so its per-token speed is unchanged. Data parallelism adds throughput, not per-request speed.",
    })}`,
  mount(root, ctx) {
    let R = 1, A = 6, queue = 0, acc = 0, dots = [], served = [];
    const svg = $("#t27-svg", root), per = 3;
    const draw = () => {
      let s = `<rect x="10" y="100" width="130" height="60" rx="8" fill="var(--surface-2)" stroke="var(--line)"/>
        <text x="75" y="95" text-anchor="middle" font-size="12" font-weight="600">Request queue</text>`;
      const qShow = Math.min(queue, 24);
      for (let i = 0; i < qShow; i++) s += `<circle cx="${22 + (i % 8) * 15}" cy="${113 + Math.floor(i / 8) * 16}" r="5" fill="var(--accent)"/>`;
      if (queue > 24) s += `<text x="75" y="178" text-anchor="middle" font-size="11" style="fill:var(--bad)">+${Math.round(queue - 24)} more</text>`;
      s += `<rect x="170" y="110" width="80" height="40" rx="6" fill="var(--surface)" stroke="var(--ink)"/><text x="210" y="134" text-anchor="middle" font-size="11" font-weight="600">router</text>`;
      for (let i = 0; i < 4; i++) {
        const y = 20 + i * 60, on = i < R;
        s += `<line x1="250" y1="130" x2="360" y2="${y + 24}" stroke="${on ? "var(--line)" : "transparent"}" stroke-width="2"/>
          <rect x="360" y="${y}" width="230" height="48" rx="8" fill="${on ? "var(--surface)" : "transparent"}" stroke="${on ? "var(--ink)" : "var(--line)"}" stroke-dasharray="${on ? "0" : "4 4"}" stroke-width="1.5"/>
          <text x="375" y="${y + 20}" font-size="12" font-weight="700" ${on ? "" : "class='muted'"}>GPU ${i + 1}</text>
          <text x="375" y="${y + 38}" font-size="11" class="muted">${on ? "full model copy + own KV cache" : "not in use"}</text>
          ${on ? `<rect x="540" y="${y + 10}" width="40" height="28" rx="4" fill="var(--memory)" opacity=".7"/><text x="560" y="${y + 29}" text-anchor="middle" font-size="9" style="fill:#fff" font-weight="700">model</text>` : ""}`;
      }
      dots.forEach(d => s += `<circle cx="${d.x}" cy="${d.y}" r="5" fill="var(--accent)"/>`);
      s += `<text x="660" y="130" text-anchor="middle" font-size="12" font-weight="600">served</text><text x="660" y="150" text-anchor="middle" font-size="18" font-weight="700" style="fill:var(--ok)">${served.length}</text>`;
      svg.innerHTML = s;
    };
    bindSliders(root, { "t27-r": x => x, "t27-a": x => x }, v => { R = v["t27-r"]; A = v["t27-a"]; });
    let t = 0, srvAcc = 0;
    ctx.loop(dt => {
      dt = Math.min(dt, .1); t += dt;
      acc += dt * A; while (acc >= 1) { acc -= 1; queue++; }
      srvAcc += dt * R * per;
      while (srvAcc >= 1 && queue >= 1) {
        srvAcc -= 1; queue--;
        const g = Math.floor(Math.random() * R);
        dots.push({ x: 250, y: 130, tx: 590, ty: 44 + g * 60, p: 0 });
      }
      if (queue < 1) srvAcc = Math.min(srvAcc, 1);
      dots.forEach(d => { d.p += dt * 1.6; d.x = 250 + (d.tx - 250) * d.p; d.y = 130 + (d.ty - 130) * Math.min(1, d.p * 1.5); });
      dots = dots.filter(d => { if (d.p >= 1) { served.push(1); return false; } return true; });
      if (served.length > 999) served = [];
      const cap = R * per;
      $("#t27-cap", root).textContent = cap;
      $("#t27-q", root).textContent = Math.round(queue);
      $("#t27-st", root).innerHTML = cap >= A ? `<span class="pill ok">keeping up</span>` : `<span class="pill bad">falling behind</span>`;
      draw();
    });
  },
});

/* ---------- Topic 28 ---------- */
topic({
  num: 28, title: "Tensor parallelism",
  question: "How can one layer run on several GPUs at once?",
  render: () => `
    ${say(
      "<b>Tensor parallelism</b> cuts each big weight matrix into slices and puts <b>one slice on each GPU</b>. Every GPU multiplies the input by its slice at the same time, then the GPUs <b>combine their partial results</b> before the next step.",
      "The model now fits (each GPU holds only part of each matrix), and each token's maths is shared. The price: GPUs must talk to each other <b>in every layer</b>."
    )}
    ${see("What does splitting one matrix across GPUs look like?", `
      <div class="btn-row">
        ${seg("t28-n", [["2", "2 GPUs"], ["4", "4 GPUs"]], "2")}
        ${seg("t28-link", [["nvlink", "NVLink"], ["pcie", "PCIe"]], "nvlink")}
        <button type="button" class="btn primary" id="t28-play">▶ Run one layer</button>
      </div>
      <svg class="svg-block" viewBox="0 0 720 300" id="t28-svg"></svg>
      <div class="stage-panel" id="t28-cap" style="min-height:56px"></div>
      <div class="grid2">
        <div style="display:grid;gap:6px"><div class="mini-label">Time per layer</div><div class="stack" id="t28-time"></div></div>
        <div class="stat-row">
          <div class="stat"><span class="v" id="t28-mem">–</span><span class="k">70B weights per GPU (BF16)</span></div>
          <div class="stat"><span class="v" id="t28-comm">–</span><span class="k">communication share</span></div>
        </div>
      </div>
      <p class="c-muted" style="font-size:.82rem">${illus("Conceptual timing")}</p>`)}
    ${ex(`<div class="calc">70B in BF16 = 140 GB · TP = 2 → 70 GB per GPU (fits on 80 GB, tight) · TP = 4 → 35 GB per GPU<br>
      80 layers × 2 combine steps per layer = <b>160 GPU-to-GPU exchanges for every token</b></div>`)}
    ${tech(`<ul>
      <li>Typical split (Megatron-style): the first matrix in a block is split by <b>columns</b>, the second by <b>rows</b>, so only one <b>all-reduce</b> is needed per attention block and one per MLP block.</li>
      <li>Attention heads are divided among GPUs; the KV cache is sharded too.</li>
      <li>Because communication happens every layer, TP needs very fast links (NVLink/NVSwitch) and is usually kept within one server.</li>
      <li>Each GPU reads only its shard of weights, so memory bandwidth effectively adds up across GPUs, which helps memory-bound decode.</li>
    </ul>`)}
    ${inf(`<p>Tensor parallelism is the go-to way to serve a model that is too big for one GPU and to lower per-token latency. On a slow interconnect, communication can eat the gains.</p>`)}
    ${check({
      q: "Why does tensor parallelism need a fast GPU-to-GPU link?",
      opts: ["It only communicates once at the start", "GPUs must combine partial results in every layer, for every token", "It copies the whole model to each GPU", "It uses the CPU for attention"],
      a: 1, why: "Each layer's matrices are split, so partial results must be combined (all-reduce) inside every layer. That's many exchanges per token.",
    })}`,
  mount(root, ctx) {
    let N = 2, link = "nvlink", phase = -1, timer = null;
    const caps = ["Input X (the token's vector) is sent to every GPU.", "Each GPU multiplies X by <b>its own slice</b> of W, all at the same time.", "<b>All-reduce</b>: GPUs exchange and combine their partial results over the interconnect.", "The combined result Y goes on to the next layer, where it happens again."];
    const draw = () => {
      const svg = $("#t28-svg", root), gw = N === 2 ? 240 : 140, gap = N === 2 ? 60 : 30;
      const total = N * gw + (N - 1) * gap, x0 = (720 - total) / 2;
      let s = `<rect x="310" y="8" width="100" height="34" rx="6" fill="${phase === 0 ? "var(--accent)" : "var(--surface-2)"}" stroke="var(--ink)"/><text x="360" y="30" text-anchor="middle" font-size="13" font-weight="700" ${phase === 0 ? 'style="fill:#fff"' : ""}>X</text>`;
      for (let i = 0; i < N; i++) {
        const x = x0 + i * (gw + gap);
        s += `<line x1="360" y1="42" x2="${x + gw / 2}" y2="70" stroke="${phase === 0 ? "var(--accent)" : "var(--line)"}" stroke-width="2"/>
          <rect x="${x}" y="70" width="${gw}" height="130" rx="10" fill="var(--surface)" stroke="var(--ink)" stroke-width="1.5"/>
          <text x="${x + 10}" y="90" font-size="12" font-weight="700">GPU ${i + 1}</text>`;
        const cols = 8 / N;
        for (let c = 0; c < 8; c++) {
          const mine = Math.floor(c / cols) === i;
          for (let r = 0; r < 4; r++) s += `<rect x="${x + gw / 2 - 48 + c * 12}" y="${100 + r * 12}" width="10" height="10" rx="1.5" fill="${mine ? (phase === 1 ? "var(--compute)" : "var(--memory)") : "var(--surface-3)"}" opacity="${mine ? 1 : .35}"/>`;
        }
        s += `<text x="${x + gw / 2}" y="170" text-anchor="middle" font-size="11" class="muted">slice ${i + 1} of W (${Math.round(100 / N)}%)</text>
          <text x="${x + gw / 2}" y="188" text-anchor="middle" font-size="11" class="muted">+ ${Math.round(100 / N)}% of KV cache</text>`;
        if (i < N - 1) s += `<line x1="${x + gw}" y1="135" x2="${x + gw + gap}" y2="135" stroke="${phase === 2 ? (link === "nvlink" ? "var(--ok)" : "var(--bad)") : "var(--line)"}" stroke-width="${phase === 2 ? 5 : 2}" stroke-dasharray="${phase === 2 ? "8 5" : "0"}"><animate attributeName="stroke-dashoffset" from="26" to="0" dur=".6s" repeatCount="indefinite"/></line>`;
        s += `<line x1="${x + gw / 2}" y1="200" x2="360" y2="250" stroke="${phase === 3 ? "var(--ok)" : "var(--line)"}" stroke-width="2"/>`;
      }
      s += `<rect x="310" y="250" width="100" height="34" rx="6" fill="${phase === 3 ? "var(--ok)" : "var(--surface-2)"}" stroke="var(--ink)"/><text x="360" y="272" text-anchor="middle" font-size="13" font-weight="700" ${phase === 3 ? 'style="fill:#fff"' : ""}>Y</text>
        <text x="360" y="232" text-anchor="middle" font-size="11" class="muted">${link === "nvlink" ? "NVLink: fast exchange" : "PCIe: slow exchange"}</text>`;
      svg.innerHTML = s;
      $("#t28-cap", root).innerHTML = phase < 0 ? `<p class="c-muted">Press “Run one layer”.</p>` : `<div class="mini-label">Step ${phase + 1} of 4</div><p>${caps[phase]}</p>`;
      const compute = 10 / N, comm = N === 1 ? 0 : (link === "nvlink" ? 1.2 : 9) * (N === 4 ? 1.3 : 1);
      const tot = compute + comm, base = 10;
      $("#t28-time", root).innerHTML = `<span class="s-act" style="width:${compute / Math.max(base, tot) * 100}%">compute</span><span class="s-kv" style="width:${comm / Math.max(base, tot) * 100}%">${comm > 1.5 ? "communicate" : ""}</span><span class="s-free" style="flex:1">${tot < base ? `${fmt(base / tot, 1)}× faster than 1 GPU` : "slower than 1 GPU!"}</span>`;
      $("#t28-mem", root).textContent = `${140 / N} GB`;
      $("#t28-comm", root).textContent = `${Math.round(comm / tot * 100)}%`;
    };
    bindSeg(root, "t28-n", k => { N = +k; draw(); });
    bindSeg(root, "t28-link", k => { link = k; draw(); });
    $("#t28-play", root).onclick = () => {
      if (timer) ctx.stop(timer); phase = 0; draw();
      timer = ctx.every(1600, () => { phase++; if (phase > 3) { ctx.stop(timer); timer = null; phase = 3; } draw(); });
    };
    draw();
  },
});

/* ---------- Topic 29 ---------- */
topic({
  num: 29, title: "Pipeline parallelism",
  question: "What if each GPU holds different layers?",
  render: () => `
    ${say(
      "<b>Pipeline parallelism</b> gives each GPU a <b>consecutive group of layers</b>, like stations on an assembly line. A token's activations flow from GPU 1 to GPU 2 to GPU 3.",
      "GPUs only talk at the <b>boundaries</b> between groups, so slower links are OK. The catch: a GPU sits idle while it waits for the previous station. Those idle gaps are <b>pipeline bubbles</b>."
    )}
    ${see("Where do the bubbles come from?", `
      ${flow([{ t: "GPU 1 · layers 1–20", c: "memory" }, { t: "GPU 2 · layers 21–40", c: "memory" }, { t: "GPU 3 · layers 41–60", c: "memory" }], { id: "t29-flow" })}
      <div class="controls">
        ${slider("t29-m", "Micro-batches sent through the pipeline", 1, 8, 1, 1)}
        ${slider("t29-p", "Pipeline stages (GPUs)", 2, 4, 1, 3)}
      </div>
      <div class="scroll-x"><div class="gantt" id="t29-gantt"></div></div>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t29-idle">–</span><span class="k">GPU time idle (bubbles)</span></div>
        <div class="stat"><span class="v" id="t29-mem">–</span><span class="k">weights per GPU (70B BF16)</span></div>
      </div>
      <p class="c-muted" style="font-size:.82rem">Each column is one time step. Hatched cells = a GPU waiting. Forward-only schedule, as in inference.</p>`)}
    ${ex(`<div class="calc">3 stages, 1 batch → each GPU works 1 of 3 time steps → <b>67% idle</b><br>
      3 stages, 8 micro-batches → idle ≈ 2 ÷ 10 = <b>20%</b>. Keep the pipeline full and bubbles shrink.</div>`)}
    ${tech(`<ul>
      <li>Only activations at stage boundaries cross GPUs: much less communication than tensor parallelism, so it works across servers.</li>
      <li>Splitting the batch into <b>micro-batches</b> keeps every stage busy; with continuous serving the pipeline stays mostly full.</li>
      <li>Pipeline parallelism adds latency per token (every stage in turn), but lets very large models span many GPUs. It's often combined with tensor parallelism inside each server.</li>
    </ul>`)}
    ${fx(["bubble fraction ≈ (stages − 1) ÷ (micro-batches + stages − 1)"])}
    ${inf(`<p>Use pipeline parallelism when a model must span servers, where the links between machines are too slow for tensor parallelism in every layer.</p>`)}
    ${check({
      q: "What is a pipeline bubble?",
      opts: ["A memory leak in the KV cache", "Time when a GPU in the pipeline is idle, waiting for work from the previous stage", "A corrupted activation", "A request that is dropped"],
      a: 1, why: "Stages depend on earlier stages, so a GPU idles until its input arrives. Feeding more micro-batches fills those gaps.",
    })}`,
  mount(root, ctx) {
    let k = 0; ctx.every(700, () => setFlow($("#t29-flow", root), k++ % 3));
    bindSliders(root, { "t29-m": x => x, "t29-p": x => x }, v => {
      const m = v["t29-m"], p = v["t29-p"], cols = m + p - 1;
      let html = `<div class="gr" style="--cols:${cols}"><span class="gl"></span>${Array.from({ length: cols }, (_, t) => `<span class="gl" style="text-align:center">t${t + 1}</span>`).join("")}</div>`;
      for (let s = 0; s < p; s++) {
        html += `<div class="gr" style="--cols:${cols}"><span class="gl">GPU ${s + 1}</span>`;
        for (let t = 0; t < cols; t++) {
          const mb = t - s;
          html += mb >= 0 && mb < m ? `<span class="gc" style="background:${REQ_COLORS[mb]}">${mb + 1}</span>` : `<span class="gc idle"></span>`;
        }
        html += `</div>`;
      }
      $("#t29-gantt", root).innerHTML = html;
      $("#t29-idle", root).textContent = `${Math.round((p - 1) / cols * 100)}%`;
      $("#t29-mem", root).textContent = `${fmt(140 / p, 1)} GB`;
    });
  },
});

/* ---------- Topic 30 ---------- */
topic({
  num: 30, title: "Expert parallelism",
  question: "How do Mixture-of-Experts models spread across GPUs?",
  render: () => `
    ${say(
      "A <b>Mixture-of-Experts (MoE)</b> model replaces one big MLP with many smaller MLPs called <b>experts</b>. A small <b>router</b> looks at each token and picks only the <b>top few experts</b> (top-k) to process it.",
      "So the model can have huge total knowledge (many experts) while each token only pays for a few. <b>Expert parallelism</b> places different experts on different GPUs and sends each token to wherever its chosen experts live."
    )}
    ${see("Where does each token go?", `
      <div class="btn-row">
        ${seg("t30-k", [["1", "top-1"], ["2", "top-2"]], "2")}
        <button type="button" class="btn primary" id="t30-go">Route next token</button>
        <button type="button" class="btn" id="t30-auto">▶ Auto</button>
      </div>
      <svg class="svg-block" viewBox="0 0 720 320" id="t30-svg"></svg>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t30-active">–</span><span class="k">expert weights used per token</span></div>
        <div class="stat"><span class="v" id="t30-tok">–</span><span class="k">tokens routed</span></div>
      </div>`)}
    ${ex(`<div class="calc">Mixtral-8x7B-style model: 8 experts per layer, top-2 routing<br>
      total parameters ≈ 47B (all must be stored) · active per token ≈ 13B (what each token computes)<br>
      → memory like a ~47B model, compute per token like a ~13B model</div>`)}
    ${tech(`<ul>
      <li><b>Router (gate):</b> a small linear layer that scores every expert for a token; the top-k scores are selected and their outputs are mixed by those scores.</li>
      <li><b>Sparse activation:</b> unselected experts do no work for that token.</li>
      <li><b>Expert parallelism:</b> experts are sharded across GPUs; tokens are dispatched to the right GPUs and gathered back with <b>all-to-all</b> communication, in every MoE layer.</li>
      <li>Challenges: uneven load (popular experts), communication cost, and all experts still need memory.</li>
    </ul>`)}
    ${inf(`<p>MoE lowers <b class="c-compute">FLOPs per token</b> but not <b class="c-memory">memory</b>: all experts must sit in HBM. Decode reads only the active experts' weights for a single token, but across a batch many different experts get touched. Expert parallelism spreads that memory and work across GPUs.</p>`)}
    ${check({
      q: "In a top-2 MoE layer with 8 experts, how many experts process a given token?",
      opts: ["All 8", "2", "1", "4"],
      a: 1, why: "Top-2 routing picks the two highest-scoring experts. The other six do no work for that token (though they still take memory).",
    })}`,
  mount(root, ctx) {
    const toks = ["The", " robot", " solved", " the", " integral", " and", " wrote", " a", " poem", " about", " GPUs"];
    let ti = 0, load = new Array(8).fill(0), last = null, auto = null;
    const topK = () => +segVal(root, "t30-k");
    const draw = () => {
      const k = topK();
      let s = `<rect x="300" y="10" width="120" height="36" rx="6" fill="var(--accent-soft)" stroke="var(--accent)"/>
        <text x="360" y="33" text-anchor="middle" font-size="13" font-weight="700">${last ? JSON.stringify(last.tok) : "token"}</text>
        <rect x="290" y="70" width="140" height="36" rx="6" fill="var(--surface)" stroke="var(--ink)" stroke-width="1.5"/>
        <text x="360" y="93" text-anchor="middle" font-size="13" font-weight="700">Router</text>
        <line x1="360" y1="46" x2="360" y2="70" stroke="var(--ink)"/>`;
      for (let g = 0; g < 4; g++) {
        const gx = 20 + g * 175;
        s += `<rect x="${gx}" y="170" width="160" height="120" rx="10" fill="var(--surface)" stroke="var(--ink)" stroke-width="1.5"/><text x="${gx + 10}" y="190" font-size="12" font-weight="700">GPU ${g + 1}</text>`;
        for (let e = 0; e < 2; e++) {
          const ei = g * 2 + e, ex = gx + 12 + e * 74, sel = last && last.top.includes(ei);
          s += `<rect x="${ex}" y="202" width="64" height="50" rx="6" fill="${sel ? "var(--compute)" : "var(--surface-2)"}" stroke="${sel ? "var(--compute)" : "var(--line)"}"/>
            <text x="${ex + 32}" y="224" text-anchor="middle" font-size="11" font-weight="700" ${sel ? 'style="fill:#fff"' : ""}>Expert ${ei + 1}</text>
            <text x="${ex + 32}" y="242" text-anchor="middle" font-size="10" ${sel ? 'style="fill:#fff"' : 'class="muted"'}>${last ? `score ${last.scores[ei].toFixed(2)}` : ""}</text>
            <rect x="${ex}" y="258" width="64" height="6" rx="3" fill="var(--surface-3)"/><rect x="${ex}" y="258" width="${Math.min(64, load[ei] * 8)}" height="6" rx="3" fill="var(--kv)"/>`;
          if (sel) s += `<line x1="360" y1="106" x2="${ex + 32}" y2="202" stroke="var(--compute)" stroke-width="2.5" stroke-dasharray="6 4"><animate attributeName="stroke-dashoffset" from="20" to="0" dur=".5s" repeatCount="indefinite"/></line>`;
        }
      }
      s += `<text x="20" y="308" font-size="10" class="muted">purple bars = how many tokens each expert has handled (load)</text>`;
      $("#t30-svg", root).innerHTML = s;
      $("#t30-active", root).textContent = `${k} of 8 experts (${k * 100 / 8}%)`;
      $("#t30-tok", root).textContent = ti;
    };
    const route = () => {
      const tok = toks[ti % toks.length];
      let h = 0; for (const c of tok + ti) h = (h * 31 + c.charCodeAt(0)) % 9973;
      const raw = Array.from({ length: 8 }, (_, i) => Math.abs(Math.sin(h * (i + 1) * 12.9898)) * 3);
      const ex = raw.map(Math.exp), sum = ex.reduce((a, b) => a + b, 0), scores = ex.map(v => v / sum);
      const top = scores.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, topK()).map(x => x[1]);
      top.forEach(i => load[i]++);
      last = { tok, scores, top }; ti++; draw();
    };
    bindSeg(root, "t30-k", () => { if (last) { last.top = last.scores.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, topK()).map(x => x[1]); } draw(); });
    $("#t30-go", root).onclick = route;
    $("#t30-auto", root).onclick = () => {
      if (auto) { ctx.stop(auto); auto = null; $("#t30-auto", root).textContent = "▶ Auto"; return; }
      $("#t30-auto", root).textContent = "❚❚ Pause"; route(); auto = ctx.every(1200, route);
    };
    draw();
  },
});
