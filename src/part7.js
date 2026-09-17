/* ============================================================
   PART 7 — Advanced Inference Optimizations (topics 31–35)
   ============================================================ */
part({
  num: 7, title: "Advanced Inference Optimizations",
  blurb: "Serving tricks that shrink the KV cache, keep the GPU full, and avoid repeating work.",
  story: "Advanced serving optimizations <b>improve utilization and reduce repeated work</b>: smaller KV caches, fuller batches, reused prefixes, smoother scheduling.",
});

/* ---------- Topic 31 ---------- */
topic({
  num: 31, title: "MHA vs MQA vs GQA",
  question: "How can the KV cache get smaller without changing the model size much?",
  render: () => `
    ${say(
      "Attention has many <b>heads</b>, each looking for different patterns. Each head normally has its own Query, Key and Value.",
      "But only <b class='c-kv'>K and V</b> go into the cache. If several Query heads <b>share</b> the same K/V head, there's less to store for every token. That's the whole idea."
    )}
    ${see("What changes when Query heads share K/V heads?", `
      <div class="btn-row">${seg("t31-mode", [["8", "MHA: 8 K/V heads"], ["2", "GQA: 2 K/V heads"], ["1", "MQA: 1 K/V head"]], "8")}</div>
      <svg class="svg-block" viewBox="0 0 720 220" id="t31-svg"></svg>
      <div class="controls">
        ${slider("t31-ctx", "Context length (tokens)", 0, 5, 1, 2)}
        ${slider("t31-users", "Concurrent users", 1, 64, 1, 8)}
      </div>
      <div class="bars-h" id="t31-bars"></div>
      <p class="c-muted" style="font-size:.82rem">Diagram shows 8 Query heads. Numbers use a 70B-class shape: 80 layers, 64 Query heads, head size 128, FP16 KV cache; MHA = 64 K/V heads, GQA = 8, MQA = 1. Red line = 80 GB. ${illus("Approximate")}</p>`)}
    ${ex(`<div class="calc">KV cache per token = 2 × 80 layers × K/V heads × 128 × 2 bytes<br>
      MHA (64 heads): <b>2.6 MB</b> per token · GQA (8): <b>0.33 MB</b> · MQA (1): <b>0.04 MB</b><br>
      4,096-token conversation: MHA ≈ 10.7 GB · GQA ≈ 1.3 GB · MQA ≈ 0.17 GB</div>`)}
    ${tech(`<ul>
      <li><b>MHA</b> (Multi-Head Attention): every Query head has its own K and V heads.</li>
      <li><b>GQA</b> (Grouped-Query Attention): Query heads are split into groups; each group shares one K/V head. Used by many modern LLMs.</li>
      <li><b>MQA</b> (Multi-Query Attention): all Query heads share a single K/V head. Smallest cache, but can cost more quality.</li>
      <li>GQA usually keeps quality close to MHA while cutting KV memory by the grouping factor.</li>
    </ul>`)}
    ${flow([{ t: "Fewer K/V heads", c: "kv" }, { t: "Smaller KV cache", c: "kv" }, { t: "Less memory + fewer bytes read per step", c: "memory" }, { t: "More users, longer contexts, faster decode" }])}
    ${inf(`<p>This is a <b>model architecture</b> choice made before training. As a serving engineer you don't switch it on, but it explains why two models of the same size can have very different KV cache needs and batch capacity.</p>`)}
    ${check({
      q: "A model switches from MHA with 32 K/V heads to GQA with 8 K/V heads. What happens to the KV cache per token?",
      opts: ["Unchanged", "About 4× smaller", "About 4× larger", "About 32× smaller"],
      a: 1, why: "KV cache size is proportional to the number of K/V heads: 32 → 8 is a 4× reduction.",
    })}`,
  mount(root, ctx) {
    const ctxs = [1024, 2048, 4096, 8192, 32768, 131072];
    const draw = () => {
      const n = +segVal(root, "t31-mode");
      let s = `<text x="20" y="28" font-size="12" font-weight="700">Query heads</text><text x="20" y="178" font-size="12" font-weight="700" style="fill:var(--kv)">K/V heads (cached)</text>`;
      const qx = i => 170 + i * 70, kx = j => n === 1 ? 415 : 170 + (j + .5) * (560 / n) - 35;
      for (let i = 0; i < 8; i++) {
        const g = Math.floor(i / (8 / n));
        s += `<line x1="${qx(i)}" y1="52" x2="${kx(g)}" y2="150" stroke="${REQ_COLORS[g]}" stroke-width="2" opacity=".7"/>
          <circle cx="${qx(i)}" cy="36" r="16" fill="var(--surface)" stroke="${REQ_COLORS[g]}" stroke-width="2.5"/><text x="${qx(i)}" y="41" text-anchor="middle" font-size="11" font-weight="700">Q${i + 1}</text>`;
      }
      for (let j = 0; j < n; j++) s += `<rect x="${kx(j) - 26}" y="150" width="52" height="40" rx="6" fill="${REQ_COLORS[j]}"/><text x="${kx(j)}" y="175" text-anchor="middle" font-size="11" font-weight="700" style="fill:#fff">K${j + 1} V${j + 1}</text>`;
      s += `<text x="700" y="210" text-anchor="end" font-size="12" class="muted">${n === 8 ? "each Q head has its own K/V" : n === 1 ? "all Q heads share one K/V" : `${8 / n} Q heads share each K/V`}</text>`;
      $("#t31-svg", root).innerHTML = s;
    };
    bindSeg(root, "t31-mode", draw); draw();
    const m = MODELS["70B"];
    bindSliders(root, { "t31-ctx": x => ctxs[x].toLocaleString("en-US"), "t31-users": x => x }, v => {
      const tokens = ctxs[v["t31-ctx"]] * v["t31-users"];
      const rows = [["MHA (64)", 64], ["GQA (8)", 8], ["MQA (1)", 1]].map(([n, h]) => [n, kvBytesPerToken(m, 2, h) * tokens]);
      const scale = Math.max(rows[0][1], 80e9) * 1.05;
      $("#t31-bars", root).innerHTML = rows.map(([n, b]) => `<div class="row" style="grid-template-columns:80px 1fr 80px"><span class="mono">${n}</span>
        <div style="position:relative;height:18px;background:var(--surface-2);border-radius:3px"><div class="b" style="height:100%;width:${b / scale * 100}%;background:${b > 80e9 ? "var(--bad)" : "var(--kv)"}"></div>
        <div style="position:absolute;top:-3px;bottom:-3px;left:${80e9 / scale * 100}%;width:2px;background:var(--bad)"></div></div><span class="v">${gb(b)}</span></div>`).join("");
    });
  },
});

/* ---------- Topic 32 ---------- */
const CB_REQS = [
  { id: "A", arr: 0, len: 8 }, { id: "B", arr: 0, len: 3 }, { id: "C", arr: 0, len: 5 }, { id: "D", arr: 1, len: 4 },
  { id: "E", arr: 2, len: 6 }, { id: "F", arr: 3, len: 2 }, { id: "G", arr: 4, len: 5 }, { id: "H", arr: 6, len: 3 },
];
function simulateBatching(mode, slots = 3) {
  const grid = Array.from({ length: slots }, () => []), running = new Array(slots).fill(null), queue = [], finish = {};
  let t = 0, wasted = 0;
  while (Object.keys(finish).length < CB_REQS.length && t < 60) {
    CB_REQS.filter(r => r.arr === t).forEach(r => queue.push({ ...r, left: r.len }));
    if (mode === "static") {
      if (running.every(x => !x)) for (let s = 0; s < slots && queue.length; s++) running[s] = queue.shift();
    } else {
      for (let s = 0; s < slots && queue.length; s++) if (!running[s]) running[s] = queue.shift();
    }
    for (let s = 0; s < slots; s++) {
      const r = running[s];
      if (r) { grid[s][t] = r.id; r.left--; if (r.left === 0) { finish[r.id] = t + 1; running[s] = null; } }
      else { grid[s][t] = null; if (queue.length) wasted++; }
    }
    t++;
  }
  const avg = CB_REQS.reduce((a, r) => a + (finish[r.id] - r.arr), 0) / CB_REQS.length;
  return { grid, T: t, avg, wasted };
}
topic({
  num: 32, title: "Continuous batching",
  question: "Why make new requests wait for the whole batch to finish?",
  render: () => `
    ${say(
      "With simple <b>static batching</b>, the GPU takes a group of requests and runs them <b>until every one is finished</b>. A short answer finishes early, but its slot sits empty while the long answer keeps going, and new requests wait outside.",
      "<b>Continuous batching</b> checks after <b>every decode step</b>: if a request finished, a waiting request takes its slot immediately. The batch stays full."
    )}
    ${see("How do the two schedules compare on the same requests?", `
      <div class="btn-row"><button type="button" class="btn primary" id="t32-play">▶ Play timeline</button><button type="button" class="btn" id="t32-all">Show all</button></div>
      <div class="tokens" id="t32-reqs"></div>
      <div class="mini-label">Static batching (3 slots)</div>
      <div class="scroll-x"><div class="gantt" id="t32-static"></div></div>
      <div class="mini-label">Continuous batching (3 slots)</div>
      <div class="scroll-x"><div class="gantt" id="t32-cont"></div></div>
      <div class="scroll-x"><table class="cmp-table" id="t32-stats"></table></div>
      <p class="c-muted" style="font-size:.82rem">Each column is one decode step. Letters = which request is using the slot. Hatched = an empty slot. Numbers in the chips = arrival step, output length.</p>`)}
    ${ex(`<div class="calc">B needs 3 tokens, A needs 8. Static: B's slot is empty for 5 steps while D, E and F wait.<br>
      Continuous: D takes B's slot at step 4. All 8 requests finish in <b>13 steps instead of 19</b>.</div>`)}
    ${tech(`<ul>
      <li>Also called <b>in-flight</b> or <b>iteration-level</b> batching: the scheduler re-forms the batch at every forward pass.</li>
      <li>New requests' prefill can be mixed into the same step as other requests' decode (see chunked prefill).</li>
      <li>Needs flexible KV cache management, since requests of different lengths come and go (see paged KV cache).</li>
    </ul>`)}
    ${inf(`<p>Remember topic 19: decode is cheapest per token when the batch is <b>full</b>, because the weights are read once for everyone. Continuous batching keeps the batch full all the time, raising throughput and GPU utilization and cutting waiting time.</p>`)}
    ${check({
      q: "What is the key change in continuous batching?",
      opts: ["Requests are grouped by length before starting", "Finished requests leave and new ones join after every decode step", "Each request gets its own GPU", "Prefill is skipped"],
      a: 1, why: "The batch is updated at every step, so empty slots are refilled immediately instead of waiting for the longest request.",
    })}`,
  mount(root, ctx) {
    const st = simulateBatching("static"), co = simulateBatching("continuous"), cols = Math.max(st.T, co.T);
    const color = id => REQ_COLORS[id.charCodeAt(0) - 65];
    $("#t32-reqs", root).innerHTML = CB_REQS.map(r => `<span class="tok" style="background:${color(r.id)};color:#fff;border-color:transparent">${r.id} · arrives ${r.arr} · ${r.len} tokens</span>`).join("");
    const render = (el, sim, upto) => {
      let html = `<div class="gr" style="--cols:${cols}"><span class="gl">step</span>${Array.from({ length: cols }, (_, t) => `<span class="gl" style="text-align:center">${t + 1}</span>`).join("")}</div>`;
      sim.grid.forEach((row, s) => {
        html += `<div class="gr" style="--cols:${cols}"><span class="gl">slot ${s + 1}</span>`;
        for (let t = 0; t < cols; t++) {
          const id = row[t], hide = t >= upto ? "hide" : "";
          html += t >= sim.T ? `<span class="gc ${hide}" style="background:transparent"></span>` : id ? `<span class="gc ${hide}" style="background:${color(id)}">${id}</span>` : `<span class="gc idle ${hide}"></span>`;
        }
        html += `</div>`;
      });
      el.innerHTML = html;
    };
    const draw = (upto) => { render($("#t32-static", root), st, upto); render($("#t32-cont", root), co, upto); };
    $("#t32-stats", root).innerHTML = `<tr><th></th><th>Steps to finish all</th><th>Average time per request</th><th>Empty slot-steps while requests waited</th></tr>
      <tr><td><b>Static</b></td><td class="mono">${st.T}</td><td class="mono">${fmt(st.avg, 1)} steps</td><td class="mono c-bad">${st.wasted}</td></tr>
      <tr><td><b>Continuous</b></td><td class="mono">${co.T}</td><td class="mono">${fmt(co.avg, 1)} steps</td><td class="mono c-ok">${co.wasted}</td></tr>`;
    draw(cols);
    let timer = null;
    $("#t32-play", root).onclick = () => {
      if (timer) ctx.stop(timer); let u = 0; draw(0);
      timer = ctx.every(380, () => { u++; draw(u); if (u >= cols) { ctx.stop(timer); timer = null; } });
    };
    $("#t32-all", root).onclick = () => { if (timer) { ctx.stop(timer); timer = null; } draw(cols); };
  },
});

/* ---------- Topic 33 ---------- */
topic({
  num: 33, title: "Prefix caching",
  question: "Why prefill the same system prompt a thousand times?",
  render: () => `
    ${say(
      "Many requests start with the <b>same text</b>: a long system prompt, a company policy, a shared document, few-shot examples.",
      "Prefill of that shared start produces the same KV cache every time. <b>Prefix caching</b> keeps that KV cache after the first request and <b>reuses it</b> for later requests that start with exactly the same tokens. Only the different ending needs prefill."
    )}
    ${see("How much prefill is saved when requests share a prefix?", `
      <div class="btn-row">${seg("t33-mode", [["off", "Prefix cache OFF"], ["on", "Prefix cache ON"]], "on")}
        <button type="button" class="btn primary" id="t33-play">▶ Send 3 requests</button></div>
      <div id="t33-rows" style="display:grid;gap:10px"></div>
      <div class="stat-row">
        <div class="stat"><span class="v c-compute" id="t33-comp">–</span><span class="k">prompt tokens prefilled</span></div>
        <div class="stat"><span class="v c-kv" id="t33-reuse">–</span><span class="k">prompt tokens reused from cache</span></div>
        <div class="stat"><span class="v" id="t33-save">–</span><span class="k">prefill work saved</span></div>
      </div>`)}
    ${ex(`<div class="calc">Shared policy text = 2,000 tokens; questions = 30, 25, 40 tokens<br>
      Without cache: 3 × 2,000 + 95 = <b>6,095</b> tokens of prefill<br>
      With cache: 2,000 + 95 = <b>2,095</b> tokens → about <b>66% less</b> prefill, much lower TTFT for requests 2 and 3</div>`)}
    ${tech(`<ul>
      <li>Only an <b>exact token-for-token prefix from the start</b> can be reused: KV entries depend on every earlier token.</li>
      <li>Engines usually match prefixes in fixed-size <b>blocks</b> of tokens (e.g. hashing each block), which fits naturally with paged KV cache.</li>
      <li>Cached prefixes take GPU memory, so they are evicted (for example least-recently-used) when space is needed.</li>
      <li>Tip: put the stable content (system prompt, documents) <b>first</b> and the changing content <b>last</b> to get more cache hits.</li>
    </ul>`)}
    ${inf(`<p>Prefix caching removes repeated <b class="c-compute">compute-bound prefill work</b>, which cuts TTFT and frees compute for other requests. It trades a bit of GPU memory (the stored KV) for a lot of skipped computation: the same bargain as the KV cache itself.</p>`)}
    ${check({
      q: "Request 1: “[policy] What is the refund window?” Request 2: “[policy] How do I file a claim?” What can prefix caching reuse for Request 2?",
      opts: ["Nothing: the questions differ", "The KV cache for the shared [policy] tokens", "The whole answer from Request 1", "Only the last token"],
      a: 1, why: "The shared beginning produces identical K/V entries, so they can be reused. The different question still needs its own prefill, and the answer is generated fresh.",
    })}`,
  mount(root, ctx) {
    const reqs = [["What is the refund window?", 30], ["How do I file a claim?", 25], ["Who approves exceptions?", 40]], P = 2000;
    let timer = null;
    const draw = (shown) => {
      const on = segVal(root, "t33-mode") === "on";
      let comp = 0, reuse = 0;
      $("#t33-rows", root).innerHTML = reqs.map(([q, n], i) => {
        const vis = i < shown, cached = on && i > 0;
        if (vis) { comp += n + (cached ? 0 : P); reuse += cached ? P : 0; }
        return `<div style="display:grid;grid-template-columns:90px 1fr;gap:10px;align-items:center;opacity:${vis ? 1 : .3};transition:opacity .3s">
          <span class="mono" style="font-size:.8rem">Request ${i + 1}</span>
          <div style="display:flex;gap:3px;height:38px">
            <div style="flex:0 0 72%;border-radius:5px;display:grid;place-items:center;color:#fff;font-size:.78rem;font-weight:600;background:${cached ? "var(--kv)" : "var(--compute)"}">
              ${cached ? "↺ reuse cached KV: Company policy… (2,000 tokens)" : "compute prefill: Company policy… (2,000 tokens)"}</div>
            <div style="flex:1;border-radius:5px;display:grid;place-items:center;font-size:.74rem;background:var(--compute-soft);color:var(--compute);border:1px solid var(--compute);padding:0 6px;text-align:center;line-height:1.1">${q} (${n})</div>
          </div></div>`;
      }).join("");
      $("#t33-comp", root).textContent = comp.toLocaleString("en-US");
      $("#t33-reuse", root).textContent = reuse.toLocaleString("en-US");
      $("#t33-save", root).textContent = shown ? `${Math.round(reuse / (comp + reuse) * 100)}%` : "–";
    };
    bindSeg(root, "t33-mode", () => draw(3));
    $("#t33-play", root).onclick = () => {
      if (timer) ctx.stop(timer); let k = 0; draw(0);
      timer = ctx.every(900, () => { k++; draw(k); if (k >= 3) { ctx.stop(timer); timer = null; } });
    };
    draw(3);
  },
});

/* ---------- Topic 34 ---------- */
topic({
  num: 34, title: "Chunked prefill",
  question: "Why should one huge prompt not freeze everyone else's stream?",
  render: () => `
    ${say(
      "Imagine three users happily streaming answers. Then someone pastes a <b>10,000-token document</b>. Its prefill is one enormous pass, and while the GPU runs it, the three streams <b>freeze</b>.",
      "<b>Chunked prefill</b> cuts the big prompt into smaller pieces and processes <b>one piece per step, alongside the ongoing decode tokens</b>. Everyone keeps moving."
    )}
    ${see("What happens to the other users' streams?", `
      <div class="btn-row">${seg("t34-chunk", [["10000", "No chunking"], ["4000", "4,000-token chunks"], ["2000", "2,000-token chunks"], ["1000", "1,000-token chunks"]], "10000")}</div>
      <svg class="svg-block" viewBox="0 0 720 230" id="t34-svg"></svg>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t34-itl">–</span><span class="k">worst gap between tokens for users A–C</span></div>
        <div class="stat"><span class="v" id="t34-ttft">–</span><span class="k">new user's time to first token</span></div>
        <div class="stat"><span class="v" id="t34-steps">–</span><span class="k">steps with the new prompt</span></div>
      </div>
      <p class="c-muted" style="font-size:.82rem">Time units are made up: prefill costs 1 unit per 1,000 tokens, a decode step for the 3 users costs 0.1. Each tick = a token for users A–C. ${illus("Conceptual timing")}</p>`)}
    ${ex(`<div class="calc">10,000-token prompt, no chunking → one 10-unit step → users A–C wait <b>~10 units</b> for their next token<br>
      1,000-token chunks → 10 steps of 1.1 units → users A–C wait at most <b>~1.1 units</b>; new user's TTFT grows slightly (~11 units)</div>`)}
    ${tech(`<ul>
      <li>The scheduler sets a <b>token budget per step</b>. Decode tokens are included first; leftover budget goes to prefill chunks.</li>
      <li>Each chunk attends to the KV cache of the earlier chunks, so the result is the same as a single prefill.</li>
      <li>Trade-off: much smoother inter-token latency for everyone, a slightly longer TTFT for the long prompt, and better GPU utilization by mixing compute-heavy prefill with memory-heavy decode.</li>
    </ul>`)}
    ${inf(`<p>Mixed workloads (short chats plus long documents) are the norm in real serving. Chunked prefill, combined with continuous batching, keeps latency predictable.</p>`)}
    ${check({
      q: "What is the main benefit of chunked prefill for users who are already streaming?",
      opts: ["Their answers become more accurate", "Their token stream isn't frozen by someone else's huge prompt", "Their KV cache disappears", "Their prompt is skipped"],
      a: 1, why: "Chunking stops one huge prefill from blocking decode steps, so everyone else's inter-token latency stays low.",
    })}`,
  mount(root, ctx) {
    const draw = (chunk) => {
      chunk = +chunk;
      const iters = []; let t = 0;
      for (let i = 0; i < 3; i++) { iters.push({ t0: t, d: 0.1, pre: 0 }); t += 0.1; }
      const arrive = t; let left = 10000;
      while (left > 0) { const c = Math.min(chunk, left); left -= c; const d = 0.1 + c / 1000; iters.push({ t0: t, d, pre: c }); t += d; }
      const firstTok = t;
      for (let i = 0; i < 4; i++) { iters.push({ t0: t, d: 0.13, pre: 0, newUser: true }); t += 0.13; }
      const T = t, X = v => 110 + v / T * 590;
      let s = `<text x="10" y="42" font-size="12" font-weight="600">Users A–C</text><text x="10" y="112" font-size="12" font-weight="600">New user</text><text x="10" y="126" font-size="11" class="muted">(10k prompt)</text>
        <line x1="110" y1="190" x2="700" y2="190" stroke="var(--line)"/>`;
      let maxGap = 0, prev = 0;
      iters.forEach(it => {
        const end = it.t0 + it.d;
        s += `<rect x="${X(it.t0) + .5}" y="160" width="${Math.max(1, X(end) - X(it.t0) - 1)}" height="22" rx="2" fill="${it.pre ? "var(--compute)" : "var(--memory)"}" opacity="${it.pre ? .85 : .5}"/>`;
        for (let u = 0; u < 3; u++) s += `<line x1="${X(end)}" x2="${X(end)}" y1="${26 + u * 12}" y2="${34 + u * 12}" stroke="var(--memory)" stroke-width="2"/>`;
        maxGap = Math.max(maxGap, end - prev); prev = end;
        if (it.pre) s += `<rect x="${X(it.t0) + 1}" y="96" width="${Math.max(2, X(end) - X(it.t0) - 2)}" height="26" rx="3" fill="var(--compute)"/>${X(end) - X(it.t0) > 40 ? `<text x="${(X(it.t0) + X(end)) / 2}" y="113" text-anchor="middle" font-size="10" font-weight="700" style="fill:#fff">${(it.pre / 1000)}k</text>` : ""}`;
        if (it.newUser) s += `<line x1="${X(end)}" x2="${X(end)}" y1="100" y2="118" stroke="var(--ok)" stroke-width="2.5"/>`;
      });
      s += `<line x1="${X(arrive)}" x2="${X(arrive)}" y1="86" y2="190" stroke="var(--ink)" stroke-dasharray="3 3"/><text x="${X(arrive) + 4}" y="84" font-size="10" class="muted">prompt arrives</text>
        <line x1="${X(firstTok)}" x2="${X(firstTok)}" y1="92" y2="130" stroke="var(--ok)" stroke-width="2"/><text x="${Math.min(X(firstTok) + 4, 640)}" y="142" font-size="10" style="fill:var(--ok)">first token</text>
        <text x="110" y="206" font-size="11" class="muted">GPU steps (orange = includes prefill chunk, blue = decode only) → time</text>`;
      $("#t34-svg", root).innerHTML = s;
      $("#t34-itl", root).innerHTML = `<span class="${maxGap > 3 ? "c-bad" : "c-ok"}">${fmt(maxGap, 1)} units</span>`;
      $("#t34-ttft", root).textContent = `${fmt(firstTok - arrive, 1)} units`;
      $("#t34-steps", root).textContent = Math.ceil(10000 / chunk);
    };
    bindSeg(root, "t34-chunk", draw); draw(10000);
  },
});

/* ---------- Topic 35 ---------- */
topic({
  num: 35, title: "KV cache optimizations: paged KV cache",
  question: "Why does KV cache memory get wasted, and how do we stop it?",
  render: () => `
    ${say(
      "Nobody knows in advance how long an answer will be. A simple server <b>reserves one continuous chunk of memory for the longest possible answer</b>. Most of it stays empty, and the gaps left behind by finished requests are often too small to reuse.",
      "<b>Paged KV cache</b> (the idea behind PagedAttention) works like an operating system's virtual memory: split memory into small fixed-size <b>blocks</b>, give a request a new block only when it actually needs one, and let its blocks sit <b>anywhere</b>. A small table remembers where each request's blocks are."
    )}
    ${see("Same events, two memory managers: where does space get wasted?", `
      <div class="btn-row">
        <button type="button" class="btn" id="t35-back">← Back</button>
        <button type="button" class="btn primary" id="t35-next">Next event →</button>
        <button type="button" class="btn" id="t35-reset">Reset</button>
      </div>
      <div class="stage-panel" id="t35-event" style="min-height:56px"></div>
      <div class="grid2">
        <div style="display:grid;gap:8px"><h4>Contiguous reservation</h4><div class="blocks" id="t35-naive"></div><div class="stat-row" id="t35-ns"></div></div>
        <div style="display:grid;gap:8px"><h4>Paged KV cache</h4><div class="blocks" id="t35-paged"></div><div class="stat-row" id="t35-ps"></div><div id="t35-table" class="mono" style="font-size:.78rem"></div></div>
      </div>
      <div class="btn-row" style="gap:14px;font-size:.8rem"><span><span class="pill" style="background:var(--memory);color:#fff">A</span> block holding real KV data</span><span><span class="pill bad">▨</span> reserved but empty (wasted)</span><span><span class="pill" style="background:var(--surface-3)">&nbsp;</span> free</span></div>`)}
    ${ex(`<div class="calc">40 memory blocks. A, B, C reserve 10, 14 and 10 blocks but use only 2–5 each.<br>
      A finishes → 16 blocks are free, but the largest continuous gap is only 10.<br>
      D needs a 14-block reservation → <span class="c-bad">rejected</span> with contiguous reservation · <span class="c-ok">served</span> with paging</div>`)}
    ${tech(`<ul>
      <li><b>Internal fragmentation:</b> memory reserved for a request but not used (over-reserving for max length).</li>
      <li><b>External fragmentation:</b> free memory broken into gaps too small for a new contiguous reservation.</li>
      <li><b>Paged KV cache:</b> fixed-size KV blocks (e.g. 16 tokens each), allocated on demand, mapped by a per-request block table; waste is limited to the last, partly-filled block.</li>
      <li>Blocks can be <b>shared</b> between requests (prefix caching, parallel sampling) with copy-on-write, and freed or evicted cleanly.</li>
    </ul>`)}
    ${inf(`<p>Less wasted KV memory means <b>more requests fit in a batch</b>. Since batching is what makes memory-bound decode efficient, better KV memory management turns directly into higher throughput on the same GPU.</p>`)}
    ${check({
      q: "Why can paged KV cache serve a new request when contiguous reservation cannot, even with the same total free memory?",
      opts: ["It compresses the KV cache to 1 bit", "Its blocks don't need to be next to each other, and it only allocates what is actually used", "It moves the KV cache to disk", "It deletes old requests"],
      a: 1, why: "Paging avoids both over-reservation and the need for one continuous gap, so scattered free blocks are still usable.",
    })}`,
  mount(root, ctx) {
    const TOTAL = 40;
    const colors = { A: REQ_COLORS[0], B: REQ_COLORS[1], C: REQ_COLORS[2], D: REQ_COLORS[3] };
    const events = [
      { txt: "Start: GPU KV memory is empty. 40 blocks.", fn: () => { } },
      { txt: "Request <b>A</b> arrives. Could grow to 10 blocks; right now needs 2.", fn: s => add(s, "A", 10, 2) },
      { txt: "Request <b>B</b> arrives. Could grow to 14 blocks; needs 3.", fn: s => add(s, "B", 14, 3) },
      { txt: "Request <b>C</b> arrives. Could grow to 10 blocks; needs 2.", fn: s => add(s, "C", 10, 2) },
      { txt: "All three generate more tokens: A now needs 4 blocks, B 5, C 3.", fn: s => { grow(s, "A", 4); grow(s, "B", 5); grow(s, "C", 3); } },
      { txt: "Request <b>A</b> finishes and frees its memory.", fn: s => free(s, "A") },
      { txt: "Request <b>D</b> arrives. Could grow to 14 blocks; needs 3.", fn: s => add(s, "D", 14, 3) },
    ];
    const fresh = () => ({ naive: new Array(TOTAL).fill(null), paged: new Array(TOTAL).fill(null), res: {}, table: {}, rejected: [] });
    function add(s, id, max, need) {
      let start = -1, run = 0;
      for (let i = 0; i < TOTAL; i++) { run = s.naive[i] ? 0 : run + 1; if (run === max) { start = i - max + 1; break; } }
      if (start >= 0) { s.res[id] = { start, max, used: need }; for (let i = 0; i < max; i++) s.naive[start + i] = { id, used: i < need }; }
      else s.rejected.push(id);
      s.table[id] = [];
      grow(s, id, need, true);
    }
    function grow(s, id, need, pagedOnly) {
      if (!pagedOnly && s.res[id]) { s.res[id].used = need; for (let i = 0; i < s.res[id].max; i++) s.naive[s.res[id].start + i] = { id, used: i < need }; }
      while (s.table[id].length < need) { const f = s.paged.findIndex(x => !x); if (f < 0) break; s.paged[f] = id; s.table[id].push(f); }
    }
    function free(s, id) {
      if (s.res[id]) { for (let i = 0; i < s.res[id].max; i++) s.naive[s.res[id].start + i] = null; delete s.res[id]; }
      s.table[id].forEach(b => s.paged[b] = null); delete s.table[id];
    }
    let step = 0;
    const draw = () => {
      const s = fresh(); for (let i = 0; i <= step; i++) events[i].fn(s);
      $("#t35-event", root).innerHTML = `<div class="mini-label">Event ${step + 1} of ${events.length}</div><p>${events[step].txt}</p>`;
      $("#t35-naive", root).innerHTML = s.naive.map(b => !b ? `<i></i>` : b.used ? `<i style="background:${colors[b.id]}">${b.id}</i>` : `<i class="reserved">${b.id}</i>`).join("");
      $("#t35-paged", root).innerHTML = s.paged.map(b => b ? `<i style="background:${colors[b]}">${b}</i>` : `<i></i>`).join("");
      const nUsed = s.naive.filter(b => b && b.used).length, nRes = s.naive.filter(b => b && !b.used).length, nFree = TOTAL - nUsed - nRes;
      let big = 0, run = 0; s.naive.forEach(b => { run = b ? 0 : run + 1; big = Math.max(big, run); });
      const pUsed = s.paged.filter(Boolean).length;
      const rej = s.rejected.length ? `<div class="stat"><span class="v c-bad">${s.rejected.join(", ")} rejected</span><span class="k">no continuous gap big enough</span></div>` : "";
      $("#t35-ns", root).innerHTML = `<div class="stat"><span class="v">${nUsed}</span><span class="k">blocks with data</span></div><div class="stat"><span class="v c-bad">${nRes}</span><span class="k">reserved, empty</span></div><div class="stat"><span class="v">${nFree} (largest gap ${big})</span><span class="k">free</span></div>${rej}`;
      $("#t35-ps", root).innerHTML = `<div class="stat"><span class="v">${pUsed}</span><span class="k">blocks with data</span></div><div class="stat"><span class="v c-ok">0</span><span class="k">reserved, empty</span></div><div class="stat"><span class="v">${TOTAL - pUsed}</span><span class="k">free (all usable)</span></div>`;
      $("#t35-table", root).innerHTML = Object.keys(s.table).length ? `<div class="mini-label" style="margin-bottom:4px">Block table</div>` + Object.entries(s.table).map(([id, bl]) => `<div><b style="color:${colors[id]}">${id}</b>: logical 0…${bl.length - 1} → physical [${bl.join(", ")}]</div>`).join("") : "";
      $("#t35-back", root).disabled = step === 0; $("#t35-next", root).disabled = step === events.length - 1;
    };
    $("#t35-next", root).onclick = () => { step = Math.min(step + 1, events.length - 1); draw(); };
    $("#t35-back", root).onclick = () => { step = Math.max(step - 1, 0); draw(); };
    $("#t35-reset", root).onclick = () => { step = 0; draw(); };
    draw();
  },
});
