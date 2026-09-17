/* ============================================================
   PART 1 — How LLM Inference Works (topics 1–4)
   ============================================================ */
part({
  num: 1, title: "How LLM Inference Works",
  blurb: "What the model actually does when it answers: one forward pass per new token, with prefill first and decode after.",
  story: "LLM inference → <b>prefill + decode</b> → these two phases do very different kinds of work. Everything later builds on this split.",
});

/* ---------- Topic 1 ---------- */
topic({
  num: 1, title: "What happens during LLM inference",
  question: "What exactly happens, step by step, when a model answers?",
  render: () => `
    ${say(
      "Inference means <b>using</b> a trained model to produce text. The weights are fixed; nothing is learned.",
      "The model reads the text so far, predicts <b>one</b> next token, adds it to the text, and runs again. Press <b>Play</b> to watch every step for the prompt “Why is the sky”."
    )}
    ${see("What happens between sending a prompt and getting the answer?", `
      ${player("t1-pl")}
      <div class="caption" id="t1-cap"></div>
      <div class="inf-top">
        <div style="display:grid;gap:6px"><div class="mini-label">Text so far (outlined = being processed in this pass)</div><div class="textbox" id="t1-text"></div></div>
        <div class="counters">
          <div class="stat"><span class="v" id="t1-phase">–</span><span class="k">phase</span></div>
          <div class="stat"><span class="v" id="t1-pass">0</span><span class="k">forward passes run</span></div>
          <div class="stat"><span class="v" id="t1-gen">0</span><span class="k">tokens generated</span></div>
          <div class="stat"><span class="v c-kv" id="t1-kv">0</span><span class="k">KV cache entries</span></div>
        </div>
      </div>
      <div class="scroll-x">${flow(["Text", "Tokens", "Token IDs", "Embeddings", { t: "Transformer layers (GPU)", c: "compute" }, "Next-token scores", "Pick a token", "Append &amp; repeat"], { id: "t1-flow" })}</div>
      <div class="stage-view" id="t1-view"></div>`)}
    ${see("How is this different from training?", `
      <div class="grid2">
        <div class="loop-box"><h4>Training <span class="pill memory">weights change</span></h4>
          ${flow([{ t: "Data batch" }, { t: "Model", c: "compute" }, { t: "Loss" }, { t: "Update weights", c: "memory" }, { t: "↺ repeat" }], { id: "t1-train" })}
          <p class="c-muted" style="font-size:.85rem">Forward pass + backward pass + weight update, millions of times.</p></div>
        <div class="loop-box"><h4>Inference <span class="pill ok">weights frozen</span></h4>
          ${flow([{ t: "Text so far" }, { t: "Model", c: "compute" }, { t: "Next token", c: "kv" }, { t: "Add to text" }, { t: "↺ repeat" }], { id: "t1-inf" })}
          <p class="c-muted" style="font-size:.85rem">Forward pass only: one pass per new token.</p></div>
      </div>`)}
    ${ex(`<div class="calc">Prompt “Why is the sky” = 4 tokens → <b>1 prefill pass</b> reads all 4 together → first token “ blue”<br>
      Each further token = <b>1 decode pass</b> → a 200-token answer ≈ 200 passes through the whole model</div>`)}
    ${tech(`<ul>
      <li><b>Forward pass:</b> running the input through every layer, from input to output, using the weights.</li>
      <li><b>Next-token prediction:</b> the last layer gives a score (logit) for every token in the vocabulary; a sampler picks one.</li>
      <li><b>Autoregressive generation:</b> each output token is appended and becomes part of the next input.</li>
      <li><b>Prefill</b> processes the prompt; <b>decode</b> produces one token per pass. Both are covered in detail in topic 3.</li>
    </ul>`)}
    ${inf(`<p>The GPU runs the model <b>again and again</b>, once per new token, and every pass reads the model's weights. How fast each pass runs decides how fast text appears.</p>`)}
    ${check({
      q: "During inference, what happens to the model weights?",
      opts: ["They are updated after every token", "They stay frozen and are only read", "They are updated after each full response", "They are deleted after the prompt is read"],
      a: 1, why: "Inference never changes the weights. It only reads them, for every token. That repeated reading will matter a lot when we get to GPU memory.",
    })}`,
  mount(root, ctx) {
    let li = 0;
    ctx.every(900, () => { setFlow($("#t1-train", root), li % 5); setFlow($("#t1-inf", root), li % 5); li++; });

    const P = ["Why", " is", " the", " sky"], G = [" blue", " because", " of", " scattering"];
    const IDS = { "Why": 5195, " is": 318, " the": 262, " sky": 6766, " blue": 4171, " because": 780, " of": 286, " scattering": 45765 };
    const CANDS = [
      [[" blue", 71], [" so", 9], [" red", 5], [" dark", 4], [" clear", 3]],
      [[" because", 48], ["?", 22], [" during", 12], [",", 8], [" in", 4]],
      [[" of", 62], [" sunlight", 20], [" the", 7], [" air", 5], [" light", 3]],
      [[" scattering", 55], [" Rayleigh", 25], [" air", 8], [" light", 6], [" dust", 2]],
    ];
    const frames = [];
    for (let s = 0; s <= 7; s++) frames.push({ p: 0, s });
    for (let p = 1; p < G.length; p++) [2, 3, 4, 5, 6, 7].forEach(s => frames.push({ p, s }));
    frames.push({ p: G.length, s: 8 });

    const esc = t => t.replace(/ /g, "␣");
    const vec = (t) => { let h = 7; for (const c of t) h = (h * 31 + c.charCodeAt(0)) % 10007; return Array.from({ length: 8 }, (_, i) => Math.sin(h * (i + 3) * 0.37)); };
    const cellColor = v => v >= 0 ? `rgba(47,107,220,${0.15 + Math.abs(v) * 0.75})` : `rgba(223,116,20,${0.15 + Math.abs(v) * 0.75})`;

    const render = (fi) => {
      const f = frames[fi], done = f.s === 8;
      const appended = done ? G.length : f.p + (f.s >= 7 ? 1 : 0);
      const all = P.concat(G.slice(0, appended));
      const procIdx = f.p === 0 ? [0, 1, 2, 3] : [P.length + f.p - 1];
      const procToks = procIdx.map(i => P.concat(G)[i]);
      const processing = !done && f.s >= 1 && f.s <= 4;
      $("#t1-text", root).innerHTML = all.map((t, i) =>
        `<span class="tok ${i < P.length ? "prompt" : "gen"} ${processing && procIdx.includes(i) ? "proc" : ""} ${f.s === 7 && i === all.length - 1 ? "new" : ""}">${t}</span>`).join("");
      $("#t1-phase", root).textContent = done ? "Done" : f.p === 0 ? "Prefill" : "Decode";
      $("#t1-pass", root).textContent = done ? G.length : f.p + (f.s >= 4 ? 1 : 0);
      $("#t1-gen", root).textContent = appended;
      $("#t1-kv", root).textContent = done ? P.length + G.length - 1 : f.s >= 4 ? P.length + f.p : (f.p === 0 ? 0 : P.length + f.p - 1);
      setFlow($("#t1-flow", root), done ? -1 : f.s);

      const passName = f.p === 0 ? "Prefill pass" : `Decode pass ${f.p}`;
      let cap = "", view = "";
      const cand = CANDS[Math.min(f.p, CANDS.length - 1)];
      switch (done ? 8 : f.s) {
        case 0:
          cap = "The prompt arrives as plain text. A computer can't do maths on letters, so it must be turned into numbers first.";
          view = `<h4>Plain text</h4><div class="tokens"><span class="tok" style="font-size:1.3rem;padding:6px 14px">Why is the sky</span></div>`;
          break;
        case 1:
          cap = "The tokenizer splits the text into tokens: whole words or pieces of words. Spaces belong to the token after them.";
          view = `<h4>Tokens</h4><div class="tokens">${P.map(t => `<span class="tok prompt new" style="font-size:1.1rem">${esc(t)}</span>`).join("")}</div><p class="c-muted">4 tokens. (␣ marks a space.)</p>`;
          break;
        case 2:
          cap = f.p === 0 ? "Each token is looked up in the vocabulary and replaced by its ID number." : `${passName}: the newest token “${procToks[0].trim()}” is already a token, so it just needs its ID. The earlier tokens are not sent again.`;
          view = `<h4>Token IDs</h4><div class="tokens">${procToks.map(t => `<span class="tok ${f.p ? "gen" : "prompt"}" style="font-size:1rem">${esc(t)} → <b>${IDS[t]}</b></span>`).join("")}</div><p class="c-muted" style="font-size:.85rem">IDs are illustrative; every tokenizer numbers its vocabulary differently.</p>`;
          break;
        case 3:
          cap = "Each ID selects a row from the embedding table: a list of numbers (a vector) that stands for the token's meaning.";
          view = `<h4>Embeddings (${procToks.length} vector${procToks.length > 1 ? "s" : ""})</h4><div class="embed">${procToks.map(t => `<div class="erow"><span>${esc(t)}</span>${vec(t).map(v => `<i style="background:${cellColor(v)}" title="${v.toFixed(2)}"></i>`).join("")}</div>`).join("")}</div>
            <p class="c-muted" style="font-size:.85rem">8 numbers shown per token; a 70B-class model uses 8,192.</p>`;
          break;
        case 4:
          cap = f.p === 0
            ? "Prefill: all 4 prompt vectors go through every Transformer layer together, in parallel. Each layer reads its weights from GPU memory, and the Keys and Values of all 4 tokens are saved in the KV cache."
            : `${passName}: only the ONE new vector goes through the layers. The earlier tokens are not recomputed: attention reads their Keys and Values from the KV cache, and this token's K and V are added to it.`;
          view = `<h4>Transformer layers on the GPU</h4>
            <div class="grid2" style="grid-template-columns:minmax(0,1fr) 250px;align-items:start">
              <div style="display:grid;gap:8px">
                <div class="layer-stack layer-run">${[1, 2, 3, 4, 5, 6].map((n, i) => `<div class="lyr" style="animation-delay:${i * 0.25}s"><b>Layer ${n}</b><span class="a">attention</span><span class="m">MLP</span></div>`).join("")}<div class="lyr" style="border-style:dashed"><b>…</b><span>up to</span><b>80</b></div></div>
                <p style="font-size:.9rem">Processing <b>${procToks.length} token${procToks.length > 1 ? "s" : ""}</b> in this pass: ${procToks.map(t => `<code>${esc(t)}</code>`).join(" ")}</p>
                <p class="c-muted" style="font-size:.85rem">Topic 10 follows one token through these layers step by step and shows which GPU parts do each step.</p>
              </div>
              <div class="mem-side"><div class="mini-label">GPU memory (HBM)</div>
                <div class="mem-chip w"><span>Model weights</span><b>read by every layer</b></div>
                <div class="mem-chip kvc"><span>KV cache</span><b>${f.p === 0 ? "write 4" : `read ${P.length + f.p - 1}, write 1`}</b></div>
              </div>
            </div>`;
          break;
        case 5:
          cap = "The last layer turns the final token's vector into a score for every token in the vocabulary (often 100,000+). Softmax turns the scores into probabilities.";
          view = `<h4>Next-token probabilities (top 5 of the whole vocabulary)</h4><div class="bars-h">${cand.map(([t, p]) => `<div class="row"><span class="mono">${esc(t)}</span><div><div class="b" style="width:${p}%"></div></div><span class="v">${p}%</span></div>`).join("")}</div>`;
          break;
        case 6:
          cap = `A sampler picks one token. Here we take the most likely one: “${cand[0][0].trim()}”.`;
          view = `<h4>Pick a token</h4><div class="bars-h">${cand.map(([t, p], i) => `<div class="row ${i === 0 ? "pick" : ""}"><span class="mono">${esc(t)}</span><div><div class="b" style="width:${p}%;opacity:${i === 0 ? 1 : .3}"></div></div><span class="v">${i === 0 ? "✓ " : ""}${p}%</span></div>`).join("")}</div>`;
          break;
        case 7:
          cap = f.p < G.length - 1
            ? `“${G[f.p].trim()}” is appended. ${passName} is finished. The new token is now the input for the next pass, and everything repeats.`
            : `“${G[f.p].trim()}” is appended. In a real run this continues until an end token or a length limit.`;
          view = `<h4>Append and repeat</h4><div class="tokens" style="font-size:1.1rem">${all.map((t, i) => `<span class="tok ${i < P.length ? "prompt" : "gen"} ${i === all.length - 1 ? "new" : ""}">${t}</span>`).join("")}</div>
            <p>${f.p < G.length - 1 ? `Next: <b>Decode pass ${f.p + 1}</b> takes only “${G[f.p].trim()}” as new input.` : "Generation continues the same way."}</p>`;
          break;
        default:
          cap = "Summary: one prefill pass read the whole prompt. Then each decode pass handled one new token and reused earlier work from the KV cache.";
          view = `<h4>What happened</h4><div class="scroll-x"><table class="cmp-table">
            <tr><th>Pass</th><th>Input tokens processed</th><th>Output token</th><th>KV cache after</th></tr>
            ${G.map((g, p) => `<tr><td>${p === 0 ? "Prefill" : `Decode ${p}`}</td><td class="mono">${p === 0 ? P.map(esc).join(" ") : esc(G[p - 1])}</td><td class="mono">${esc(g)}</td><td class="mono">${P.length + p} entries</td></tr>`).join("")}
          </table></div>`;
      }
      $("#t1-cap", root).innerHTML = `<span class="cap-step">${done ? "Finished" : `${passName} · step ${f.s + 1}`}</span><span>${cap}</span>`;
      $("#t1-view", root).innerHTML = view;
    };
    bindPlayer(root, "t1-pl", ctx, { frames: frames.length, render, interval: 2600 });
  },
});

/* ---------- Topic 2 ---------- */
topic({
  num: 2, title: "What happens when you send a prompt?",
  question: "What steps turn my text into the next word?",
  render: () => `
    ${say("Computers can't do maths on words. So your text is cut into pieces (<b>tokens</b>), each piece becomes a number, each number becomes a list of numbers (a <b>vector</b>), and the model transforms those vectors layer by layer. At the end it scores every possible next token and picks one.")}
    ${see("What happens to the prompt “The cat sat” at each stage?", `
      <div class="pipeline-stages" id="t2-stages"></div>
      <div class="stage-panel" id="t2-panel"></div>
      <div class="btn-row">
        <button type="button" class="btn" id="t2-prev">← Stage</button>
        <button type="button" class="btn primary" id="t2-next">Next stage →</button>
        <button type="button" class="btn" id="t2-play">▶ Play all</button>
      </div>`, "Click each stage")}
    ${ex(`<div class="calc">"The cat sat" → 3 tokens → 3 IDs → 3 vectors → through every layer → scores for ~100k tokens → pick " on"<br>
      Then: "The cat sat on" → run again → " the" → …</div>`)}
    ${tech(`<ul>
      <li><b>Tokenizer:</b> splits text into sub-word pieces from a fixed vocabulary (often 32k–200k entries).</li>
      <li><b>Embedding lookup:</b> a big table (vocabulary × hidden size). Token ID <i>i</i> selects row <i>i</i>.</li>
      <li><b>Transformer layer:</b> attention (tokens exchange information) followed by an MLP (each token is processed on its own). Both are mostly matrix multiplications.</li>
      <li><b>Logits:</b> one raw score per vocabulary token. Softmax turns them into probabilities. <b>Sampling</b> (greedy, top-p, temperature) picks the token.</li>
    </ul>`)}
    ${inf(`<p>Almost all GPU time is spent inside the <b class="c-compute">Transformer layers</b>. Tokenization and sampling are cheap by comparison. The layers are where the weights are read and where the matrix maths happens.</p>`)}
    ${check({
      q: "What are logits?",
      opts: ["The token IDs of the prompt", "One score per possible next token", "The model weights for the last layer", "The cached attention values"],
      a: 1, why: "Logits are raw scores, one for every token in the vocabulary. Softmax turns them into probabilities, and a sampler picks the next token.",
    })}`,
  mount(root, ctx) {
    const stages = [
      { n: "Prompt", h: `<p>You type plain text.</p><div class="tokens"><span class="tok" style="font-size:1.1rem">The cat sat</span></div>` },
      { n: "Tokenization", h: `<p>The text is split into <b>tokens</b>: common words or pieces of words. Note the leading spaces: they are part of the token.</p>
          <div class="tokens"><span class="tok prompt">The</span><span class="tok prompt"> cat</span><span class="tok prompt"> sat</span></div>` },
      { n: "Token IDs", h: `<p>Each token is looked up in the vocabulary and replaced by its ID number.</p>
          <div class="tokens"><span class="tok prompt">The → 464</span><span class="tok prompt"> cat → 3797</span><span class="tok prompt"> sat → 3332</span></div>
          <p class="c-muted" style="font-size:.8rem">IDs shown are illustrative; every tokenizer has its own numbering.</p>` },
      { n: "Embeddings", h: `<p>Each ID picks a row from the embedding table, giving a <b>vector</b>: a list of numbers that stands for the token's meaning.</p>
          <div style="display:grid;gap:6px">
            <div>464 → <span class="vec"><span>0.12</span><span>-0.80</span><span>0.33</span><span>0.05</span><span>…</span></span></div>
            <div>3797 → <span class="vec"><span>0.91</span><span>0.14</span><span>-0.42</span><span>0.60</span><span>…</span></span></div>
            <div>3332 → <span class="vec"><span>-0.27</span><span>0.55</span><span>0.08</span><span>-0.71</span><span>…</span></span></div>
          </div><p class="c-muted" style="font-size:.8rem">Real models use thousands of numbers per token (for example 4,096 or 8,192), not 4.</p>` },
      { n: "Transformer layers", h: `<p>The vectors pass through a stack of identical layers. In each layer, <b>attention</b> lets tokens look at earlier tokens, and an <b>MLP</b> transforms each token. Both steps are big <b class="c-compute">matrix multiplications with the weights</b>.</p>
          <div class="layers" id="t2-layers">${"<i></i>".repeat(24)}</div><p class="c-muted" style="font-size:.8rem">A 70B-class model has about 80 of these layers.</p>` },
      { n: "Logits", h: `<p>The last token's final vector is turned into one score for every token in the vocabulary. Higher score means more likely next.</p>
          <div class="bars-h">${[[" on", 62], [" down", 18], [" in", 7], [" there", 5], [" quietly", 3]].map(([t, p], i) => `<div class="row ${i === 0 ? "pick" : ""}"><span class="mono">"${t}"</span><div><div class="b" style="width:${p}%"></div></div><span class="v">${p}%</span></div>`).join("")}</div>` },
      { n: "Next token", h: `<p>A sampler picks one token. With greedy decoding it takes the highest score.</p>
          <div class="tokens"><span class="tok prompt">The</span><span class="tok prompt"> cat</span><span class="tok prompt"> sat</span><span class="tok gen new"> on</span></div>` },
      { n: "Repeat", h: `<p>The new token is added to the text, and the model runs again to get the next one. This loop continues until a stop token or a length limit.</p>
          <div class="tokens"><span class="tok prompt">The</span><span class="tok prompt"> cat</span><span class="tok prompt"> sat</span><span class="tok gen"> on</span><span class="tok gen new"> the</span><span class="tok ghost"> mat</span><span class="tok ghost"> …</span></div>
          <p class="c-muted" style="font-size:.85rem">Coming up: the model does <i>not</i> need to redo all the old work each time. That is what the KV cache is for.</p>` },
    ];
    let cur = 0, seen = new Set([0]), layerTimer = null, play = null;
    const show = (i) => {
      cur = clamp(i, 0, stages.length - 1); seen.add(cur);
      $("#t2-stages", root).innerHTML = stages.map((s, j) =>
        (j ? `<span class="c-muted">→</span>` : "") + `<button type="button" class="stage-btn ${j === cur ? "on" : seen.has(j) ? "seen" : ""}" data-s="${j}">${s.n}</button>`).join("");
      $("#t2-panel", root).innerHTML = `<div class="mini-label">Stage ${cur + 1} of ${stages.length}</div><h4 style="font-size:1.1rem">${stages[cur].n}</h4>${stages[cur].h}`;
      if (layerTimer) ctx.stop(layerTimer);
      if (stages[cur].n === "Transformer layers") {
        let k = 0; const ls = $$("#t2-layers i", root);
        layerTimer = ctx.every(90, () => { ls.forEach((l, j) => l.classList.toggle("on", j <= k % (ls.length + 6))); k++; });
      }
      $("#t2-prev", root).disabled = cur === 0;
    };
    $("#t2-stages", root).onclick = e => { const b = e.target.closest(".stage-btn"); if (b) show(+b.dataset.s); };
    $("#t2-prev", root).onclick = () => show(cur - 1);
    $("#t2-next", root).onclick = () => show(cur + 1 >= stages.length ? 0 : cur + 1);
    $("#t2-play", root).onclick = () => {
      if (play) { ctx.stop(play); play = null; $("#t2-play", root).textContent = "▶ Play all"; return; }
      show(0); $("#t2-play", root).textContent = "❚❚ Pause";
      play = ctx.every(2600, () => { if (cur >= stages.length - 1) { ctx.stop(play); play = null; $("#t2-play", root).textContent = "▶ Play all"; } else show(cur + 1); });
    };
    show(0);
  },
});

/* ---------- Topic 3 ---------- */
topic({
  num: 3, title: "Prefill vs Decode",
  question: "Why does the answer start slowly, then flow word by word?",
  render: () => `
    ${say(
      "Every request has two phases.",
      "<b class='c-compute'>Prefill</b> = read the prompt you already wrote. All those tokens are known, so the GPU can process them <b>together, in parallel</b>, in one big pass.",
      "<b class='c-kv'>Decode</b> = write the answer. Token 5 cannot be computed before token 4 exists, so decode happens <b>one token per pass</b>, in sequence."
    )}
    ${see("How is the work different in the two phases?", `
      <div class="grid2">
        <div class="loop-box">
          <h4>Prefill <span class="pill compute">all prompt tokens at once</span></h4>
          <div class="tokens" id="t3-prompt"></div>
          <div class="model-box" id="t3-pmodel">GPU</div>
          <div class="stat"><span class="v" id="t3-prows">–</span><span class="k">tokens processed in this pass</span></div>
        </div>
        <div class="loop-box">
          <h4>Decode <span class="pill memory">one token per pass</span></h4>
          <div class="tokens" id="t3-out"><span class="c-muted" style="font-size:.85rem">waiting for prefill…</span></div>
          <div class="model-box" id="t3-dmodel">GPU</div>
          <div class="stat"><span class="v" id="t3-drows">–</span><span class="k">new tokens processed in this pass</span></div>
        </div>
      </div>
      <div>
        <div class="mini-label" style="margin-bottom:6px">Timeline (what the user experiences)</div>
        <div id="t3-timeline" style="display:flex;gap:3px;align-items:flex-end;height:64px;border-bottom:1.5px solid var(--line);padding-bottom:2px"></div>
        <div id="t3-marks" style="position:relative;height:36px;font-size:.78rem"></div>
      </div>
      <div class="btn-row">
        <button type="button" class="btn primary" id="t3-play">▶ Send the prompt</button>
        ${seg("t3-len", [["short", "Short prompt (5 tokens)"], ["long", "Long prompt (2,000 tokens)"]], "long")}
      </div>
      <p class="c-muted" style="font-size:.85rem">Bar height = how many tokens a pass processes. Bar width = roughly how long it takes. ${illus("Conceptual timing")}</p>`)}
    ${ex(`<div class="calc">Prompt = 2,000 tokens, answer = 300 tokens<br>
      Prefill: <b>1 pass</b> over 2,000 tokens → first token appears (TTFT)<br>
      Decode: <b>about 300 passes</b>, 1 new token each → the rest of the answer streams in</div>`)}
    ${tech(`<ul>
      <li><b>TTFT (Time To First Token):</b> time from sending the request to seeing the first output token. Mostly prefill time, plus any time spent waiting in a queue.</li>
      <li><b>Inter-token latency (ITL)</b>, also called time per output token (TPOT): the gap between consecutive output tokens. Set by how long one decode pass takes.</li>
      <li>Total time for a response ≈ TTFT + (number of output tokens − 1) × ITL.</li>
    </ul>`)}
    ${inf(`<p>These phases put very different kinds of pressure on the GPU. Prefill is a lot of <b class="c-compute">computation</b> in one go. Decode is a small amount of computation that <b>repeats hundreds of times</b>, and each repeat still reads the whole model. Parts 2 and 3 explain why that difference matters so much.</p>`)}
    ${check({
      q: "A user says: “The answer takes a long time to START, but then streams quickly.” Which phase is most likely slow?",
      opts: ["Decode", "Prefill (or waiting before prefill)", "Tokenization", "Sampling"],
      a: 1, why: "Time to first token is mostly queueing plus prefill. Long prompts make prefill heavier. Fast streaming afterwards means decode is fine.",
    })}`,
  mount(root, ctx) {
    const prompt = ["Explain", " how", " neural", " networks", " work"];
    const out = [" Neural", " networks", " learn", " by", " adjusting", " weights"];
    let running = false;
    const drawPrompt = (lit) => $("#t3-prompt", root).innerHTML = prompt.map(t => `<span class="tok prompt ${lit ? "lit" : ""}">${t}</span>`).join("");
    drawPrompt(false);
    const run = () => {
      if (running) return; running = true;
      const long = segVal(root, "t3-len") === "long";
      const tl = $("#t3-timeline", root), marks = $("#t3-marks", root);
      tl.innerHTML = ""; marks.innerHTML = "";
      $("#t3-out", root).innerHTML = ""; $("#t3-drows", root).textContent = "–";
      const pW = long ? 150 : 34;
      drawPrompt(true);
      $("#t3-pmodel", root).classList.add("busy"); $("#t3-pmodel", root).textContent = "Prefill pass: all tokens in parallel";
      $("#t3-prows", root).textContent = long ? "2,000" : "5";
      const bar = document.createElement("div");
      Object.assign(bar.style, { width: "0px", height: "60px", background: "var(--compute)", borderRadius: "3px 3px 0 0", transition: `width ${long ? 1.4 : .5}s linear` });
      tl.appendChild(bar); requestAnimationFrame(() => bar.style.width = pW + "px");
      ctx.after(long ? 1450 : 550, () => {
        drawPrompt(false);
        $("#t3-pmodel", root).classList.remove("busy"); $("#t3-pmodel", root).textContent = "GPU";
        marks.innerHTML = `<div style="position:absolute;left:${pW - 2}px;top:0;border-left:2px solid var(--ink);height:14px"></div><div style="position:absolute;left:${Math.max(0, pW - 60)}px;top:14px;white-space:nowrap"><b>TTFT</b>: first token</div>`;
        let k = 0;
        const dstep = () => {
          if (k >= out.length) { running = false; return; }
          const dm = $("#t3-dmodel", root); dm.classList.add("busy"); dm.textContent = `Decode pass ${k + 1}`;
          $("#t3-drows", root).textContent = "1";
          const b = document.createElement("div");
          Object.assign(b.style, { width: "0px", height: "8px", background: "var(--memory)", borderRadius: "2px 2px 0 0", transition: "width .35s linear" });
          tl.appendChild(b); requestAnimationFrame(() => b.style.width = "22px");
          ctx.after(420, () => {
            dm.classList.remove("busy"); dm.textContent = "GPU";
            const o = $("#t3-out", root); o.insertAdjacentHTML("beforeend", `<span class="tok gen new">${out[k]}</span>`);
            if (k === 1) marks.insertAdjacentHTML("beforeend", `<div style="position:absolute;left:${pW + 30}px;top:14px;white-space:nowrap;color:var(--memory)">← gap between tokens = <b>ITL</b></div>`);
            k++; ctx.after(160, dstep);
          });
        };
        dstep();
      });
    };
    $("#t3-play", root).onclick = run;
    bindSeg(root, "t3-len", () => { });
  },
});

/* ---------- Topic 4 ---------- */
topic({
  num: 4, title: "KV Cache",
  question: "What exactly is stored in the KV cache, and why does it make decode faster?",
  render: () => `
    ${say(
      "Inside every attention layer, each token is turned into three vectors by multiplying it with three weight matrices: a <b style='color:#1f4fa8'>Query</b> (what this token is looking for), a <b style='color:#99530b'>Key</b> (a label saying what this token offers) and a <b style='color:#6433bf'>Value</b> (the information it hands over).",
      "To produce a new token, its Query is compared with the Keys of <b>all earlier tokens</b>, and their Values are mixed together. The Keys and Values of earlier tokens <b>never change</b>, so we compute them once and <b class='c-kv'>keep them in GPU memory</b>. That store is the KV cache."
    )}
    ${see("What happens inside one attention layer during a decode step?", `
      <div class="btn-row">${seg("t4-mode", [["yes", "With KV cache"], ["no", "Without KV cache"]], "yes")}</div>
      ${player("t4-pl")}
      <div class="caption" id="t4-cap"></div>
      <div class="scroll-x"><div class="kvs" id="t4-scene"></div></div>
      <div class="stat-row">
        <div class="stat"><span class="v c-compute" id="t4-now">–</span><span class="k">tokens multiplied by W<sub>K</sub>, W<sub>V</sub> this step</span></div>
        <div class="stat"><span class="v c-compute" id="t4-tot">–</span><span class="k">total K/V computations so far (decode)</span></div>
        <div class="stat"><span class="v c-kv" id="t4-rows">–</span><span class="k">rows kept in memory between steps</span></div>
      </div>
      <p class="c-muted" style="font-size:.82rem">Real numbers, tiny sizes: 3 numbers per vector instead of thousands, one attention head, one layer. Real models repeat this for every head in every layer, and each has its own cache.</p>`)}
    ${see("How much work does the cache save over a whole answer?", `
      <div class="btn-row">
        <button type="button" class="btn primary" id="t4-step">Generate next token</button>
        <button type="button" class="btn" id="t4-auto">▶ Play</button>
        <button type="button" class="btn" id="t4-reset">Restart</button></div>
      <div class="grid2">
        <div><div class="mini-label" style="margin-bottom:4px">Without KV cache</div><div class="scroll-x"><div class="kv-grid" id="t4-grid-no"></div></div></div>
        <div><div class="mini-label" style="margin-bottom:4px">With KV cache</div><div class="scroll-x"><div class="kv-grid" id="t4-grid-yes"></div></div></div>
      </div>
      <div class="btn-row" style="gap:16px;font-size:.82rem">
        <span><span class="pill compute">K,V</span> computed in this pass</span>
        <span><span class="pill" style="background:var(--kv-soft);color:var(--kv)">↺</span> read from cache</span></div>
      <div class="stat-row">
        <div class="stat"><span class="v c-compute" id="t4-c-no">0</span><span class="k">K/V computations without cache</span></div>
        <div class="stat"><span class="v c-kv" id="t4-c-yes">0</span><span class="k">K/V computations with cache</span></div>
      </div>`)}
    ${see("How big does the KV cache get?", `
      <div class="controls">
        ${slider("t4-ctx", "Context length (tokens)", 0, 7, 1, 3)}
        ${slider("t4-batch", "Concurrent requests", 1, 64, 1, 1)}
      </div>
      <div class="stack" id="t4-stack"></div>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t4-per">–</span><span class="k">KV cache per token</span></div>
        <div class="stat"><span class="v c-kv" id="t4-total">–</span><span class="k">total KV cache</span></div>
        <div class="stat"><span class="v" id="t4-fit">–</span><span class="k">vs an 80 GB GPU</span></div>
      </div>
      <p class="c-muted" style="font-size:.82rem">70B-class model: 80 layers, 8 K/V heads, 128 numbers per head, FP16. The bar ignores the weights (another ~140 GB). ${illus("Approximate")}</p>`)}
    ${ex(`<div class="calc">Generate 8 tokens:<br>
      without cache: 1+2+…+8 = <b>36</b> K/V computations · with cache: <b>8</b><br>
      70B-class model: 2 × 80 layers × 8 heads × 128 × 2 bytes ≈ <b>0.33 MB per token</b> → 4,096 tokens ≈ 1.3 GB per conversation</div>`)}
    ${fx(["KV cache bytes = 2 × layers × KV heads × head size × bytes per number × tokens × requests"], "The 2 is for storing both K and V.")}
    ${tech(`<ul>
      <li>Per token, per layer: <code>q = x·W<sub>Q</sub></code>, <code>k = x·W<sub>K</sub></code>, <code>v = x·W<sub>V</sub></code>.</li>
      <li>Attention output = softmax(q·Kᵀ / √d) · V, where K and V stack the keys and values of all tokens so far.</li>
      <li>Causal masking means earlier tokens never look at later ones, so their K and V never change. Caching is exact: same answer, less work.</li>
      <li>Queries are not cached: a Query is only needed for the token that is being processed right now.</li>
    </ul>`)}
    ${inf(`<p>Prefill writes the prompt's Keys and Values into the cache. Each decode step computes K and V for <b>one</b> token, appends them, and <b>reads the whole cache</b>. So the cache saves a huge amount of computation, but it grows with every token and every user, and all of it sits in GPU memory.</p>`)}
    ${key(`The KV cache <b>trades memory for computation</b>: store old Keys and Values once, and never recompute them.`)}
    ${check({
      q: "Which vectors are stored in the KV cache for earlier tokens?",
      opts: ["Queries only", "Keys and Values", "Queries, Keys and Values", "The model weights"],
      a: 1, why: "Only Keys and Values are needed again by future tokens. The Query is used once, by the token being processed, and then discarded.",
    })}
    ${check({
      q: "You double the context length of one request. Roughly what happens to its KV cache memory?",
      opts: ["Stays the same", "Doubles", "Quadruples", "Halves"],
      a: 1, why: "The cache holds one row per token (per layer and head), so it grows linearly with context length.",
    })}`,
  mount(root, ctx) {
    /* ---- scene: real tiny attention ---- */
    const SEQ = ["Make", " me", " sound", " smart", " today", "."];
    const EMB = [[0.9, 0.1, 0.3], [0.2, 0.8, 0.1], [0.4, 0.3, 0.9], [0.7, 0.6, 0.2], [0.1, 0.5, 0.7], [0.3, 0.2, 0.4]];
    const WQ = [[1, 0, .5], [0, 1, .5], [.5, .5, 0]], WK = [[.5, 1, 0], [1, 0, .5], [0, .5, 1]], WV = [[1, .5, 0], [0, 1, 1], [.5, 0, 1]];
    const mul = (x, W) => [0, 1, 2].map(j => x[0] * W[0][j] + x[1] * W[1][j] + x[2] * W[2][j]);
    const K = EMB.map(x => mul(x, WK)), V = EMB.map(x => mul(x, WV)), Q = EMB.map(x => mul(x, WQ));
    const PROMPT = 3, STEPS = 3; // decode steps process tokens 3,4,5
    const frames = [{ intro: true }];
    for (let d = 0; d < STEPS; d++) for (let s = 0; s < 6; s++) frames.push({ d, s });

    const esc = t => t.replace(/ /g, "␣");
    const vcells = (v, c) => v.map(n => `<span class="cell ${c}">${n.toFixed(2)}</span>`).join("");
    const matHtml = (W, c, name) => `<div class="col-title">${name}</div><div class="mat ${c}">${W.flat().map(n => `<i>${n}</i>`).join("")}</div>`;

    const render = (fi) => {
      const f = frames[fi], cache = segVal(root, "t4-mode") === "yes";
      const d = f.intro ? -1 : f.d, s = f.intro ? -1 : f.s;
      const cur = PROMPT + d;                       // index of token being processed
      const rows = f.intro ? PROMPT : (s >= 2 ? cur + 1 : cur); // rows present in cache
      const next = cur + 1 < SEQ.length ? SEQ[cur + 1] : "⟨end⟩";
      const on = f.intro ? ["list", "kc", "vc"] : [["list", "x"], ["x", "w", "qkv"], ["qkv", "kc", "vc"], ["q", "kc", "comb"], ["vc", "comb"], ["comb", "list"]][s];
      const g = (name) => on.includes(name) || (name === "q" && on.includes("qkv")) ? "hl" : "dim";

      // attention numbers
      let scores = [], weights = [], out = [0, 0, 0];
      if (!f.intro) {
        const q = Q[cur];
        scores = K.slice(0, cur + 1).map(k => (q[0] * k[0] + q[1] * k[1] + q[2] * k[2]) / Math.sqrt(3));
        const ex = scores.map(Math.exp), sum = ex.reduce((a, b) => a + b, 0);
        weights = ex.map(e => e / sum);
        weights.forEach((w, i) => V[i].forEach((vv, j) => out[j] += w * vv));
      }
      const showScores = !f.intro && s >= 3, showWeights = !f.intro && s >= 4;
      const recompute = !cache && !f.intro && (s === 1 || s === 2);

      // token list
      const list = SEQ.map((t, i) => {
        const st = f.intro ? (i < PROMPT ? "cached" : i === PROMPT ? "cur" : "future")
          : i < cur ? "cached" : i === cur ? "cur" : (i === cur + 1 && s === 5) ? "cur" : "future";
        const tag = st === "cached" ? (cache ? "K,V cached" : "no cache") : st === "cur" ? (i === cur + 1 ? "next" : "now") : "";
        return `<div class="tk ${st === "cached" && !cache ? "" : st}"><span>${esc(t)}</span><span style="font-size:.62rem">${tag}</span></div>`;
      }).join("");

      // x column: current token only, or all tokens when recomputing
      const xRows = (recompute ? EMB.slice(0, cur + 1).map((x, i) => [SEQ[i], x]) : f.intro ? [[SEQ[PROMPT], EMB[PROMPT]]] : [[SEQ[cur], EMB[cur]]])
        .map(([t, x]) => `<div class="vrow"><span class="lbl">${esc(t)}</span>${vcells(x, "x")}</div>`).join("");
      const xCol = (r) => `<div class="part ${g("x")}" style="grid-column:2;grid-row:${r}"><div class="col-title">token vector x</div>${xRows}</div>`;

      const tableRows = (M, c, withExtra) => M.slice(0, rows).map((v, i) => {
        const isNew = !f.intro && i === cur && s === 2;
        const extra = withExtra === "score" && showScores ? `<span class="score" style="width:${Math.max(4, (scores[i] + 0.2) * 40)}px"></span><span class="wt">${scores[i].toFixed(2)}</span>`
          : withExtra === "weight" && showWeights ? `<span class="wt"><b>${Math.round(weights[i] * 100)}%</b></span>` : "";
        return `<div class="vrow ${isNew && cache ? "new" : ""} ${recompute ? "recomp" : ""}"><span class="lbl">${esc(SEQ[i])}</span>${vcells(M[i], c)}${extra}</div>`;
      }).join("");

      let comb = "";
      if (f.intro) comb = `<div class="col-title">Attention</div><p style="font-size:.9rem">Prefill already stored a Key row and a Value row for <b>Make</b>, <b>me</b> and <b>sound</b>, and produced the first new token <b>“smart”</b>.</p><p style="font-size:.9rem">Press <b>Play</b> to process “smart”.</p>`;
      else if (s < 3) comb = `<div class="col-title">Attention</div><p class="c-muted" style="font-size:.9rem">Waiting for this token's Query, Key and Value.</p>`;
      else {
        comb = `<div class="col-title">Attention</div>
          <div style="font-size:.85rem"><b>1.</b> Scores = q · each cached Key ÷ √3<br><span class="c-muted">highest: “${esc(SEQ[scores.indexOf(Math.max(...scores))])}” (${Math.max(...scores).toFixed(2)})</span></div>`;
        if (s >= 4) comb += `<div style="font-size:.85rem"><b>2.</b> Softmax → weights that add to 100%<br><b>3.</b> Output = Σ weight × Value</div>
          <div class="vrow"><span class="lbl">output</span>${vcells(out, "o")}</div>`;
        if (s >= 5) comb += `<div style="font-size:.85rem"><b>4.</b> The output continues through the rest of the model, which picks the next token:</div>
          <div class="tokens"><span class="tok gen new" style="font-size:1.05rem">${esc(next)}</span></div>`;
      }

      $("#t4-scene", root).innerHTML = `
        <div class="part ${g("list")}" style="grid-column:1;grid-row:1 / span 3;align-self:start"><div class="col-title">Tokens</div><div class="tok-list">${list}</div></div>
        ${xCol(1)}<div class="sym" style="grid-column:3;grid-row:1">×</div>
        <div class="part ${g("w")}" style="grid-column:4;grid-row:1">${matHtml(WQ, "q", "W_Query")}</div>
        <div class="sym" style="grid-column:5;grid-row:1">=</div>
        <div class="part ${g("q")}" style="grid-column:6;grid-row:1"><div class="col-title">Query q</div>${f.intro ? `<span class="c-muted">–</span>` : `<div class="vrow">${vcells(Q[cur], "q")}</div>`}</div>
        <div class="sym" style="grid-column:7;grid-row:1">→</div>
        <div class="part ${g("q")}" style="grid-column:8;grid-row:1;font-size:.8rem;color:var(--muted)">The Query is <b>not stored</b>. It is only used now, to look up the cached Keys.</div>

        ${xCol(2)}<div class="sym" style="grid-column:3;grid-row:2">×</div>
        <div class="part ${g("w")}" style="grid-column:4;grid-row:2">${matHtml(WK, "k", "W_Key")}</div>
        <div class="sym" style="grid-column:5;grid-row:2">=</div>
        <div class="part ${g("qkv")}" style="grid-column:6;grid-row:2"><div class="col-title">Key k</div>${f.intro ? `<span class="c-muted">–</span>` : `<div class="vrow">${vcells(K[cur], "k")}</div>`}</div>
        <div class="sym" style="grid-column:7;grid-row:2">→</div>
        <div class="part ${g("kc")}" style="grid-column:8;grid-row:2"><div class="col-title">${cache ? "Key cache (previous tokens)" : "Keys (rebuilt every step)"}</div><div class="cache-box">${tableRows(K, "k", "score")}</div></div>

        ${xCol(3)}<div class="sym" style="grid-column:3;grid-row:3">×</div>
        <div class="part ${g("w")}" style="grid-column:4;grid-row:3">${matHtml(WV, "v", "W_Value")}</div>
        <div class="sym" style="grid-column:5;grid-row:3">=</div>
        <div class="part ${g("qkv")}" style="grid-column:6;grid-row:3"><div class="col-title">Value v</div>${f.intro ? `<span class="c-muted">–</span>` : `<div class="vrow">${vcells(V[cur], "v")}</div>`}</div>
        <div class="sym" style="grid-column:7;grid-row:3">→</div>
        <div class="part ${g("vc")}" style="grid-column:8;grid-row:3"><div class="col-title">${cache ? "Value cache (previous tokens)" : "Values (rebuilt every step)"}</div><div class="cache-box">${tableRows(V, "v", "weight")}</div></div>

        <div class="sym" style="grid-column:9;grid-row:1 / span 3">→</div>
        <div class="part ${g("comb")}" style="grid-column:10;grid-row:1 / span 3;align-self:stretch"><div class="combine">${comb}</div></div>`;

      // caption + counters
      const n = cur + 1, tok = f.intro ? "" : SEQ[cur].trim();
      const caps = f.intro
        ? ["Starting point", "Prefill has already run on “Make me sound”. The KV cache holds one Key row and one Value row for each of those tokens."]
        : [
          [`Decode step ${d + 1} · the new token`, `The newest token “${tok}” enters the attention layer as a vector x.`],
          [`Decode step ${d + 1} · multiply by the weights`, cache
            ? `x is multiplied by W_Query, W_Key and W_Value. Only this <b>one</b> token is multiplied; earlier tokens are already done.`
            : `Nothing was saved from earlier steps, so <b>all ${n} tokens</b> must be multiplied by W_Key and W_Value again.`],
          [`Decode step ${d + 1} · ${cache ? "append to the cache" : "rebuild Keys and Values"}`, cache
            ? `The new Key and Value are <b>appended as a new row</b>. The cache now has ${n} rows.`
            : `All ${n} Key and Value rows are rebuilt from scratch, used once, and thrown away after this step.`],
          [`Decode step ${d + 1} · compare`, `The Query q is compared with <b>every cached Key</b> (dot product). A bigger score means “that token matters more to me”.`],
          [`Decode step ${d + 1} · mix the Values`, `Softmax turns the scores into weights. The output is a weighted mix of the Value rows.`],
          [`Decode step ${d + 1} · next token`, `The output flows through the rest of the model, which predicts “${next.trim()}”. That token is the input to the next decode step.`],
        ][s];
      $("#t4-cap", root).innerHTML = `<span class="cap-step">${caps[0]}</span><span>${caps[1]}</span>`;
      const doneSteps = f.intro ? 0 : d + (s >= 1 ? 1 : 0);
      let totNo = 0; for (let k = 0; k < doneSteps; k++) totNo += PROMPT + k + 1;
      $("#t4-now", root).textContent = f.intro || s < 1 ? "–" : cache ? "1 token" : `${n} tokens`;
      $("#t4-tot", root).textContent = cache ? doneSteps : totNo;
      $("#t4-rows", root).textContent = cache ? `${rows} K + ${rows} V` : "0 (nothing kept)";
    };
    const pl = bindPlayer(root, "t4-pl", ctx, { frames: frames.length, render, interval: 2600 });
    bindSeg(root, "t4-mode", () => pl.show(pl.i));

    /* ---- cost over many tokens ---- */
    const N = 8, words8 = ["The", " cat", " sat", " on", " the", " mat", " and", " purred"];
    let n = 0, auto = null;
    const tri = k => k * (k + 1) / 2;
    const grid = (el, mode) => {
      el.style.setProperty("--n", N);
      let html = `<div class="r"><span></span>${words8.map(w => `<span class="c empty" style="color:var(--ink)">${w}</span>`).join("")}</div>`;
      for (let r = 1; r <= N; r++) {
        html += `<div class="r"><span class="gl">${r <= n ? `pass ${r}` : ""}</span>`;
        for (let c = 1; c <= N; c++) {
          if (r > n || c > r) { html += `<span class="c empty"></span>`; continue; }
          const comp = mode === "no" || c === r;
          html += `<span class="c ${comp ? "comp" : "cache"}">${comp ? "K,V" : "↺"}</span>`;
        }
        html += `</div>`;
      }
      el.innerHTML = html;
    };
    const draw = () => {
      grid($("#t4-grid-no", root), "no"); grid($("#t4-grid-yes", root), "yes");
      $("#t4-c-no", root).textContent = tri(n); $("#t4-c-yes", root).textContent = n;
    };
    const stopAuto = () => { if (auto) { ctx.stop(auto); auto = null; $("#t4-auto", root).textContent = "▶ Play"; } };
    const step = () => { if (n >= N) return stopAuto(); n++; draw(); };
    $("#t4-step", root).onclick = () => { stopAuto(); step(); };
    $("#t4-reset", root).onclick = () => { stopAuto(); n = 0; draw(); };
    $("#t4-auto", root).onclick = () => {
      if (auto) return stopAuto();
      if (n >= N) n = 0;
      $("#t4-auto", root).textContent = "❚❚ Pause"; auto = ctx.every(800, step);
    };
    draw();

    /* ---- size calculator ---- */
    const ctxs = [512, 1024, 2048, 4096, 8192, 32768, 65536, 131072];
    const m = MODELS["70B"], per = kvBytesPerToken(m, 2);
    bindSliders(root, { "t4-ctx": v => ctxs[v].toLocaleString("en-US"), "t4-batch": v => v }, v => {
      const total = per * ctxs[v["t4-ctx"]] * v["t4-batch"], cap = DEMO.hbmGB * 1e9;
      $("#t4-per", root).textContent = gb(per);
      $("#t4-total", root).textContent = gb(total);
      $("#t4-fit", root).innerHTML = total <= cap ? `<span class="c-ok">${Math.round(total / cap * 100)}% of 80 GB</span>` : `<span class="c-bad">${fmt(total / cap, 1)}× too big</span>`;
      const w = Math.min(100, total / cap * 100);
      $("#t4-stack", root).innerHTML = `<span class="s-kv" style="width:${w}%">${w > 12 ? "KV cache" : ""}</span><span class="s-free" style="width:${100 - w}%">${100 - w > 20 ? "rest of 80 GB" : ""}</span>`;
    });
  },
});
