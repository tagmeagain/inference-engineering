/* ============================================================
   Core: course registry, navigation, teaching-beat helpers,
   quizzes and shared visual components.
   Each part file calls part({...}) then topic({...}).
   A topic = { title, question, render(ctx) -> html, mount(root, ctx) }
   ============================================================ */
const COURSE = { parts: [], topics: [] };
function part(p) { COURSE.parts.push(Object.assign({ topics: [] }, p)); }
function topic(t) {
  t.partIndex = COURSE.parts.length - 1;
  if (t.num !== null) t.num = COURSE.topics.filter(x => x.num !== null).length + 1;
  t.index = COURSE.topics.length;
  COURSE.topics.push(t);
  COURSE.parts[t.partIndex].topics.push(t);
}

/* ---------- tiny DOM helpers ---------- */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
const clamp = (x, a, b) => Math.max(a, Math.min(b, x));

/* ---------- number formatting ---------- */
function fmt(n, d = 1) {
  if (!isFinite(n)) return "∞";
  const a = Math.abs(n);
  if (a >= 100) return Math.round(n).toLocaleString("en-US");
  if (a >= 10) return n.toFixed(Math.max(0, d - 0)).replace(/\.0+$/, "");
  if (a >= 1) return n.toFixed(d).replace(/\.0+$/, "");
  if (a === 0) return "0";
  return n.toPrecision(2);
}
function words(n) {
  // 1.4e11 -> "140 billion"
  const units = [[1e18, "quintillion"], [1e15, "quadrillion"], [1e12, "trillion"], [1e9, "billion"], [1e6, "million"], [1e3, "thousand"]];
  for (const [v, w] of units) if (Math.abs(n) >= v) return `${fmt(n / v, 1)} ${w}`;
  return fmt(n, 1);
}
function gb(bytes) {
  const g = bytes / 1e9;
  if (g >= 1000) return `${fmt(g / 1000, 2)} TB`;
  if (g >= 1) return `${fmt(g, 1)} GB`;
  if (bytes >= 1e6) return `${fmt(bytes / 1e6, 1)} MB`;
  if (bytes >= 1e3) return `${fmt(bytes / 1e3, 1)} KB`;
  return `${fmt(bytes, 0)} bytes`;
}
function sci(n) {
  if (n === 0) return "0";
  const e = Math.floor(Math.log10(Math.abs(n)));
  const m = n / Math.pow(10, e);
  const sup = String(e).split("").map(c => "⁰¹²³⁴⁵⁶⁷⁸⁹"["0123456789".indexOf(c)] || (c === "-" ? "⁻" : c)).join("");
  return `${m.toFixed(1)}×10${sup}`;
}

/* ---------- teaching beats (the progression) ---------- */
const beat = (body, cls = "") => `<section class="beat ${cls}">${body}</section>`;
const say = (...paras) => beat(paras.map(p => `<p class="lede">${p}</p>`).join(""), "b-intuit");
const see = (question, html) => beat(`<div class="viz"><div class="viz-q">${question}</div>${html}</div>`, "b-viz");
const ex = (html) => beat(`<div class="example"><div class="ex-h">Worked example</div>${html}</div>`, "b-ex");
const tech = (html, summary = "Exact definitions") =>
  beat(`<details class="tech"><summary>${summary}</summary><div>${html}</div></details>`, "b-tech");
const fx = (lines, note = "") =>
  beat(`<div class="formula">${[].concat(lines).map(l => `<div>${l}</div>`).join("")}${note ? `<div class="fnote">${note}</div>` : ""}</div>`, "b-fx");
const inf = (html) => beat(`<div class="inf">${html}</div>`, "b-inf");
const key = (html) => beat(`<div class="callout key"><span class="tag">Key idea</span><div>${html}</div></div>`, "b-key");
const note = (html, tag = "Simplified model") => `<div class="callout note"><span class="tag">${tag}</span><div>${html}</div></div>`;
const big = (html, tag = "Core idea") => `<div class="callout big"><span class="tag">${tag}</span><div>${html}</div></div>`;
const illus = (t = "Illustrative numbers") => `<span class="illus">${t}</span>`;

function flow(items, opts = {}) {
  const arrow = opts.vertical ? "↓" : "→";
  return `<div class="flow ${opts.vertical ? "vertical" : ""}" ${opts.id ? `id="${opts.id}"` : ""}>` +
    items.map((it, i) => {
      const o = typeof it === "string" ? { t: it } : it;
      return (i ? `<span class="arr" data-a="${i}">${arrow}</span>` : "") +
        `<div class="node ${o.c || ""}" data-n="${i}">${o.t}</div>`;
    }).join("") + `</div>`;
}
function setFlow(root, idx) {
  if (!root) return;
  $$(".node", root).forEach(n => {
    const i = +n.dataset.n;
    n.classList.toggle("active", i === idx);
    n.classList.toggle("past", i < idx);
  });
  $$(".arr", root).forEach(a => a.classList.toggle("active", +a.dataset.a === idx));
}

const meter = (label, cls, id) =>
  `<div class="meter ${cls}"><div class="mh"><span>${label}</span><span id="${id}-v">0%</span></div><div class="track"><i id="${id}" style="width:0%"></i></div></div>`;
function setMeter(root, id, frac, text) {
  const bar = $("#" + id, root), v = $("#" + id + "-v", root);
  if (bar) bar.style.width = clamp(frac * 100, 0, 100) + "%";
  if (v) v.textContent = text !== undefined ? text : Math.round(clamp(frac, 0, 1) * 100) + "%";
}

function slider(id, label, min, max, step, value) {
  return `<div class="ctl"><label for="${id}"><span>${label}</span><output id="${id}-o"></output></label>
    <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${value}"></div>`;
}
function seg(id, options, active) {
  return `<div class="seg" id="${id}" role="group">${options.map(([v, t]) =>
    `<button type="button" data-v="${v}" class="${String(v) === String(active) ? "on" : ""}">${t}</button>`).join("")}</div>`;
}
function bindSeg(root, id, cb) {
  const el = $("#" + id, root);
  el.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    $$("button", el).forEach(x => x.classList.toggle("on", x === b));
    cb(b.dataset.v);
  });
}
const segVal = (root, id) => $(`#${id} button.on`, root).dataset.v;

/* ---------- quizzes ---------- */
const QUIZ = {}; let quizSeq = 0;
function check(o) {
  const id = "q" + (quizSeq++);
  QUIZ[id] = o;
  return beat(`<div class="quiz" data-quiz="${id}">
    <div class="quiz-h">Quick check</div>
    <p class="quiz-q">${o.q}</p>
    <div class="quiz-opts">${o.opts.map((t, i) => `<button type="button" class="quiz-opt" data-i="${i}"><span class="quiz-letter">${"ABCD"[i]}</span><span>${t}</span></button>`).join("")}</div>
    <div class="quiz-why" hidden></div></div>`, "b-check");
}
document.addEventListener("click", e => {
  const retry = e.target.closest(".quiz-retry");
  if (retry) {
    const qz = retry.closest(".quiz");
    $$(".quiz-opt", qz).forEach(b => { b.disabled = false; b.classList.remove("right", "wrong"); });
    $(".quiz-why", qz).hidden = true;
    return;
  }
  const btn = e.target.closest(".quiz-opt"); if (!btn) return;
  const qz = btn.closest(".quiz"), o = QUIZ[qz.dataset.quiz], i = +btn.dataset.i;
  $$(".quiz-opt", qz).forEach(b => {
    b.disabled = true;
    if (+b.dataset.i === o.a) b.classList.add("right");
  });
  if (i !== o.a) btn.classList.add("wrong");
  const why = $(".quiz-why", qz);
  why.innerHTML = `<div>${i === o.a ? `<b class="ok">Correct.</b>` : `<b class="no">Not quite.</b> The answer is <b>${"ABCD"[o.a]}</b>.`} ${o.why}</div>
    <div><button type="button" class="btn small quiz-retry">Reset question</button></div>`;
  why.hidden = false;
});

/* ---------- shared reference numbers (illustrative) ---------- */
/* An imaginary accelerator used across topics so numbers stay consistent.
   Not a real GPU spec. */
const DEMO = {
  hbmGB: 80,            // memory capacity
  bwGBs: 1000,          // memory bandwidth, GB per second
  flops: 2e14,          // peak compute, FLOPs per second (knee at 200 FLOPs/byte)
};
const MODELS = {
  "8B":  { name: "8B-class",  N: 8e9,   L: 32,  d: 4096,  qHeads: 32,  kvHeads: 8, headDim: 128 },
  "70B": { name: "70B-class", N: 70e9,  L: 80,  d: 8192,  qHeads: 64,  kvHeads: 8, headDim: 128 },
  "405B":{ name: "405B-class",N: 405e9, L: 126, d: 16384, qHeads: 128, kvHeads: 8, headDim: 128 },
};
const kvBytesPerToken = (m, bytes = 2, kvHeads = m.kvHeads) => 2 * m.L * kvHeads * m.headDim * bytes;

/* ---------- GPU diagram (shared by several topics) ---------- */
function gpuDiagram(opts = {}) {
  const sm = (i) => `<div class="sm" data-part="sm">
      <div class="sml">SM ${i + 1}</div>
      <div class="cores" data-part="cuda">${"<i></i>".repeat(12)}</div>
      <div class="tcore" data-part="tensor">Tensor Cores</div>
      <div class="mem-row"><div class="reg" data-part="reg">REG</div><div class="sram" data-part="sram">SRAM</div></div>
    </div>`;
  const hbm = (n) => Array.from({ length: n }, (_, i) => `<div class="hbm-chip" data-part="hbm"><div class="fill"></div><span>HBM</span></div>`).join("");
  return `<div class="scroll-x"><div class="gpu-wrap">
    <div class="host">
      <div class="hl">Host server</div>
      <div class="part-box" data-part="cpu">CPU</div>
      <div class="part-box" data-part="ram">CPU RAM<br><span class="c-muted" style="font-weight:400">(system memory)</span></div>
      <div class="part-box" data-part="disk">Disk / storage</div>
    </div>
    <div class="bus" data-part="pcie"><span>PCIe</span></div>
    <div class="board" data-part="board">
      <div class="bl">GPU (simplified conceptual layout)</div>
      <div class="hbm-row">${hbm(4)}</div>
      <div class="die">
        <div class="dl" data-part="die">GPU chip</div>
        <div class="sm-grid">${Array.from({ length: 8 }, (_, i) => sm(i)).join("")}</div>
        <div class="l2" data-part="l2">L2 cache</div>
      </div>
      <div class="hbm-row">${hbm(4)}</div>
      ${opts.nvlink === false ? "" : `<div class="nvl" data-part="nvlink" title="NVLink"></div>`}
    </div>
  </div></div>`;
}

/* ============================================================
   App shell / navigation
   ============================================================ */
const STORE_KEY = "llm-gpu-course-v1";
const state = { cur: 0, done: new Set(), visited: new Set() };
function loadState() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE_KEY) || "{}");
    if (Array.isArray(s.done)) s.done.forEach(i => state.done.add(i));
    if (Array.isArray(s.visited)) s.visited.forEach(i => state.visited.add(i));
    if (Number.isInteger(s.cur)) state.cur = clamp(s.cur, 0, COURSE.topics.length - 1);
  } catch (e) { /* storage unavailable: start fresh */ }
  const m = location.hash.match(/t=(\d+)/);
  if (m) state.cur = clamp(+m[1] - 1, 0, COURSE.topics.length - 1);
}
function saveState() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify({ cur: state.cur, done: [...state.done], visited: [...state.visited] })); } catch (e) { }
}

/* timers registered by a topic are cleared on navigation */
function makeCtx() {
  const timers = [], rafs = [];
  const ctx = {
    every(ms, fn) { const t = setInterval(fn, ms); timers.push(t); return t; },
    after(ms, fn) { const t = setTimeout(fn, ms); timers.push(t); return t; },
    stop(t) { clearInterval(t); clearTimeout(t); },
    loop(fn) {
      let alive = true, last = performance.now();
      const tick = (now) => { if (!alive) return; fn((now - last) / 1000, now); last = now; rafs.push(requestAnimationFrame(tick)); };
      rafs.push(requestAnimationFrame(tick));
      return () => { alive = false; };
    },
    dispose() { timers.forEach(t => { clearInterval(t); clearTimeout(t); }); rafs.forEach(cancelAnimationFrame); ctx.disposed = true; },
    go(i) { navigate(i); },
  };
  return ctx;
}
let currentCtx = null;

function topicNumberLabel(t) { return t.num ? `Topic ${t.num}` : "All together"; }

function renderShell() {
  document.getElementById("app").innerHTML = `
    <header class="topbar">
      <div class="topbar-row">
        <div class="brand"><h1>LLM Inference &amp; GPU Architecture</h1><span class="sub">What the GPU is really doing when a model writes text</span></div>
        <div class="overall"><span id="overall-t" class="tnum"></span><div class="overall-bar"><i id="overall-b"></i></div>
          <button type="button" class="icon-btn" id="reset-progress" title="Clear progress">Reset</button></div>
      </div>
      <nav class="parts" id="parts" aria-label="Course parts"></nav>
    </header>
    <div class="layout">
      <aside class="sidebar" id="sidebar"></aside>
      <main class="main"><div class="stage" id="stage"></div></main>
    </div>
    <div class="footer-nav"><div class="footer-inner">
      <button type="button" class="btn" id="prev">← Previous</button>
      <div class="where"><div id="where"></div><div class="dots" id="dots"></div></div>
      <button type="button" class="btn primary" id="next">Complete &amp; continue →</button>
    </div></div>`;
  $("#prev").onclick = () => navigate(state.cur - 1);
  $("#next").onclick = () => {
    state.done.add(state.cur); saveState();
    if (state.cur < COURSE.topics.length - 1) navigate(state.cur + 1); else renderNav();
  };
  $("#reset-progress").onclick = () => {
    if (!confirm("Clear all progress checkmarks?")) return;
    state.done.clear(); state.visited.clear(); state.visited.add(state.cur); saveState(); renderNav();
  };
  document.addEventListener("keydown", e => {
    if (e.target.matches("input, select, textarea")) return;
    if (e.key === "ArrowRight" && e.altKey) $("#next").click();
    if (e.key === "ArrowLeft" && e.altKey) navigate(state.cur - 1);
  });
  window.addEventListener("hashchange", () => {
    const m = location.hash.match(/t=(\d+)/);
    if (m && +m[1] - 1 !== state.cur) navigate(+m[1] - 1);
  });
}

function renderNav() {
  const t = COURSE.topics[state.cur], p = COURSE.parts[t.partIndex];
  $("#parts").innerHTML = COURSE.parts.map((pp, i) => {
    const d = pp.topics.filter(x => state.done.has(x.index)).length;
    const label = pp.num ? `Part ${pp.num} / 7` : "Wrap-up";
    return `<button type="button" class="part-tab ${i === t.partIndex ? "active" : ""} ${d === pp.topics.length ? "done" : ""}" data-p="${i}">
      <span class="pn">${label}</span><span class="pt">${pp.title}</span>
      <span class="pbar"><i style="width:${d / pp.topics.length * 100}%"></i></span></button>`;
  }).join("");
  $$("#parts .part-tab").forEach(b => b.onclick = () => {
    const pp = COURSE.parts[+b.dataset.p];
    const firstUndone = pp.topics.find(x => !state.done.has(x.index)) || pp.topics[0];
    navigate(firstUndone.index);
  });
  $("#sidebar").innerHTML = `
    <div class="part-head"><div class="pn">${p.num ? `Part ${p.num} of 7` : "All together"}</div><h2>${p.title}</h2><p>${p.blurb}</p></div>
    <ul class="topic-list">${p.topics.map(x => `<li><button type="button" class="topic-link ${x.index === state.cur ? "active" : ""} ${state.done.has(x.index) ? "done" : state.visited.has(x.index) ? "visited" : ""}" data-t="${x.index}">
      <span class="tn">${state.done.has(x.index) ? "✓" : (x.num || "★")}</span><span>${x.title}</span></button></li>`).join("")}</ul>
    <div class="story"><span class="story-h">Where this fits</span>${p.story}</div>`;
  $$("#sidebar .topic-link").forEach(b => b.onclick = () => navigate(+b.dataset.t));
  const total = COURSE.topics.length, done = state.done.size;
  $("#overall-t").textContent = `${done} / ${total} topics complete`;
  $("#overall-b").style.width = (done / total * 100) + "%";
  const pos = p.topics.indexOf(t);
  $("#where").textContent = `${p.num ? `Part ${p.num} / 7` : "All together"} · ${pos + 1} of ${p.topics.length} in this part`;
  $("#dots").innerHTML = p.topics.map(x => `<i class="${state.done.has(x.index) ? "done" : ""} ${x.index === state.cur ? "cur" : ""}"></i>`).join("");
  $("#prev").disabled = state.cur === 0;
  const last = state.cur === total - 1;
  const isDone = state.done.has(state.cur);
  $("#next").innerHTML = last ? (isDone ? "Course complete ✓" : "Complete course ✓") : (isDone ? "Continue →" : "Complete &amp; continue →");
}

function navigate(i) {
  if (i < 0 || i >= COURSE.topics.length) return;
  if (currentCtx) currentCtx.dispose();
  state.cur = i; state.visited.add(i); saveState();
  if (location.hash !== `#t=${i + 1}`) history.replaceState(null, "", `#t=${i + 1}`);
  const t = COURSE.topics[i], p = COURSE.parts[t.partIndex];
  currentCtx = makeCtx();
  $("#stage").innerHTML = `
    <div class="topic-head">
      <div class="eyebrow">${p.num ? `Part ${p.num} · ${p.title}` : "All together"} · ${topicNumberLabel(t)}</div>
      <h2>${t.title}</h2>
      ${t.question ? `<div class="question-chip"><b>Answers</b><span>${t.question}</span></div>` : ""}
    </div>
    ${t.render(currentCtx)}`;
  if (t.mount) t.mount($("#stage"), currentCtx);
  renderNav();
  window.scrollTo({ top: 0 });
}

function boot() {
  renderShell();
  loadState();
  navigate(state.cur);
}

/* bind range sliders: spec = { id: (value, allValues) => label } */
function bindSliders(root, spec, update) {
  const read = () => { const v = {}; for (const id in spec) v[id] = +$("#" + id, root).value; return v; };
  const run = () => {
    const v = read();
    for (const id in spec) { const o = $("#" + id + "-o", root); if (o) o.textContent = spec[id](v[id], v); }
    update(v);
  };
  for (const id in spec) $("#" + id, root).addEventListener("input", run);
  run();
  return run;
}

/* ---------- playable visualizations ----------
   player(id) renders controls; bindPlayer wires them to a frame renderer. */
function player(id) {
  return `<div class="player" id="${id}">
    <button type="button" class="btn" data-a="restart" title="Restart">⏮ Restart</button>
    <button type="button" class="btn" data-a="back">◀ Step</button>
    <button type="button" class="btn primary" data-a="play">▶ Play</button>
    <button type="button" class="btn" data-a="fwd">Step ▶</button>
    <input type="range" data-a="scrub" min="0" max="0" value="0" aria-label="Scrub timeline">
    <span class="pl-count"></span>
    <select data-a="speed" aria-label="Speed"><option value="1.7">Slow</option><option value="1" selected>Normal</option><option value="0.55">Fast</option></select>
  </div>`;
}
function bindPlayer(root, id, ctx, o) {
  const el = $("#" + id, root), q = a => $(`[data-a="${a}"]`, el);
  let i = 0, timer = null, frames = o.frames;
  const show = (k) => {
    i = clamp(k, 0, frames - 1);
    q("scrub").max = frames - 1; q("scrub").value = i;
    $(".pl-count", el).textContent = `${i + 1} / ${frames}`;
    q("back").disabled = i === 0; q("fwd").disabled = i === frames - 1;
    o.render(i);
  };
  const stop = () => { if (timer) { ctx.stop(timer); timer = null; } q("play").textContent = "▶ Play"; };
  const play = () => {
    if (i >= frames - 1) show(0);
    q("play").textContent = "❚❚ Pause";
    timer = ctx.every((o.interval || 1800) * +q("speed").value, () => { if (i >= frames - 1) stop(); else show(i + 1); });
  };
  q("play").onclick = () => timer ? stop() : play();
  q("fwd").onclick = () => { stop(); show(i + 1); };
  q("back").onclick = () => { stop(); show(i - 1); };
  q("restart").onclick = () => { stop(); show(0); };
  q("scrub").oninput = () => { stop(); show(+q("scrub").value); };
  q("speed").onchange = () => { if (timer) { stop(); play(); } };
  show(0);
  return { show, stop, get i() { return i; }, setFrames(n) { frames = n; show(Math.min(i, n - 1)); } };
}
