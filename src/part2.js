/* ============================================================
   PART 2 — How GPUs Execute LLMs (topics 5–9)
   ============================================================ */
part({
  num: 2, title: "How GPUs Execute LLMs",
  blurb: "What is physically inside the GPU, where the weights and KV cache sit, and how data reaches the circuits that do the maths.",
  story: "GPU executes matrix operations → <b>weights + KV cache live in GPU memory</b> → SMs run the work → Tensor Cores speed up matrix maths → the memory hierarchy decides how fast data reaches compute.",
});

/* Shared component descriptions (used by topics 5 and 7) */
const GPU_PARTS = {
  cpu: { name: "CPU", what: "The server's general-purpose processor.", where: "On the host motherboard, outside the GPU.", does: "Runs the serving program: receives requests, tokenizes text, schedules work, and sends commands to the GPU.", inf: "Orchestrates inference but does almost none of the heavy maths." },
  ram: { name: "CPU RAM", what: "The server's main system memory.", where: "On the host motherboard, next to the CPU.", does: "Holds programs and data for the CPU. Large, but far from the GPU.", inf: "Model files are read from disk into CPU RAM and then copied to the GPU once, at load time. Some systems offload spare KV cache here, at a speed cost." },
  disk: { name: "Disk / storage", what: "Long-term storage (SSD or network storage).", where: "Attached to the host server.", does: "Keeps model files when nothing is running.", inf: "Only used when loading the model. Far too slow to read weights from during generation." },
  pcie: { name: "PCIe link", what: "The standard connection between CPU and GPU.", where: "Between the host motherboard and the GPU card.", does: "Moves data between CPU RAM and GPU memory.", inf: "Used to load weights and send inputs and outputs. Much slower than the GPU's own memory path, so we avoid using it during every token." },
  board: { name: "GPU", what: "A processor built for doing the same maths on lots of data at the same time.", where: "A card or module plugged into the server, with its own chip and its own memory.", does: "Runs thousands of simple calculations in parallel.", inf: "Runs every Transformer layer: the matrix multiplications, attention, and all supporting maths for prefill and decode." },
  die: { name: "GPU chip", what: "The silicon chip where the computation happens.", where: "At the centre of the GPU package, surrounded by HBM.", does: "Holds the Streaming Multiprocessors, their cores, and small fast on-chip memory.", inf: "Every FLOP of inference happens here. But the chip holds only a tiny amount of data at once." },
  hbm: { name: "HBM (High Bandwidth Memory)", what: "Stacks of memory chips built to move lots of data quickly.", where: "On the same package as the GPU chip, right beside it, connected by very short, wide wiring.", does: "Holds large amounts of data the GPU works on.", inf: "Holds the <b>model weights</b>, the <b>KV cache</b>, and <b>activations</b>. The biggest and slowest (still very fast) memory on the GPU." },
  sm: { name: "SM (Streaming Multiprocessor)", what: "A major execution unit of the GPU. A GPU has many of them.", where: "Inside the GPU chip, repeated side by side.", does: "Receives work, creates and schedules threads, and runs them on its cores using its local fast memory.", inf: "When a kernel (for example a matrix multiply for one layer) is launched, its threads are spread over the SMs, which run them in parallel." },
  cuda: { name: "CUDA cores", what: "The many small parallel processing cores in the GPU.", where: "Inside each SM.", does: "Run general-purpose arithmetic on many numbers at the same time.", inf: "Run the supporting operations: normalization (RMSNorm/LayerNorm), activation functions, softmax, residual additions, rotary position embeddings, sampling." },
  tensor: { name: "Tensor Cores", what: "Specialized circuits built for fast matrix multiplication, especially in low precision (FP16, BF16, FP8).", where: "Inside the GPU chip, alongside the general cores (shown here inside each SM).", does: "Perform high-throughput matrix operations: lots of multiply-and-add work.", inf: "Accelerate the big multiplications: X×W<sub>q</sub>, X×W<sub>k</sub>, X×W<sub>v</sub>, X×W<sub>o</sub>, and the MLP projections. These make up most of the FLOPs." },
  reg: { name: "Registers", what: "The tiniest, fastest storage: a few values held right inside the processing circuits.", where: "Inside the cores.", does: "Hold the exact numbers being worked on this instant, and the thread's own bookkeeping.", inf: "Hold the handful of values in the current multiply-add, such as one element of X, one element of W, and the running sum." },
  sram: { name: "SRAM / shared memory", what: "Small, very fast memory on the chip, shared by the threads working together in an SM.", where: "On the GPU chip, next to the cores.", does: "Stages blocks of data so many threads can reuse them without going back to HBM.", inf: "Fast kernels (for example tiled matrix multiply or FlashAttention) load chunks of weights or activations here, crunch them, then move on." },
  l2: { name: "L2 cache", what: "In this course's model: an intermediate memory level, larger and slower than on-chip SRAM.", where: "Between the compute units and HBM.", does: "Keeps recently or soon-needed data closer to compute than HBM.", inf: "Buffers data travelling between HBM and the cores. Real GPU designs differ here; treat it as the “middle layer” of the hierarchy." },
  nvlink: { name: "NVLink", what: "A high-speed link that connects GPUs directly to each other.", where: "On the edge of the GPU module, wired to other GPUs (or to NVSwitch chips).", does: "Moves data between GPUs much faster than PCIe.", inf: "Needed when one model is split across several GPUs, because the GPUs must exchange results at every layer." },
};
function partInfo(k) {
  const p = GPU_PARTS[k]; if (!p) return "";
  return `<h4 style="font-size:1.05rem">${p.name}</h4><div class="qa4">
    <div class="card"><span class="mini-label">What is it?</span><p>${p.what}</p></div>
    <div class="card"><span class="mini-label">Where is it?</span><p>${p.where}</p></div>
    <div class="card"><span class="mini-label">What does it do?</span><p>${p.does}</p></div>
    <div class="card"><span class="mini-label">Role in LLM inference</span><p>${p.inf}</p></div></div>`;
}
function wireDiagram(root, scope, onPick) {
  scope.addEventListener("click", e => {
    const el = e.target.closest("[data-part]"); if (!el || !scope.contains(el)) return;
    e.stopPropagation();
    const k = el.dataset.part;
    $$("[data-part]", scope).forEach(x => x.classList.toggle("sel", x.dataset.part === k));
    onPick(k);
  });
}

/* ---------- Threads & warps player (shared by the GPU topics) ---------- */
function twHTML(p) {
  return `${player(p + "-pl")}
    <div class="caption" id="${p}-cap"></div>
    <div class="tw">
      <div style="display:grid;gap:6px;align-content:start">
        <div class="mini-label">Output Y = X × W · 4 rows × 8 columns · one cell = one output number</div>
        <div class="scroll-x"><div class="tw-grid" id="${p}-grid"></div></div>
        <div class="tw-legend" id="${p}-legend"></div>
      </div>
      <div class="tw-side">
        <div class="mini-label">The program every thread runs</div>
        <div class="tw-steps" id="${p}-steps"></div>
        <div class="mini-label">Streaming Multiprocessors</div>
        <div class="tw-sms" id="${p}-sms"></div>
      </div>
    </div>`;
}
function mountTW(root, ctx, p) {
  const X = [[1, 2], [3, 1], [2, 2], [1, 3]];
  const W = [[1, 0, 2, 1, 3, 0, 1, 2], [0, 1, 1, 2, 0, 3, 2, 1]];
  const Y = X.map(r => W[0].map((_, c) => r[0] * W[0][c] + r[1] * W[1][c]));
  const WN = ["A", "B", "C", "D"], WC = ["#2f6bdc", "#df7414", "#8753e6", "#1f8a4c"], SM_OF = [0, 0, 1, 1];
  const frames = [
    { k: "job" }, { k: "threads" }, { k: "one" }, { k: "warps" }, { k: "sms" },
    { k: "run", rows: [0, 2], step: 0 }, { k: "run", rows: [0, 2], step: 1 }, { k: "run", rows: [0, 2], step: 2 },
    { k: "run", rows: [1, 3], step: 0 }, { k: "run", rows: [1, 3], step: 1 }, { k: "run", rows: [1, 3], step: 2 },
    { k: "done" },
  ];
  const STEPS = ["Load its 4 numbers into registers", "Multiply the two pairs", "Add them and write the result"];
  const order = ["job", "threads", "one", "warps", "sms", "run", "done"];
  const render = (i) => {
    const f = frames[i], at = order.indexOf(f.k);
    const round2 = f.k === "run" && f.rows[0] === 1;
    const rowDone = r => f.k === "done" || (f.k === "run" && ((round2 && (r === 0 || r === 2)) || (f.rows.includes(r) && f.step === 2)));
    const rowRun = r => f.k === "run" && f.rows.includes(r) && f.step < 2;
    let html = "";
    for (let r = 0; r < 4; r++) for (let c = 0; c < 8; c++) {
      const n = r * 8 + c + 1, warps = at >= 3;
      const focus = f.k === "one" && r === 1 && c === 2;
      const run = rowRun(r), done = rowDone(r);
      let text = at === 0 ? "?" : `T${n}`;
      if (run) text = f.step === 0 ? "load" : "×";
      if (done) text = String(Y[r][c]);
      const style = [
        warps ? `border-color:${WC[r]}` : "",
        warps && !run && !done ? `background:${WC[r]}14` : "",
        run ? `background:${WC[r]};color:#fff` : "",
        done ? `background:${WC[r]}33;color:var(--ink);font-weight:700;font-size:.85rem` : "",
        f.k === "one" && !focus ? "opacity:.3" : "",
      ].join(";");
      html += `<div class="tw-cell ${focus ? "focus" : ""} ${run ? "run" : ""}" style="${style}">${text}</div>`;
    }
    $("#" + p + "-grid", root).innerHTML = html;
    $("#" + p + "-legend", root).innerHTML = at >= 3
      ? WN.map((w, r) => `<span><i style="background:${WC[r]}"></i>Warp ${w}: T${r * 8 + 1}–T${r * 8 + 8}${at >= 4 ? ` → SM ${SM_OF[r] + 1}` : ""}</span>`).join("")
      : at >= 1 ? `<span>32 threads · each one computes exactly one cell</span>` : `<span>Nothing computed yet</span>`;

    let steps = STEPS.map((s, j) => `<div class="${f.k === "run" && f.step === j ? "on" : ""}"><b>${j + 1}</b> ${s}</div>`).join("");
    if (f.k === "one") steps = `<div class="code">T11 → row 1, column 2<br>y[1][2] = x[1][0]·w[0][2] + x[1][1]·w[1][2]<br>= 3·2 + 1·1 = <b>7</b></div>` + steps;
    else if (at >= 1) steps = `<div class="code">Tn → its own row r and column c<br>y[r][c] = x[r][0]·w[0][c] + x[r][1]·w[1][c]</div>` + steps;
    $("#" + p + "-steps", root).innerHTML = at === 0 ? `<p class="c-muted" style="font-size:.85rem">Not created yet.</p>` : steps;

    $("#" + p + "-sms", root).innerHTML = at < 4 ? `<p class="c-muted" style="font-size:.85rem">No work assigned yet.</p>` : [0, 1].map(s => `<div class="tw-sm"><b>SM ${s + 1}</b>${[0, 1, 2, 3].filter(r => SM_OF[r] === s).map(r => {
      const active = rowRun(r) || (f.k === "run" && f.rows.includes(r));
      return `<span class="chip ${active ? "active" : ""}" style="border-color:${WC[r]};${active ? `background:${WC[r]};color:#fff` : ""}">Warp ${WN[r]} ${rowDone(r) ? "✓" : ""}</span>`;
    }).join("")}</div>`).join("");

    const caps = {
      job: ["The job", "A <b>kernel</b> (a small GPU program) is launched: compute <b>Y = X × W</b>. Y has 4 rows and 8 columns, so 32 output numbers. Each one can be computed without waiting for the others."],
      threads: ["Threads", "The GPU creates <b>one thread per output number</b>: T1 to T32. A thread is not a piece of hardware. It is one copy of the same small program, pointed at a different cell."],
      one: ["One thread up close", "Thread <b>T11</b> computes row 1, column 2. Every thread runs exactly these steps. Only the row and column differ."],
      warps: ["Warps", "Threads are grouped into <b>warps</b>: here 4 warps of 8 threads, one row each. All threads in a warp execute the <b>same instruction at the same moment</b>. (NVIDIA's classic design used 32 threads per warp.)"],
      sms: ["Assign to SMs", "Each <b>Streaming Multiprocessor</b> receives warps to run on its cores: SM 1 gets warps A and B, SM 2 gets C and D."],
      done: ["Finished", "All 32 numbers were produced in <b>2 rounds × 3 instructions = 6 steps</b>, instead of 32 × 3 = 96 steps one at a time. A real GPU does this with thousands of threads per kernel, for every matrix in every layer."],
    };
    let cap;
    if (f.k === "run") {
      const names = f.rows.map(r => WN[r]).join(" and ");
      cap = [`Round ${round2 ? 2 : 1} · instruction ${f.step + 1} of 3`,
        [`SM 1 runs warp ${WN[f.rows[0]]} while SM 2 runs warp ${WN[f.rows[1]]}, at the same time. All 16 threads <b>load their numbers</b> together.`,
          `All 16 threads in warps ${names} <b>multiply their pairs</b> at the same instant.`,
          `All 16 threads <b>add and write</b> their result. Rows ${names} are finished.`][f.step]];
    } else cap = caps[f.k];
    $("#" + p + "-cap", root).innerHTML = `<span class="cap-step">${cap[0]}</span><span>${cap[1]}</span>`;
  };
  return bindPlayer(root, p + "-pl", ctx, { frames: frames.length, render, interval: 2600 });
}

/* ---------- Topic: What is a GPU? ---------- */
topic({
  num: 0, title: "What is a GPU?",
  question: "What is a GPU, and why do LLMs run on one?",
  render: () => `
    ${say(
      "A processor does tiny maths steps (add, multiply) billions of times a second. Chips differ in <b>how many steps they do at the same time</b>.",
      "<b>CPU = a few experts:</b> 8–128 flexible cores, great at decision-heavy work → tens of things at once. <b>GPU = a huge team:</b> thousands of simple cores doing the same step on different data → tens of thousands at once.",
      "An LLM needs ~140 billion identical, independent steps per token. That's a job for the team."
    )}
    ${see("How exactly does a GPU split a real problem across its cores?", `
      <div class="btn-row">${seg("g1-prob", [["photo", "Brighten a photo (64 pixels)"], ["matmul", "Multiply matrices, like an LLM layer (16 numbers)"]], "photo")}</div>
      <div class="caption" id="g1-task"></div>
      ${player("g1-pl")}
      <div class="caption" id="g1-cap"></div>
      <div class="grid2">
        <div class="loop-box">
          <h4>CPU <span class="pill mixed">4 fast cores</span></h4>
          <div class="pgrid" id="g1-cpu-grid"></div>
          <div class="corelist" id="g1-cpu-cores"></div>
          <div class="stat-row"><div class="stat"><span class="v tnum" id="g1-cpu-s">0</span><span class="k">rounds of work</span></div><div class="stat"><span class="v tnum" id="g1-cpu-t">0 ms</span><span class="k">time (1 ms per round)</span></div></div>
        </div>
        <div class="loop-box">
          <h4>GPU <span class="pill compute">1 simple core per item</span></h4>
          <div class="pgrid" id="g1-gpu-grid"></div>
          <div class="corelist" id="g1-gpu-cores"></div>
          <div class="stat-row"><div class="stat"><span class="v tnum" id="g1-gpu-s">0</span><span class="k">rounds of work</span></div><div class="stat"><span class="v tnum" id="g1-gpu-t">0 ms</span><span class="k">time (2 ms per round: slower cores)</span></div></div>
        </div>
      </div>
      <p class="c-muted" style="font-size:.82rem">Each item is independent: no pixel or output number needs another one's result. A CPU core finishes one item in 1 ms; a simpler GPU core needs 2 ms. The GPU still wins because every item gets its own core. Real GPUs have thousands of cores and real LLM matrices have millions of numbers. ${illus("Toy timings")}</p>`)}
    ${see("Why is an LLM a perfect job for a GPU?", `
      <div class="grid3">
        <div class="card"><h4>An LLM is mostly matrix multiplication</h4><p>Every layer multiplies token vectors by weight matrices: X × W<sub>q</sub>, X × W<sub>k</sub>, X × W<sub>v</sub>, and more.</p></div>
        <div class="card"><h4>Every output number uses the same recipe</h4><p>Y[i][j] = row i of X · column j of W. Same steps, different numbers.</p></div>
        <div class="card"><h4>The numbers don't wait for each other</h4><p>Y[0][0] and Y[5][7] can be computed at the same moment, so thousands of cores can work together.</p></div>
      </div>`)}
    ${see("What does a GPU need besides its cores?", `
      <svg class="svg-block" viewBox="0 0 720 210" id="g1-parts"></svg>
      <div class="grid3">
        <div class="card"><h4 class="c-compute">Cores that compute</h4><p>Do the arithmetic: thousands of multiplies and adds at the same time.</p></div>
        <div class="card"><h4 class="c-memory">Memory that holds the numbers</h4><p>The weights and the KV cache must sit in the GPU's own memory, close to the cores.</p></div>
        <div class="card"><h4>Connections</h4><p>A link to the CPU that sends the work, and links to other GPUs when one isn't enough.</p></div>
      </div>
      <p style="font-size:.95rem">A GPU is only as fast as its cores <b>and</b> the speed at which numbers reach those cores. That second part turns out to matter a lot for LLMs.</p>`)}
    ${ex(`<div class="calc">One token through a 70B-class model ≈ 140 billion multiplies and adds<br>
      One at a time, even at a billion per second → 140 seconds per token<br>
      Thousands of cores working together → a small fraction of a second per token</div>`)}
    ${tech(`<ul>
      <li><b>Core:</b> an independent processing unit inside a chip that executes instructions.</li>
      <li><b>CPU:</b> few (typically 8–128) complex cores, each optimized to run one instruction stream with low latency on varied, branching tasks. Cores run in parallel, so a CPU handles tens of tasks at once.</li>
      <li><b>GPU:</b> thousands of simpler cores, optimized for <b>throughput</b> on data-parallel tasks: the same operation applied to many data elements (SIMD / SIMT execution).</li>
      <li>A GPU is an <b>accelerator</b>: the CPU still runs the program and hands heavy maths to the GPU.</li>
    </ul>`)}
    ${inf(`<p>For an LLM, the CPU receives the request and sends work to the GPU. The GPU runs the maths of every layer for every token. To do that it needs two things: <b class="c-compute">cores to compute</b> and <b class="c-memory">memory holding the weights</b> close by. The rest of this part opens up the GPU to see both.</p>`)}
    ${check({
      q: "Why are GPUs better than CPUs for running LLMs?",
      opts: ["Each GPU core is much smarter than a CPU core", "LLMs are huge numbers of identical, independent calculations, which thousands of GPU cores can do at the same time", "GPUs have no memory limits", "CPUs can't do multiplication"],
      a: 1, why: "GPU cores are individually simpler, but matrix multiplication splits into many identical independent pieces, so doing them all at once wins by a huge margin.",
    })}`,
  mount(root, ctx) {
    /* two real problems: brighten a photo, multiply matrices */
    const PHOTO = [
      "........", "..####..", ".#....#.", "#.#..#.#", "#......#", "#.#..#.#", ".#.##.#.", "..####..",
    ].map(r => r.split("").map(ch => ch === "#" ? 150 : 40));
    const X = [[1, 2], [3, 1], [2, 2], [1, 3]], W = [[2, 0, 1, 3], [1, 2, 2, 0]];
    const PROBS = {
      photo: {
        task: "<b>Task:</b> brighten a photo. Rule for <b>every</b> pixel: <code>new = old + 60</code>. The photo is 8 × 8 = 64 pixels, shown with their brightness numbers.",
        rows: 8, cols: 8,
        before: (r, c) => PHOTO[r][c], after: (r, c) => Math.min(255, PHOTO[r][c] + 60),
        work: (r, c) => `pixel (${r},${c}): ${PHOTO[r][c]} + 60 = ${Math.min(255, PHOTO[r][c] + 60)}`,
        color: v => `rgb(${v},${v},${v})`,
      },
      matmul: {
        task: "<b>Task:</b> Y = X × W. X = [[1,2],[3,1],[2,2],[1,3]], W = [[2,0,1,3],[1,2,2,0]]. Rule for <b>every</b> output cell: <code>y[r][c] = x[r][0]·w[0][c] + x[r][1]·w[1][c]</code>. Y has 4 × 4 = 16 numbers.",
        rows: 4, cols: 4,
        before: () => "?", after: (r, c) => X[r][0] * W[0][c] + X[r][1] * W[1][c],
        work: (r, c) => `y[${r}][${c}] = ${X[r][0]}·${W[0][c]} + ${X[r][1]}·${W[1][c]} = ${X[r][0] * W[0][c] + X[r][1] * W[1][c]}`,
        color: null,
      },
    };
    let prob = "photo", pl = null;
    const CPU_CORES = 4;
    const cellHTML = (P, r, c, state, core) => {
      const v = state === "done" ? P.after(r, c) : P.before(r, c);
      let style = "";
      if (P.color) { const g = typeof v === "number" ? v : 40; style = `background:${P.color(g)};color:${g > 120 ? "#111" : "#fff"}`; }
      return `<div class="pc ${state}" style="${style}">${state === "work" ? `<b>C${core}</b>` : v}</div>`;
    };
    const render = (f) => {
      const P = PROBS[prob], n = P.rows * P.cols, cpuRounds = Math.ceil(n / CPU_CORES);
      const frames = cpuRounds + 2;
      // frame 0 = before, frames 1..cpuRounds = CPU rounds, last = summary
      const round = Math.min(f, cpuRounds);
      const cpuDone = Math.min(n, Math.max(0, (round - (f <= cpuRounds && f > 0 ? 1 : 0)) * CPU_CORES));
      const cpuWorking = f >= 1 && f <= cpuRounds ? [...Array(CPU_CORES)].map((_, k) => cpuDone + k).filter(i => i < n) : [];
      const gpuState = f === 0 ? "todo" : f === 1 ? "work" : "done";
      const grid = (who) => {
        let h = "";
        for (let i = 0; i < n; i++) {
          const r = Math.floor(i / P.cols), c = i % P.cols;
          if (who === "cpu") {
            const st = cpuWorking.includes(i) ? "work" : i < cpuDone || f > cpuRounds ? "done" : "todo";
            h += cellHTML(P, r, c, st, cpuWorking.indexOf(i) + 1);
          } else h += cellHTML(P, r, c, gpuState, i + 1);
        }
        return h;
      };
      $("#g1-cpu-grid", root).style.gridTemplateColumns = `repeat(${P.cols}, 1fr)`;
      $("#g1-gpu-grid", root).style.gridTemplateColumns = `repeat(${P.cols}, 1fr)`;
      $("#g1-cpu-grid", root).innerHTML = grid("cpu");
      $("#g1-gpu-grid", root).innerHTML = grid("gpu");
      $("#g1-cpu-cores", root).innerHTML = cpuWorking.length
        ? cpuWorking.map((i, k) => `<div><b>Core ${k + 1}</b> ${P.work(Math.floor(i / P.cols), i % P.cols)}</div>`).join("")
        : `<div class="c-muted">${f === 0 ? "Waiting to start." : "All 4 cores finished."}</div>`;
      $("#g1-gpu-cores", root).innerHTML = f === 1
        ? `<div><b>Core 1</b> ${P.work(0, 0)}</div><div><b>Core 2</b> ${P.work(0, 1)}</div><div class="c-muted">… cores 3 to ${n - 1} …</div><div><b>Core ${n}</b> ${P.work(P.rows - 1, P.cols - 1)}</div>`
        : `<div class="c-muted">${f === 0 ? "Waiting to start." : `Finished in round 1. Idle while the CPU keeps working.`}</div>`;
      const cpuR = Math.min(f, cpuRounds), gpuR = Math.min(f, 1);
      $("#g1-cpu-s", root).textContent = cpuR; $("#g1-cpu-t", root).textContent = `${cpuR} ms`;
      $("#g1-gpu-s", root).textContent = gpuR; $("#g1-gpu-t", root).textContent = `${gpuR * 2} ms`;
      const cap = f === 0 ? ["Before", `Nothing is computed yet. Both chips get the same ${n} independent items.`]
        : f === 1 ? ["Round 1", `The CPU's 4 cores take the first 4 items. The GPU gives <b>each of the ${n} items its own core</b> and computes all of them at the same moment.`]
        : f <= cpuRounds ? [`Round ${f}`, `The CPU moves on to the next 4 items (${cpuDone + 1}–${Math.min(n, cpuDone + 4)}). The GPU finished long ago.`]
        : ["Result", `Same answer on both. CPU: ${cpuRounds} rounds × 1 ms = <b>${cpuRounds} ms</b>. GPU: 1 round × 2 ms = <b>2 ms</b>, about <b>${fmt(cpuRounds / 2, 1)}× faster</b>, even though each GPU core is slower.`];
      $("#g1-cap", root).innerHTML = `<span class="cap-step">${cap[0]}</span><span>${cap[1]}</span>`;
      $("#g1-task", root).innerHTML = `<span>${P.task}</span>`;
      return frames;
    };
    const framesFor = p => Math.ceil(PROBS[p].rows * PROBS[p].cols / CPU_CORES) + 2;
    pl = bindPlayer(root, "g1-pl", ctx, { frames: framesFor(prob), render, interval: 1300 });
    bindSeg(root, "g1-prob", v => { prob = v; pl.stop(); pl.setFrames(framesFor(v)); pl.show(0); });
    $("#g1-parts", root).innerHTML = `
      <rect x="250" y="40" width="220" height="130" rx="12" fill="var(--compute-soft)" stroke="var(--compute)" stroke-width="2"/>
      <text x="360" y="95" text-anchor="middle" font-size="16" font-weight="700">GPU chip</text>
      <text x="360" y="116" text-anchor="middle" font-size="12" class="muted">thousands of cores: compute</text>
      <rect x="500" y="40" width="200" height="130" rx="12" fill="var(--memory-soft)" stroke="var(--memory)" stroke-width="2"/>
      <text x="600" y="95" text-anchor="middle" font-size="16" font-weight="700">GPU memory</text>
      <text x="600" y="116" text-anchor="middle" font-size="12" class="muted">holds weights and KV cache</text>
      <line x1="470" y1="105" x2="500" y2="105" stroke="var(--memory)" stroke-width="6"/>
      <rect x="20" y="40" width="190" height="130" rx="12" fill="var(--surface-2)" stroke="var(--line)" stroke-width="2" stroke-dasharray="6 4"/>
      <text x="115" y="95" text-anchor="middle" font-size="16" font-weight="700">CPU + server</text>
      <text x="115" y="116" text-anchor="middle" font-size="12" class="muted">sends the work</text>
      <line x1="210" y1="105" x2="250" y2="105" stroke="var(--muted)" stroke-width="3" stroke-dasharray="5 4"/>
      <text x="230" y="95" text-anchor="middle" font-size="10" class="muted">PCIe</text>
      <text x="360" y="195" text-anchor="middle" font-size="12" class="muted">The next topic opens each box.</text>`;
  },
});

/* ---------- Topic: Inside a GPU, one part at a time ---------- */
const BUILD = [
  { parts: ["board"], name: "The GPU card", kind: "", what: "A separate board plugged into the server. It carries its own processor chip and its own memory.", like: "A factory building: machines, storage and loading docks under one roof.", size: "About the size of a book.", inf: "The whole LLM runs here. A server usually holds 1 to 8 of these." },
  { parts: ["die"], name: "The GPU chip", kind: "compute", what: "The piece of silicon where every calculation happens. Everything we add next, until the memory, lives inside it.", like: "The factory floor, where all the production work happens.", size: "A few centimetres across.", inf: "Every multiply and add of every layer, for every token, happens on this chip." },
  { parts: ["cuda"], name: "CUDA cores", kind: "compute", what: "Thousands of small, simple processing cores. Each can do basic arithmetic on numbers it is given.", like: "Thousands of general-purpose machines, each doing simple operations.", size: "Thousands per chip (only a few drawn).", inf: "Do the general maths around the big multiplications: normalization, activation functions, softmax, adding results together, picking the next token." },
  { parts: ["tensor"], name: "Tensor Cores", kind: "compute", what: "Circuits specialized for one job: multiplying matrices, especially with low-precision numbers (FP16, FP8).", like: "Specialized high-speed machines built for one repeated job.", size: "Part of the chip, next to the general cores.", inf: "Speed up the big weight multiplications (X×W<sub>q</sub>, X×W<sub>k</sub>, X×W<sub>v</sub>, X×W<sub>o</sub>, the MLP), which are most of an LLM's maths." },
  { parts: ["sm"], name: "SMs (Streaming Multiprocessors)", kind: "compute", what: "The cores are organized into groups called Streaming Multiprocessors. Each SM receives work, splits it into small pieces and runs those pieces on its cores.", like: "Production cells, each with a supervisor who assigns work orders to its machines.", size: "Many per chip (8 drawn).", inf: "When the GPU is told “multiply this layer's matrices”, the SMs share the job out and run it in parallel." },
  { parts: ["cuda", "sm"], name: "Threads and warps", kind: "compute", threads: true, what: "A <b>thread</b> is one small piece of work, such as computing one output number. SMs run threads in groups called <b>warps</b>, where every thread in the group does the same instruction at the same time.", like: "A thread is one work order. A warp is a batch of identical work orders that machines carry out in lockstep.", size: "Not a physical part: a way of organizing work.", inf: "One matrix multiply becomes thousands of threads, one for each piece of the result, run warp by warp across the SMs." },
  { parts: ["reg"], name: "Registers", kind: "memory", level: 0, what: "The tiniest memory: a few slots right inside the processing circuit, holding the exact numbers being worked on this instant.", like: "The part currently held in a machine's gripper.", size: "Tiny, extremely fast.", inf: "Hold the two numbers being multiplied and the running total, for example one weight and one input value." },
  { parts: ["sram"], name: "SRAM / shared memory", kind: "memory", level: 1, what: "Small, very fast memory on the chip, shared by the threads of an SM.", like: "The parts tray right at the machines of a cell.", size: "Small, very fast.", inf: "Holds a block of weights or token vectors that many threads are working on together, so they don't each fetch it." },
  { parts: ["l2"], name: "L2 cache", kind: "memory", level: 2, what: "A larger but slower layer of memory between the on-chip memory and HBM (a simplified level in this course).", like: "A parts rack beside the production line.", size: "Larger, slower than on-chip memory.", inf: "Buffers the chunks of weights and activations that are about to be needed or were just produced." },
  { parts: ["hbm"], name: "HBM (High Bandwidth Memory)", kind: "memory", level: 3, what: "Stacks of memory chips on the same package as the GPU chip, right beside it, with very many short, wide connections.", like: "The warehouse attached to the factory, with many wide loading docks.", size: "Huge (tens to hundreds of GB), slower than on-chip memory, but still very fast.", inf: "Where the <b>model weights</b>, the <b>KV cache</b> and the <b>activations</b> live. Every token needs data from here." },
  { parts: ["host", "cpu", "ram", "disk", "pcie"], name: "CPU, CPU RAM and the PCIe link", kind: "", what: "The server around the GPU: its CPU, its main memory, its storage, and the PCIe connection to the GPU.", like: "Head office and an off-site storage facility, connected by a public road.", size: "PCIe is far slower than the GPU's own memory path.", inf: "Weights are copied from disk → CPU RAM → PCIe → HBM once at startup. The CPU schedules requests and sends work, but the per-token maths stays on the GPU." },
  { parts: ["nvlink"], name: "NVLink", kind: "", what: "A fast, direct connection from this GPU to other GPUs.", like: "A private high-speed conveyor to the neighbouring factory.", size: "Much faster than PCIe for GPU-to-GPU traffic.", inf: "Needed when one model is split across several GPUs that must exchange results at every layer (Part 6)." },
  { parts: [], all: true, name: "The whole picture", kind: "", what: "Compute (cores, Tensor Cores, SMs) sits on the chip. Small fast memory sits inside and beside the cores. Big memory (HBM) sits next to the chip. Links connect the GPU to the CPU and to other GPUs.", like: "Machines on the floor, parts in hand and in trays, a rack nearby, an attached warehouse, roads and conveyors outside.", size: "Closer to the cores = smaller and faster.", inf: "Next: where exactly the model's weights go, and what happens when the chip needs them." },
];
topic({
  num: 0, title: "Inside a GPU, one part at a time",
  question: "What are the parts of a GPU, and what does each one do?",
  render: () => `
    ${say(
      "A GPU has two kinds of parts: parts that <b class='c-compute'>compute</b> (do arithmetic) and parts that <b class='c-memory'>remember</b> (hold numbers). Plus connections to the outside world.",
      "We'll build the GPU up piece by piece. For each part: what it is, how big and fast it is, and what it does for an LLM. Press <b>Play</b> or step through."
    )}
    ${see("What does each part of the GPU do?", `
      ${player("g2-pl")}
      <div class="caption" id="g2-cap"></div>
      <div class="gpu-build" id="g2-diag">${gpuDiagram()}</div>
      <div id="g2-extra"></div>
      <div class="qa4" id="g2-info"></div>`)}
    ${see("How does one multiply become threads and warps?", twHTML("g2tw"))}
    ${see("How do the memory parts compare?", `
      <div class="scroll-x"><table class="cmp-table">
        <tr><th>Part</th><th>Where</th><th>Size</th><th>Speed</th><th>Holds during inference</th></tr>
        <tr><td><b>Registers</b></td><td>inside each core</td><td>a few numbers</td><td>extremely fast</td><td>the numbers in the current multiply</td></tr>
        <tr><td><b>SRAM / shared memory</b></td><td>on the chip, beside the cores</td><td>small</td><td>very fast</td><td>a block of weights or vectors being worked on</td></tr>
        <tr><td><b>L2 cache</b></td><td>between the cores and HBM</td><td>larger</td><td>fast</td><td>chunks about to be used</td></tr>
        <tr><td><b>HBM</b></td><td>beside the chip, same package</td><td>tens to hundreds of GB</td><td>slower than on-chip</td><td>weights, KV cache, activations</td></tr>
      </table></div>
      ${note("The layout is a teaching model, not a real floor plan. Real GPUs have far more cores and SMs, and memory designs differ between GPU families.")}`)}
    ${tech(`<ul>
      <li><b>Compute:</b> CUDA cores (general parallel arithmetic), Tensor Cores (accelerated matrix multiply), organized into Streaming Multiprocessors that schedule threads in warps.</li>
      <li><b>Memory hierarchy:</b> registers → on-chip SRAM / shared memory → cache → HBM. Capacity grows and speed drops as you move away from the cores.</li>
      <li><b>Interconnect:</b> PCIe to the host CPU, NVLink (and NVSwitch) to other GPUs.</li>
    </ul>`)}
    ${key("Cores can only compute on numbers that are right in front of them. The big data of an LLM lives further away in HBM. <b>Getting numbers to the cores</b> is as important as the cores themselves.")}
    ${check({
      q: "Which part of the GPU is specialized for the large matrix multiplications that make up most of an LLM's work?",
      opts: ["CUDA cores", "Tensor Cores", "HBM", "PCIe link"],
      a: 1, why: "Tensor Cores are circuits built for high-throughput matrix multiplication. CUDA cores do general arithmetic, HBM stores data, and PCIe connects to the CPU.",
    })}
    ${check({
      q: "Why is HBM placed on the same package, right beside the GPU chip?",
      opts: ["To make the GPU card smaller", "Short, wide connections let data reach the cores faster", "Because HBM also does calculations", "So that the CPU can read it directly"],
      a: 1, why: "The closer and wider the connection, the more data per second can flow between memory and compute. For LLMs, that data flow is often the bottleneck.",
    })}`,
  mount(root, ctx) {
    const diag = $("#g2-diag", root);
    // tag containers that have no data-part so they can be revealed too
    $(".die", diag).dataset.part = "die";
    $(".host", diag).dataset.part = "host";
    const firstFrame = {};
    BUILD.forEach((b, i) => b.parts.forEach(p => { if (!(p in firstFrame)) firstFrame[p] = i; }));
    let anim = null;
    const render = (i) => {
      const b = BUILD[i];
      if (anim) { ctx.stop(anim); anim = null; }
      $$("[data-part]", diag).forEach(el => {
        const p = el.dataset.part, f = firstFrame[p] ?? 0;
        const vis = b.all || i >= f;
        el.classList.toggle("bh", !vis); el.classList.toggle("bv", vis);
        el.classList.toggle("sel", !b.all && b.parts.includes(p) && p !== "host");
      });
      $$(".cores i", diag).forEach(c => c.classList.remove("on"));
      $$(".tcore", diag).forEach(c => c.classList.remove("on"));
      let extra = "";
      if (b.parts.includes("cuda") && !b.threads) anim = ctx.every(180, () => $$(".cores i", diag).forEach(c => c.classList.toggle("on", Math.random() < .5)));
      if (b.parts.includes("tensor")) anim = ctx.every(300, () => $$(".tcore", diag).forEach(c => c.classList.toggle("on")));
      if (b.threads) {
        extra = `<div class="tw-summary">
          <div class="tws"><b>1 kernel</b><span>“compute Y = X × W”</span></div><span class="arr">→</span>
          <div class="tws"><b>32 threads</b><span>one per output number</span></div><span class="arr">→</span>
          <div class="tws"><b>4 warps</b><span>8 threads each, run in lockstep</span></div><span class="arr">→</span>
          <div class="tws"><b>2 SMs</b><span>each runs its warps on its cores</span></div>
        </div><p class="c-muted" style="font-size:.85rem">Step through this in detail in <b>“How does one multiply become threads and warps?”</b> below.</p>`;
      }
      if (b.level !== undefined) {
        const lv = [["Registers", 4, 100], ["SRAM", 18, 80], ["L2 cache", 40, 55], ["HBM", 100, 30]];
        extra = `<div class="grid2">${[["Size", 1, "var(--memory)"], ["Speed", 2, "var(--compute)"]].map(([t, idx, c]) => `<div class="bars-h"><div class="mini-label">${t} (relative)</div>${lv.map((l, j) => `<div class="row" style="grid-template-columns:80px 1fr;${j === b.level ? "font-weight:700" : "opacity:.55"}"><span>${l[0]}</span><div><div class="b" style="width:${l[idx]}%;background:${c}"></div></div></div>`).join("")}</div>`).join("")}</div>`;
      }
      $("#g2-extra", root).innerHTML = extra;
      $("#g2-cap", root).innerHTML = `<span class="cap-step">Part ${i + 1} of ${BUILD.length}${b.kind ? ` · ${b.kind === "compute" ? "computes" : "remembers"}` : ""}</span><span><b>${b.name}.</b> ${b.what}</span>`;
      $("#g2-info", root).innerHTML = `
        <div class="card"><span class="mini-label">Comparable to</span><p>${b.like}</p></div>
        <div class="card"><span class="mini-label">Size and speed</span><p>${b.size}</p></div>
        <div class="card" style="grid-column:span 2"><span class="mini-label">Job in LLM inference</span><p>${b.inf}</p></div>`;
    };
    bindPlayer(root, "g2-pl", ctx, { frames: BUILD.length, render, interval: 5200 });
    mountTW(root, ctx, "g2tw");
  },
});

/* ---------- Topic 5 ---------- */
topic({
  num: 5, title: "Where do model weights live?",
  question: "Where are my model weights, physically?",
  render: () => `
    ${say(
      "A model file is really just a <b>very long list of numbers</b>: the weights. A 70B model has 70 billion of them.",
      "The GPU's cores can only use numbers that are in the GPU's own memory. Disk and CPU memory are connected through much slower links, so reading the weights from there for every token would make generation many times slower.",
      "So when the server starts, the weights are copied <b>once</b> into the GPU's big memory, <b class='c-memory'>HBM</b>, and stay there. While serving, HBM also fills up with two more things: the <b class='c-kv'>KV cache</b> and temporary <b class='c-compute'>activations</b>."
    )}
    ${see("How do the weights get into the GPU, and what fills HBM while serving?", `
      ${player("t5-pl")}
      <div class="caption" id="t5-cap"></div>
      <div id="t5-diag">${gpuDiagram()}</div>
      <div><div class="mini-label" style="margin-bottom:4px">Contents of HBM (80 GB GPU, 8B-class model in FP16)</div><div class="stack" id="t5-hbm"></div></div>`)}
    ${see("Does my model fit on one GPU?", `
      <div class="btn-row">${seg("t5-model", [["8", "8B"], ["13", "13B"], ["70", "70B"], ["405", "405B"]], "70")}
        <span class="c-muted" style="font-size:.85rem">parameters, in FP16 (2 bytes each)</span></div>
      <div class="stack" id="t5-fit"></div>
      <div id="t5-fitmsg" style="font-size:.95rem"></div>`, "Try it")}
    ${ex(`<div class="calc">70B model in FP16<br>70,000,000,000 parameters × 2 bytes = <b>140 GB</b> of weights<br>
      One 80 GB GPU → <span class="c-bad">does not fit</span> (and we still need room for the KV cache)</div>`)}
    ${tech(`<ul>
      <li><b>CPU RAM</b> (system memory) is large but connected to the GPU over PCIe, which is slow compared with the GPU's own memory.</li>
      <li><b>HBM</b> (High Bandwidth Memory) is stacked memory placed on the GPU package, next to the chip, with very wide, short connections. Capacity is tens to hundreds of GB.</li>
      <li>During serving, HBM holds: <b class="c-memory">weights</b> (fixed size), <b class="c-kv">KV cache</b> (grows with tokens and users), <b class="c-compute">activations / intermediate tensors</b> (temporary results inside each layer), plus framework workspace.</li>
      <li>Actual usage is a bit higher than parameters × bytes, because of runtime buffers and memory kept in reserve.</li>
    </ul>`)}
    ${fx(["Weight memory ≈ number of parameters × bytes per parameter"])}
    ${inf(`<p>Every forward pass reads the weights from HBM. They never go back to the CPU while serving. So <b>the size of HBM</b> decides whether a model fits, and <b>the speed of HBM</b> (coming in topic 12) decides how quickly those weights can be read.</p>`)}
    ${check({
      q: "Where do a model's weights live while it is generating tokens?",
      opts: ["On disk, read for every token", "In CPU RAM, copied over PCIe for every token", "In the GPU's HBM", "Inside the registers"],
      a: 2, why: "Weights are loaded once into GPU memory (HBM) and stay there. Disk and CPU RAM are far too slow to use for every token, and registers hold only a few values.",
    })}`,
  mount(root, ctx) {
    const diag = $("#t5-diag", root);
    const steps = [
      { on: ["disk"], cap: "The model file sits on disk: 16 GB of numbers for an 8B-class model in FP16. Disk is huge but far too slow to read from during generation.", hbm: [0, 0, 0] },
      { on: ["disk", "ram"], cap: "At startup, the server reads the file into <b>CPU RAM</b>, the server's main memory.", hbm: [0, 0, 0] },
      { on: ["ram", "pcie"], cap: "The weights are copied across the <b>PCIe link</b> into the GPU. This road is slow compared with the GPU's own memory path, so it is used once, not for every token.", hbm: [0.5, 0, 0] },
      { on: ["hbm"], cap: "The weights are now in <b>HBM</b>, right beside the GPU chip. Loading is done. From here on, the weights never go back to the CPU.", hbm: [1, 0, 0] },
      { on: ["hbm", "tensor", "cuda"], cap: "A request arrives. For every layer, the chip reads that layer's weights from HBM and computes. Temporary results (<b>activations</b>) use some HBM while each layer runs.", hbm: [1, 0.3, 1] },
      { on: ["hbm", "tensor"], cap: "Every layer also produces a Key and a Value vector for each token. These are saved in the <b>KV cache</b>, which is stored in HBM right next to the weights. Unlike the weights, it grows with every token and every user.", hbm: [1, 1, 1] },
      { on: ["hbm"], cap: "Summary: HBM holds <b>weights</b> (fixed size), <b>KV cache</b> (grows) and <b>activations</b> (temporary). What is left is room for more users or longer contexts.", hbm: [1, 1, 1] },
    ];
    const render5 = (i) => {
      const s = steps[i];
      $$("[data-part]", diag).forEach(el => el.classList.toggle("sel", s.on.includes(el.dataset.part)));
      $$(".hbm-chip .fill", diag).forEach(x => { x.style.background = "var(--memory)"; x.style.height = (s.hbm[0] * 55 + s.hbm[1] * 25) + "%"; });
      $$(".cores i", diag).forEach(c => c.classList.toggle("on", s.on.includes("cuda") && Math.random() < .5));
      $$(".tcore", diag).forEach(c => c.classList.toggle("on", s.on.includes("tensor")));
      const w = 16 * s.hbm[0], kv = 20 * s.hbm[1], act = 4 * s.hbm[2];
      $("#t5-hbm", root).innerHTML = `<span class="s-w" style="width:${w / 80 * 100}%">${w ? `weights ${fmt(w, 0)} GB` : ""}</span><span class="s-kv" style="width:${kv / 80 * 100}%">${kv > 8 ? `KV cache ${fmt(kv, 0)} GB` : ""}</span><span class="s-act" style="width:${act / 80 * 100}%"></span><span class="s-free" style="flex:1">free ${fmt(80 - w - kv - act, 0)} GB</span>`;
      $("#t5-cap", root).innerHTML = `<span class="cap-step">Step ${i + 1} of ${steps.length}</span><span>${s.cap}</span>`;
    };
    bindPlayer(root, "t5-pl", ctx, { frames: steps.length, render: render5, interval: 3600 });
    const draw = (b) => {
      const w = +b * 2, cap = DEMO.hbmGB, scale = Math.max(w, cap) * 1.05;
      $("#t5-fit", root).innerHTML = `<span class="s-w" style="width:${w / scale * 100}%">${w} GB weights</span><span class="s-free" style="flex:1"></span><div class="cap-line" style="left:${cap / scale * 100}%" title="80 GB"></div>`;
      const n = Math.ceil(w / (cap * 0.9));
      $("#t5-fitmsg", root).innerHTML = w <= cap * 0.9
        ? `<span class="c-ok">Fits</span> on one 80 GB GPU, leaving ${cap - w} GB for KV cache and activations.`
        : `<span class="c-bad">Does not fit</span> on one 80 GB GPU (red line). Weights alone need at least <b>${n} GPUs</b>, or fewer bytes per parameter (Part 4).`;
    };
    bindSeg(root, "t5-model", draw); draw("70");
  },
});

/* ---------- Topic 6 ---------- */
topic({
  num: 6, title: "What happens when a Transformer operation runs?",
  question: "What happens to one matrix multiply, Y = X·W?",
  render: () => `
    ${say(
      "Almost every step inside a Transformer is a matrix multiplication, <b>Y = X × W</b>: the token's numbers <b>X</b> times a weight matrix <b>W</b>.",
      "A multiplication happens inside a core, and a core can only multiply numbers that are <b>physically inside it at that moment</b>.",
      "But W is not in the core. It is stored in <b class='c-memory'>HBM</b>, the memory chips beside the GPU chip. So every operation is really <b>two jobs</b>: <b class='c-memory'>1. bring the numbers to the core</b>, then <b class='c-compute'>2. do the arithmetic</b>. Both take time. Press <b>Play</b> to follow the numbers."
    )}
    ${see("How do X and W get from memory to the circuits that multiply them?", `
      ${player("t6-pl")}
      ${note("Conceptual data-flow model. Real operations don't all take this exact path: which memory levels are used depends on the operation, the kernel and the GPU design. Use it to understand the hierarchy, not as a literal route.")}
      <div class="grid2" style="grid-template-columns:minmax(200px,260px) 1fr">
        ${flow([
          { t: "HBM<br><small>weights, KV, activations</small>", c: "memory" }, { t: "L2 cache", c: "memory" },
          { t: "SRAM / shared memory", c: "memory" }, { t: "Registers", c: "kv" },
          { t: "Tensor Cores", c: "compute" }, { t: "Matrix computation", c: "compute" }, { t: "Result Y" }], { id: "t6-flow", vertical: true })}
        <div style="display:grid;gap:12px;align-content:start">
          <div class="caption" id="t6-panel" style="min-height:120px"></div>
          <div id="t6-mat" class="scroll-x"></div>
          <div class="stat-row">
            <div class="stat"><span class="v c-memory" id="t6-bytes">0 bytes</span><span class="k">data moved toward compute</span></div>
            <div class="stat"><span class="v c-compute" id="t6-flops">0</span><span class="k">FLOPs done (multiplies + adds)</span></div>
          </div>
        </div>
      </div>
`)}
    ${ex(`<div class="calc">X = [[1, 2], [3, 4]] (2 tokens, 2 numbers each)   W = [[0.5, −1], [2, 1]]<br>
      Y[0][0] = 1×0.5 + 2×2 = 4.5 → <b>2 multiplies + 1 add = 3 FLOPs</b><br>
      4 output numbers × 3 = <b>12 FLOPs</b>. Data moved: 8 numbers × 2 bytes = <b>16 bytes</b>.</div>`)}
    ${tech(`<ul>
      <li><b>HBM</b> stores the large data: full weight matrices and the KV cache.</li>
      <li><b>Caches</b> (L2, SRAM / shared memory) keep a working subset close to the cores so it can be reused without another trip to HBM.</li>
      <li><b>Registers</b> hold the exact operands of the arithmetic happening right now.</li>
      <li><b>Tensor Cores</b> run the multiply-accumulate loops of matrix multiplication at very high throughput, many in parallel.</li>
      <li>Fast kernels are written to <b>reduce data movement</b>: they split matrices into blocks that fit in fast memory (tiling), and fuse several operations so intermediate results never go back to HBM.</li>
    </ul>`)}
    ${inf(`<p>Scale the toy up: in a 70B-class model one projection matrix is 8,192 × 8,192 ≈ 67 million numbers (about <b class="c-memory">134 MB</b> in FP16). For a single token that projection needs about <b class="c-compute">134 million FLOPs</b>. A model has several such matrices in each of its ~80 layers, and they all have to be read for every generated token.</p>`)}
    ${key("Computation requires data, and <b>moving that data has a cost</b>. A GPU can only compute as fast as the numbers reach its cores.")}
    ${check({
      q: "In this model, where do the full weight matrices sit before an operation needs them?",
      opts: ["Registers", "SRAM / shared memory", "HBM", "Tensor Cores"],
      a: 2, why: "Full weight matrices are far too big for on-chip memory. They live in HBM, and only the part needed right now is brought closer to the cores.",
    })}`,
  mount(root, ctx) {
    const X = [[1, 2], [3, 4]], W = [[0.5, -1], [2, 1]];
    const Y = X.map(r => W[0].map((_, j) => r[0] * W[0][j] + r[1] * W[1][j]));
    const steps = [
      { h: "<span class='cap-step'>Job 1 · where the numbers start</span><span><b>HBM</b> stores the whole weight matrix <b>W</b> and the token's numbers <b>X</b>. It is big enough to hold everything, but it sits outside the chip, the furthest point from the cores.</span>", bytes: 0 },
      { h: "<span class='cap-step'>Job 1 · move closer</span><span>Fetching numbers one at a time from HBM would be far too slow. So the next <b>chunk</b> of X and W is pulled into the <b>L2 cache</b>, and the cores don't wait on HBM for every number.</span>", bytes: 16 },
      { h: "<span class='cap-step'>Job 1 · onto the chip</span><span>A block small enough to fit is placed in <b>SRAM / shared memory</b>, right beside the cores. Every thread working on this multiply can reach it instantly.</span>", bytes: 16 },
      { h: "<span class='cap-step'>Job 1 · into the circuit</span><span>The exact two numbers for one multiply, for example x = 1 and w = 0.5, are loaded into <b>registers</b>. Only now can arithmetic happen.</span>", bytes: 16 },
      { h: "<span class='cap-step'>Job 2 · start computing</span><span>The <b>Tensor Cores</b> run many multiply-and-add operations at the same time, one thread per output number.</span>", bytes: 16 },
      { h: "<span class='cap-step'>Job 2 · the arithmetic</span><span>Each output number = a row of X combined with a column of W: multiply the pairs, then add. Watch the FLOP counter: 3 operations per output number.</span>", bytes: 16, compute: true },
      { h: "<span class='cap-step'>Done</span><span><b>Y</b> is ready and becomes the input to the next operation. Notice that 16 bytes had to travel from HBM to the cores before a single FLOP could run. In a real model that is hundreds of MB for each matrix.</span>", bytes: 16, done: true },
    ];
    let cur = 0, auto = null, cellT = null, cellsDone = 0;
    const mat = (M, cls = "", hi = -1, lbl = "") => `<div style="display:inline-grid;gap:4px;vertical-align:middle">
      <div class="mini-label" style="text-align:center">${lbl}</div>
      <div style="display:grid;grid-template-columns:repeat(2,44px);gap:3px">${M.flat().map((v, i) =>
        `<span class="tok ${i === hi ? "lit" : cls}" style="text-align:center;padding:4px 0">${v === null ? "?" : v}</span>`).join("")}</div></div>`;
    const drawMat = (showY, hiCell) => {
      const Yshow = Y.map((r, i) => r.map((v, j) => (i * 2 + j) < showY ? v : null));
      const hiRow = hiCell >= 0 ? Math.floor(hiCell / 2) : -1;
      $("#t6-mat", root).innerHTML = `<div style="display:flex;gap:10px;align-items:center;font-family:var(--f-mono);white-space:nowrap">
        ${mat(X, "prompt", -1, "X")}<span>×</span>${mat(W, "", -1, "W")}<span>=</span>${mat(Yshow, "gen", hiCell, "Y")}</div>
        ${hiCell >= 0 ? `<p class="mono" style="font-size:.85rem;margin-top:6px">Y[${hiRow}][${hiCell % 2}] = ${X[hiRow][0]}×${W[0][hiCell % 2]} + ${X[hiRow][1]}×${W[1][hiCell % 2]} = ${Y[hiRow][hiCell % 2]}</p>` : ""}`;
    };
    const show = (i) => {
      cur = clamp(i, 0, steps.length - 1);
      const s = steps[cur];
      setFlow($("#t6-flow", root), cur);
      $("#t6-panel", root).innerHTML = s.h;
      $("#t6-bytes", root).textContent = `${s.bytes} bytes`;
      if (cellT) { ctx.stop(cellT); cellT = null; }
      if (s.compute) {
        cellsDone = 0; drawMat(0, 0); $("#t6-flops", root).textContent = "0";
        cellT = ctx.every(650, () => {
          cellsDone++;
          $("#t6-flops", root).textContent = cellsDone * 3;
          drawMat(cellsDone, cellsDone < 4 ? cellsDone : -1);
          if (cellsDone >= 4) { ctx.stop(cellT); cellT = null; }
        });
      } else {
        drawMat(s.done ? 4 : 0, -1);
        $("#t6-flops", root).textContent = s.done ? "12" : "0";
      }
    };
    bindPlayer(root, "t6-pl", ctx, { frames: steps.length, render: show, interval: 4200 });
  },
});

/* ---------- Topic: The CPU's role, from request to token ---------- */
topic({
  num: 0, title: "The CPU's role: from request to token",
  question: "Where does the program start, and what do disk, CPU, CPU RAM and PCIe actually do?",
  render: () => `
    ${say(
      "An LLM server is a <b>program</b>, and every program runs on the <b>CPU</b>. The CPU is the manager: it receives requests, prepares the work and tells the GPU what to compute. The <b>GPU</b> is the specialist that does the heavy maths.",
      "The GPU never starts anything on its own. It only runs work the CPU hands it. So every token follows the same route: <b>CPU prepares → GPU computes → CPU delivers</b>."
    )}
    ${see("What does each part of the server do?", `
      <div class="scroll-x"><table class="cmp-table">
        <tr><th>Part</th><th>Hardware or software?</th><th>What it does in LLM serving</th><th>When it's used</th></tr>
        <tr><td><b>Disk / storage</b></td><td>hardware</td><td>Stores the model files (weights), the serving program and the tokenizer while nothing is running. Keeps them when the power is off.</td><td><span class="pill mixed">startup</span></td></tr>
        <tr><td><b>CPU</b></td><td>hardware</td><td>Runs the serving program: receives requests, tokenizes text, decides which requests run together (scheduling), launches GPU kernels, turns token IDs back into text, streams the answer.</td><td><span class="pill ok">every token</span></td></tr>
        <tr><td><b>CPU RAM</b></td><td>hardware</td><td>The CPU's working memory: holds the running program, request text, token IDs and the scheduler's bookkeeping. At startup the weights pass through it piece by piece on their way to the GPU, so it does not need to hold the whole model at once.</td><td><span class="pill ok">every token</span></td></tr>
        <tr><td><b>Network card</b></td><td>hardware</td><td>Receives user requests and sends the generated text back.</td><td><span class="pill ok">every request</span></td></tr>
        <tr><td><b>PCIe link</b></td><td>hardware</td><td>The connection between CPU and GPU. Carries the weights once at startup, then only small things: token IDs in, kernel launch commands, sampled token IDs out.</td><td><span class="pill ok">every token (tiny data)</span></td></tr>
        <tr><td><b>GPU cores</b></td><td>hardware</td><td>Run the kernels: all the matrix maths of every layer.</td><td><span class="pill ok">every token</span></td></tr>
        <tr><td><b>GPU HBM</b></td><td>hardware</td><td>Holds the weights, the KV cache and activations right beside the GPU cores.</td><td><span class="pill ok">every token</span></td></tr>
        <tr><td><b>Serving engine</b> (vLLM, TensorRT-LLM, SGLang…)</td><td>software, runs on the CPU</td><td>The program that does all the CPU jobs above and decides which kernels to launch.</td><td><span class="pill ok">always running</span></td></tr>
        <tr><td><b>Kernel</b></td><td>software, runs on the GPU</td><td>A small GPU program for one operation (explained below).</td><td><span class="pill ok">many per token</span></td></tr>
      </table></div>`)}
    ${see("What happens from server start to a streamed token?", `
      ${player("cp-pl")}
      <div class="caption" id="cp-cap"></div>
      <div class="scroll-x"><svg class="svg-block" viewBox="0 0 920 330" id="cp-svg" style="min-width:760px"></svg></div>
      <div class="stat-row">
        <div class="stat"><span class="v" id="cp-when">–</span><span class="k">how often this happens</span></div>
        <div class="stat"><span class="v" id="cp-where">–</span><span class="k">where the work happens</span></div>
        <div class="stat"><span class="v" id="cp-data">–</span><span class="k">data moved in this step</span></div>
      </div>`)}
    ${see("What is a kernel, and how does it get from the CPU to the GPU?", `
      <div class="grid2">
        <div class="card"><h4>A kernel is software</h4><p>A <b>kernel</b> is a small, pre-compiled program that runs <b>on the GPU</b> and does <b>one operation</b>: multiply two matrices, apply softmax, compute attention, add two tensors. It is <b>not hardware</b>, and it is <b>not</b> the operating-system kernel (a different use of the same word).</p></div>
        <div class="card"><h4>Who writes and runs it</h4><p>GPU experts write kernels (in CUDA C++ or Triton) and ship them in libraries such as <b>cuBLAS</b> (matrix maths) and <b>FlashAttention</b>. The CPU decides <b>which</b> kernel to run and <b>on which data</b>; the GPU's cores <b>execute</b> it, as thousands of threads.</p></div>
      </div>
      ${player("kn-pl")}
      <div class="caption" id="kn-cap"></div>
      <div class="kn">
        <div class="kn-stack" id="kn-stack"></div>
        <div class="kn-code" id="kn-code"></div>
      </div>`)}
    ${ex(`<div class="calc">One decode step of a 70B-class model ≈ 80 layers × ~15 operations ≈ <b>1,000+ kernel launches</b> from the CPU<br>
      Data sent over PCIe per token: a few KB (token IDs, launch commands) · weights moved over PCIe per token: <b>0</b><br>
      That's why real engines work hard to make kernel launching cheap (fused kernels, CUDA graphs): a slow CPU side can leave the GPU waiting.</div>`)}
    ${tech(`<ul>
      <li><b>Host</b> = the CPU side; <b>device</b> = the GPU side. Host code runs on the CPU, device code (kernels) runs on the GPU.</li>
      <li>Launching a kernel is <b>asynchronous</b>: the CPU places a command in a GPU queue and moves on; the GPU works through the queue.</li>
      <li><b>Kernel fusion</b> combines several small operations into one kernel, which means fewer launches and less data movement. <b>CUDA graphs</b> record a whole decode step once and replay it with a single launch.</li>
      <li>When a kernel runs, the GPU creates many <b>threads</b> (one copy of the kernel's code per piece of data) and runs them on its cores in warps (topic 6).</li>
    </ul>`)}
    ${inf(`<p>The CPU does the orchestration: text in, tokens out, scheduling, launching kernels. The GPU does the heavy maths on weights and KV cache that never leave HBM. Only tiny amounts of data cross PCIe per token. If the CPU side is slow, the GPU sits idle, so serving engines keep that path lean.</p>`)}
    ${check({
      q: "Is a GPU kernel hardware or software?",
      opts: ["Hardware: a special core inside the GPU", "Software: a small program for one operation that runs on the GPU's cores", "The operating system's kernel", "A part of CPU RAM"],
      a: 1, why: "A kernel is a small compiled program (like “multiply these matrices”). The CPU launches it; the GPU's cores execute it as many parallel threads.",
    })}
    ${check({
      q: "Which part turns the user's text into token IDs and streams the answer text back?",
      opts: ["The GPU's Tensor Cores", "HBM", "The CPU, running the serving program", "The disk"],
      a: 2, why: "Tokenizing, scheduling and detokenizing are CPU jobs in the serving program. The GPU only runs the model's maths.",
    })}`,
  mount(root, ctx) {
    /* ---------- end-to-end system player ---------- */
    const N = {
      user: [20, 125, 110, 80, "User app"],
      cpu: [200, 55, 170, 90, "CPU"],
      ram: [200, 205, 170, 80, "CPU RAM"],
      disk: [395, 205, 150, 80, "Disk / storage"],
      chip: [650, 55, 250, 90, "GPU chip (cores)"],
      hbm: [650, 205, 250, 80, "GPU HBM"],
    };
    const E = {
      userCpu: "M130,165 L200,110", cpuUser: "M200,110 L130,165",
      diskCpu: "M470,205 L470,170 L370,120", diskRam: "M395,245 L370,245",
      ramHbm: "M370,265 L610,265 L650,255", cpuChip: "M370,95 L610,95 L650,95", chipCpu: "M650,110 L610,110 L370,110",
      hbmChip: "M775,205 L775,145", chipHbm: "M790,145 L790,205", cpuRam: "M285,145 L285,205",
    };
    const F = [
      { phase: "Startup", when: "once", where: "Disk → CPU", data: "program files", on: ["disk", "cpu", "ram"], edges: ["diskCpu"], pkt: "program", cap: "The server starts. The <b>serving program</b> (for example vLLM) is read from <b>disk</b> and starts running on the <b>CPU</b>. Everything begins on the CPU." },
      { phase: "Startup", when: "once", where: "Disk → CPU RAM", data: "one piece at a time (e.g. a ~5 GB shard)", on: ["disk", "ram", "cpu"], edges: ["diskRam"], pkt: "shard", cap: "The weights are stored on <b>disk</b> as several files (shards). The loader <b>memory-maps</b> them (for example safetensors files): each piece is read from disk into <b>CPU RAM</b> only when it is needed. The whole model does <b>not</b> have to fit in CPU RAM." },
      { phase: "Startup", when: "once", where: "CPU RAM → PCIe → HBM", data: "~140 GB in total, piece by piece", on: ["ram", "hbm", "pcie"], edges: ["ramHbm"], pkt: "shard", cap: "Each piece is copied across <b>PCIe</b> into <b>GPU HBM</b>, then the next piece follows, until all ~140 GB are on the GPU. This is the only time the weights cross PCIe. (With <b>GPUDirect Storage</b>, pieces can go from fast storage straight to HBM, skipping CPU RAM.)" },
      { phase: "Startup", when: "once", where: "GPU HBM", data: "none", on: ["hbm", "cpu"], edges: [], cap: "The CPU asks the GPU to reserve the remaining HBM for the <b>KV cache</b> and loads the <b>kernels</b> the model needs. The server is ready." },
      { phase: "Request", when: "per request", where: "Network → CPU", data: "a few KB of text", on: ["user", "cpu"], edges: ["userCpu"], pkt: "text", cap: "A user sends “Why is the sky blue?”. The request arrives over the network and is received by the <b>CPU</b>." },
      { phase: "Request", when: "per request", where: "CPU + CPU RAM", data: "a few KB", on: ["cpu", "ram"], edges: ["cpuRam"], pkt: "IDs", cap: "The CPU <b>tokenizes</b> the text into token IDs, a short list of numbers kept in <b>CPU RAM</b>." },
      { phase: "Per step", when: "every step", where: "CPU", data: "none", on: ["cpu"], edges: [], cap: "The <b>scheduler</b> on the CPU decides which requests run together in the next GPU step (the batch) and which KV cache blocks each one uses." },
      { phase: "Per step", when: "every step", where: "CPU → PCIe → GPU", data: "token IDs + ~1,000 launch commands", on: ["cpu", "chip", "pcie"], edges: ["cpuChip"], pkt: "launch", cap: "The CPU sends the token IDs to the GPU and <b>launches the kernels</b> for every operation in every layer. These are commands like “run matrix-multiply on these tensors”, not data-heavy transfers." },
      { phase: "Per step", when: "every step", where: "GPU (HBM ↔ cores)", data: "~140 GB weights + KV cache read", on: ["chip", "hbm"], edges: ["hbmChip", "chipHbm"], pkt: "weights", cap: "The <b>GPU runs the kernels</b>: it reads weights and KV cache from <b>HBM</b>, computes on its cores, and writes new K and V back. This is the heavy part (topic 10 shows it in detail). The CPU waits or prepares the next step." },
      { phase: "Per step", when: "every step", where: "GPU → PCIe → CPU", data: "a few bytes (token ID)", on: ["chip", "cpu", "pcie"], edges: ["chipCpu"], pkt: "ID", cap: "The GPU picks the next token and sends back <b>only its ID</b>, a few bytes, over PCIe." },
      { phase: "Per step", when: "every token", where: "CPU → Network", data: "a few bytes of text", on: ["cpu", "user"], edges: ["cpuUser"], pkt: "“blue”", cap: "The CPU turns the ID back into text (<b>detokenizes</b>) and streams it to the user. The word appears on screen." },
      { phase: "Decode loop", when: "every token", where: "CPU ⇄ GPU", data: "tiny over PCIe, huge inside the GPU", on: ["cpu", "chip", "hbm", "pcie", "user"], edges: ["cpuChip", "hbmChip", "chipCpu", "cpuUser"], pkt: "loop", cap: "Steps 7–11 repeat for every new token: <b>CPU schedules and launches → GPU computes with HBM → CPU streams</b>, until the answer is finished. Disk is not touched again." },
    ];
    const draw = (i) => {
      const f = F[i];
      const box = (k) => {
        const [x, y, w, h, label] = N[k], on = f.on.includes(k);
        const col = k === "chip" ? "var(--compute)" : k === "hbm" ? "var(--memory)" : "var(--accent)";
        return `<rect x="${x}" y="${y}" width="${w}" height="${h}" rx="8" fill="${on ? (k === "chip" ? "var(--compute-soft)" : k === "hbm" ? "var(--memory-soft)" : "var(--accent-soft)") : "var(--surface)"}" stroke="${on ? col : "var(--line)"}" stroke-width="${on ? 3 : 1.5}"/>
          <text x="${x + w / 2}" y="${y + h / 2 + 5}" text-anchor="middle" font-size="14" font-weight="700" ${on ? "" : 'class="muted"'}>${label}</text>`;
      };
      let s = `<rect x="180" y="20" width="385" height="295" rx="12" fill="none" stroke="var(--line)" stroke-dasharray="6 5"/>
        <text x="190" y="40" font-size="11" class="muted">SERVER (host)</text>
        <rect x="630" y="20" width="285" height="295" rx="12" fill="none" stroke="var(--ink)" stroke-width="1.5"/>
        <text x="640" y="40" font-size="11" class="muted">GPU (device)</text>
        <rect x="575" y="80" width="45" height="200" rx="6" fill="${f.on.includes("pcie") ? "var(--warn-soft)" : "var(--surface-2)"}" stroke="${f.on.includes("pcie") ? "var(--warn)" : "var(--line)"}"/>
        <text x="597" y="185" text-anchor="middle" font-size="12" font-weight="700" transform="rotate(-90 597 185)" ${f.on.includes("pcie") ? "" : 'class="muted"'}>PCIe link</text>
        <text x="75" y="115" text-anchor="middle" font-size="11" class="muted">network</text>`;
      Object.keys(N).forEach(k => s += box(k));
      s += `<text x="285" y="128" text-anchor="middle" font-size="11" class="muted">runs the serving program</text>`;
      f.edges.forEach(e => {
        s += `<path d="${E[e]}" fill="none" stroke="var(--ink)" stroke-width="2.5" stroke-dasharray="7 5"><animate attributeName="stroke-dashoffset" from="24" to="0" dur=".6s" repeatCount="indefinite"/></path>`;
        if (f.pkt) s += `<g><rect x="-26" y="-11" width="52" height="22" rx="11" fill="var(--ink)"/><text x="0" y="4" text-anchor="middle" font-size="10" font-weight="700" style="fill:var(--bg)">${f.pkt}</text>
          <animateMotion dur="1.6s" repeatCount="indefinite" path="${E[e]}"/></g>`;
      });
      $("#cp-svg", root).innerHTML = s;
      $("#cp-cap", root).innerHTML = `<span class="cap-step">Step ${i + 1} of ${F.length} · ${f.phase}</span><span>${f.cap}</span>`;
      $("#cp-when", root).textContent = f.when;
      $("#cp-where", root).textContent = f.where;
      $("#cp-data", root).textContent = f.data;
    };
    bindPlayer(root, "cp-pl", ctx, { frames: F.length, render: draw, interval: 4200 });

    /* ---------- kernel player ---------- */
    const LAYERS = [
      ["Serving engine (Python)", "software · CPU"],
      ["PyTorch", "software · CPU"],
      ["Kernel library (cuBLAS, FlashAttention…)", "software · CPU picks the kernel"],
      ["CUDA driver + PCIe", "software + hardware · CPU → GPU"],
      ["SMs create threads", "hardware · GPU"],
      ["Cores run the kernel code", "hardware · GPU"],
    ];
    const K = [
      { on: 0, cap: "The serving engine, running on the <b>CPU</b>, reaches the Q projection of layer 1. In Python this is one line.", code: `<span class="c-muted"># runs on the CPU</span>\nq = x @ W_q      <span class="c-muted"># multiply token vectors by W_q</span>` },
      { on: 1, cap: "PyTorch sees that <b>x</b> and <b>W_q</b> live in GPU memory, so the multiply must happen on the GPU. It does not do the maths itself.", code: `<span class="c-muted"># inside PyTorch (CPU)</span>\nx.device    → cuda:0   <span class="c-muted"># data is in HBM</span>\nW_q.device  → cuda:0\n→ dispatch to a GPU matrix-multiply kernel` },
      { on: 2, cap: "A library picks the best pre-compiled <b>kernel</b> for this shape and precision, for example a cuBLAS FP16 matrix-multiply kernel that uses Tensor Cores. The kernel is <b>software</b>: compiled GPU code shipped in the library.", code: `<span class="c-muted">// a kernel: GPU code, simplified (CUDA C++)</span>\n<span class="c-muted">// the same code runs in every thread</span>\nvoid matmul_kernel(x, W, y) {\n  r, c = my_thread_position();\n  y[r][c] = sum_k x[r][k] * W[k][c];\n}` },
      { on: 3, cap: "The CPU sends a small <b>launch command</b> over PCIe: “run matmul_kernel on tensors at these HBM addresses”. It is queued on the GPU; the CPU immediately moves on to the next operation.", code: `<span class="c-muted">// launch command (a few bytes)</span>\nkernel:  matmul_kernel\ninputs:  x @ HBM 0x7f3a…, W_q @ HBM 0x7e10…\noutput:  q @ HBM 0x7f51…\nthreads: one per output number` },
      { on: 4, cap: "On the GPU, the SMs create <b>threads</b>: one running copy of the kernel's code for each piece of the output, grouped into warps.", code: `thread 0   → matmul_kernel(r=0, c=0)\nthread 1   → matmul_kernel(r=0, c=1)\n…\nthread N-1 → matmul_kernel(r=…, c=…)` },
      { on: 5, cap: "The <b>Tensor Cores</b> and CUDA cores execute all those threads in parallel, reading W_q from HBM. The result q is written to HBM, ready for the next kernel (the K projection). This repeats for every operation in every layer.", code: `q is ready in HBM ✓\nnext launch: matmul_kernel(x, W_k) …\n<span class="c-muted">≈ 1,000+ launches per decode step for a 70B model</span>` },
    ];
    const drawK = (i) => {
      const k = K[i];
      $("#kn-stack", root).innerHTML = LAYERS.map(([n, t], j) => `<div class="kn-layer ${j === k.on ? "on" : j < k.on ? "done" : ""} ${j >= 4 ? "hw" : ""}"><b>${n}</b><span>${t}</span></div>${j < LAYERS.length - 1 ? `<div class="kn-arr">↓</div>` : ""}`).join("");
      $("#kn-code", root).innerHTML = `<pre>${k.code}</pre>`;
      $("#kn-cap", root).innerHTML = `<span class="cap-step">Step ${i + 1} of ${K.length}</span><span>${k.cap}</span>`;
    };
    bindPlayer(root, "kn-pl", ctx, { frames: K.length, render: drawK, interval: 4200 });
  },
});

/* ---------- Topic 7 ---------- */
topic({
  num: 7, title: "The GPU at work on one token",
  question: "Which parts are busy when my model processes a token?",
  render: () => `
    ${say(
      "Generating one token means running the token through every layer of the model. Each step of a layer is done by specific parts of the GPU: <b class='c-memory'>memory parts</b> bring the weights and KV cache to the chip, and <b class='c-compute'>compute parts</b> do the maths.",
      "Press <b>Play</b> to follow one token from the CPU, through layer 1 in detail, through the other 79 layers, and back out as the next token. Watch which parts light up and how much data moves."
    )}
    ${see("What does the GPU do, step by step, to generate one token?", `
      ${player("tg-pl")}
      <div class="caption" id="tg-cap"></div>
      <div class="tg">
        <div class="tg-arch">
          <div class="mini-label">Model step (70B-class, 80 layers)</div>
          <div id="tg-ops"></div>
        </div>
        <div style="display:grid;gap:10px;align-content:start">
          <div class="mini-label">GPU parts busy in this step</div>
          <div id="tg-diag">${gpuDiagram()}</div>
          <div class="tg-busy" id="tg-busy"></div>
          <div class="stat-row">
            <div class="stat"><span class="v c-memory tnum" id="tg-bytes">0</span><span class="k">weights read from HBM for this token</span></div>
            <div class="stat"><span class="v c-kv tnum" id="tg-kv">0</span><span class="k">KV cache read / written</span></div>
            <div class="stat"><span class="v c-compute tnum" id="tg-flops">0</span><span class="k">FLOPs computed</span></div>
          </div>
        </div>
      </div>
      <p class="c-muted" style="font-size:.82rem">One decode step for a Llama-3-70B-style model in FP16 (hidden size 8,192, 8 KV heads, MLP size 28,672) with 2,000 tokens already in the KV cache. Byte and FLOP counts are rounded. The memory path shown (HBM → L2 → SRAM → registers) is the course's simplified model. ${illus("Approximate")}</p>`)}
    ${see("Which part of the GPU does which job?", `
      <div class="btn-row" id="t7-pick">
        ${[["board", "GPU"], ["sm", "SMs"], ["cuda", "CUDA cores"], ["tensor", "Tensor Cores"], ["threads", "Threads &amp; warps"]].map(([k, t], i) =>
          `<button type="button" class="btn ${i === 0 ? "on" : ""}" data-k="${k}">${t}</button>`).join("")}
      </div>
      <div id="t7-visual"></div>
      <div id="t7-info"></div>`, "Pick a component")}
    ${ex(`<div class="calc">One layer of a 70B-class model, one token:<br>
      X×W<sub>q</sub>, X×W<sub>k</sub>, X×W<sub>v</sub>, X×W<sub>o</sub>, MLP gate, MLP up, MLP down → <b>7 weight multiplies</b><br>
      × 80 layers = <b>560 big matrix multiplies</b> for every single token</div>`)}
    ${tech(`<ul>
      <li><b>Kernel:</b> a small GPU program (for example “multiply these two matrices”). Launching a kernel creates many <b>threads</b> that run the same code on different elements of the data.</li>
      <li><b>SM (Streaming Multiprocessor):</b> an execution unit that creates, schedules and runs those threads on its cores, using its registers and shared memory.</li>
      <li><b>Warp:</b> a group of threads that an SM schedules and runs together, all doing the same instruction on different data. In NVIDIA's classic design a warp was 32 threads; the size is a hardware detail. In CUDA code you'll see threads grouped into <i>blocks</i> instead.</li>
      <li><b>Tensor Cores</b> accelerate mixed-precision matrix multiply. Libraries such as cuBLAS and inference engines route the big projections to them.</li>
    </ul>`)}
    ${inf(`<p>When your model generates a token: the CPU launches kernels layer by layer → SMs split each kernel into threads and warps → <b class="c-compute">Tensor Cores</b> crunch the projections → <b>CUDA cores</b> handle norms, activations and softmax → results flow into the next layer. All of it needs data brought from <b class="c-memory">HBM</b>.</p>`)}
    ${check({
      q: "Which GPU component is specialized for the large matrix multiplications like X×W<sub>q</sub> and the MLP projections?",
      opts: ["Registers", "Tensor Cores", "PCIe", "CPU RAM"],
      a: 1, why: "Tensor Cores are circuits built for high-throughput matrix multiplication, which is where most of the FLOPs in a Transformer layer go.",
    })}`,
  mount(root, ctx) {
    /* ---- one token through the GPU ---- */
    const OPS = [
      { id: "in", label: "Token ID arrives", group: "start" },
      { id: "emb", label: "Embedding lookup", group: "start" },
      { id: "norm1", label: "RMSNorm", group: "attn" },
      { id: "loadqkv", label: "Load W_Q, W_K, W_V", group: "attn" },
      { id: "qkv", label: "Q, K, V projections", group: "attn" },
      { id: "kvw", label: "Write K, V to KV cache", group: "attn" },
      { id: "attn", label: "Attention over the KV cache", group: "attn" },
      { id: "o", label: "Output projection + add", group: "attn" },
      { id: "mlp", label: "RMSNorm + MLP (gate, up, down) + add", group: "mlp" },
      { id: "rep", label: "Repeat for layers 2 … 80", group: "rep" },
      { id: "head", label: "Final norm + vocabulary scores", group: "end" },
      { id: "sample", label: "Sample next token → CPU", group: "end" },
    ];
    const MB = 1e6, GB = 1e9;
    // cumulative [weights bytes, kv bytes, flops] after each step (FP16, one layer then ×80)
    const L = { qkv: 168 * MB, o: 134 * MB, mlp: 1410 * MB, kvw: 0.33 * MB, kvr: 655 * MB / 1000 * 1000 / 80 };
    const STEPS = [
      { op: "in", parts: ["cpu", "pcie"], cap: "The CPU (running the serving program, topic 9) sends the ID of the newest token over <b>PCIe</b> and launches the <b>kernels</b> (small pre-compiled GPU programs, one per operation) for every layer. Only a few bytes cross PCIe: the weights are already on the GPU.", w: 0, kv: 0, f: 0 },
      { op: "emb", parts: ["hbm", "l2", "sram"], read: true, cap: "The token ID selects <b>one row</b> of the embedding table, which is stored in <b>HBM</b>. That row (8,192 numbers, about 16 KB) is copied toward the chip and becomes the token's vector x.", w: 16e3, kv: 0, f: 0 },
      { op: "norm1", parts: ["cuda", "sram", "reg"], cap: "<b>CUDA cores</b> normalize x (RMSNorm): a few simple operations on 8,192 numbers, done in parallel. The numbers sit in <b>SRAM</b> and <b>registers</b> right next to the cores.", w: 16e3, kv: 0, f: 3e4 },
      { op: "loadqkv", parts: ["hbm", "l2", "sram"], read: true, cap: "Layer 1 needs its attention weights. <b>W_Q, W_K and W_V</b> (about 84 million numbers, ~168 MB) stream from <b>HBM</b> through the <b>L2 cache</b> into <b>SRAM</b> in blocks. This is data movement, not maths.", w: 16e3 + 168 * MB, kv: 0, f: 3e4 },
      { op: "qkv", parts: ["tensor", "sram", "reg"], cap: "<b>Tensor Cores</b> multiply x by W_Q, W_K and W_V. The job is split into thousands of threads that run at the same time. Result: this token's Query, Key and Value. About <b>168 million FLOPs</b>.", w: 16e3 + 168 * MB, kv: 0, f: 3e4 + 168e6 },
      { op: "kvw", parts: ["hbm"], kvWrite: true, cap: "The new <b>Key and Value</b> are written into layer 1's section of the <b>KV cache in HBM</b>, so later tokens never have to recompute them.", w: 16e3 + 168 * MB, kv: 4.1e3, f: 3e4 + 168e6 },
      { op: "attn", parts: ["hbm", "l2", "sram", "tensor", "cuda"], read: true, cap: "Attention needs the Keys and Values of all <b>2,000 earlier tokens</b> in this layer: about 8 MB read from the <b>KV cache in HBM</b>. Tensor Cores compare the Query with every Key; CUDA cores apply softmax and mix the Values.", w: 16e3 + 168 * MB, kv: 4.1e3 + 8.2 * MB, f: 3e4 + 168e6 + 33e6 },
      { op: "o", parts: ["hbm", "l2", "sram", "tensor", "cuda"], read: true, cap: "<b>W_O</b> (~134 MB) is loaded from HBM and <b>Tensor Cores</b> multiply by it. <b>CUDA cores</b> add the result back to x (the residual connection).", w: 16e3 + 302 * MB, kv: 4.1e3 + 8.2 * MB, f: 3e4 + 302e6 + 33e6 },
      { op: "mlp", parts: ["hbm", "l2", "sram", "tensor", "cuda"], read: true, cap: "The MLP is the biggest part of each layer. Its three matrices (~705 million numbers, <b>~1.4 GB</b>) stream from HBM. <b>Tensor Cores</b> run gate, up and down projections; <b>CUDA cores</b> apply the SiLU activation and add. Layer 1 done: <b>~1.7 GB moved, ~1.7 billion FLOPs</b>.", w: 16e3 + 1712 * MB, kv: 4.1e3 + 8.2 * MB, f: 3e4 + 1712e6 + 33e6 },
      { op: "rep", parts: ["hbm", "l2", "sram", "tensor", "cuda", "reg"], read: true, kvWrite: true, cap: "The same sequence runs in <b>all 80 layers</b>, one after another. Each layer loads its own weights from HBM, reads and extends its own KV cache, and computes. The totals grow 80×.", w: 137 * GB, kv: 660 * MB, f: 137e9 + 2.6e9 },
      { op: "head", parts: ["hbm", "l2", "sram", "tensor"], read: true, cap: "A final RMSNorm, then the vector is multiplied by the <b>vocabulary matrix</b> (128,000 × 8,192 numbers, ~2.1 GB from HBM) to score every possible next token. Tensor Cores do this multiply.", w: 139 * GB, kv: 660 * MB, f: 139e9 + 2.6e9 },
      { op: "sample", parts: ["cuda", "pcie", "cpu"], cap: "<b>CUDA cores</b> turn the scores into probabilities and pick the next token. Only its <b>ID</b> (a few bytes) goes back to the CPU over PCIe. The CPU turns it into text and streams it to the user, then launches the next step.", w: 139 * GB, kv: 660 * MB, f: 139e9 + 2.6e9 },
      { op: null, parts: ["hbm", "tensor"], summary: true, cap: "<b>One token = ~140 GB of weights read from HBM + ~140 billion FLOPs.</b> The Tensor Cores can do that maths in a small fraction of a second; moving 140 GB takes much longer. That is why decode speed is usually set by how fast memory can deliver data (Part 3).", w: 139 * GB, kv: 660 * MB, f: 139e9 + 2.6e9 },
    ];
    const NAMES = { cpu: "CPU", pcie: "PCIe", hbm: "HBM", l2: "L2 cache", sram: "SRAM", reg: "Registers", tensor: "Tensor Cores", cuda: "CUDA cores" };
    const diagTG = $("#tg-diag", root);
    let curStep = STEPS[0];
    const renderTG = (i) => {
      const st = STEPS[i]; curStep = st;
      const idx = st.op ? OPS.findIndex(o => o.id === st.op) : OPS.length;
      const groupLabel = { attn: "Layer 1 · attention", mlp: "Layer 1 · MLP" };
      let lastGroup = "";
      $("#tg-ops", root).innerHTML = OPS.map((o, j) => {
        const head = (o.group === "attn" || o.group === "mlp") && o.group !== lastGroup ? `<div class="tg-group">${groupLabel[o.group]}</div>` : "";
        lastGroup = o.group;
        return head + `<div class="tg-op ${j === idx ? "on" : j < idx ? "done" : ""} ${o.group}">${j < idx ? "✓ " : ""}${o.label}</div>`;
      }).join("");
      $$("[data-part]", diagTG).forEach(el => el.classList.toggle("sel", st.parts.includes(el.dataset.part)));
      $$(".hbm-chip", diagTG).forEach(c => c.classList.toggle("reading", !!st.read));
      $$(".hbm-chip .fill", diagTG).forEach(x => { x.style.height = st.kvWrite ? "100%" : "0"; x.style.background = "var(--kv)"; });
      $("#tg-busy", root).innerHTML = Object.keys(NAMES).map(k => `<span class="pill ${st.parts.includes(k) ? (["tensor", "cuda", "reg"].includes(k) ? "compute" : k === "cpu" || k === "pcie" ? "mixed" : "memory") : ""}" style="${st.parts.includes(k) ? "" : "background:var(--surface-2);color:var(--muted);opacity:.6"}">${NAMES[k]}</span>`).join("");
      $("#tg-bytes", root).textContent = gb(st.w);
      $("#tg-kv", root).textContent = gb(st.kv);
      $("#tg-flops", root).textContent = st.f ? words(st.f) : "0";
      $("#tg-cap", root).innerHTML = `<span class="cap-step">${st.summary ? "Summary" : `Step ${i + 1} of ${STEPS.length - 1}`}</span><span>${st.cap}</span>`;
    };
    ctx.every(170, () => {
      const on = curStep.parts;
      $$(".cores i", diagTG).forEach(c => c.classList.toggle("on", on.includes("cuda") && Math.random() < .55));
      $$(".tcore", diagTG).forEach(c => c.classList.toggle("on", on.includes("tensor") && Math.random() < .8));
    });
    bindPlayer(root, "tg-pl", ctx, { frames: STEPS.length, render: renderTG, interval: 4200 });

    /* ---- component explorer ---- */
    let anim = null;
    const vis = $("#t7-visual", root), info = $("#t7-info", root);
    let twPlayer = null;
    const stopAnim = () => { if (anim) { ctx.stop(anim); anim = null; } if (twPlayer) { twPlayer.stop(); twPlayer = null; } };
    const threadGrid = (groups) => {
      const rows = 4, cols = 16;
      return `<div class="scroll-x"><div style="display:grid;grid-template-columns:repeat(${cols},minmax(18px,1fr));gap:3px;min-width:420px" id="t7-threads">
        ${Array.from({ length: rows * cols }, (_, i) => `<span data-g="${groups ? Math.floor((i % cols) / 8) + rows * 0 + Math.floor(i / cols) * 2 : i}" style="aspect-ratio:1;border-radius:3px;background:var(--surface-3);display:grid;place-items:center;font-size:.55rem;font-family:var(--f-mono);transition:background .2s"></span>`).join("")}
      </div></div>`;
    };
    const views = {
      board: () => { vis.innerHTML = gpuDiagram(); $$("[data-part='board']", vis).forEach(x => x.classList.add("sel")); info.innerHTML = partInfo("board"); },
      sm: () => {
        vis.innerHTML = gpuDiagram(); info.innerHTML = partInfo("sm");
        const sms = $$(".sm", vis); let k = 0;
        anim = ctx.every(350, () => { sms.forEach((s, i) => s.style.boxShadow = i === k % sms.length ? "0 0 0 3px var(--accent)" : "none"); k++; });
      },
      cuda: () => {
        vis.innerHTML = gpuDiagram() + `<div class="tokens">${["RMSNorm", "SiLU / GELU", "softmax", "residual add", "rotary embedding", "sampling"].map(o => `<span class="tok">${o}</span>`).join("")}</div>`;
        info.innerHTML = partInfo("cuda");
        const cores = $$(".cores i", vis);
        anim = ctx.every(160, () => cores.forEach(c => c.classList.toggle("on", Math.random() < .45)));
      },
      tensor: () => {
        const ops = ["X × W<sub>q</sub>", "X × W<sub>k</sub>", "X × W<sub>v</sub>", "attention (Q·Kᵀ, then × V)", "X × W<sub>o</sub>", "MLP gate", "MLP up", "MLP down"];
        vis.innerHTML = gpuDiagram() + `<div class="grid2" style="grid-template-columns:1fr auto;align-items:center">
          <div class="tokens" id="t7-ops">${ops.map(o => `<span class="tok">${o}</span>`).join("")}</div>
          <div class="stat"><span class="v c-compute" id="t7-layer">layer 1 / 80</span><span class="k">one token's forward pass</span></div></div>`;
        info.innerHTML = partInfo("tensor");
        const tc = $$(".tcore", vis), opEls = $$("#t7-ops .tok", vis); let k = 0;
        anim = ctx.every(260, () => {
          const o = k % ops.length; opEls.forEach((e, i) => e.classList.toggle("lit", i === o));
          tc.forEach(t => t.classList.toggle("on", o !== 3 || Math.random() < .5));
          $("#t7-layer", vis).textContent = `layer ${Math.floor(k / ops.length) % 80 + 1} / 80`; k++;
        });
      },
      threads: () => {
        vis.innerHTML = twHTML("t7tw");
        info.innerHTML = `<div class="qa4">
          <div class="card"><span class="mini-label">Thread</span><p>One copy of a GPU program (kernel) working on one piece of the data, such as one output number.</p></div>
          <div class="card"><span class="mini-label">Warp</span><p>A group of threads that an SM runs together, all executing the same instruction at the same moment.</p></div>
          <div class="card"><span class="mini-label">Who manages them</span><p>The SMs create and schedule them; CUDA libraries and the driver decide the grouping. You never manage warps yourself.</p></div>
          <div class="card"><span class="mini-label">Role in LLM inference</span><p>Every matrix multiply of every layer becomes thousands of threads, run warp by warp across many SMs.</p></div></div>`;
        twPlayer = mountTW(root, ctx, "t7tw");
      },
    };
    const pick = (k) => { stopAnim(); $$("#t7-pick .btn", root).forEach(b => b.classList.toggle("on", b.dataset.k === k)); views[k](); };
    $("#t7-pick", root).onclick = e => { const b = e.target.closest("button"); if (b) pick(b.dataset.k); };
    pick("board");
  },
});

/* ---------- Topic 8 ---------- */
topic({
  num: 8, title: "GPU memory hierarchy",
  question: "Why does the GPU have several kinds of memory?",
  render: () => `
    ${say(
      "Memory can be <b>big</b> or <b>fast</b>, but not both. The fastest memory must sit right next to the circuits, where there is very little room, and signals that travel further take longer.",
      "So the GPU arranges memory in <b>levels</b>: <b>registers</b> inside each core, <b>SRAM</b> on the chip beside the cores, an <b>L2 cache</b> between, and <b>HBM</b> beside the chip. Each level further from the cores holds more but takes longer to reach."
    )}
    ${see("How big and how fast is each level? Click a level.", `
      <div class="grid2" style="grid-template-columns:minmax(240px,1fr) minmax(260px,1.2fr);align-items:start">
        <div class="pyramid" id="t8-pyr"></div>
        <div id="t8-detail" class="stage-panel"></div>
      </div>
      <div class="scroll-x"><table class="cmp-table" id="t8-table"></table></div>
      ${note("Sizes and speeds are shown as relative orders of magnitude, not measured specs. Operations do <b>not</b> literally pass through every level in order: data moves from larger, slower memory toward smaller, faster memory near the cores when and where the kernel needs it.")}`)}
    ${see("Which levels does one decode step lean on?", `
      <div class="btn-row"><button type="button" class="btn primary" id="t8-play">▶ Walk through one decode step</button></div>
      <div class="stage-panel" id="t8-walk" style="min-height:70px"><p class="c-muted">Press play to follow a token through one layer.</p></div>`, "Watch it")}
    ${ex(`<div class="calc">70B-class model in FP16: 140 GB of weights → only HBM is big enough<br>
      One 8,192-number token vector: 16 KB → fits easily in on-chip SRAM<br>
      One multiply: two numbers → registers</div>`)}
    ${tech(`<ul>
      <li>Moving closer to the cores: capacity goes <b>down</b>, speed goes <b>up</b>, cost per byte goes <b>up</b>.</li>
      <li>HBM is “slow” only compared with on-chip memory. It is still much faster than CPU RAM reached over PCIe.</li>
      <li>Well-designed kernels keep hot data in fast memory as long as possible and avoid round trips to HBM (for example FlashAttention computes attention in blocks that fit in SRAM).</li>
    </ul>`)}
    ${inf(`<p>The data that dominates inference, the <b class="c-memory">weights</b> and the <b class="c-kv">KV cache</b>, is so large that it can only live in HBM. That makes the <b>HBM → compute</b> path the busiest road in LLM inference.</p>`)}
    ${check({
      q: "A 70B model's FP16 weights (about 140 GB) are needed for every token. Which level is large enough to hold them?",
      opts: ["Registers", "SRAM / shared memory", "L2 cache", "HBM"],
      a: 3, why: "Only HBM has tens to hundreds of GB. The faster levels are orders of magnitude smaller and only ever hold a small working piece.",
    })}`,
  mount(root, ctx) {
    const L = [
      { k: "Registers", size: "Tiny", sizeN: 1, speed: 100, color: "var(--kv)", width: 34, where: "Inside the processing circuits of each core.", stores: "The few values being computed this instant, plus thread bookkeeping.", when: "Every single multiply-add: the current weight element, input element and running sum.", sum: "Tiny + extremely fast" },
      { k: "SRAM / shared memory", size: "Small", sizeN: 2, speed: 85, color: "var(--accent)", width: 56, where: "On the GPU chip, beside the cores of an SM.", stores: "Blocks of data that a group of threads is working on together.", when: "Staging and reusing chunks of weights, activations or attention blocks inside fast kernels.", sum: "Small + very fast" },
      { k: "L2 cache", size: "Larger", sizeN: 3, speed: 55, color: "var(--memory)", width: 78, where: "Between on-chip memory and HBM (simplified level in this course).", stores: "Recently used or soon-needed data travelling between HBM and the cores.", when: "Buffering streams of weight and activation data headed to or from compute.", sum: "Larger + slower than on-chip" },
      { k: "HBM", size: "Huge", sizeN: 5, speed: 28, color: "#28409a", width: 100, where: "Beside the GPU chip, on the same package.", stores: "Model weights, KV cache, activations and other large data.", when: "Always: every forward pass reads weights from here, and every decode step reads (and appends to) the KV cache here.", sum: "Huge + much slower than on-chip" },
    ];
    const pyr = $("#t8-pyr", root);
    pyr.innerHTML = L.map((l, i) => `<div class="pyr-level" data-i="${i}" style="width:${l.width}%;background:${l.color}">${l.k}<small>${l.sum}</small></div>`).join("");
    const bar = (v, c) => `<div style="height:8px;border-radius:4px;background:var(--surface-3);width:120px"><div style="height:100%;width:${v}%;background:${c};border-radius:4px"></div></div>`;
    $("#t8-table", root).innerHTML = `<tr><th>Level</th><th>Relative size</th><th>Relative speed</th><th>What's stored</th></tr>` +
      L.map(l => `<tr><td><b>${l.k}</b></td><td>${bar(l.sizeN * 20, "var(--memory)")}<span class="c-muted" style="font-size:.78rem">${l.size}</span></td><td>${bar(l.speed, "var(--compute)")}</td><td>${l.stores}</td></tr>`).join("");
    const pick = (i) => {
      $$(".pyr-level", pyr).forEach((p, j) => p.classList.toggle("sel", i === j));
      const l = L[i];
      $("#t8-detail", root).innerHTML = `<h4 style="font-size:1.1rem">${l.k}</h4>
        <div class="stat-row"><div class="stat"><span class="v">${l.size}</span><span class="k">relative size</span></div><div class="stat"><span class="v">${["Extremely fast", "Very fast", "Fast", "Slower (still fast)"][i]}</span><span class="k">relative speed</span></div></div>
        <p><b>Where:</b> ${l.where}</p><p><b>Stores:</b> ${l.stores}</p><p><b>Used during inference:</b> ${l.when}</p>`;
    };
    pyr.onclick = e => { const p = e.target.closest(".pyr-level"); if (p) pick(+p.dataset.i); };
    pick(3);
    const walk = [
      [3, "The layer's weight matrices and this request's KV cache are in <b>HBM</b>."],
      [2, "The needed data streams toward the chip, buffered in the <b>L2 cache</b>."],
      [1, "Blocks of weights and the token's vector are staged in <b>SRAM / shared memory</b> for the threads."],
      [0, "Individual numbers sit in <b>registers</b> while Tensor Cores multiply and add."],
      [3, "New K and V for this token are written back into the KV cache in <b>HBM</b>. Then the next layer starts, and the trip repeats ×80 layers."],
    ];
    let t = null;
    $("#t8-play", root).onclick = () => {
      if (t) ctx.stop(t);
      let k = 0;
      const stepW = () => { if (k >= walk.length) { ctx.stop(t); t = null; return; } pick(walk[k][0]); $("#t8-walk", root).innerHTML = `<div class="mini-label">Step ${k + 1} of ${walk.length}</div><p>${walk[k][1]}</p>`; k++; };
      stepW(); t = ctx.every(2200, stepW);
    };
  },
});

/* ---------- Topic 9 ---------- */
topic({
  num: 9, title: "Memory capacity vs memory bandwidth",
  question: "Is it about how much memory I have, or how fast it is?",
  render: () => `
    ${say(
      "Two different questions: <b class='c-memory'>Capacity</b>: “How much data can I store?” <b class='c-memory'>Bandwidth</b>: “How quickly can I move data out?”",
      "Think of a warehouse. Its <b>size</b> is capacity. The <b>number of trucks leaving per second</b> is bandwidth. A giant warehouse with one small road is still slow to empty."
    )}
    ${see("What limits a warehouse: its size or its road?", `
      <svg class="svg-block" viewBox="0 0 720 210" id="t9-svg" role="img" aria-label="Warehouse with road to GPU chip"></svg>
      <div class="controls">
        ${slider("t9-cap", "Capacity: warehouse size (GB)", 24, 192, 8, 80)}
        ${slider("t9-bw", "Bandwidth: road speed (GB/s)", 200, 2000, 100, 1000)}
        <div class="ctl"><label>Model in the warehouse</label>${seg("t9-model", [["16", "8B FP16 (16 GB)"], ["140", "70B FP16 (140 GB)"]], "16")}</div>
      </div>
      <div class="stat-row">
        <div class="stat"><span class="v" id="t9-fit">–</span><span class="k">does it fit? (capacity)</span></div>
        <div class="stat"><span class="v c-memory" id="t9-time">–</span><span class="k">time to read all weights once (bandwidth)</span></div>
        <div class="stat"><span class="v" id="t9-tps">–</span><span class="k">max tokens/sec at batch 1 (upper bound)</span></div>
      </div>
      <p class="c-muted" style="font-size:.82rem">${illus()} Upper bound ignores the KV cache and compute time: it only asks how often the full weights could be delivered.</p>`)}
    ${ex(`<div class="calc">Two GPUs, both 141 GB capacity ${illus()}<br>
      GPU A bandwidth 1,000 GB/s → 70B FP16 (140 GB): about <b>7 tokens/sec</b> max at batch 1<br>
      GPU B bandwidth 1,400 GB/s → about <b>10 tokens/sec</b><br>
      Same capacity, same model, <b>different decode speed</b>.</div>`)}
    ${tech(`<ul>
      <li><b>Memory capacity:</b> total bytes the GPU memory can hold (GB).</li>
      <li><b>Memory bandwidth:</b> how many bytes per second can move between memory and the compute chip (GB/s). It depends on the memory type, how many parallel connections there are, and the signalling speed.</li>
    </ul>`)}
    ${fx(["Decode tokens/sec (batch 1) ≲ memory bandwidth ÷ bytes read per token", "bytes read per token ≈ weight bytes + KV cache bytes"], "≲ means “at most about”. Real speed is lower because of compute time and overheads.")}
    ${inf(`<p>Weights live in HBM. Each decode step needs the weights of <b>every layer</b> for just one new token. So at small batch sizes, how fast HBM can deliver data, not how much it can hold, often sets the speed limit.</p>`)}
    ${check({
      q: "Two GPUs have the same memory capacity, but GPU B has 40% higher memory bandwidth. For batch-1 decode of a large dense model, which is likely faster?",
      opts: ["GPU A: capacity is what matters", "GPU B: it can deliver weights faster", "They're the same: capacity is equal", "Can't be faster: bandwidth only matters for training"],
      a: 1, why: "Capacity decides what fits. Bandwidth decides how quickly the weights reach the compute. Batch-1 decode of a large dense model is usually limited by bandwidth.",
    })}`,
  mount(root, ctx) {
    const svg = $("#t9-svg", root);
    let v = { cap: 80, bw: 1000, model: 16 }, trucks = [], spawn = 0;
    const draw = () => {
      const wW = 60 + v.cap * 1.0, fillFrac = Math.min(1, v.model / v.cap);
      const lanes = Math.max(1, Math.round(v.bw / 400));
      let s = `<rect x="10" y="${100 - wW / 3}" width="${wW}" height="${wW * 2 / 3}" rx="6" fill="var(--memory-soft)" stroke="var(--memory)" stroke-width="2"/>
        <rect x="10" y="${100 - wW / 3 + wW * 2 / 3 * (1 - fillFrac)}" width="${wW}" height="${wW * 2 / 3 * fillFrac}" rx="4" fill="${v.model > v.cap ? "var(--bad)" : "var(--memory)"}" opacity=".55"/>
        <text x="${10 + wW / 2}" y="${100 - wW / 3 - 8}" text-anchor="middle" font-size="13" font-weight="600">HBM warehouse · ${v.cap} GB</text>
        <text x="${10 + wW / 2}" y="104" text-anchor="middle" font-size="12" font-weight="700" style="fill:#fff">${v.model > v.cap ? "OVERFLOW" : "weights " + v.model + " GB"}</text>`;
      const rx = 20 + wW, rw = 560 - rx, laneH = 14;
      const top = 100 - lanes * laneH / 2;
      for (let i = 0; i < lanes; i++) s += `<rect x="${rx}" y="${top + i * laneH}" width="${rw}" height="${laneH - 3}" fill="var(--surface-3)"/>`;
      s += `<text x="${rx + rw / 2}" y="${top - 8}" text-anchor="middle" font-size="12" class="muted">road: ${v.bw} GB/s (${lanes} lane${lanes > 1 ? "s" : ""})</text>`;
      s += `<g id="t9-trucks"></g>
        <rect x="570" y="50" width="140" height="100" rx="8" fill="var(--compute-soft)" stroke="var(--compute)" stroke-width="2"/>
        <text x="640" y="95" text-anchor="middle" font-size="13" font-weight="700">GPU chip</text>
        <text x="640" y="113" text-anchor="middle" font-size="11" class="muted">(compute)</text>`;
      svg.innerHTML = s;
      svg._geo = { rx, rw, top, laneH, lanes };
    };
    const stopLoop = ctx.loop((dt) => {
      const g = svg._geo; if (!g) return;
      spawn += dt * v.bw / 250;
      while (spawn > 1) { spawn -= 1; trucks.push({ x: g.rx, lane: Math.floor(Math.random() * g.lanes) }); }
      trucks.forEach(t => t.x += dt * 160);
      trucks = trucks.filter(t => t.x < g.rx + g.rw - 16).slice(-80);
      const tg = $("#t9-trucks", svg);
      if (tg) tg.innerHTML = trucks.map(t => `<rect x="${t.x}" y="${g.top + t.lane * g.laneH + 1}" width="14" height="${g.laneH - 5}" rx="2" fill="var(--memory)"/>`).join("");
    });
    const update = () => {
      const fits = v.model <= v.cap;
      $("#t9-fit", root).innerHTML = fits ? `<span class="c-ok">Yes</span>` : `<span class="c-bad">No</span>`;
      $("#t9-time", root).textContent = `${fmt(v.model / v.bw, 2)} s`;
      $("#t9-tps", root).innerHTML = fits ? `${fmt(v.bw / v.model, 1)} tok/s` : `<span class="c-bad">n/a: doesn't fit</span>`;
      draw();
    };
    bindSliders(root, { "t9-cap": x => x, "t9-bw": x => x.toLocaleString("en-US") }, s => { v.cap = s["t9-cap"]; v.bw = s["t9-bw"]; update(); });
    bindSeg(root, "t9-model", m => { v.model = +m; update(); });
  },
});
