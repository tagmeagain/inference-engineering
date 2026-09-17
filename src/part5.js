/* ============================================================
   PART 5 — What makes newer GPUs better for inference? (24–25)
   ============================================================ */
part({
  num: 5, title: "GPU Types & Generations",
  blurb: "Newer GPUs improve on five separate dimensions. Which ones matter depends on whether your workload is compute-bound or memory-bound.",
  story: "Newer GPUs improve <b>compute + HBM capacity + bandwidth + precision support + interconnect</b>. Each one fixes a different bottleneck.",
});

const DIMENSIONS = [
  { k: "compute", name: "More compute", c: "var(--compute)", what: "Faster and more capable Tensor Cores and cores overall.", helps: ["Faster prefill (lower TTFT for long prompts)", "Higher throughput at large batch sizes", "Compute-bound workloads"], },
  { k: "cap", name: "More HBM capacity", c: "var(--memory)", what: "More GB of memory on each GPU.", helps: ["Larger models on fewer GPUs", "Longer contexts (bigger KV cache)", "More concurrent users per GPU"], },
  { k: "bw", name: "Higher HBM bandwidth", c: "var(--memory)", what: "More GB per second between memory and compute.", helps: ["Faster weight and KV cache movement", "Faster low-batch decode (memory-bound)", "Better tokens/sec per user"], },
  { k: "prec", name: "Better low-precision support", c: "var(--kv)", what: "Hardware that computes natively in FP16 → BF16 → FP8 → FP4, plus fast INT8/INT4 kernels.", helps: ["Fewer bytes per weight → less memory and data movement", "Faster matrix maths in low precision", "Bigger models on the same hardware"], },
  { k: "link", name: "Better GPU-to-GPU interconnect", c: "var(--accent)", what: "PCIe → NVLink → NVSwitch: faster, wider links between GPUs.", helps: ["Efficient tensor parallelism across GPUs", "Serving models too big for one GPU", "Less time lost communicating each layer"], },
];

/* ---------- Topic 24 ---------- */
topic({
  num: 24, title: "Why are newer GPUs better?",
  question: "Which GPU improvement actually helps my workload?",
  render: () => `
    ${say(
      "“Newer is faster” hides the real story. A GPU generation can improve <b>five different things</b>, and each one fixes a <b>different bottleneck</b>.",
      "If your workload is memory-bound, more compute barely helps. If it's compute-bound, more bandwidth barely helps. So first ask: <b>what is my workload waiting on?</b>"
    )}
    ${see("Which dimension fixes which bottleneck? Pick a workload.", `
      <div class="btn-row" id="t24-work">
        ${[["chat", "Chat with a large model, few users"], ["rag", "Very long prompts (RAG, documents)"], ["many", "Many users, long conversations"], ["huge", "Model too big for one GPU"]].map(([k, t], i) => `<button type="button" class="btn ${i === 0 ? "on" : ""}" data-k="${k}">${t}</button>`).join("")}
      </div>
      <div class="stage-panel" id="t24-why" style="min-height:60px"></div>
      <div class="grid3" id="t24-dims"></div>`)}
    ${ex(`<div class="calc">Two GPUs with the <b>same compute</b>, but GPU B has more and faster HBM ${illus("Conceptual")}<br>
      Long-prompt prefill → roughly the same TTFT on both (compute-bound)<br>
      Batch-1 decode of a 70B model → GPU B streams tokens faster (bandwidth-bound)<br>
      Many users with long contexts → GPU B fits more KV cache (capacity-bound)</div>`)}
    ${tech(`<ul>
      <li><b>Compute:</b> peak FLOPS, from more and better cores and Tensor Cores.</li>
      <li><b>Capacity:</b> GB of HBM (or GDDR on smaller cards).</li>
      <li><b>Bandwidth:</b> GB/s between memory and the GPU chip.</li>
      <li><b>Precision support:</b> native number formats (FP16 → BF16/TF32 → FP8 → FP4) and fast low-bit kernels.</li>
      <li><b>Interconnect:</b> PCIe (general-purpose, slower) → NVLink (direct GPU links) → NVSwitch (every GPU in a group linked at full speed).</li>
    </ul>`)}
    ${inf(`<p>Match the upgrade to the bottleneck: prefill-heavy → compute and precision; low-batch decode → bandwidth and precision; many long conversations → capacity; giant models → capacity and interconnect.</p>`)}
    ${check({
      q: "Your service runs a big dense model at batch 1 and the output streams slowly. Compute utilization is very low. Which upgrade is most likely to help?",
      opts: ["More peak compute", "Higher HBM bandwidth (or lower-precision weights)", "More CPU cores", "More disk"],
      a: 1, why: "Very low compute utilization at batch 1 points to a memory-bandwidth bottleneck. Faster HBM, or fewer bytes per weight, helps directly.",
    })}`,
  mount(root, ctx) {
    const W = {
      chat: { on: ["bw", "prec"], why: "Few users means small batches, so decode is memory-bandwidth-bound. Faster HBM and fewer bytes per weight help most." },
      rag: { on: ["compute", "prec"], why: "Long prompts make prefill dominate. Prefill is compute-bound, so more compute and fast low-precision maths cut TTFT." },
      many: { on: ["cap", "bw", "compute"], why: "Many long conversations means a huge KV cache (capacity), large batches (compute rises), and lots of KV reads per step (bandwidth)." },
      huge: { on: ["cap", "link", "prec"], why: "If weights don't fit, you need more GB per GPU, lower precision, or several GPUs connected by a fast interconnect." },
    };
    const draw = (k) => {
      $$("#t24-work .btn", root).forEach(b => b.classList.toggle("on", b.dataset.k === k));
      $("#t24-why", root).innerHTML = `<p>${W[k].why}</p>`;
      $("#t24-dims", root).innerHTML = DIMENSIONS.map(d => {
        const on = W[k].on.includes(d.k);
        return `<div class="card" style="border-width:2px;border-color:${on ? d.c : "var(--line)"};opacity:${on ? 1 : .5};transition:all .3s">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:6px"><h4>${d.name}</h4>${on ? `<span class="pill ok">helps here</span>` : ""}</div>
          <p class="c-muted">${d.what}</p><ul style="padding-left:1.1em;margin:0">${d.helps.map(h => `<li>${h}</li>`).join("")}</ul></div>`;
      }).join("");
    };
    $("#t24-work", root).onclick = e => { const b = e.target.closest("button"); if (b) draw(b.dataset.k); };
    draw("chat");
  },
});

/* ---------- Topic 25 ---------- */
const GENS = [
  { k: "T4", arch: "Turing", mem: "16 GB GDDR6", lv: [1, 1, 1, 2, 1], feats: ["Tensor Cores", "FP16 / INT8", "PCIe only", "Low power"], enabled: "Cheap, power-efficient inference for small models and classic deep learning. Still a sensible choice when the model is small." },
  { k: "A10", arch: "Ampere", mem: "24 GB GDDR6", lv: [2, 2, 2, 3, 1], feats: ["More compute", "BF16 / TF32", "PCIe only"], enabled: "A 7B-class model in 16-bit fits on a single card, for affordable single-GPU serving." },
  { k: "A100", arch: "Ampere", mem: "40 / 80 GB HBM", lv: [3, 3, 3, 3, 3], feats: ["HBM", "BF16 / TF32", "NVLink + NVSwitch", "MIG partitioning"], enabled: "Large models served across 8-GPU NVLink nodes. Made tensor parallelism practical at scale." },
  { k: "H100", arch: "Hopper", mem: "80 GB HBM", lv: [4, 3, 3, 4, 4], feats: ["Much more compute", "FP8 (Transformer Engine)", "Faster NVLink"], enabled: "FP8 inference and much faster prefill and large-batch serving." },
  { k: "H200", arch: "Hopper", mem: "141 GB HBM", lv: [4, 4, 4, 4, 4], feats: ["Same compute class as H100", "More and faster HBM"], enabled: "A clean example of memory mattering: similar compute to H100, but bigger KV caches, bigger models per GPU and faster memory-bound decode." },
  { k: "B200", arch: "Blackwell", mem: "192 GB HBM", lv: [5, 5, 5, 5, 5], feats: ["More compute", "FP4 support", "Newer NVLink, larger GPU domains"], enabled: "Very large models and long contexts on fewer GPUs, FP4 serving, and big NVLink-connected groups for huge MoE models." },
];
topic({
  num: 25, title: "What are newer GPUs enabling?",
  question: "What does each generation make possible that the last one didn't?",
  render: () => `
    ${say("Each generation pushes several dimensions forward at once. What matters isn't the spec sheet; it's what those improvements <b>unlock</b>: bigger models, longer contexts, more users, lower latency.")}
    ${see("How did GPU generations change what inference can do? Click a GPU.", `
      <div class="arch-seq" id="t25-gens"></div>
      <div class="stage-panel" id="t25-detail"></div>
      <p class="c-muted" style="font-size:.82rem">${illus("Relative, qualitative levels")} Dots compare generations loosely; they are not measured specs. Memory sizes are the common configurations.</p>`)}
    ${see("How do the improvements add up?", `
      <div class="grid2" style="align-items:center">
        ${flow([{ t: "More compute", c: "compute" }, { t: "+ More memory", c: "memory" }, { t: "+ More bandwidth", c: "memory" }, { t: "+ Better precision support", c: "kv" }, { t: "+ Better interconnect" }], { vertical: true, id: "t25-in" })}
        ${flow([{ t: "Larger models" }, { t: "+ Longer context" }, { t: "+ Higher throughput" }, { t: "+ Lower latency" }, { t: "+ More concurrent users" }, { t: "+ More efficient multi-GPU serving" }], { vertical: true, id: "t25-out" })}
      </div>`, "The evolution")}
    ${ex(`<div class="calc">70B model in BF16 (140 GB of weights):<br>
      T4 / A10 → needs many cards over PCIe (impractical)<br>
      A100 / H100 80 GB → at least 2 GPUs with NVLink<br>
      H200 141 GB → weights barely fit, leaving no room for KV cache → in practice 2 GPUs<br>
      B200 192 GB → fits on one GPU, with room for KV cache<br>
      In FP8 (70 GB) → fits on a single 80 GB GPU (tight)</div>`)}
    ${tech(`<ul>
      <li>T4 and A10 use GDDR6 memory on the card rather than HBM on the chip package: less capacity and bandwidth, but cheaper and lower power.</li>
      <li>FP8 matrix maths needs Ada/Hopper-class hardware or newer; FP4 needs Blackwell-class.</li>
      <li>NVLink and NVSwitch matter once a model is split across GPUs (Part 6).</li>
    </ul>`)}
    ${inf(`<p>${big("Newer is <b>not</b> automatically better for every workload. A small model on an older, cheaper GPU may give the best cost per token. Pick hardware by the bottleneck and the price, not by the release date.", "Accuracy")}</p>`)}
    ${check({
      q: "H200 has a similar compute class to H100 but more and faster HBM. Which workload benefits most?",
      opts: ["Short-prompt prefill of a small model", "Low-batch decode of a large model with long context", "CPU tokenization", "Nothing: compute is the same"],
      a: 1, why: "More capacity holds bigger KV caches and models; more bandwidth speeds memory-bound decode. Compute-bound prefill gains little.",
    })}`,
  mount(root, ctx) {
    const lvl = (n, cls) => `<span class="lvl ${cls}">${[1, 2, 3, 4, 5].map(i => `<i class="${i <= n ? "on" : ""}"></i>`).join("")}</span>`;
    const names = [["Compute", "c"], ["Capacity", "m"], ["Bandwidth", "m"], ["Precision", "k"], ["Interconnect", "a"]];
    $("#t25-gens", root).innerHTML = GENS.map((g, i) => `<button type="button" class="gen-card" data-i="${i}"><span class="mini-label">${g.arch}</span><span class="gname">${g.k}</span>
      <span style="display:grid;gap:3px;font-size:.72rem">${names.map(([n, c], j) => `<span style="display:flex;justify-content:space-between;gap:6px;align-items:center"><span class="c-muted">${n}</span>${lvl(g.lv[j], c)}</span>`).join("")}</span></button>`).join("");
    const pick = (i) => {
      $$("#t25-gens .gen-card", root).forEach((c, j) => c.classList.toggle("sel", i === j));
      const g = GENS[i], prev = GENS[i - 1];
      const up = prev ? names.map(([n], j) => g.lv[j] > prev.lv[j] ? n : null).filter(Boolean) : [];
      $("#t25-detail", root).innerHTML = `<div style="display:flex;gap:10px;align-items:baseline;flex-wrap:wrap"><h4 style="font-size:1.2rem">${g.k}</h4><span class="c-muted">${g.arch} · ${g.mem}</span></div>
        <div class="tokens">${g.feats.map(f => `<span class="tok">${f}</span>`).join("")}</div>
        <p><b>What it enabled:</b> ${g.enabled}</p>
        ${prev ? `<p class="c-muted" style="font-size:.88rem">Improved over ${prev.k} in: ${up.length ? up.join(", ") : "cost and efficiency more than raw capability"}</p>` : ""}`;
    };
    $("#t25-gens", root).onclick = e => { const c = e.target.closest(".gen-card"); if (c) pick(+c.dataset.i); };
    pick(4);
    let k = 0;
    ctx.every(800, () => { setFlow($("#t25-in", root), k % 6 < 5 ? k % 6 : 4); setFlow($("#t25-out", root), Math.max(-1, k % 6 - 0)); k++; });
  },
});
