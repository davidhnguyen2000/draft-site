// Curriculum explorer: one row per robot, a task switcher, a curriculum slider,
// the rung's video, and a torque–speed plot of the whole run beside it.
//
// The ladders (rungs, labels, pass counts) come from data/curriculum.json, which
// scripts/site/build_curriculum.py exports from the r10 recordings. Each rung may
// also have a video and a joint trace (scripts/site/build_explorer.py); where one
// is missing the row says so rather than showing anything in its place.

(() => {
  const ROBOTS = {
    cheetah: { name: "Cheetah", color: "#4c72b0", facts: "33 kg · 0.48 m leg · QDD, 72 N·m, 6:1" },
    bear: { name: "Bear", color: "#dd8452", facts: "110 kg · 0.74 m leg · MidGear, 450 N·m, 20:1" },
    giraffe: { name: "Giraffe", color: "#55a868", facts: "91 kg · 1.30 m leg · High GR, 332 N·m, 80:1" },
  };
  // Seaborn "deep", as the paper's figures: grey, purple and red for the three joints.
  const JOINT_COLOR = { roll: "#8c8c8c", pitch: "#8172b3", knee: "#c44e52" };

  let data = null;
  const rows = [];
  const sync = document.getElementById("sync");

  // ------------------------------------------------------------ joint traces

  // A trace is {dt, tau: [[12]...], omega: [[12]...]}, each value signed and
  // divided by the joint's peak torque or no-load speed.
  const traceCache = new Map();

  function loadTrace(rung) {
    if (!rung.joints_ready) return Promise.resolve(null);
    if (!traceCache.has(rung.joints)) {
      traceCache.set(rung.joints, fetch(rung.joints, { cache: "no-cache" }).then((r) => (r.ok ? r.json() : null)).catch(() => null));
    }
    return traceCache.get(rung.joints);
  }

  // ------------------------------------------------------------ torque–speed plot

  function pct(values, q) {
    const v = [...values].sort((a, b) => a - b);
    return v[Math.min(v.length - 1, Math.floor(q * (v.length - 1)))];
  }

  // The whole run at once: every sample of every joint as a faint dot, and each
  // joint's 95th-percentile torque and speed (the paper's utilisation statistic)
  // as a ringed marker. `message` stands in the plot when there is no trace.
  function drawPlot(row, trace, message) {
    const canvas = row.canvas, ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const pad = { l: 44, r: 10, t: 10, b: 36 }, max = 1.2;
    const X = (v) => pad.l + (Math.min(v, max) / max) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - (Math.min(v, max) / max) * (h - pad.t - pad.b);

    ctx.fillStyle = "rgba(196, 78, 82, 0.07)";
    ctx.beginPath();
    ctx.moveTo(X(0), Y(1)); ctx.lineTo(X(1), Y(0)); ctx.lineTo(X(max), Y(0));
    ctx.lineTo(X(max), Y(max)); ctx.lineTo(X(0), Y(max)); ctx.closePath(); ctx.fill();
    ctx.strokeStyle = "#e4e6eb"; ctx.lineWidth = 1;
    ctx.font = "11px 'Noto Sans', sans-serif"; ctx.fillStyle = "#7a8190";
    for (let v = 0; v <= max + 1e-9; v += 0.25) {
      ctx.beginPath(); ctx.moveTo(X(v), Y(0)); ctx.lineTo(X(v), Y(max)); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(X(0), Y(v)); ctx.lineTo(X(max), Y(v)); ctx.stroke();
      if (Math.abs(v * 100 % 50) < 1e-6) {
        ctx.textAlign = "center"; ctx.fillText(`${v * 100}%`, X(v), h - pad.b + 14);
        ctx.textAlign = "right"; ctx.fillText(`${v * 100}%`, pad.l - 6, Y(v) + 4);
      }
    }
    ctx.textAlign = "center";
    ctx.fillText("speed, % of no-load", X(max / 2), h - 6);
    ctx.save(); ctx.translate(12, Y(max / 2)); ctx.rotate(-Math.PI / 2);
    ctx.fillText("torque, % of peak", 0, 0); ctx.restore();
    ctx.strokeStyle = "#1f2328"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(X(0), Y(1)); ctx.lineTo(X(1), Y(0)); ctx.stroke();
    ctx.fillStyle = "#4b5260"; ctx.textAlign = "left";
    ctx.fillText("motor limit", X(0.56) + 6, Y(0.44) - 4);

    if (!trace) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.fillRect(pad.l + 1, pad.t + 1, w - pad.l - pad.r - 2, h - pad.t - pad.b - 2);
      ctx.fillStyle = "#4b5260"; ctx.textAlign = "center"; ctx.font = "12px 'Noto Sans', sans-serif";
      const lines = message.split("\n");
      lines.forEach((ln, i) => ctx.fillText(ln, (pad.l + w - pad.r) / 2, (pad.t + h - pad.b) / 2 + (i - (lines.length - 1) / 2) * 17));
      row.readout.textContent = "";
      return;
    }

    const names = data.joint_names;
    const typeOf = (n) => (n.includes("roll") ? "roll" : n.includes("pitch") ? "pitch" : "knee");
    ctx.globalAlpha = 0.16;
    for (let j = 0; j < names.length; j++) {
      ctx.fillStyle = JOINT_COLOR[typeOf(names[j])];
      for (let i = 0; i < trace.tau.length; i++) {
        ctx.beginPath();
        ctx.arc(X(Math.abs(trace.omega[i][j])), Y(Math.abs(trace.tau[i][j])), 1.6, 0, 2 * Math.PI);
        ctx.fill();
      }
    }
    ctx.globalAlpha = 1;
    let worst = null;
    for (let j = 0; j < names.length; j++) {
      const tq = trace.tau.map((r) => Math.abs(r[j])), om = trace.omega.map((r) => Math.abs(r[j]));
      const p = { j, tq: pct(tq, 0.95), om: pct(om, 0.95), sum: pct(tq.map((t, i) => t + om[i]), 0.95) };
      ctx.beginPath(); ctx.arc(X(p.om), Y(p.tq), 4.5, 0, 2 * Math.PI);
      ctx.fillStyle = JOINT_COLOR[typeOf(names[j])]; ctx.fill();
      ctx.lineWidth = 1.5; ctx.strokeStyle = "#fff"; ctx.stroke();
      if (!worst || p.sum > worst.sum) worst = p;
    }
    const pretty = names[worst.j].replace(/_/g, " ").replace(/^(\w\w)/, (m) => m.toUpperCase());
    row.readout.textContent = `Nearest the limit (95th percentile over the run): ${pretty} · `
      + `τ ${Math.round(worst.tq * 100)}% · ω ${Math.round(worst.om * 100)}% · τ+ω ${worst.sum.toFixed(2)}`;
  }

  // ------------------------------------------------------------ rows

  function build(robotKey) {
    const node = document.getElementById("row-template").content.firstElementChild.cloneNode(true);
    const robot = ROBOTS[robotKey];
    node.style.setProperty("--robot", robot.color);
    node.querySelector(".xrow-name").textContent = robot.name;
    node.querySelector(".xrow-facts").textContent = robot.facts;

    const row = {
      key: robotKey, node, task: 0, level: 0, trace: null,
      video: node.querySelector("video"), ph: node.querySelector(".ph-live"),
      canvas: node.querySelector(".ts-plot"), readout: node.querySelector(".plot-readout"),
      slider: node.querySelector(".level"), rungsEl: node.querySelector(".rungs"),
    };

    node.querySelectorAll(".arrow").forEach((b) =>
      b.addEventListener("click", () => setTask(row, row.task + Number(b.dataset.step), true)));
    row.slider.addEventListener("input", () => setLevel(row, Number(row.slider.value), true));
    node.addEventListener("keydown", (e) => {
      if (e.target === row.slider) return;
      if (e.key === "ArrowLeft") setTask(row, row.task - 1, true);
      if (e.key === "ArrowRight") setTask(row, row.task + 1, true);
    });
    node.tabIndex = -1;
    document.getElementById("explorer-rows").appendChild(node);
    rows.push(row);
    return row;
  }

  const ladderOf = (row) => data.robots[row.key][data.tasks[row.task].key];

  function renderLadder(row) {
    const task = data.tasks[row.task], ladder = ladderOf(row);
    row.node.querySelector(".task-name b").textContent = task.name;
    row.node.querySelector(".task-count").textContent = `${row.task + 1} / ${data.tasks.length}`;
    row.slider.max = ladder.rungs.length - 1;
    row.rungsEl.innerHTML = "";
    row.rungsEl.style.setProperty("--n", ladder.rungs.length);
    ladder.rungs.forEach((r, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "rung";
      const rate = r.reached ? r.passed / r.n : null;
      b.dataset.state = !r.reached ? "none" : rate >= 0.75 ? "pass" : rate >= 0.25 ? "partial" : "fail";
      b.innerHTML = `<span>${r.chip}</span><i style="--p:${rate ?? 0}"></i>`;
      b.title = r.reached ? `${r.label}: ${r.passed}/${r.n} passed` : "not attempted: the ladder stops after two failed rungs";
      b.addEventListener("click", () => setLevel(row, i, true));
      row.rungsEl.appendChild(b);
    });
    row.node.querySelector(".rung-axis")?.remove();
    const axis = document.createElement("span");
    axis.className = "rung-axis";
    axis.textContent = ladder.axis;
    row.rungsEl.after(axis);
  }

  function setTask(row, task, fromUser) {
    const n = data.tasks.length;
    row.task = ((task % n) + n) % n;
    renderLadder(row);
    setLevel(row, Math.min(row.level, ladderOf(row).rungs.length - 1), false);
    if (fromUser && sync.checked) rows.forEach((r) => r !== row && (r.task !== row.task) && setTask(r, row.task, false));
  }

  async function setLevel(row, level, fromUser) {
    const ladder = ladderOf(row), rung = ladder.rungs[level], task = data.tasks[row.task];
    row.level = level;
    row.slider.value = level;
    window.paintRange(row.slider);
    row.rungsEl.querySelectorAll(".rung").forEach((b, i) => b.classList.toggle("is-on", i === level));

    const robot = ROBOTS[row.key].name;
    const caption = rung.reached
      ? `Rung ${level + 1}: ${rung.label}. <b>${rung.passed}/${rung.n}</b> replicas passed.`
      : `Rung ${level + 1} (${rung.chip} ${ladder.axis.replace(/\s*\(.*\)/, "")}): not attempted. The ladder stops after two failed rungs.`;
    row.node.querySelector(".rung-caption").innerHTML = caption;

    // Video, or its placeholder.
    const ready = rung.video_ready;
    row.video.hidden = !ready;
    row.ph.hidden = ready;
    if (ready) {
      if (row.video.getAttribute("src") !== rung.video) { row.video.src = rung.video; row.video.play().catch(() => {}); }
    } else {
      row.video.removeAttribute("src");
      row.ph.querySelector(".ph-title").textContent = `${robot} · ${task.name} · rung ${level + 1}`;
      row.ph.querySelector(".ph-desc").textContent = rung.reached
        ? `${rung.label}. Goal: ${task.goal}.`
        : "Not attempted in the recording: the ladder stops after two failed rungs.";
      row.ph.querySelector(".ph-path").textContent = rung.video;
    }

    const trace = await loadTrace(rung);
    if (row.level !== level) return;          // moved on while the trace loaded
    row.trace = trace;
    row.message = !rung.reached ? "Not attempted: the ladder stops\nafter two failed rungs."
      : !rung.video_ready ? "This rung has not been rendered yet."
      : "Torque and speed were not logged\nin this recording.";
    drawPlot(row, row.trace, row.message);

    if (fromUser && sync.checked) rows.forEach((r) => r !== row && r.level !== level && setLevel(r, level, false));
  }

  // Redraw the static plots when their canvases change size.
  new ResizeObserver(() => rows.forEach((r) => r.message !== undefined && drawPlot(r, r.trace, r.message)))
    .observe(document.getElementById("explorer-rows"));

  fetch("data/curriculum.json", { cache: "no-cache" })
    .then((r) => r.json())
    .then((json) => {
      data = json;
      document.getElementById("explorer-note").textContent =
        `Pass rates from 16 replicas per rung, ${json.round} policies, seed ${json.seed}.`;
      for (const key of Object.keys(ROBOTS)) {
        const row = build(key);
        setTask(row, 0, false);
        setLevel(row, 3, false);
      }
    })
    .catch(() => {
      document.getElementById("explorer-rows").textContent = "Could not load data/curriculum.json.";
    });
})();
