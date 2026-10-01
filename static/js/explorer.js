// Curriculum explorer: one row per robot, a task switcher, a curriculum slider,
// the rung's video, and beside it how hard each joint type worked over the run:
// torque and speed as shares of the motor's limits. (draft-site-fixes)
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

  // ------------------------------------------------------------ utilization bars

  function pct(values, q) {
    const v = [...values].sort((a, b) => a - b);
    return v[Math.min(v.length - 1, Math.floor(q * (v.length - 1)))];
  }

  // Torque and speed, each as a share of its motor's limit, in the robot's own
  // render colours: torque in its motor colour, speed in its torso colour
  // (motor_color and torso_color in experiments/quadruped_variants/<robot>.yaml).
  const METRIC = [
    { key: "tau", name: "torque", unit: "of peak torque" },
    { key: "omega", name: "speed", unit: "of no-load speed" },
  ];
  const SHADES = {
    cheetah: { tau: "#283b5c", omega: "#4c72b0" },
    bear: { tau: "#73452b", omega: "#dd8452" },
    giraffe: { tau: "#2c5736", omega: "#55a868" },
  };
  const TYPES = [["hip_roll", "hip roll"], ["hip_pitch", "hip pitch"], ["knee", "knee"]];
  const LEG = { fl: "front left", fr: "front right", rl: "rear left", rr: "rear right" };

  // Per joint type and quantity: the 95th percentile of |value| over the run, with
  // the four legs' samples pooled (the paper's utilisation statistic), and each
  // leg's own 95th percentile for the tooltip.
  function utilization(trace) {
    const names = data.joint_names;
    return TYPES.map(([type, label]) => {
      const cols = names.map((n, j) => [n, j]).filter(([n]) => n.endsWith(type));
      const bars = METRIC.map((m) => {
        const legs = cols.map(([n, j]) => ({ leg: LEG[n.slice(0, 2)] || n, v: pct(trace[m.key].map((r) => Math.abs(r[j])), 0.95) }));
        const pooled = pct(cols.flatMap(([, j]) => trace[m.key].map((r) => Math.abs(r[j]))), 0.95);
        return { metric: m, v: pooled, legs };
      });
      return { label, bars };
    });
  }

  // `message` stands in the plot when there is no trace.
  function drawPlot(row, trace, message) {
    const canvas = row.canvas, ctx = canvas.getContext("2d");
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    row.hits = [];
    row.tip.hidden = true;

    const pad = { l: 44, r: 8, t: 18, b: 30 }, max = 1.1;
    const Y = (v) => h - pad.b - (Math.min(v, max) / max) * (h - pad.t - pad.b);
    const font = (px, weight = 400) => `${weight} ${px}px 'Noto Sans', sans-serif`;

    // Recessive grid, with the motor limit (named in the legend) as the one
    // emphasised line.
    ctx.font = font(11); ctx.fillStyle = "#7a8190"; ctx.textAlign = "right";
    for (let v = 0; v <= 1 + 1e-9; v += 0.25) {
      ctx.strokeStyle = "#eef0f3"; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, Y(v)); ctx.lineTo(w - pad.r, Y(v)); ctx.stroke();
      ctx.fillText(`${Math.round(v * 100)}%`, pad.l - 6, Y(v) + 4);
    }
    ctx.strokeStyle = "#1f2328"; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(pad.l, Y(1)); ctx.lineTo(w - pad.r, Y(1)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.save(); ctx.translate(12, Y(0.5)); ctx.rotate(-Math.PI / 2); ctx.textAlign = "center";
    ctx.fillStyle = "#7a8190"; ctx.fillText("share of limit, 95th pct.", 0, 0); ctx.restore();

    const groupW = (w - pad.l - pad.r) / TYPES.length;
    const gap = 2, barW = Math.min(46, (groupW * 0.62 - gap) / 2);
    ctx.textAlign = "center"; ctx.font = font(12); ctx.fillStyle = "#4b5260";
    TYPES.forEach(([, label], g) => ctx.fillText(label, pad.l + groupW * (g + 0.5), h - pad.b + 18));

    if (!trace) {
      ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
      ctx.fillRect(pad.l + 1, pad.t, w - pad.l - pad.r - 2, h - pad.t - pad.b - 1);
      ctx.fillStyle = "#4b5260"; ctx.font = font(12);
      const lines = message.split("\n");
      lines.forEach((ln, i) => ctx.fillText(ln, (pad.l + w - pad.r) / 2, (pad.t + h - pad.b) / 2 + (i - (lines.length - 1) / 2) * 17));
      row.readout.textContent = "";
      return;
    }

    const groups = utilization(trace);
    let top = null;
    groups.forEach((grp, g) => {
      const x0 = pad.l + groupW * (g + 0.5) - barW - gap / 2;
      grp.bars.forEach((bar, k) => {
        const x = x0 + k * (barW + gap), y = Y(bar.v), y0 = Y(0);
        ctx.fillStyle = SHADES[row.key][bar.metric.key];
        ctx.beginPath();
        ctx.roundRect(x, y, barW, Math.max(0, y0 - y), [4, 4, 0, 0]);
        ctx.fill();
        ctx.font = font(11, 600); ctx.lineJoin = "round";
        ctx.strokeStyle = "#fff"; ctx.lineWidth = 4;
        ctx.strokeText(`${Math.round(bar.v * 100)}%`, x + barW / 2, y - 5);
        ctx.fillStyle = "#1f2328";
        ctx.fillText(`${Math.round(bar.v * 100)}%`, x + barW / 2, y - 5);
        // Hover target: the whole column above the baseline, wider than the bar.
        row.hits.push({ x0: x - gap / 2, x1: x + barW + gap / 2, y: Math.min(y, Y(1)) - 16, y0, grp, bar });
        if (!top || bar.v > top.bar.v) top = { grp, bar };
      });
    });

    row.readout.textContent = `Nearest its limit: ${top.grp.label} ${top.bar.metric.name}, `
      + `${Math.round(top.bar.v * 100)}% ${top.bar.metric.unit} (95th percentile over the run).`;
  }

  function hover(row, event) {
    const r = row.canvas.getBoundingClientRect();
    const x = event.clientX - r.left, y = event.clientY - r.top;
    const hit = (row.hits || []).find((b) => x >= b.x0 && x <= b.x1 && y >= b.y && y <= b.y0);
    if (!hit) { row.tip.hidden = true; return; }
    const { grp, bar } = hit;
    row.tip.innerHTML = `<b>${grp.label} ${bar.metric.name}</b>`
      + `<span class="tip-main">${Math.round(bar.v * 100)}% ${bar.metric.unit}</span>`
      + bar.legs.map((l) => `<span>${l.leg}<em>${Math.round(l.v * 100)}%</em></span>`).join("");
    row.tip.hidden = false;
    const tw = row.tip.offsetWidth;
    const left = Math.min(Math.max(0, (hit.x0 + hit.x1) / 2 - tw / 2), r.width - tw);
    row.tip.style.left = `${row.canvas.offsetLeft + left}px`;
    row.tip.style.top = `${row.canvas.offsetTop + Math.max(0, hit.y - row.tip.offsetHeight + 8)}px`;
  }

  // ------------------------------------------------------------ rows

  function build(robotKey) {
    const node = document.getElementById("row-template").content.firstElementChild.cloneNode(true);
    const robot = ROBOTS[robotKey];
    node.style.setProperty("--robot", robot.color);
    node.style.setProperty("--tau-c", SHADES[robotKey].tau);
    node.style.setProperty("--omega-c", SHADES[robotKey].omega);
    node.querySelector(".xrow-name").textContent = robot.name;
    node.querySelector(".xrow-facts").textContent = robot.facts;

    const row = {
      key: robotKey, node, task: 0, level: 0, trace: null,
      video: node.querySelector("video"), ph: node.querySelector(".ph-live"),
      canvas: node.querySelector(".ts-plot"), readout: node.querySelector(".plot-readout"),
      tip: node.querySelector(".plot-tip"),
      slider: node.querySelector(".level"), rungsEl: node.querySelector(".rungs"),
    };

    node.querySelectorAll(".arrow").forEach((b) =>
      b.addEventListener("click", () => setTask(row, row.task + Number(b.dataset.step), true)));
    row.slider.addEventListener("input", () => setLevel(row, Number(row.slider.value), true));
    row.canvas.addEventListener("mousemove", (e) => hover(row, e));
    row.canvas.addEventListener("mouseleave", () => { row.tip.hidden = true; });
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
    // On the ladder, not the rungs: the slider above them is sized from it too.
    row.node.querySelector(".xrow-ladder").style.setProperty("--n", ladder.rungs.length);
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
