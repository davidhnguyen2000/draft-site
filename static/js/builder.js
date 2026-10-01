// Quadruped builder. Each slider move is generated live by Draft's own Python
// generator in a worker (builder_worker.js: Pyodide + MuJoCo's WebAssembly
// build); this file draws the result and its feasibility report.

(() => {
  const CHECK_LABEL = {
    actuator_mass_fraction: ["Actuator share of mass", "pct"],
    trunk_mass_fraction: ["Trunk share of mass", "pct"],
    total_mass_vs_size_kg: ["Total mass for its leg length", "kg"],
  };

  const $ = (id) => document.getElementById(id);
  const host = $("b-canvas");
  if (!host || !window.THREE) return;

  // Per robot: its sliders, the values they are at, and the last design that passed.
  let robots = null, robot = "quadruped", axes = null, palette = null, fitData = null;
  const valuesOf = {}, lastGoodOf = {};
  let values = {}, lastGood = null;
  //: Mass bar parts: report key, label, index into the robot's palette
  //: (link, actuator, detail, trunk, foot, ...), as draft_site.palette lists it.
  const MASS_PARTS = [["torso", "Trunk", 3], ["motor", "Actuators", 1], ["link", "Links", 0],
                      ["head", "Head", 3], ["hand", "Hands", 0], ["foot", "Feet", 4]];

  // ------------------------------------------------------------ scene

  const renderer = new THREE.WebGLRenderer({ antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.shadowMap.enabled = true;
  renderer.setClearColor(0xf6f7f9);
  host.appendChild(renderer.domElement);

  const scene = new THREE.Scene();
  THREE.Object3D.DefaultUp.set(0, 0, 1);
  const camera = new THREE.PerspectiveCamera(35, 1, 0.05, 50);
  camera.up.set(0, 0, 1);
  camera.position.set(2.3, -2.6, 1.35);
  const controls = new THREE.OrbitControls(camera, renderer.domElement);
  controls.target.set(0, 0, 0.45);
  controls.enableDamping = true;
  controls.minDistance = 0.8;
  controls.maxDistance = 8;
  controls.update();

  scene.add(new THREE.HemisphereLight(0xffffff, 0xb8bcc4, 0.75));
  const sun = new THREE.DirectionalLight(0xffffff, 0.75);
  sun.position.set(2, -1.5, 4);
  sun.castShadow = true;
  sun.shadow.mapSize.set(1024, 1024);
  Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2 });
  scene.add(sun);

  const ground = new THREE.Mesh(new THREE.PlaneGeometry(12, 12), new THREE.ShadowMaterial({ opacity: 0.12 }));
  ground.receiveShadow = true;
  scene.add(ground);
  const grid = new THREE.GridHelper(6, 30, 0xd5d8de, 0xe4e6eb);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);

  const body3d = new THREE.Group();
  scene.add(body3d);

  function resize() {
    const w = host.clientWidth, h = host.clientHeight;
    renderer.setSize(w, h, false);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }
  new ResizeObserver(resize).observe(host);
  resize();

  // Reframe only when a design would overflow the view or shrink too far into
  // it, so that growing a leg is seen as growth rather than silently refitted.
  let aim = null;
  function frame(force) {
    const box = new THREE.Box3().setFromObject(body3d);
    if (box.isEmpty()) return;
    const radius = box.getBoundingSphere(new THREE.Sphere()).radius;
    const fov = (camera.fov * Math.PI) / 180;
    const need = radius / Math.sin(Math.min(fov, fov * camera.aspect) / 2);
    const dist = camera.position.distanceTo(controls.target);
    if (!force && dist > need * 1.02 && dist < need * 2.4) return;
    const target = box.getCenter(new THREE.Vector3());
    const dir = camera.position.clone().sub(controls.target).normalize();
    aim = { target, position: target.clone().add(dir.multiplyScalar(need * 1.25)) };
  }

  (function loop() {
    if (aim) {
      controls.target.lerp(aim.target, 0.12);
      camera.position.lerp(aim.position, 0.12);
      if (camera.position.distanceTo(aim.position) < 1e-3) aim = null;
    }
    controls.update();
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  })();

  // MuJoCo geoms, in MuJoCo's frame (z up): [type, size, pos, xmat (row-major), colour, mesh?]
  function geomMesh(g, refused) {
    const [type, size, pos, R, colour, mesh] = g;
    let geo;
    if (type === 2) geo = new THREE.SphereGeometry(size[0], 20, 14);
    else if (type === 3 || type === 5) {
      geo = new THREE.CylinderGeometry(size[0], size[0], 2 * size[1], type === 5 ? 28 : 20);
      geo.rotateX(Math.PI / 2);          // three's cylinder runs along y, MuJoCo's along z
    } else if (type === 6) geo = new THREE.BoxGeometry(2 * size[0], 2 * size[1], 2 * size[2]);
    else if (type === 7) {
      geo = new THREE.BufferGeometry();
      geo.setAttribute("position", new THREE.Float32BufferAttribute(mesh.v, 3));
      geo.setIndex(mesh.f);
      geo = geo.toNonIndexed();
      geo.computeVertexNormals();
    } else return null;
    const [r, gr, b] = palette[colour];
    const color = new THREE.Color(r, gr, b);
    if (refused) color.lerp(new THREE.Color(0xc44e52), 0.55);
    const m = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color, roughness: 0.6, metalness: 0.05, flatShading: type === 7,
      transparent: refused, opacity: refused ? 0.45 : 1,
    }));
    m.matrixAutoUpdate = false;
    m.matrix.set(R[0], R[1], R[2], pos[0], R[3], R[4], R[5], pos[1], R[6], R[7], R[8], pos[2], 0, 0, 0, 1);
    m.castShadow = !refused;
    return m;
  }

  function drawRobot(geoms, refused) {
    for (const child of body3d.children) { child.geometry.dispose(); child.material.dispose(); }
    body3d.clear();
    for (const g of geoms) {
      const m = geomMesh(g, refused);
      if (m) body3d.add(m);
    }
  }

  // ------------------------------------------------------------ engine

  //: The worker message format this page speaks (builder_worker.js API).
  const API = 5;
  let worker = null, busy = false, queued = false, requestId = 0, framed = false;
  let fitsWorker = null, fitsReady = false, pointsBusy = false, pointsQueued = false, pointsId = 0;
  //: The slider being dragged, if any, and the actuator classes it drives.
  let dragging = null;

  function requestPoints() {
    if (!fitsReady || !axes) return;
    if (pointsBusy) { pointsQueued = true; return; }
    pointsBusy = true;
    const start = Object.fromEntries(axes.map((a) => [a.key, a.start]));
    fitsWorker.postMessage({ type: "points", id: ++pointsId, robot, values: { ...values }, start });
  }

  function onFits(event) {
    const msg = event.data;
    if (msg.type === "ready") { fitsReady = true; return; }
    if (msg.type === "points") {
      pointsBusy = false;
      if (msg.id === pointsId && msg.robot === robot && dragging) drawFits(msg.points);
      if (pointsQueued) { pointsQueued = false; requestPoints(); }
    }
    if (msg.type === "error") pointsBusy = false;
  }

  function fail(text) {
    $("b-loading").hidden = false;
    $("b-loading").textContent = text;
  }

  function request() {
    if (!axes) return;
    if (busy) { queued = true; return; }     // only the latest position matters
    busy = true;
    $("b-time").textContent = "generating…";
    worker.postMessage({ type: "generate", id: ++requestId, robot, values: { ...values } });
  }

  function onMessage(event) {
    try {
      handle(event.data);
    } catch (err) {
      fail(`The builder could not start: ${err.message}`);
      console.error(err);
    }
  }

  function handle(msg) {
    if (msg.type === "status") $("b-loading").textContent = msg.text;
    if (msg.type === "ready") {
      if (msg.api !== API || !msg.axes || !msg.axes.robots) {
        fail("This page and the builder's files are out of step, probably a stale browser "
             + "cache. Reload the page; if it persists, rebuild with scripts/site/build_pyodide.py.");
        return;
      }
      robots = msg.axes.robots;
      fitData = msg.axes.fits;
      for (const [key, r] of Object.entries(robots)) {
        valuesOf[key] = Object.fromEntries(r.axes.map((a) => [a.key, a.start]));
      }
      const q = new URLSearchParams(location.search).get("robot");
      buildTabs();
      selectRobot(robots[q] ? q : "quadruped", true);
      $("b-loading").hidden = true;
    }
    if (msg.type === "result") {
      busy = false;
      if (msg.id === requestId && msg.robot === robot) show(msg.design, msg.ms);
      if (queued) { queued = false; request(); }
    }
    if (msg.type === "error") {
      busy = false;
      if (msg.id === undefined) {
        $("b-loading").hidden = false;
        $("b-loading").textContent = `The builder could not start: ${msg.text}`;
        return;
      }
      // One design failed to generate: say so, and carry on with the next move.
      $("b-refused").hidden = false;
      $("b-refused-list").innerHTML = `<li>This design could not be generated: ${msg.text.trim().split("\n").pop()}</li>`;
      $("b-status").textContent = "error";
      $("b-status").className = "b-status bad";
      if (queued) { queued = false; request(); }
    }
  }

  // Start the worker on the bundle's current version, so a cached old bundle is
  // never paired with this page.
  fetch("py/bundle.json", { cache: "no-store" })
    .then((r) => (r.ok ? r.json() : Promise.reject(new Error(`py/bundle.json: HTTP ${r.status}`))))
    .then(({ version }) => {
      worker = new Worker(`static/js/builder_worker.js?v=${version}`, { type: "module" });
      worker.onmessage = onMessage;
      worker.onerror = (e) => fail(`The builder could not start: ${e.message || "worker error"}`);
      // The second worker only runs the actuator trends, fast enough to move the
      // fit plots on every step of a drag while the first is still generating.
      fitsWorker = new Worker(`static/js/builder_worker.js?v=${version}&role=fits`, { type: "module" });
      fitsWorker.onmessage = onFits;
    })
    .catch((err) => fail(`The builder could not start: ${err.message} (run scripts/site/build_pyodide.py)`));

  // ------------------------------------------------------------ controls

  const decimals = (a) => Math.max(0, -Math.floor(Math.log10(a.step) + 1e-9));
  function fmt(a, v) {
    if (v === null || v === undefined) return "auto";
    const s = Number(v).toFixed(decimals(a));
    return a.unit ? `${s} ${a.unit}` : s;
  }

  // Log-scaled axes move evenly in log space; the slider runs 0..1000 on them.
  const toSlider = (a, v) => (a.log ? Math.round((1000 * Math.log(v / a.min)) / Math.log(a.max / a.min)) : v);
  function fromSlider(a, s) {
    if (!a.log) return Number(s);
    const v = a.min * Math.pow(a.max / a.min, Number(s) / 1000);
    return Number((Math.round(v / a.step) * a.step).toFixed(decimals(a)));
  }

  function buildSliders() {
    const box = $("b-sliders");
    box.innerHTML = "";
    for (const a of axes) {
      const wrap = document.createElement("div");
      wrap.className = "b-slider";
      const range = a.log ? `min="0" max="1000" step="1"` : `min="${a.min}" max="${a.max}" step="${a.step}"`;
      wrap.innerHTML = `<span class="b-slider-head"><span>${a.label}</span><b data-val="${a.key}"></b></span>
        <input type="range" ${range} data-key="${a.key}" aria-label="${a.label}">
        <span class="b-slider-ends"><span>${fmt(a, a.min)}</span>
          ${a.auto ? `<label class="b-auto"><input type="checkbox" data-auto="${a.key}"> from the trend</label>` : ""}
          <span>${fmt(a, a.max)}</span></span>`;
      const input = wrap.querySelector("input[type=range]");
      const auto = wrap.querySelector("[data-auto]");
      const sync = () => {
        wrap.querySelector(`[data-val="${a.key}"]`).textContent = fmt(a, values[a.key]);
        if (auto) { auto.checked = values[a.key] === null; input.disabled = auto.checked; }
      };
      input.value = toSlider(a, values[a.key] ?? 0.6);
      window.paintRange(input);
      input.addEventListener("input", () => {
        values[a.key] = fromSlider(a, input.value); sync(); toURL(); request();
        if (a.affects.length) requestPoints();
      });
      input.addEventListener("pointerdown", () => {
        if (!a.affects.length) return;
        dragging = { key: a.key, affects: new Set(a.affects) };
        if (lastActuators) drawFits(lastActuators);
      });
      if (auto) {
        auto.addEventListener("change", () => {
          values[a.key] = auto.checked ? null : fromSlider(a, input.value);
          sync(); toURL(); request();
        });
      }
      sync();
      box.appendChild(wrap);
    }
  }

  // ------------------------------------------------------------ robots

  function buildTabs() {
    const box = $("b-robot-tabs");
    box.innerHTML = "";
    for (const [key, r] of Object.entries(robots)) {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "seg-btn";
      b.dataset.value = key;
      b.setAttribute("role", "tab");
      const icon = { quadruped: "fa-dog", humanoid: "fa-person" }[key] || "fa-robot";
      b.innerHTML = `<i class="fas ${icon}"></i><span>${r.label}</span>`;
      b.addEventListener("click", () => selectRobot(key, false));
      box.appendChild(b);
    }
  }

  function selectRobot(key, fromLoad) {
    if (!fromLoad && key === robot && axes) return;
    robot = key;
    axes = robots[key].axes;
    palette = robots[key].palette;
    // The builder's controls take the robot's own colour (its trunk's, palette[3]).
    const [cr, cg, cb] = palette[3].map((x) => Math.round(x * 255));
    $("builder").style.setProperty("--robot-accent", `rgb(${cr}, ${cg}, ${cb})`);
    $("b-preset").textContent = `Starts as ${robots[key].preset}. Every slider begins at its value.`;
    document.querySelector("#b-reset span").textContent = `Reset to ${robots[key].preset.replace(/^the /, "")}`;
    values = valuesOf[key];
    lastGood = lastGoodOf[key] || null;
    document.querySelectorAll("#b-robot-tabs .seg-btn").forEach((t) => {
      const on = t.dataset.value === key;
      t.classList.toggle("is-active", on);
      t.setAttribute("aria-selected", on);
    });
    if (fromLoad) fromURL();
    buildSliders();
    toURL();
    framed = false;
    request();
  }

  // A design is shareable: the robot and every slider are in the URL.
  function fromURL() {
    const q = new URLSearchParams(location.search);
    for (const a of axes) {
      if (!q.has(a.key)) continue;
      const raw = q.get(a.key);
      if (a.auto && raw === "auto") { values[a.key] = null; continue; }
      const v = Number(raw);
      if (Number.isFinite(v)) values[a.key] = Math.min(a.max, Math.max(a.min, v));
    }
  }

  function toURL() {
    const q = new URLSearchParams();
    q.set("robot", robot);
    for (const a of axes) q.set(a.key, values[a.key] === null ? "auto" : values[a.key]);
    history.replaceState(null, "", `${location.pathname}?${q}${location.hash}`);
  }

  $("b-reset").addEventListener("click", () => {
    if (!axes) return;
    for (const a of axes) values[a.key] = a.start;
    buildSliders();
    toURL();
    framed = false;
    request();
  });

  // ------------------------------------------------------------ report

  function renderReport(rep) {
    $("b-mass").textContent = rep.total_mass_kg.toFixed(1);

    const parts = MASS_PARTS.filter(([k]) => rep.mass_kg[k] > 0);
    const total = rep.total_mass_kg;
    $("b-massbar").innerHTML = parts.map(([k, , c]) => {
      const [r, g, b] = palette[c].map((x) => Math.round(x * 255));
      return `<span style="flex:${rep.mass_kg[k]};background:rgb(${r},${g},${b})"></span>`;
    }).join("");
    $("b-masslegend").innerHTML = parts.map(([k, name]) =>
      `<span>${name} <b>${rep.mass_kg[k].toFixed(1)} kg</b> · ${Math.round((100 * rep.mass_kg[k]) / total)}%</span>`).join("");

    $("b-checks").innerHTML = rep.checks.map((c) => {
      const [label, unit] = CHECK_LABEL[c.quantity] || [c.quantity.replace(/_/g, " "), ""];
      const show = (v) => (unit === "pct" ? `${Math.round(v * 100)}%` : `${v.toFixed(1)} kg`);
      const lo = Math.min(c.lo, c.value), hi = Math.max(c.hi, c.value);
      const span = hi - lo || 1, x0 = lo - 0.15 * span, x1 = hi + 0.15 * span;
      const at = (v) => `${(100 * (v - x0)) / (x1 - x0)}%`;
      const ok = c.status === "ok";
      return `<div class="check">
        <div class="check-head"><span>${label}</span><b class="${ok ? "ok" : "warn"}">${show(c.value)}</b></div>
        <div class="band"><span class="band-in" style="left:${at(c.lo)};right:calc(100% - ${at(c.hi)})"></span>
          <span class="band-mark ${ok ? "" : "warn"}" style="left:${at(c.value)}"></span></div>
        <div class="check-foot"><span>surveyed ${show(c.lo)}–${show(c.hi)}</span>${ok ? "" : `<span class="warn">${c.status.replace(/_/g, " ")}</span>`}${c.extrapolated ? "<span>extrapolated</span>" : ""}</div>
      </div>`;
    }).join("");

    const st = rep.stretched || [];
    $("b-stretch-box").hidden = !st.length;
    $("b-stretch").innerHTML = st.map((o) => `<div class="override">
        <span>${o.param}</span>
        <span><s>${o.from.toFixed(3)}</s> → <b>${o.to.toFixed(3)}</b> m</span>
        <em>so the actuators at either end do not overlap</em></div>`).join("");

    $("b-warn-count").textContent = rep.warnings.length;
    $("b-warnings").innerHTML = rep.warnings.map((w) => `<li>${w}</li>`).join("");
  }


  // ------------------------------------------------------------ actuator fits

  // The four fits of the supplementary video (the paper's Fig. 3): the surveyed
  // catalogue, the trend fitted through it, and where this design's actuators
  // land. Log-log, redrawn on every design.
  const FIT_PANELS = {
    a: { title: "Reduction from speed", x: "no-load speed ω (rad/s)", y: "reduction N",
         law: "N ∝ ω⁻¹·⁰⁷", lawAt: "right", px: (a) => a.omega, py: (a) => a.N },
    b: { title: "Mass from torque", x: "peak torque τ (N·m)", y: "mass m (kg)",
         law: "m ∝ τ⁰·⁶⁷", px: (a) => a.tau, py: (a) => a.m },
    c: { title: "Volume from torque and gear", x: "peak torque τ (N·m)", y: "volume πr²ℓ (cm³)",
         law: "V ∝ τ⁰·⁷⁰ N⁻⁰·¹⁷", px: (a) => a.tau, py: (a) => a.V },
    d: { title: "Rotor inertia from radius", x: "housing radius r (m)", y: "rotor inertia J (kg·m²)",
         law: "J ∝ r⁴ (imposed)", px: (a) => a.r, py: (a) => a.J },
  };
  const CAT_COLOUR = { QDD: "#4c72b0", MidGear: "#dd8452", Harmonic: "#55a868" };
  let lastActuators = null;
  const hover = {};

  function logRange(values, pad = 1.35) {
    const lo = Math.min(...values), hi = Math.max(...values);
    return [Math.log10(lo / pad), Math.log10(hi * pad)];
  }

  function tickLabel(e) {
    const v = Math.pow(10, e);
    if (e >= 0 && e <= 4) return String(v);
    if (e < 0 && e >= -2) return v.toFixed(-e);
    return `1e${e}`;
  }

  function drawPanel(canvas, key, acts) {
    const P = FIT_PANELS[key], pts = fitData.points[key], law = fitData.laws[key];
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w) return;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);

    const mine = acts.map((a) => [P.px(a), P.py(a), a.cls, a.N]);
    const [x0, x1] = logRange([...pts.map((p) => p[0]), ...mine.map((m) => m[0])]);
    const [y0, y1] = logRange([...pts.map((p) => p[1]), ...mine.map((m) => m[1])]);
    const pad = { l: 46, r: 10, t: 26, b: 34 };
    const X = (v) => pad.l + ((Math.log10(v) - x0) / (x1 - x0)) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - ((Math.log10(v) - y0) / (y1 - y0)) * (h - pad.t - pad.b);

    ctx.font = "600 12px 'Google Sans', 'Noto Sans', sans-serif";
    ctx.fillStyle = "#1f2328"; ctx.textAlign = "left";
    ctx.fillText(`(${key}) ${P.title}`, pad.l, 14);

    // decade grid and ticks
    ctx.font = "10px 'Noto Sans', sans-serif"; ctx.fillStyle = "#7a8190";
    ctx.strokeStyle = "#e9ebef"; ctx.lineWidth = 1;
    for (let e = Math.ceil(x0); e <= Math.floor(x1); e++) {
      const x = X(Math.pow(10, e));
      ctx.beginPath(); ctx.moveTo(x, pad.t); ctx.lineTo(x, h - pad.b); ctx.stroke();
      ctx.textAlign = "center"; ctx.fillText(tickLabel(e), x, h - pad.b + 12);
    }
    for (let e = Math.ceil(y0); e <= Math.floor(y1); e++) {
      const y = Y(Math.pow(10, e));
      ctx.beginPath(); ctx.moveTo(pad.l, y); ctx.lineTo(w - pad.r, y); ctx.stroke();
      ctx.textAlign = "right"; ctx.fillText(tickLabel(e), pad.l - 5, y + 3);
    }
    ctx.strokeStyle = "#c9cdd4"; ctx.strokeRect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b);
    ctx.fillStyle = "#4b5260"; ctx.textAlign = "center";
    ctx.fillText(P.x, (pad.l + w - pad.r) / 2, h - 6);
    ctx.save(); ctx.translate(11, (pad.t + h - pad.b) / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText(P.y, 0, 0); ctx.restore();

    ctx.save();
    ctx.beginPath(); ctx.rect(pad.l, pad.t, w - pad.l - pad.r, h - pad.t - pad.b); ctx.clip();
    // the catalogue
    for (const [x, y, cat] of pts) {
      ctx.globalAlpha = 0.7; ctx.fillStyle = CAT_COLOUR[cat] || "#8c8c8c";
      ctx.beginPath(); ctx.arc(X(x), Y(y), 3, 0, 2 * Math.PI); ctx.fill();
    }
    ctx.globalAlpha = 1;
    // the fitted trend; for volume, one line at each of this design's reductions
    const line = (f, colour, width, alpha = 1) => {
      ctx.strokeStyle = colour; ctx.lineWidth = width; ctx.globalAlpha = alpha; ctx.beginPath();
      for (let i = 0; i <= 60; i++) {
        const lx = Math.pow(10, x0 + ((x1 - x0) * i) / 60);
        const px = X(lx), py = Y(f(lx));
        if (i) ctx.lineTo(px, py); else ctx.moveTo(px, py);
      }
      ctx.stroke(); ctx.globalAlpha = 1;
    };
    const focus = dragging && dragging.affects;
    const lit = (names) => !focus || names.some((n) => focus.has(n));
    if (key === "c") {
      const byN = new Map();
      for (const [, , cls, N] of mine) {
        const k = N.toFixed(2);
        if (!byN.has(k)) byN.set(k, { N, names: [] });
        byN.get(k).names.push(cls);
      }
      for (const { N, names } of byN.values()) {
        line((t) => law.coef * Math.pow(t, law.exp) * Math.pow(N, law.gear_exp), "#1f2328", 1.5,
             lit(names) ? 0.75 : 0.15);
      }
    } else {
      line((v) => law.coef * Math.pow(v, law.exp), "#1f2328", 1.75);
    }
    // this design: identical units drawn once, named together
    const groups = new Map();
    for (const [x, y, cls] of mine) {
      const k = `${x.toPrecision(5)}|${y.toPrecision(5)}`;
      if (!groups.has(k)) groups.set(k, { x, y, names: [] });
      groups.get(k).names.push(cls);
    }
    hover[key] = [];
    // Faded points first, so the ones being moved are drawn over them.
    const order = [...groups.values()].sort((p, q) => lit(p.names) - lit(q.names));
    for (const g of order) {
      const px = X(g.x), py = Y(g.y);
      ctx.globalAlpha = lit(g.names) ? 1 : 0.22;
      ctx.beginPath(); ctx.arc(px, py, 6, 0, 2 * Math.PI);
      ctx.fillStyle = "#1f2328"; ctx.fill();
      ctx.lineWidth = 2; ctx.strokeStyle = "#fff"; ctx.stroke();
      hover[key].push({ px, py, g });
    }
    ctx.globalAlpha = 1;
    ctx.restore();
    // In the corner the line leaves empty; a system face, since the web fonts lack ∝.
    ctx.font = "11px ui-sans-serif, system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.fillStyle = "#4b5260";
    const right = P.lawAt === "right";
    ctx.textAlign = right ? "right" : "left";
    ctx.fillText(P.law, right ? w - pad.r - 6 : pad.l + 6, pad.t + 14);
  }

  const CLASS_NAME = { L: "every joint", HP: "hip pitch", HR: "hip roll", HY: "hip yaw", KN: "knee",
    AP: "ankle pitch", AR: "ankle roll", TR: "torso roll", TP: "torso pitch", TY: "torso yaw",
    SP: "shoulder pitch", SR: "shoulder roll", SY: "shoulder yaw", EL: "elbow",
    WR: "wrist roll", WP: "wrist pitch", WY: "wrist yaw" };

  function drawFits(acts) {
    if (!fitData || !acts) return;
    lastActuators = acts;
    document.querySelectorAll("#b-fits canvas").forEach((c) => drawPanel(c, c.dataset.panel, acts));
  }

  document.querySelectorAll("#b-fits canvas").forEach((c) => {
    c.addEventListener("mousemove", (e) => {
      const r = c.getBoundingClientRect(), mx = e.clientX - r.left, my = e.clientY - r.top;
      const near = (hover[c.dataset.panel] || []).find((h) => Math.hypot(h.px - mx, h.py - my) < 9);
      const tip = near ? near.g.names.map((n) => CLASS_NAME[n] || n).join(", ") : "";
      c.title = tip;
      c.style.cursor = near ? "help" : "";
    });
  });
  new ResizeObserver(() => lastActuators && drawFits(lastActuators)).observe($("b-fits"));

  // A drag ends wherever the pointer is let go. The plots then return to the
  // finished design's report, which says the same as the fast path.
  window.addEventListener("pointerup", () => {
    if (!dragging) return;
    dragging = null;
    if (lastGood && lastGood.report) drawFits(lastGood.report.actuators);
    else if (lastActuators) drawFits(lastActuators);
  });

  // ------------------------------------------------------------ result

  function show(d, ms) {
    $("b-time").textContent = `generated in ${Math.round(ms)} ms`;
    const refused = Boolean(d.refused);
    $("b-refused").hidden = !refused;
    $("b-report").classList.toggle("is-stale", refused);
    document.querySelector(".b-mass").classList.toggle("is-stale", refused);
    if (refused) {
      $("b-refused-list").innerHTML = d.refused.map((r) => `<li>${r}</li>`).join("");
      $("b-status").textContent = "refused";
      $("b-status").className = "b-status bad";
      if (lastGood) drawRobot(lastGood.geoms, true);
      $("b-fits").classList.add("is-stale");
      return;
    }
    lastGood = lastGoodOf[robot] = d;
    $("b-fits").classList.remove("is-stale");
    const warned = d.report.checks.some((c) => c.status !== "ok");
    $("b-status").textContent = warned ? "feasible, with warnings" : "feasible";
    $("b-status").className = `b-status ${warned ? "warn" : "good"}`;
    drawRobot(d.geoms, false);
    frame(!framed);
    framed = true;
    renderReport(d.report);
    if (!dragging) drawFits(d.report.actuators);
  }
})();
