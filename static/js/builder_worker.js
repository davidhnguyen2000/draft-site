// The live builder's engine: Draft's own Python, running in Pyodide. The page
// starts two of these, so a slow generation never holds up the plots:
//
//   builder_worker.js?v=<bundle>            the generator, with MuJoCo's WebAssembly
//                                           build compiling the models it emits
//   builder_worker.js?v=<bundle>&role=fits  only the actuator trends (draft_design):
//                                           no MuJoCo, numpy or trimesh, a few
//                                           milliseconds a step, for dragging
//
// Messages in:  {type: "generate", id, robot, values}          (generator)
//               {type: "points", id, robot, values, start}     (fits)
// Messages out: {type: "status", text} while loading, {type: "ready", api, role, axes?},
//               {type: "result", id, robot, design, ms}, {type: "points", id, robot, points},
//               {type: "error", id?, text}

import loadMujoco from "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.13.0/mujoco.js";
import { loadPyodide } from "https://cdn.jsdelivr.net/pyodide/v314.0.7/full/pyodide.mjs";

//: The message format; the page refuses a worker that speaks another one.
const API = 5;
const params = new URL(import.meta.url).searchParams;
const ROLE = params.get("role") || "generator";
const status = (text) => ROLE === "generator" && postMessage({ type: "status", text });

// Download progress for the page's loading bar: the bytes of every file this
// worker has fetched so far (from the network or the browser's cache).
if (ROLE === "generator") {
  let bytes = 0;
  new PerformanceObserver((list) => {
    for (const e of list.getEntries()) bytes += e.encodedBodySize || e.transferSize || 0;
    postMessage({ type: "progress", bytes });
  }).observe({ type: "resource", buffered: true });
}

async function unpackBundle(py) {
  // The page starts this worker as builder_worker.js?v=<bundle version>; the same
  // version on the bundle's URL keeps a cached old bundle out.
  const v = params.get("v") || Date.now();
  const res = await fetch(new URL(`../../py/bundle.zip?v=${v}`, import.meta.url), { cache: "no-cache" });
  if (!res.ok) throw new Error(`py/bundle.zip: HTTP ${res.status} (run scripts/site/build_pyodide.py)`);
  py.unpackArchive(await res.arrayBuffer(), "zip", { extractDir: "/home/pyodide/draft_site" });
  py.runPython(`import sys; sys.path.insert(0, "/home/pyodide/draft_site")`);
}

async function bootGenerator() {
  status("Loading MuJoCo (WebAssembly)…");
  globalThis.mujoco_wasm = await loadMujoco();

  status("Loading Python…");
  const py = await loadPyodide();
  status("Loading numpy, scipy, trimesh…");
  await py.loadPackage(["numpy", "scipy", "pyyaml", "networkx", "shapely", "micropip"]);
  await py.runPythonAsync(`
import micropip
await micropip.install("trimesh")
`);

  status("Loading Draft…");
  await unpackBundle(py);
  py.runPython(`
import browser_compat
browser_compat.install()
import draft_site

def _sim_inputs(robot):
    # The design generated last for this robot: its MJCF, its mesh files and the
    # pose the page draws it in, for the simulate button.
    import json
    from draft.generation.mjcf_assets import read_assets
    path = (draft_site._OUT / robot / f"{robot}.xml").resolve()
    return path.read_text(), read_assets(path), json.dumps(draft_site.ROBOTS[robot]["pose"])
`);
  const axes = JSON.parse(py.runPython("draft_site.axes_json()"));
  const simInputs = py.globals.get("_sim_inputs");
  return { axes, generate: py.runPython("draft_site.generate_json"),
           simInputs: (robot) => {
             const t = simInputs(robot);
             try { return t.toJs({ dict_converter: Object.fromEntries }); } finally { t.destroy(); }
           } };
}

async function bootFits() {
  const py = await loadPyodide();
  await py.loadPackage(["pyyaml"]);
  await unpackBundle(py);
  py.runPython("import draft_design");
  return { points: py.runPython("draft_design.points_json") };
}

// ------------------------------------------------------------ simulate
//
// The design just generated, dropped onto a floor and held in its pose by the
// motor model the RL tasks train with (draft.tasks.quadruped.quadruped_entity):
// a PD loop, kp = (100/70)·peak torque and kd = (5/70)·peak torque, clipped to
// the DC motor's torque-speed band, tau in [sat(-1 - w/w_NL), sat(1 - w/w_NL)]
// within +/- the peak torque. Nothing balances it. The robot touches only the
// floor, never itself, as in training. With the motors off no torque is applied
// at all: only gravity, the joints' own friction and damping, and the floor.
// The loop runs here in JavaScript,
// against the same MuJoCo build the generator compiles with.

const SIM_SECONDS = 6, SIM_DROP_M = 0.02, SIM_FRAME_MS = 16, SIM_MAX_STEPS = 400;
const KP_PER_NM = 100 / 70, KD_PER_NM = 5 / 70;
const FLOOR = '<geom name="sim_floor" type="plane" size="0 0 1" contype="0" conaffinity="1"/>';
let sim = null;

function simStop(reason) {
  if (!sim) return;
  clearTimeout(sim.timer);
  const { id, m, d } = sim;
  sim = null;
  d.delete(); m.delete();
  postMessage({ type: "simend", id, reason });
}

function nameAt(m, adr) {
  const all = m.names;
  let end = adr;
  while (all[end] !== 0) end++;
  return String.fromCharCode(...all.subarray(adr, end));
}

// How far a geom reaches below its centre (draft_site._half_height).
function below(type, size, R, rbound) {
  if (type === 2) return size[0];
  if (type === 3) return Math.abs(R[8]) * size[1] + size[0];
  if (type === 5) return Math.abs(R[8]) * size[1] + size[0] * Math.sqrt(Math.max(0, 1 - R[8] ** 2));
  if (type === 6) return Math.abs(R[6]) * size[0] + Math.abs(R[7]) * size[1] + Math.abs(R[8]) * size[2];
  return rbound;
}

function simStart(id, robot, inputs, motors = true) {
  simStop("restarted");
  const mj = globalThis.mujoco_wasm;
  const [xml0, assets, poseJson] = inputs;
  const at = xml0.lastIndexOf("</worldbody>");
  const xml = xml0.slice(0, at) + FLOOR + xml0.slice(at);
  const vfs = new mj.MjVFS();
  let m;
  try {
    for (const [name, buf] of Object.entries(assets)) vfs.addBuffer(name, buf);
    m = mj.MjModel.from_xml_string(xml, vfs);
  } finally {
    vfs.delete();
  }
  const d = new mj.MjData(m);

  // The robot's geoms, in the order the page drew them (every geom but the floor).
  const types = m.geom_type, geoms = [];
  for (let g = 0; g < m.ngeom; g++) if (types[g] !== 0) geoms.push(g);
  const contype = m.geom_contype, conaff = m.geom_conaffinity;
  for (const g of geoms) { contype[g] = 1; conaff[g] = 0; }

  const numeric = {};
  for (let k = 0; k < m.nnumeric; k++) numeric[nameAt(m, m.name_numericadr[k])] = m.numeric_data[m.numeric_adr[k]];
  const pose = JSON.parse(poseJson), joints = [];
  const jtype = m.jnt_type, qadr = m.jnt_qposadr, dadr = m.jnt_dofadr, frc = m.jnt_actfrcrange;
  const qpos = d.qpos, qpos0 = m.qpos0;
  for (let j = 0; j < m.njnt; j++) {
    if (jtype[j] !== 3) continue;                       // hinges only; the root is free
    const name = nameAt(m, m.name_jntadr[j]);
    const key = Object.keys(pose).find((k) => name === k || name.endsWith("_" + k));
    const target = key !== undefined ? pose[key] : qpos0[qadr[j]];
    qpos[qadr[j]] = target;
    const effort = frc[2 * j + 1];
    joints.push({ qa: qadr[j], da: dadr[j], target, effort,
                  vnl: numeric[name + "_velocity_limit"] ?? Infinity });
  }

  // Lowest point SIM_DROP_M above the floor, as the page drew it (on the floor).
  mj.mj_forward(m, d);
  const size = m.geom_size, rb = m.geom_rbound, xp = d.geom_xpos, xm = d.geom_xmat;
  let low = Infinity;
  for (const g of geoms) {
    low = Math.min(low, xp[3 * g + 2] - below(types[g], size.subarray(3 * g, 3 * g + 3), xm.subarray(9 * g, 9 * g + 9), rb[g]));
  }
  d.qpos[2] += SIM_DROP_M - low;
  mj.mj_forward(m, d);

  sim = { id, m, d, geoms, joints: motors ? joints : [], wall0: performance.now(), dt: m.opt.timestep, timer: 0 };
  postMessage({ type: "simstart", id, n: geoms.length, seconds: SIM_SECONDS });
  simTick();
}

function simTick() {
  const s = sim;
  if (!s) return;
  const mj = globalThis.mujoco_wasm, { m, d, joints } = s;
  const until = Math.min(SIM_SECONDS, (performance.now() - s.wall0) / 1000);
  let steps = 0;
  while (d.time + s.dt / 2 < until && steps < SIM_MAX_STEPS) {
    const qpos = d.qpos, qvel = d.qvel, f = d.qfrc_applied;
    for (const j of joints) {
      const w = qvel[j.da], sat = j.effort;
      const tau = KP_PER_NM * sat * (j.target - qpos[j.qa]) - KD_PER_NM * sat * w;
      const hi = Math.min(sat * (1 - w / j.vnl), sat), lo = Math.max(sat * (-1 - w / j.vnl), -sat);
      f[j.da] = Math.min(Math.max(tau, lo), hi);
    }
    mj.mj_step(m, d);
    steps++;
  }
  // Behind real time (a slow machine, or a generation in between): slow the
  // clock down rather than try to catch up.
  if (steps === SIM_MAX_STEPS) s.wall0 = performance.now() - d.time * 1000;

  const xp = d.geom_xpos, xm = d.geom_xmat, n = s.geoms.length, out = new Float32Array(12 * n);
  let ok = true;
  s.geoms.forEach((g, k) => {
    for (let a = 0; a < 3; a++) out[12 * k + a] = xp[3 * g + a];
    for (let a = 0; a < 9; a++) out[12 * k + 3 + a] = xm[9 * g + a];
    ok = ok && Number.isFinite(xp[3 * g + 2]);
  });
  postMessage({ type: "simframe", id: s.id, t: d.time, x: out }, [out.buffer]);
  if (!ok) return simStop("unstable");
  if (d.time + s.dt / 2 >= SIM_SECONDS) return simStop("done");
  s.timer = setTimeout(simTick, SIM_FRAME_MS);
}

let engine = null;
const booting = (ROLE === "fits" ? bootFits() : bootGenerator())
  .then((e) => { engine = e; postMessage({ type: "ready", api: API, role: ROLE, axes: e.axes }); })
  .catch((err) => postMessage({ type: "error", role: ROLE, text: String(err && err.message || err) }));

onmessage = async (event) => {
  const msg = event.data;
  await booting;
  if (!engine) return;
  if (msg.type === "simstop") return simStop("stopped");
  if (msg.type === "simulate" && engine.simInputs) {
    try {
      simStart(msg.id, msg.robot, engine.simInputs(msg.robot), msg.motors !== false);
    } catch (err) {
      postMessage({ type: "simend", id: msg.id, reason: "error", text: String(err && err.message || err) });
    }
    return;
  }
  const t0 = performance.now();
  try {
    if (msg.type === "generate" && engine.generate) {
      const design = JSON.parse(engine.generate(JSON.stringify(msg.values), msg.robot));
      postMessage({ type: "result", id: msg.id, robot: msg.robot, design, ms: performance.now() - t0 });
    } else if (msg.type === "points" && engine.points) {
      const points = JSON.parse(engine.points(JSON.stringify(msg.values), msg.robot, JSON.stringify(msg.start)));
      postMessage({ type: "points", id: msg.id, robot: msg.robot, points });
    }
  } catch (err) {
    postMessage({ type: "error", id: msg.id, role: ROLE, text: String(err && err.message || err) });
  }
};
