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
`);
  const axes = JSON.parse(py.runPython("draft_site.axes_json()"));
  return { axes, generate: py.runPython("draft_site.generate_json") };
}

async function bootFits() {
  const py = await loadPyodide();
  await py.loadPackage(["pyyaml"]);
  await unpackBundle(py);
  py.runPython("import draft_design");
  return { points: py.runPython("draft_design.points_json") };
}

let engine = null;
const booting = (ROLE === "fits" ? bootFits() : bootGenerator())
  .then((e) => { engine = e; postMessage({ type: "ready", api: API, role: ROLE, axes: e.axes }); })
  .catch((err) => postMessage({ type: "error", role: ROLE, text: String(err && err.message || err) }));

onmessage = async (event) => {
  const msg = event.data;
  await booting;
  if (!engine) return;
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
