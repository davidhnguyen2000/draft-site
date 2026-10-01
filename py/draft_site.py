"""The project site's robot builder: one design in, what Draft made of it out.

Runs in the browser under Pyodide (with the stand-ins in `browser_compat.py`) and
under CPython for `scripts/site/check_mjshim.py`, which is what proves the two
agree. Nothing here sizes anything: the numbers all come from Draft's own
`RobotGenerator` and the report it writes.

Each robot starts from a real design, its PRESET, and every slider starts at the
value that design has, so with nothing moved the builder generates exactly it:

  quadruped  the lineup's cheetah (experiments/quadruped_variants/cheetah.yaml)
  humanoid   the Unitree G1 twin (presets/g1_twin.yaml, from draft.twins)

A slider overrides its parameter only once it has moved. An actuator slider that
moves drops the preset's catalogue part and, for a speed, its declared gear, so the
trends size the unit from what is stated and infer the gear from the speed.
Frontier refusal is on (`allow_hypothetical: false`), as `draft-generate` refuses.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import tempfile
from pathlib import Path

import numpy as np
import yaml

import mujoco  # the real bindings, or mjshim in the browser

from draft.generation.generator import RobotGenerator
from draft.generation.mjcf_assets import load_model
from draft.robots import robot_dir
from draft.trends.feasibility import FeasibilityViolation

try:
    import draft_design
except ModuleNotFoundError:          # loaded by path from a checkout: it sits beside this file
    import importlib.util
    import sys
    _spec = importlib.util.spec_from_file_location(
        "draft_design", Path(__file__).resolve().parent / "draft_design.py")
    draft_design = importlib.util.module_from_spec(_spec)
    sys.modules["draft_design"] = draft_design
    _spec.loader.exec_module(draft_design)
from draft_design import BASE_CLASS, ROBOTS, preset

HERE = Path(__file__).resolve().parent

#: Links the builder lengthens when the actuators they join would collide. A
#: basic link draws its actuator at the far end of its own length, so when two
#: consecutive actuators interpenetrate the length that separates them is the
#: child's: keyed by the child's actuator geom (leg prefix off), valued by the
#: parameter and what the page calls it. A pair not listed here is refused as
#: `draft-generate` refuses it.
STRETCH = {
    "quadruped": {"pitch_link_motor": ("hip_roll_link_length", "Hip offset")},
    "humanoid": {
        "upper_hip_link_motor": ("upper_hip_link_length", "Upper hip link"),
        "lower_hip_link_motor": ("lower_hip_link_length", "Lower hip link"),
        "upper_shoulder_link_motor": ("upper_shoulder_link_length", "Upper shoulder link"),
        "lower_shoulder_link_motor": ("lower_shoulder_link_length", "Lower shoulder link"),
        # The torso's units and the ankle's are leg-sized, so a stronger leg
        # crowds them too, and no slider reaches these. They are the lengths
        # draft.twins itself lengthens when it packs the G1 twin.
        "upper_core_link_motor": ("upper_core_link_length", "Upper torso link"),
        "lower_core_link_motor": ("lower_core_link_length", "Lower torso link"),
        "pelvis_link_motor": ("pelvis_link_length", "Pelvis link"),
        "ankle_link_motor": ("ankle_link_length", "Ankle link"),
        "lower_wrist_link_motor": ("lower_wrist_link_length", "Wrist link"),
        # Two the sliders set. A stronger arm or leg crowds these at once, so the
        # slider's value is kept as a minimum and the report says what it became.
        "forearm_link_motor": ("forearm_link_length", "Forearm"),
        "lower_leg_link_motor": ("lower_leg_link_length", "Shank"),
    },
}
#: Each round moves a link out by its overlap times this, plus the margin: the
#: overlap is measured along whichever axis separates least, not along the link,
#: so an exact correction usually needs another round.
STRETCH_OVERSHOOT = 1.3
STRETCH_MARGIN_M = 0.002
STRETCH_ROUNDS = 8

LEG_PREFIX = re.compile(r"\b(fl|fr|rl|rr|left|right)_")
#: The cheetah's joints are all on class L; [M] and [S] repeat whatever [L] says.
OTHER_CLASS = re.compile(r"^\s*-?\s*\[(M|S)\]")
_GEN: dict[str, RobotGenerator] = {}
_OUT = Path(tempfile.mkdtemp(prefix="draft_site_"))
_START: dict[str, dict] = {}


def _generator(robot: str) -> RobotGenerator:
    if robot not in _GEN:
        gen = RobotGenerator(robot_dir(robot))
        if robot == "humanoid":
            # As draft.twins does: the twin's per-joint classes replace L/M/S, which
            # would otherwise still be sized and frontier-checked with no joint on them.
            for key in [k for k in gen.params if BASE_CLASS.match(str(k))]:
                gen.params.pop(key)
        _GEN[robot] = gen
    return _GEN[robot]


def overrides(values: dict, robot: str = "quadruped") -> dict:
    """The preset with the moved sliders laid over it (see draft_design)."""
    return draft_design.overrides(values, robot, starts(robot))


def starts(robot: str) -> dict:
    """Every slider's value in the preset, read off the generated preset itself
    (several of the G1 twin's lengths are expressions)."""
    if robot not in _START:
        out = _OUT / robot
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            _generator(robot).generate(out, params_override=preset(robot))
        resolved = yaml.safe_load((out / "parameters_resolved.yaml").read_text())
        s = {}
        for a in ROBOTS[robot]["axes"]:
            if a.get("auto"):
                s[a["key"]] = None
            elif "scale" in a:
                s[a["key"]] = 1.0
            else:
                s[a["key"]] = round(float(resolved[a["param"]]), 4)
        _START[robot] = s
    return _START[robot]


def colour_index(rgba, palette) -> int:
    return int(np.argmin([np.abs(np.array(c) - rgba[:3]).sum() for c in palette]))


def _half_height(t: int, size, R) -> float:
    """How far a geom reaches below its centre along world z."""
    r = size[0]
    if t == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        return r
    if t == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        return abs(R[2, 2]) * size[1] + r
    if t == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        return abs(R[2, 2]) * size[1] + r * np.sqrt(max(0.0, 1 - R[2, 2] ** 2))
    if t == int(mujoco.mjtGeom.mjGEOM_BOX):
        return float(np.abs(R[2]) @ size)
    return 0.0


def palette(robot: str) -> list:
    return [list(c) for c in dict.fromkeys(ROBOTS[robot]["colours"].values())]


def geometry(xml: Path, robot: str) -> list:
    """Every visible geom in the robot's pose, lowest point on z = 0, as
    [type, size, pos, xmat, colour(, mesh)] with MuJoCo's type codes and colour an
    index into the robot's palette."""
    pose, pal = ROBOTS[robot]["pose"], palette(robot)
    # Not from_xml_path: MuJoCo caches assets by path, and every design rewrites
    # the torso mesh at the same one.
    m = load_model(xml)
    d = mujoco.MjData(m)
    qpos = np.array(d.qpos, dtype=float)
    adr = np.ravel(m.jnt_qposadr)
    for j in range(int(m.njnt)):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
        for key, angle in pose.items():
            if name == key or name.endswith("_" + key):
                qpos[int(adr[j])] = angle
    d.qpos = qpos
    mujoco.mj_kinematics(m, d)

    types = np.ravel(m.geom_type)
    rgba = np.asarray(m.geom_rgba).reshape(-1, 4)
    size = np.asarray(m.geom_size).reshape(-1, 3)
    xpos = np.asarray(d.geom_xpos).reshape(-1, 3)
    xmat = np.asarray(d.geom_xmat).reshape(-1, 9)
    out, lowest = [], np.inf
    for g in range(int(m.ngeom)):
        t = int(types[g])
        if t == int(mujoco.mjtGeom.mjGEOM_PLANE) or rgba[g][3] == 0:
            continue
        entry = [t, np.round(size[g], 4).tolist(), xpos[g].copy(), np.round(xmat[g], 4).tolist(),
                 colour_index(rgba[g], pal)]
        if t == int(mujoco.mjtGeom.mjGEOM_MESH):
            mid = int(np.ravel(m.geom_dataid)[g])
            va, vn = int(np.ravel(m.mesh_vertadr)[mid]), int(np.ravel(m.mesh_vertnum)[mid])
            fa, fn = int(np.ravel(m.mesh_faceadr)[mid]), int(np.ravel(m.mesh_facenum)[mid])
            verts = np.asarray(m.mesh_vert).reshape(-1, 3)[va:va + vn]
            faces = np.asarray(m.mesh_face).reshape(-1, 3)[fa:fa + fn]
            entry.append({"v": np.round(verts, 4).ravel().tolist(), "f": faces.ravel().tolist()})
            z = (xmat[g].reshape(3, 3) @ verts.T)[2] + xpos[g][2]
            lowest = min(lowest, float(z.min()))
        else:
            lowest = min(lowest, xpos[g][2] - _half_height(t, size[g], xmat[g].reshape(3, 3)))
        out.append(entry)
    for entry in out:
        entry[2] = np.round(entry[2] - [0, 0, lowest], 4).tolist()
    return out


def collapse_legs(messages: list[str]) -> list[str]:
    """The same message for all four legs, or both sides, said once."""
    seen, out = set(), []
    for msg in messages:
        key = LEG_PREFIX.sub("", msg)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def summary(report: dict, robot: str = "quadruped") -> dict:
    """The part of feasibility_report.yaml the page shows: what the model is
    made of, and how it fares against the checks."""
    checks = [{"quantity": c["quantity"], "value": c["value"], "status": c["status"],
               "lo": c.get("p10", c.get("lo_2sigma")), "hi": c.get("p90", c.get("hi_2sigma")),
               "extrapolated": c.get("extrapolated", False)}
              for c in report.get("mass_composition_check", {}).get("checks", [])]
    single = robot == "quadruped"
    classes = report["actuator_checks"]["classes"]
    shown = ["L"] if single else sorted(classes)
    return {
        # Where each of the design's actuators lands on the four fits.
        "actuators": [draft_design.actuator_point(
            c, classes[c]["effort_Nm"], classes[c]["velocity_rad_s"], classes[c]["gear_ratio"],
            {"mass": classes[c]["mass_kg"], "r": classes[c]["radius_m"],
             "volume": classes[c]["volume_cm3"] / 1e6, "J_rotor": classes[c]["rotor_inertia_kgm2"]})
            for c in shown if c in classes],
        "total_mass_kg": round(report["total_mass_kg"], 2),
        "mass_kg": {k: round(v, 2) for k, v in report["mass_kg"].items()},
        "checks": checks,
        "warnings": collapse_legs([w for w in report["actuator_checks"]["warnings"]
                                   if not (single and OTHER_CLASS.match(w))
                                   and "no gear ratio declared" not in w]
                                  + report.get("segment_densities", {}).get("warnings", [])),
    }


def _stretch(gen: RobotGenerator, out: Path, ov: dict, robot: str) -> tuple[dict, list, bool]:
    """Lengthen the links in STRETCH until no actuator pair they separate collides.

    Each round generates with overlaps allowed, which is how the report comes to
    list the colliding pairs and their depths, and moves each mapped link out by
    its pair's depth plus a margin. Returns the overrides to generate for real,
    what was stretched, and whether the last round collided nowhere at all, in
    which case what it wrote IS the real generation (allowing an overlap that
    does not occur changes nothing). A pair this cannot resolve is left for the
    real generation, which refuses it.
    """
    table = STRETCH.get(robot, {})
    if not table:
        return ov, [], False
    base = gen.params
    start: dict = {}
    clean = False
    for _ in range(STRETCH_ROUNDS):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            gen.generate(out, params_override={**ov, "allow_motor_overlap": True})
        pairs = yaml.safe_load((out / "feasibility_report.yaml").read_text())["actuator_packing"]["pairs"]
        clean = not pairs
        grow: dict = {}
        for pair in pairs:
            hit = table.get(LEG_PREFIX.sub("", pair["b"]))
            if hit:
                need = STRETCH_OVERSHOOT * pair["depth_m"] + STRETCH_MARGIN_M
                grow[hit[0]] = max(grow.get(hit[0], 0.0), need)
        if not grow:
            break
        for param, by in grow.items():
            now = float(ov.get(param, base.get(param)))
            start.setdefault(param, now)
            ov = {**ov, param: round(now + by, 4)}
    labels = {param: label for param, label in table.values()}
    return ov, [{"param": labels[p], "from": round(v, 3), "to": round(float(ov[p]), 3)}
                for p, v in start.items()], clean


def generate(values: dict, robot: str = "quadruped") -> dict:
    """One design: {"geoms", "report"} or {"refused": [reasons]}."""
    gen = _generator(robot)
    out = _OUT / robot
    single = robot == "quadruped"
    try:
        try:
            ov, stretched, clean = _stretch(gen, out, overrides(values, robot), robot)
            if not clean:
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    gen.generate(out, params_override=ov)
        except FeasibilityViolation as exc:
            lines = [ln.strip().lstrip("- ").strip() for ln in str(exc).split("\n")]
            refused = collapse_legs([ln for ln in lines[1:]
                                     if ln and not (single and OTHER_CLASS.match(ln))])
            return {"refused": refused}
        report = yaml.safe_load((out / "feasibility_report.yaml").read_text())
        return {"geoms": geometry(out / f"{robot}.xml", robot),
                "report": {**summary(report, robot), "stretched": stretched}}
    finally:
        # In the browser, MuJoCo's models live on the WebAssembly heap until freed.
        if hasattr(mujoco, "release_all"):
            mujoco.release_all()


def generate_json(values_json: str, robot: str = "quadruped") -> str:
    """The browser's entry point: JSON in, JSON out."""
    return json.dumps(generate(json.loads(values_json), robot), separators=(",", ":"))


def fits() -> dict:
    """The actuator catalogue on the four fits the supplementary video shows, and
    the fitted trends through them, read from Draft's own trend data (the paper's
    Fig. 3): (a) reduction against no-load speed, (b) mass against peak torque,
    (c) volume against peak torque, (d) rotor inertia against housing radius."""
    import draft.trends
    from draft.trends import motor_fits
    data = Path(draft.trends.__file__).parent / "data"
    cat = json.loads((data / "actuator_catalog.json").read_text())
    trends = json.loads((data / "actuator_trends.json").read_text())
    laws = motor_fits()
    geared = [r for r in cat if r.get("gear")]

    def pts(rows, x, y):
        return [[round(x(r), 6), round(y(r), 9), r["category"]] for r in rows]

    return {
        "points": {
            "a": pts([r for r in geared if r.get("omega_NL_rad_s")],
                     lambda r: r["omega_NL_rad_s"], lambda r: r["gear"]),
            "b": pts([r for r in geared if r.get("mass_kg")],
                     lambda r: r["tau_peak_Nm"], lambda r: r["mass_kg"]),
            "c": pts([r for r in geared if r.get("volume_cm3")],
                     lambda r: r["tau_peak_Nm"], lambda r: r["volume_cm3"]),
            "d": pts([r for r in cat if r.get("rotor_inertia_kgm2") and r.get("radius_mm")
                      and not r.get("rotor_is_geared_inrunner")],
                     lambda r: r["radius_mm"] / 1000, lambda r: r["rotor_inertia_kgm2"]),
        },
        "laws": {
            # N from omega: the speed trend omega = c N^e, inverted.
            "a": {"coef": laws.speed_coef ** (-1 / laws.speed_exp), "exp": 1 / laws.speed_exp},
            "b": {"coef": trends["mass_trend"]["coef"], "exp": trends["mass_trend"]["tau_exp"]},
            # V = c tau^a N^b (cm3): one line per reduction, drawn at each actuator's own.
            "c": {"coef": trends["geometry_trend"]["volume_coef_m3"] * 1e6,
                  "exp": trends["geometry_trend"]["volume_tau_exp"],
                  "gear_exp": trends["geometry_trend"]["volume_gear_exp"]},
            "d": {"coef": trends["inertia_trend"]["k_areal_kg_m2"], "exp": 4},
        },
    }


def axes_json() -> str:
    """Each robot's sliders, with the preset's values as their starts, and palette."""
    robots = {}
    for key, r in ROBOTS.items():
        start = starts(key)
        axes = []
        for a in r["axes"]:
            affected = draft_design.affects(key, a)
            a = {k: v for k, v in a.items() if k not in ("param", "scale", "classes")}
            a["start"] = start[a["key"]]
            a["affects"] = affected
            if a["start"] is not None:          # the preset may sit outside the range
                a["min"], a["max"] = min(a["min"], a["start"]), max(a["max"], a["start"])
            axes.append(a)
        robots[key] = {"label": r["label"], "axes": axes, "palette": palette(key),
                       "preset": {"cheetah": "the lineup's cheetah",
                                  "g1_twin": "the Unitree G1 twin"}[r["preset"]]}
    return json.dumps({"robots": robots, "fits": fits()})
