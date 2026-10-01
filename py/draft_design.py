"""The builder's robots, as parameters: presets, sliders, and the actuators they drive.

The light half of the project site's builder, split from `draft_site.py` so it can
run without the generator: it imports nothing but Draft's parameter files and its
fitted actuator trends (`draft.trends`), no MuJoCo, numpy or trimesh. That is what
lets the page's second worker move the actuator-fit plots while a slider is still
being dragged, in a few milliseconds a step, while the first worker is busy
generating the whole robot.

`actuator_points` is the generator's own first step (`resolve_catalog_motors`, then
`motor_solve.design_class` per class, as `size_actuators` does) and nothing after
it, so a point it plots is the point the finished design reports.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

from draft.robots import robot_dir
from draft.trends.motor_catalog import resolve_catalog_motors
from draft.trends.motor_solve import design_class

HERE = Path(__file__).resolve().parent

#: Seaborn's "deep" palette.
DEEP = {"blue": (0.298, 0.447, 0.690), "orange": (0.867, 0.518, 0.322),
        "green": (0.333, 0.659, 0.408), "red": (0.769, 0.306, 0.322),
        "purple": (0.506, 0.447, 0.702), "grey": (0.549, 0.549, 0.549)}


def _mix(c, other, k):
    return tuple(round(a + (b - a) * k, 3) for a, b in zip(c, other))


def _scheme(base) -> dict:
    """A robot drawn in one hue: the trunk in it, links a tint, actuators a shade."""
    white, black = (1, 1, 1), (0, 0, 0)
    return {"link_color": _mix(base, white, 0.45), "motor_color": _mix(base, black, 0.45),
            "mot_detail_color": _mix(base, black, 0.68), "torso_color": base,
            "foot_color": _mix(DEEP["grey"], black, 0.35)}


#: Each builder robot: its sliders, the pose it is drawn in, its colours.
#: A slider sets `param` directly, or scales the effort or velocity of the actuator
#: classes in `classes` (`scale`); `auto` sliders may be None: as the preset has it.
ROBOTS = {
    "quadruped": {
        "label": "Quadruped", "preset": "cheetah",
        "axes": [
            {"key": "thigh", "label": "Thigh length", "unit": "m", "param": "upper_leg_link_length", "min": 0.15, "max": 0.75, "step": 0.01},
            {"key": "shank", "label": "Shank length", "unit": "m", "param": "lower_leg_link_length", "min": 0.15, "max": 0.75, "step": 0.01},
            {"key": "hip", "label": "Hip offset", "unit": "m", "param": "hip_roll_link_length", "min": 0.06, "max": 0.30, "step": 0.01},
            {"key": "length", "label": "Torso length", "unit": "m", "param": "torso_lx", "min": 0.35, "max": 1.00, "step": 0.01},
            {"key": "width", "label": "Torso width", "unit": "m", "param": "torso_ly", "min": 0.15, "max": 0.60, "step": 0.01},
            {"key": "torque", "label": "Actuator peak torque", "unit": "N·m", "param": "L_motor_effort", "min": 20, "max": 500, "step": 1, "log": True},
            {"key": "speed", "label": "Actuator no-load speed", "unit": "rad/s", "param": "L_motor_velocity", "min": 3, "max": 50, "step": 0.5, "log": True},
            {"key": "aspect", "label": "Actuator aspect ratio L/D", "unit": "", "param": "L_motor_aspect", "min": 0.20, "max": 2.00, "step": 0.01, "auto": True},
        ],
        #: Standing pose, radians; a key matches a joint by its name or its suffix.
        "pose": {"hip_pitch": 0.55, "knee": -1.1},
        "colours": _scheme(DEEP["blue"]),
    },
    "humanoid": {
        "label": "Humanoid", "preset": "g1_twin",
        "axes": [
            {"key": "thigh", "label": "Thigh length", "unit": "m", "param": "upper_leg_link_length", "min": 0.10, "max": 0.50, "step": 0.01},
            {"key": "shank", "label": "Shank length", "unit": "m", "param": "lower_leg_link_length", "min": 0.10, "max": 0.50, "step": 0.01},
            {"key": "torso", "label": "Torso length", "unit": "m", "param": "torso_length", "min": 0.25, "max": 0.80, "step": 0.01},
            {"key": "shoulders", "label": "Shoulder width", "unit": "m", "param": "shoulder_width", "min": 0.12, "max": 0.45, "step": 0.01},
            {"key": "upper_arm", "label": "Upper arm length", "unit": "m", "param": "bicep_link_length", "min": 0.04, "max": 0.30, "step": 0.01},
            {"key": "forearm", "label": "Forearm length", "unit": "m", "param": "forearm_link_length", "min": 0.06, "max": 0.30, "step": 0.01},
            {"key": "leg_torque", "label": "Leg actuator torque", "unit": "×", "scale": "effort", "classes": "legs", "min": 0.3, "max": 3.0, "step": 0.01, "log": True},
            {"key": "leg_speed", "label": "Leg actuator speed", "unit": "×", "scale": "velocity", "classes": "legs", "min": 0.3, "max": 3.0, "step": 0.01, "log": True},
            {"key": "arm_torque", "label": "Arm actuator torque", "unit": "×", "scale": "effort", "classes": "arms", "min": 0.3, "max": 3.0, "step": 0.01, "log": True},
            {"key": "arm_speed", "label": "Arm actuator speed", "unit": "×", "scale": "velocity", "classes": "arms", "min": 0.3, "max": 3.0, "step": 0.01, "log": True},
            {"key": "aspect", "label": "Leg actuator aspect ratio L/D", "unit": "", "scale": "aspect", "classes": "legs", "min": 0.20, "max": 2.00, "step": 0.01, "auto": True},
        ],
        #: The G1 twin's actuator classes, by the joints they drive.
        "classes": {"legs": ["HP", "HR", "HY", "KN", "AP", "AR", "TR", "TP", "TY"],
                    "arms": ["SP", "SR", "SY", "EL"]},
        # The supplementary video's display pose (scripts/icra_video2/common.py):
        # at zero the hands sit inside the pelvis.
        "pose": {"left_elbow": -0.85, "right_elbow": -0.85,
                 "left_shoulder_roll": 0.28, "right_shoulder_roll": -0.28,
                 "left_shoulder_pitch": 0.15, "right_shoulder_pitch": 0.15},
        "colours": {**_scheme(DEEP["purple"]),
                    "head_sphere_color": DEEP["purple"],
                    "hand_sphere_color": _mix(DEEP["purple"], (1, 1, 1), 0.45),
                    "left_eye_color": (0.12, 0.12, 0.14), "right_eye_color": (0.12, 0.12, 0.14)},
    },
}
#: The quadruped's hip spacing as a share of its trunk's width, as the cheetah
#: states it; the width slider carries it along.
SHOULDER_OVER_WIDTH = 0.16 / 0.20

#: The humanoid's own L/M/S classes, which the G1 twin replaces with its own.
BASE_CLASS = re.compile(r"^_?[LMS]_motor")
MOVED = 1e-9


def _preset_file(name: str) -> Path:
    """Beside this module in the browser bundle; in a checkout, the cheetah is
    read from experiments/ so there is one copy of it."""
    local = HERE / "presets" / f"{name}.yaml"
    if local.exists():
        return local
    from draft.paths import repo_root
    return repo_root() / "experiments" / "quadruped_variants" / f"{name}.yaml"


def preset(robot: str) -> dict:
    """The robot's starting design as parameter overrides, in the site's colours."""
    cfg = ROBOTS[robot]
    ov = {k: v for k, v in yaml.safe_load(_preset_file(cfg["preset"]).read_text()).items()
          if not k.endswith("_color")}
    ov.update({k: " ".join(f"{x:g}" for x in (*c, 1)) for k, c in cfg["colours"].items()})
    ov["allow_hypothetical"] = False
    return ov



def _classes(robot: str, group: str) -> list[str]:
    return ROBOTS[robot]["classes"][group]



def overrides(values: dict, robot: str, start: dict) -> dict:
    """The preset, with every slider that has moved away from it laid over it.
    `start` is each slider's value in the preset (`draft_site.starts`)."""
    ov = preset(robot)
    moved = {k for k, v in values.items()
             if (v is None) != (start.get(k) is None)
             or (v is not None and abs(float(v) - float(start[k])) > MOVED)}
    for a in ROBOTS[robot]["axes"]:
        key = a["key"]
        if key not in moved:
            continue
        v = values[key]
        if "param" in a:
            ov[a["param"]] = None if v is None else float(v)
            if robot == "quadruped" and a["param"].startswith("L_motor"):
                ov["L_motor"] = None                  # no longer the catalogue part
                if a["param"] == "L_motor_velocity":
                    ov["L_motor_gear"] = None         # the speed trend sets it
            if robot == "quadruped" and key == "width":
                ov["shoulder_width"] = round(SHOULDER_OVER_WIDTH * float(v), 4)
        else:
            for cls in _classes(robot, a["classes"]):
                if a["scale"] == "aspect":
                    ov[f"{cls}_motor_aspect"] = None if v is None else float(v)
                    continue
                field = f"{cls}_motor_{a['scale']}"
                ov[field] = round(float(ov[field]) * float(v), 4)
                if a["scale"] == "velocity":
                    ov[f"{cls}_motor_gear"] = None
    return ov




def shown_classes(robot: str, params: dict) -> list[str]:
    """The actuator classes that drive a joint: the cheetah's single class L (its
    M and S are sized but drive nothing), or every class of the G1 twin."""
    if robot == "quadruped":
        return ["L"]
    return sorted({m.group(1) for k in params for m in [CLASS_EFFORT.match(str(k))] if m})


CLASS_EFFORT = re.compile(r"^([A-Z]{1,3})_motor_effort$")


def base_params(robot: str) -> dict:
    """The robot's own parameters.yaml, as the generator loads it; the humanoid
    without the L/M/S classes the twin replaces."""
    params = yaml.safe_load((robot_dir(robot) / "parameters.yaml").read_text()) or {}
    if robot == "humanoid":
        params = {k: v for k, v in params.items() if not BASE_CLASS.match(str(k))}
    return params


def actuator_point(cls: str, tau, omega, gear, sized) -> dict:
    """One class on the four fits: what the page plots and hovers."""
    # Rounded as feasibility_report.yaml rounds them, so both paths plot one point.
    return {"cls": cls, "tau": round(tau, 4), "omega": round(omega, 6), "N": round(gear, 3),
            "m": round(sized["mass"], 6), "r": round(sized["r"], 6),
            "V": round(sized["volume"] * 1e6, 2), "J": round(sized["J_rotor"], 12)}


def actuator_points(values: dict, robot: str, start: dict) -> list[dict]:
    """Where the design's actuators land on the fits, without generating it."""
    params = {**base_params(robot), **overrides(values, robot, start)}
    params = resolve_catalog_motors(params)
    out = []
    for cls in shown_classes(robot, params):
        d = design_class(cls, params)
        out.append(actuator_point(cls, d.point.tau, d.point.omega, d.point.gear, d.sized))
    return out


def affects(robot: str, axis: dict) -> list[str]:
    """The actuator classes a slider moves, for the page to highlight."""
    if "classes" in axis:
        return list(ROBOTS[robot]["classes"][axis["classes"]])
    if axis.get("param", "").startswith("L_motor"):
        return ["L"]
    return []


def points_json(values_json: str, robot: str, start_json: str) -> str:
    """The page's fits worker's entry point: JSON in, JSON out."""
    import json
    return json.dumps(actuator_points(json.loads(values_json), robot, json.loads(start_json)),
                      separators=(",", ":"))
