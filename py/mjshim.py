"""The part of the `mujoco` Python API that Draft's generator uses, for a browser.

The project site runs Draft's own generator in Pyodide, and MuJoCo's Python
bindings do not exist there. What does is MuJoCo's official WebAssembly build
(`@mujoco/mujoco`), whose well-tested entry point is `MjModel.from_xml_string`.
So this module stands in for `mujoco` in `sys.modules`:

* `MjSpec` is a small pure-Python tree. It records what the generator builds
  (bodies, geoms, joints, sites, defaults, sensors, excludes, numerics, one mesh)
  and writes it out as MJCF at full precision. It follows the one MjSpec rule
  that changes the compiled model: an element takes its attribute values from
  the default in force WHEN IT IS ADDED, so assigning a class afterwards records
  the class name without re-applying that class's values.
* Compiling hands that XML to a backend: the WebAssembly MuJoCo in a browser, or
  the real `mujoco` package under CPython, which is how `scripts/site/check_mjshim.py`
  proves the two paths build the same robot.
* `MjModel` and `MjData` expose the fields the generator and the site read as
  numpy arrays shaped the way the real bindings shape them.

Anything the generator touches that is not implemented raises, rather than
emitting a model that quietly differs.
"""
from __future__ import annotations

import enum
import math
import sys
from pathlib import Path

import numpy as np

__version__ = "draft-mjshim"

# ── Enums (values from mujoco 3.x, which have not changed across 3.x) ──────────


class mjtGeom(enum.IntEnum):
    mjGEOM_PLANE = 0
    mjGEOM_HFIELD = 1
    mjGEOM_SPHERE = 2
    mjGEOM_CAPSULE = 3
    mjGEOM_ELLIPSOID = 4
    mjGEOM_CYLINDER = 5
    mjGEOM_BOX = 6
    mjGEOM_MESH = 7
    mjGEOM_SDF = 8


class mjtJoint(enum.IntEnum):
    mjJNT_FREE = 0
    mjJNT_BALL = 1
    mjJNT_SLIDE = 2
    mjJNT_HINGE = 3


class mjtLimited(enum.IntEnum):
    mjLIMITED_FALSE = 0
    mjLIMITED_TRUE = 1
    mjLIMITED_AUTO = 2


class mjtObj(enum.IntEnum):
    mjOBJ_UNKNOWN = 0
    mjOBJ_BODY = 1
    mjOBJ_XBODY = 2
    mjOBJ_JOINT = 3
    mjOBJ_DOF = 4
    mjOBJ_GEOM = 5
    mjOBJ_SITE = 6
    mjOBJ_CAMERA = 7
    mjOBJ_LIGHT = 8
    mjOBJ_FLEX = 9
    mjOBJ_MESH = 10
    mjOBJ_SKIN = 11
    mjOBJ_HFIELD = 12
    mjOBJ_TEXTURE = 13
    mjOBJ_MATERIAL = 14
    mjOBJ_PAIR = 15
    mjOBJ_EXCLUDE = 16
    mjOBJ_EQUALITY = 17
    mjOBJ_TENDON = 18
    mjOBJ_ACTUATOR = 19
    mjOBJ_SENSOR = 20
    mjOBJ_NUMERIC = 21


class mjtOrientation(enum.IntEnum):
    mjORIENTATION_QUAT = 0
    mjORIENTATION_AXISANGLE = 1
    mjORIENTATION_XYAXES = 2
    mjORIENTATION_ZAXIS = 3
    mjORIENTATION_EULER = 4


class mjtSensor(enum.IntEnum):
    mjSENS_TOUCH = 0
    mjSENS_ACCELEROMETER = 1
    mjSENS_VELOCIMETER = 2
    mjSENS_GYRO = 3
    mjSENS_FORCE = 4
    mjSENS_TORQUE = 5
    mjSENS_JOINTPOS = 9
    mjSENS_JOINTVEL = 10
    mjSENS_FRAMEPOS = 26
    mjSENS_FRAMEQUAT = 27
    mjSENS_FRAMELINVEL = 31
    mjSENS_FRAMEANGVEL = 32
    mjSENS_SUBTREECOM = 35
    mjSENS_SUBTREELINVEL = 36
    mjSENS_SUBTREEANGMOM = 37


class mjtSolver(enum.IntEnum):
    mjSOL_PGS = 0
    mjSOL_CG = 1
    mjSOL_NEWTON = 2


# ── Formatting ────────────────────────────────────────────────────────────────

def _num(v) -> str:
    v = float(v)
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def _vec(v) -> str:
    return " ".join(_num(x) for x in np.ravel(v))


def _bool(v) -> str:
    return "true" if v else "false"


def _limited(v) -> str:
    return {0: "false", 1: "true", 2: "auto"}[int(v)]


_GEOM_NAME = {mjtGeom.mjGEOM_PLANE: "plane", mjtGeom.mjGEOM_HFIELD: "hfield",
              mjtGeom.mjGEOM_SPHERE: "sphere", mjtGeom.mjGEOM_CAPSULE: "capsule",
              mjtGeom.mjGEOM_ELLIPSOID: "ellipsoid", mjtGeom.mjGEOM_CYLINDER: "cylinder",
              mjtGeom.mjGEOM_BOX: "box", mjtGeom.mjGEOM_MESH: "mesh"}
_JOINT_NAME = {mjtJoint.mjJNT_FREE: "free", mjtJoint.mjJNT_BALL: "ball",
               mjtJoint.mjJNT_SLIDE: "slide", mjtJoint.mjJNT_HINGE: "hinge"}
_SOLVER_NAME = {0: "PGS", 1: "CG", 2: "Newton"}
_SENSOR_TAG = {
    mjtSensor.mjSENS_ACCELEROMETER: "accelerometer", mjtSensor.mjSENS_VELOCIMETER: "velocimeter",
    mjtSensor.mjSENS_GYRO: "gyro", mjtSensor.mjSENS_FORCE: "force", mjtSensor.mjSENS_TORQUE: "torque",
    mjtSensor.mjSENS_JOINTPOS: "jointpos", mjtSensor.mjSENS_JOINTVEL: "jointvel",
    mjtSensor.mjSENS_FRAMEPOS: "framepos", mjtSensor.mjSENS_FRAMEQUAT: "framequat",
    mjtSensor.mjSENS_FRAMELINVEL: "framelinvel", mjtSensor.mjSENS_FRAMEANGVEL: "frameangvel",
    mjtSensor.mjSENS_SUBTREECOM: "subtreecom", mjtSensor.mjSENS_SUBTREELINVEL: "subtreelinvel",
    mjtSensor.mjSENS_SUBTREEANGMOM: "subtreeangmom",
}
_OBJ_NAME = {mjtObj.mjOBJ_BODY: "body", mjtObj.mjOBJ_XBODY: "xbody", mjtObj.mjOBJ_GEOM: "geom",
             mjtObj.mjOBJ_SITE: "site", mjtObj.mjOBJ_JOINT: "joint"}


# ── Attribute sets ────────────────────────────────────────────────────────────

class _Attrs:
    """An element's attributes. Only the names in FIELDS may be set; each maps
    to its MJCF attribute and a formatter."""

    FIELDS: dict = {}

    def __init__(self, **kw):
        object.__setattr__(self, "_v", {})
        for k, v in kw.items():
            setattr(self, k, v)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        if name not in self.FIELDS:
            raise AttributeError(f"mjshim: {type(self).__name__}.{name} is not implemented")
        self._v[name] = value

    def __getattr__(self, name):
        v = object.__getattribute__(self, "_v")
        if name in v:
            return v[name]
        if name in type(self).FIELDS:
            return None
        raise AttributeError(f"mjshim: {type(self).__name__}.{name} is not implemented")

    def _xml_attrs(self, keys=None) -> list[str]:
        out = []
        for k in (keys if keys is not None else self._v):
            if k not in self._v or self._v[k] is None:
                continue
            xml, fmt = self.FIELDS[k]
            out.append(f'{xml}="{fmt(self._v[k])}"')
        return out


_perdof = lambda v: _num(np.ravel(v)[0])  # noqa: E731  — hinge damping is slot 0


class _GeomDefaults(_Attrs):
    FIELDS = {"contype": ("contype", _num), "conaffinity": ("conaffinity", _num),
              "condim": ("condim", _num), "friction": ("friction", _vec),
              "solimp": ("solimp", _vec), "solref": ("solref", _vec), "group": ("group", _num),
              "density": ("density", _num), "rgba": ("rgba", _vec)}


class _JointDefaults(_Attrs):
    FIELDS = {"damping": ("damping", _perdof), "armature": ("armature", _num),
              "frictionloss": ("frictionloss", _num)}


class MjsDefault:
    def __init__(self, name: str, parent: "MjsDefault | None"):
        self.name = name
        self.parent = parent
        self.children: list[MjsDefault] = []
        self.geom = _GeomDefaults()
        self.joint = _JointDefaults()

    def resolved(self, which: str) -> dict:
        """This class's values with its ancestors' filled in."""
        base = self.parent.resolved(which) if self.parent else {}
        return {**base, **getattr(self, which)._v}

    def _xml(self, ind: str) -> list[str]:
        head = f'{ind}<default class="{self.name}">' if self.parent else f"{ind}<default>"
        lines = [head]
        for tag, attrs in (("joint", self.joint), ("geom", self.geom)):
            a = attrs._xml_attrs()
            if a:
                lines.append(f"{ind}  <{tag} {' '.join(a)}/>")
        for c in self.children:
            lines += c._xml(ind + "  ")
        lines.append(f"{ind}</default>")
        return lines


class _Alt(_Attrs):
    FIELDS = {"type": ("", None), "euler": ("euler", _vec)}


class MjsGeom(_Attrs):
    FIELDS = {"name": ("name", str), "type": ("type", lambda t: _GEOM_NAME[mjtGeom(int(t))]),
              "size": ("size", _vec), "pos": ("pos", _vec), "quat": ("quat", _vec),
              "rgba": ("rgba", _vec), "density": ("density", _num),
              "meshname": ("mesh", str), "contype": ("contype", _num),
              "conaffinity": ("conaffinity", _num), "condim": ("condim", _num),
              "friction": ("friction", _vec), "solimp": ("solimp", _vec),
              "solref": ("solref", _vec), "group": ("group", _num), "mass": ("mass", _num),
              "fromto": ("fromto", _vec), "classname": ("class", lambda c: getattr(c, "name", c))}

    def __init__(self, default: MjsDefault, **kw):
        # MjSpec copies the default in force when the element is added; a class
        # assigned later names the class but does not re-apply its values.
        object.__setattr__(self, "_created_from", default.resolved("geom"))
        object.__setattr__(self, "alt", _Alt())
        super().__init__(**kw)

    def _xml(self, spec: "MjSpec") -> str:
        attrs = self._xml_attrs(["name", "type", "size", "pos", "quat"])
        if self.alt._v.get("type") == mjtOrientation.mjORIENTATION_EULER:
            attrs.append(f'euler="{_vec(self.alt.euler)}"')
        cls = self._v.get("classname")
        if cls is not None:
            attrs.append(f'class="{getattr(cls, "name", cls)}"')
            # Pin every value the class would otherwise supply to what this geom
            # actually carries, so the compiled geom is the one MjSpec builds.
            klass = spec._find_default(getattr(cls, "name", cls)).resolved("geom")
            for k in klass:
                if k not in self._v:
                    v = self._created_from.get(k, _GEOM_BUILTIN.get(k))
                    if v is not None:
                        xml, fmt = _GeomDefaults.FIELDS[k]
                        attrs.append(f'{xml}="{fmt(v)}"')
        attrs += self._xml_attrs([k for k in self._v
                                  if k not in ("name", "type", "size", "pos", "quat", "classname")])
        return f"<geom {' '.join(attrs)}/>"


#: MuJoCo's own geom defaults, for a value a class overrides that neither the
#: geom nor the default it was created under stated.
_GEOM_BUILTIN = {"contype": 1, "conaffinity": 1, "condim": 3, "group": 0,
                 "friction": [1, 0.005, 0.0001], "density": 1000}


class MjsJoint(_Attrs):
    FIELDS = {"name": ("name", str), "type": ("type", lambda t: _JOINT_NAME[mjtJoint(int(t))]),
              "axis": ("axis", _vec), "pos": ("pos", _vec), "range": ("range", _vec),
              "limited": ("limited", _limited), "actfrclimited": ("actuatorfrclimited", _limited),
              "actfrcrange": ("actuatorfrcrange", _vec), "armature": ("armature", _num),
              "damping": ("damping", _perdof), "frictionloss": ("frictionloss", _num),
              "stiffness": ("stiffness", _perdof), "ref": ("ref", _num)}

    def _xml(self) -> str:
        return f"<joint {' '.join(self._xml_attrs())}/>"


class MjsSite(_Attrs):
    FIELDS = {"name": ("name", str), "pos": ("pos", _vec), "group": ("group", _num),
              "size": ("size", _vec), "rgba": ("rgba", _vec)}

    def _xml(self) -> str:
        return f"<site {' '.join(self._xml_attrs())}/>"


class MjsBody:
    def __init__(self, spec: "MjSpec", name: str = "", pos=None):
        self._spec = spec
        self.name = name
        self.pos = pos
        self.alt = _Alt()
        self.bodies: list[MjsBody] = []
        self.geoms: list[MjsGeom] = []
        self.joints: list = []
        self.sites: list[MjsSite] = []

    def add_body(self, name: str = "", pos=None, **kw) -> "MjsBody":
        if kw:
            raise AttributeError(f"mjshim: add_body({', '.join(kw)}) is not implemented")
        b = MjsBody(self._spec, name, pos)
        self.bodies.append(b)
        return b

    def add_freejoint(self, name: str = "") -> MjsJoint:
        # MjSpec writes a free joint with damping 0, not the default's.
        j = MjsJoint(name=name, type=mjtJoint.mjJNT_FREE, damping=[0.0, 0.0, 0.0])
        self.joints.append(j)
        return j

    def add_joint(self, **kw) -> MjsJoint:
        j = MjsJoint(**kw)
        self.joints.append(j)
        return j

    def add_geom(self, **kw) -> MjsGeom:
        g = MjsGeom(self._spec.default, **kw)
        self.geoms.append(g)
        return g

    def add_site(self, **kw) -> MjsSite:
        s = MjsSite(**kw)
        self.sites.append(s)
        return s

    def _xml(self, ind: str) -> list[str]:
        attrs = [f'name="{self.name}"'] if self.name else []
        if self.pos is not None:
            attrs.append(f'pos="{_vec(self.pos)}"')
        if self.alt._v.get("type") == mjtOrientation.mjORIENTATION_EULER:
            attrs.append(f'euler="{_vec(self.alt.euler)}"')
        lines = [f"{ind}<body {' '.join(attrs)}>"]
        inner = ind + "  "
        lines += [inner + j._xml() for j in self.joints]
        lines += [inner + g._xml(self._spec) for g in self.geoms]
        lines += [inner + s._xml() for s in self.sites]
        for b in self.bodies:
            lines += b._xml(inner)
        lines.append(f"{ind}</body>")
        return lines


class _Compiler(_Attrs):
    FIELDS = {"degree": ("angle", lambda d: "degree" if d else "radian"),
              "autolimits": ("autolimits", _bool)}


class _Option(_Attrs):
    FIELDS = {"timestep": ("timestep", _num), "iterations": ("iterations", _num),
              "solver": ("solver", lambda s: _SOLVER_NAME[int(s)]),
              "tolerance": ("tolerance", _num), "gravity": ("gravity", _vec)}


class MjSpec:
    def __init__(self):
        self.modelname = "model"
        self.compiler = _Compiler(degree=True, autolimits=True)
        self.option = _Option()
        self.default = MjsDefault("main", None)
        self.worldbody = MjsBody(self, "world")
        self.assets: dict[str, bytes] = {}
        self._meshes: list[tuple[str, str]] = []
        self._excludes: list[tuple[str, str, str]] = []
        self._numerics: list[tuple[str, list]] = []
        self._sensors: list[str] = []

    # MjSpec.from_file / attach (modules) are not needed by the quadruped.

    def add_default(self, name: str, parent: MjsDefault) -> MjsDefault:
        d = MjsDefault(name, parent)
        parent.children.append(d)
        return d

    def _find_default(self, name: str) -> MjsDefault:
        stack = [self.default]
        while stack:
            d = stack.pop()
            if d.name == name:
                return d
            stack += d.children
        raise KeyError(f"mjshim: no default class {name!r}")

    def add_mesh(self, name: str, file: str) -> None:
        self._meshes.append((name, file))

    def add_exclude(self, name: str = "", bodyname1: str = "", bodyname2: str = "") -> None:
        self._excludes.append((name, bodyname1, bodyname2))

    def add_numeric(self, name: str, data) -> None:
        self._numerics.append((name, list(np.ravel(data))))

    def add_sensor(self, name: str = "", type=None, objtype=None, objname: str = "") -> None:
        tag = _SENSOR_TAG[mjtSensor(int(type))]
        obj = _OBJ_NAME[mjtObj(int(objtype))]
        if tag.startswith("frame"):
            ref = f'objtype="{obj}" objname="{objname}"'
        else:
            ref = f'{obj}="{objname}"'
        self._sensors.append(f'<{tag} name="{name}" {ref}/>')

    def _all(self, attr: str) -> list:
        out, stack = [], [self.worldbody]
        while stack:
            b = stack.pop(0)
            out += getattr(b, attr)
            stack = b.bodies + stack
        return out

    def geom(self, name: str) -> MjsGeom:
        return next(g for g in self._all("geoms") if g.name == name)

    def joint(self, name: str) -> MjsJoint:
        return next(j for j in self._all("joints") if j.name == name)

    def to_xml(self) -> str:
        L = [f'<mujoco model="{self.modelname}">',
             f"  <compiler {' '.join(self.compiler._xml_attrs())}/>"]
        opt = self.option._xml_attrs()
        if opt:
            L.append(f"  <option {' '.join(opt)}/>")
        L += self.default._xml("  ")
        if self._numerics:
            L.append("  <custom>")
            L += [f'    <numeric name="{n}" size="{len(d)}" data="{_vec(d)}"/>' for n, d in self._numerics]
            L.append("  </custom>")
        if self._meshes:
            L.append("  <asset>")
            L += [f'    <mesh name="{n}" file="{f}"/>' for n, f in self._meshes]
            L.append("  </asset>")
        L.append("  <worldbody>")
        for b in self.worldbody.bodies:
            L += b._xml("    ")
        L.append("  </worldbody>")
        if self._excludes:
            L.append("  <contact>")
            L += [f'    <exclude name="{n}" body1="{a}" body2="{b}"/>' for n, a, b in self._excludes]
            L.append("  </contact>")
        if self._sensors:
            L.append("  <sensor>")
            L += [f"    {s}" for s in self._sensors]
            L.append("  </sensor>")
        L.append("</mujoco>")
        return "\n".join(L) + "\n"

    def compile(self) -> "MjModel":
        return MjModel.from_xml_string(self.to_xml(), dict(self.assets))

    def copy(self) -> "MjSpec":
        raise AttributeError("mjshim: MjSpec.copy is not implemented (modules only)")


# ── Compiled model and data ───────────────────────────────────────────────────

#: Row width of each per-element field the generator or the site reads; the
#: WebAssembly build hands every array back flat.
_WIDTH = {"geom_size": 3, "geom_pos": 3, "geom_quat": 4, "geom_rgba": 4, "geom_xpos": 3,
          "geom_xmat": 9, "mesh_vert": 3, "mesh_face": 3, "jnt_range": 2,
          "jnt_actfrcrange": 2, "body_pos": 3, "xpos": 3, "xmat": 9, "body_inertia": 3,
          "body_ipos": 3}


class MjModel:
    def __init__(self, native):
        self._m = native

    @staticmethod
    def from_xml_string(xml: str, assets: dict | None = None) -> "MjModel":
        return MjModel(_backend().compile(xml, assets or {}))

    @staticmethod
    def from_xml_path(path: str) -> "MjModel":
        p = Path(path)
        xml = p.read_text()
        assets = {f.name: f.read_bytes() for f in p.parent.iterdir()
                  if f.suffix.lower() in (".stl", ".obj", ".msh") and f.name in xml}
        return MjModel.from_xml_string(xml, assets)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _backend().model_field(self._m, name)


class MjData:
    def __init__(self, model: MjModel):
        self._model = model
        self._d = _backend().make_data(model._m)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return _backend().data_field(self._d, name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
        else:
            _backend().set_data_field(self._d, name, value)


def mj_forward(m: MjModel, d: MjData) -> None:
    _backend().call("mj_forward", m._m, d._d)


def mj_kinematics(m: MjModel, d: MjData) -> None:
    _backend().call("mj_kinematics", m._m, d._d)


def mj_name2id(m: MjModel, objtype, name: str) -> int:
    return int(_backend().call("mj_name2id", m._m, int(objtype), name))


def mj_id2name(m: MjModel, objtype, i: int):
    return _backend().call("mj_id2name", m._m, int(objtype), int(i)) or None


def mju_mat2Quat(quat, mat) -> None:
    """Rotation matrix (9) to unit quaternion (w, x, y, z), in place."""
    R = np.asarray(mat, dtype=float).reshape(3, 3)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = math.sqrt(1.0 + R[i, i] - R[j, j] - R[k, k]) * 2
        q = [0.0] * 4
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    quat[:] = np.array(q) * (1 if q[0] >= 0 else -1)


# ── Backends ──────────────────────────────────────────────────────────────────

class _Native:
    """The real `mujoco` package, for checking the shim under CPython."""

    def __init__(self, mj):
        self.mj = mj

    def compile(self, xml, assets):
        return self.mj.MjModel.from_xml_string(xml, assets)

    def model_field(self, m, name):
        return getattr(m, name)

    def make_data(self, m):
        return self.mj.MjData(m)

    def data_field(self, d, name):
        return getattr(d, name)

    def set_data_field(self, d, name, value):
        getattr(d, name)[:] = value

    def call(self, fn, *args):
        return getattr(self.mj, fn)(*args)

    def release(self):
        pass                     # CPython frees these itself


class _Wasm:
    """MuJoCo's WebAssembly build, reached through Pyodide's `js` bridge. The
    page loads `@mujoco/mujoco` and publishes it as `globalThis.mujoco_wasm`."""

    def __init__(self):
        import js  # noqa: F401 — only exists under Pyodide
        self.mj = js.mujoco_wasm
        # Every MjModel and MjData made, until `release`. Emscripten objects live on
        # the WebAssembly heap and are never garbage-collected: without an explicit
        # delete, each slider move leaked two or three models until MuJoCo reported
        # "Could not allocate memory".
        self._live = []

    def compile(self, xml, assets):
        from pyodide.ffi import to_js
        vfs = self.mj.MjVFS.new()
        try:
            for name, data in assets.items():
                vfs.addBuffer(name, to_js(np.frombuffer(data, dtype=np.uint8)))
            m = self.mj.MjModel.from_xml_string(xml, vfs)
        finally:
            vfs.delete()
        self._live.append(m)
        return m

    def release(self):
        while self._live:
            obj = self._live.pop()
            obj.delete()

    def _array(self, value, name, count_hint=None):
        if hasattr(value, "to_py"):
            arr = np.asarray(value.to_py()).copy()
        else:
            arr = np.asarray(value)
        w = _WIDTH.get(name)
        return arr.reshape(-1, w) if w else arr

    def model_field(self, m, name):
        v = getattr(m, name)
        return v if isinstance(v, (int, float)) else self._array(v, name)

    def make_data(self, m):
        d = self.mj.MjData.new(m)
        self._live.append(d)
        return d

    def data_field(self, d, name):
        return self._array(getattr(d, name), name)

    def set_data_field(self, d, name, value):
        view = getattr(d, name)
        flat = np.ravel(np.asarray(value, dtype=float))
        for i, x in enumerate(flat):
            view[i] = float(x)

    def call(self, fn, *args):
        return getattr(self.mj, fn)(*args)


_BACKEND = None


def release_all() -> None:
    """Free every model and data compiled since the last call. Anything read out of
    them before this is a numpy copy and stays valid; the wrappers themselves do not."""
    if _BACKEND is not None:
        _BACKEND.release()


def use_native(real_mujoco) -> None:
    """Compile through the real bindings (CPython only)."""
    global _BACKEND
    _BACKEND = _Native(real_mujoco)


def _backend():
    global _BACKEND
    if _BACKEND is None:
        if sys.platform != "emscripten":
            raise RuntimeError("mjshim: call use_native(mujoco) outside the browser")
        _BACKEND = _Wasm()
    return _BACKEND
