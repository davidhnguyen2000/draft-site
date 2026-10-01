"""Everything the browser build swaps in for a native extension Pyodide lacks.

Draft's generator runs unmodified in the project site; three of the libraries it
reaches are C/C++ extensions that do not exist under Pyodide, and each is
answered here:

* `mujoco`         -> `mjshim.py`, compiling through MuJoCo's WebAssembly build
* `mapbox_earcut`  -> `mapbox_earcut.py`, over a vendored pure-Python earcut
* trimesh's point-in-mesh test, which needs `rtree` for its ray queries
  -> `contains` below, the same parity rule in numpy

`install()` puts all three in place. The site's worker calls it; so does
`scripts/site/check_mjshim.py`, which then compares every design against the
native libraries, so what is checked is exactly what the browser runs.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

#: The ray every point casts. Oblique, so it does not run along an edge or
#: through a vertex of an axis-aligned loft.
_RAY = np.array([0.4395064455, 0.617598629942, 0.652231566745])


def _crossings(tri: np.ndarray, P: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Distinct faces-crossings of the ray from each point along `d`
    (Moller-Trumbore against every face). A ray through an edge or vertex hits
    every face sharing it, so crossings are counted once per distance."""
    v0, e1, e2 = tri[:, 0], tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(d, e2)
    a = np.einsum("fj,fj->f", e1, h)
    ok = np.abs(a) > 1e-12
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s = P[:, None, :] - v0[None, :, :]
    u = f * np.einsum("nfj,fj->nf", s, h)
    q = np.cross(s, e1[None, :, :])
    v = f * np.einsum("j,nfj->nf", d, q)
    t = f * np.einsum("fj,nfj->nf", e2, q)
    hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-12)
    T = np.sort(np.where(hit, t, np.inf), axis=1)
    return np.isfinite(T[:, 0]).astype(int) + (np.isfinite(T[:, 1:]) & (np.diff(T, axis=1) > 1e-9)).sum(axis=1)


def contains(mesh, points, direction=_RAY) -> np.ndarray:
    """Whether each point is inside a closed mesh, by trimesh's own rule
    (`trimesh.ray.ray_util.contains_points`): cast the ray both ways and count
    crossings; where the two parities agree that is the answer, where they
    disagree and one side is empty the point is outside, and the rest are cast
    again along another direction. Trimesh picks that one at random; this uses
    a fixed one, so the site is repeatable."""
    P = np.asarray(points, dtype=float).reshape(-1, 3)
    out = np.zeros(len(P), dtype=bool)
    lo, hi = mesh.bounds
    box = np.all((P >= lo) & (P <= hi), axis=1)
    if not box.any():
        return out
    tri = np.asarray(mesh.triangles, dtype=float)
    Q = P[box]
    fwd, back = _crossings(tri, Q, direction), _crossings(tri, Q, -direction)
    a, b = fwd % 2 == 1, back % 2 == 1
    agree = a == b
    res = np.where(agree, a, False)
    broken = ~agree & (fwd > 0) & (back > 0)
    if broken.any() and direction is _RAY:
        alt = np.array([0.2672612419, -0.5345224838, 0.8017837257])
        res[broken] = contains(mesh, Q[broken], alt)
    out[box] = res
    return out


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def install(native_mujoco=None) -> None:
    """Swap all three in. `native_mujoco`: compile through the real bindings
    (CPython, for checking) rather than the WebAssembly build."""
    shim = _load("mjshim")
    if native_mujoco is not None:
        shim.use_native(native_mujoco)
    sys.modules["mujoco"] = shim
    _load("_earcut")
    _load("mapbox_earcut")
    import trimesh
    trimesh.Trimesh.contains = contains
