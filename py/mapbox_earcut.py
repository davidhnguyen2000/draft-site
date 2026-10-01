"""`mapbox_earcut`'s one call, for the browser.

Draft triangulates the torso loft's end caps with `mapbox_earcut`, a C++
extension Pyodide does not ship. `_earcut.py` beside this file is a vendored
pure-Python port of the same Mapbox algorithm, and this module answers the call
the generator makes with it. `scripts/site/check_mjshim.py` runs the builder's
designs through it and through the extension and compares the results.
"""
from __future__ import annotations

import numpy as np
import _earcut


def triangulate_float64(vertices, rings):
    pts = np.asarray(vertices, dtype=np.float64)
    ends = [int(r) for r in np.ravel(rings)]
    holes = ends[:-1]
    return np.asarray(_earcut.earcut(pts.ravel().tolist(), holes, 2), dtype=np.uint32)


def triangulate_float32(vertices, rings):
    return triangulate_float64(np.asarray(vertices, dtype=np.float64), rings)
