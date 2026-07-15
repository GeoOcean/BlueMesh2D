"""Compatibility shim: the staged pipeline now lives in ``bluemesh2d``.

Every function that used to be defined here has been moved into the core
package (see ``bluemesh2d.pipeline`` for the map of stage -> module) so it
can be used from plain Python without the QGIS plugin. This module keeps
the historical import surface (``from .pipeline import ...``) working, and
still owns the two plugin-environment concerns: making the bundled
``bluemesh2d`` importable, and forcing matplotlib to a headless backend.
"""

from __future__ import annotations

import os
import sys

# --- make the bundled copy of bluemesh2d importable and matplotlib headless ---
# `bluemesh2d` sits directly in the plugin root, so add the plugin directory to
# sys.path and import it as a top-level package (works both inside QGIS and when
# this module is used standalone / headless).
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)  # must precede any bluemesh2d import (getiso)

from bluemesh2d.pipeline import (  # noqa: E402,F401
    MeshCanceled,
    MeshConfig,
    MeshResult,
    _LogWriter,
    _NullFeedback,
    _SubProgress,
    _UGRID_FILL,
    _WGS84_FILL,
    _available_ram_bytes,
    _boundary_loops,
    _check,
    _compile_custom_hfun,
    _fixed_part_from_z,
    _flag_fixed_vertices,
    _locate_fixed,
    _make_depth_hfun,
    _prune_nonoriginal_fixed,
    _raster_crs,
    _warn_if_mesh_too_big,
    _warn_if_ram_risk,
    _write_ugrid_netcdf,
    boundary_lines_from_points,
    build_hfun_constant_raster,
    build_hfun_raster,
    check_dependencies,
    classify_boundary_lines,
    classify_boundary_points,
    export_boundary_conditions,
    export_grd,
    export_grd_from_lines,
    export_ugrid,
    extract_water_polygon,
    generate_boundary_condition_points,
    generate_boundary_conditions,
    generate_mesh,
    load_hfun_raster,
    mesh_pslg,
    optional_dependencies,
    pslg_from_segments,
    read_ugrid_mesh,
    resample_boundary,
    smood_dependencies,
    write_open_boundary_files,
    write_open_boundary_pli,
)
