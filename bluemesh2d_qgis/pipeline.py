"""Compatibility shim: the staged pipeline now lives in ``bluemesh2d``.

Every function that used to be defined here has been moved into the core
package (see ``bluemesh2d.pipeline`` for the map of stage -> module) so it
can be used from plain Python without the QGIS plugin. The plugin no longer
bundles a copy of it: ``bluemesh2d`` is installed from PyPI by the
dependency dialog (see ``deps_installer``). This module keeps the historical
import surface (``from .pipeline import ...``) working, and owns the two
plugin-environment concerns that must happen before the first library
import: the pyproj/libproj load order, and a headless matplotlib backend.
"""

from __future__ import annotations

import contextlib

# pyproj must load BEFORE any pip rasterio wheel brings its own copy of
# libproj into the process: the reverse order makes pyproj misbehave with
# "TypeError: expected bytes, str found" (see the pyproj FAQ on mixing
# PROJ versions). Harmless when pyproj is missing or unusable -- the
# dependency dialog handles that case.
with contextlib.suppress(Exception):
    import pyproj  # noqa: E402
    # importing is not enough: pyproj loads libproj lazily at the first CRS
    # creation, so force it NOW, before rasterio's copy enters the process
    pyproj.CRS.from_epsg(4326)

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
    _corner_vertices,
    _fixed_part_from_z,
    _flag_fixed_vertices,
    _locate_fixed,
    _make_depth_hfun,
    _prune_nonoriginal_fixed,
    _raster_crs,
    _valid_parts,
    _warn_if_mesh_too_big,
    _warn_if_ram_risk,
    _write_ugrid_netcdf,
    apply_nc_metadata,
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
