"""Runtime dependency checks for the staged meshing pipeline."""
from __future__ import annotations




def check_dependencies():
    """Check that the required runtime dependencies are importable.

    ``xarray`` is not needed (the UGRID NetCDF is written directly with
    ``netCDF4``) and ``triangle`` is optional (without it, ``deltri`` falls
    back to the pure-scipy conforming Delaunay) -- see
    :func:`optional_dependencies`.

    Returns
    -------
    missing : list of str
        Names of required packages that failed to import. Empty if all are
        present.
    """
    missing = []
    # contourpy is matplotlib's contouring backend, used by the stage-1
    # coastline extraction; some QGIS bundles (macOS vcpkg builds) ship
    # matplotlib without it, so it is checked explicitly.
    for mod in ("numpy", "scipy", "shapely", "rasterio", "pyproj",
                "matplotlib", "contourpy", "netCDF4"):
        try:
            __import__(mod)
        except Exception:
            missing.append(mod)
    return missing


def optional_dependencies():
    """Check for optional packages that only affect speed or quality.

    Returns
    -------
    missing : list of str
        Names of optional packages that failed to import. Empty if all are
        present.
    """
    missing = []
    try:
        __import__("triangle")
    except Exception:
        missing.append("triangle")
    return missing


def smood_dependencies():
    """Check for packages required only when ``smood`` (orthogonalization)
    is used.

    ``bluemesh2d.smood`` always builds an in-memory ``xarray.Dataset`` (via
    ``ortho_merge_iterate_tria`` / ``adcirc2DFlowFM``) regardless of the
    ``merge_small_links`` option, so ``xarray`` is required to call it even
    though it is not needed anywhere else in this plugin (the UGRID NetCDF
    exports write directly with ``netCDF4``).

    Returns
    -------
    missing : list of str
        Names of packages required by ``smood`` that failed to import. Empty
        if all are present.
    """
    missing = []
    try:
        __import__("xarray")
    except Exception:
        missing.append("xarray")
    return missing


# ===========================================================================
# Stage 1: raster -> water-domain polygon
# ===========================================================================

