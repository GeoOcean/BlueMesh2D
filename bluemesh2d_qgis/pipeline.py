"""Headless, staged orchestration facade over the bundled ``bluemesh2d`` package.

The notebook workflow (raster -> coastline -> size function -> boundary ->
refine/smooth -> UGRID NetCDF) is exposed as four independent stages, so the
QGIS plugin can run them as separate Processing algorithms with inspectable
intermediate layers -- or all at once via :func:`generate_mesh`.

Stages
------
1. :func:`extract_water_polygon`
    raster -> water-domain polygon (working CRS).
2. :func:`build_hfun_raster`
    raster -> gradient-limited size raster (GeoTIFF, working CRS).
3. :func:`resample_boundary`
    polygon + hfun -> resampled boundary, PSLG nodes/edges.
4. :func:`mesh_pslg` + :func:`export_ugrid`
    PSLG + hfun -> mesh -> UGRID NetCDF.

Notes
-----
All stages are QGIS-free (progress/cancel via a ``QgsProcessingFeedback``-shaped
duck-type) and matplotlib is forced to the non-interactive ``Agg`` backend
before ``bluemesh2d`` is imported (contour extraction needs it).
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
from dataclasses import dataclass
from typing import Optional

# --- make the bundled copy of bluemesh2d importable and matplotlib headless ---
# `bluemesh2d` sits directly in the plugin root, so add the plugin directory to
# sys.path and import it as a top-level package (works both inside QGIS and when
# this module is used standalone / headless).
_PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)

import matplotlib  # noqa: E402
matplotlib.use("Agg", force=True)  # must precede any bluemesh2d import (getiso)


class MeshCanceled(Exception):
    """Exception raised when a run is cancelled.

    Raised when ``feedback.isCanceled()`` becomes ``True`` mid-run.
    """


class _NullFeedback:
    """No-op feedback so the facade runs without QGIS.

    Writes to ``sys.__stdout__`` (the interpreter's real stdout) rather than
    via ``print``, so it stays safe while ``refine``/``smooth`` output is
    captured by ``contextlib.redirect_stdout`` (printing to the redirected
    stdout would recurse).
    """

    def isCanceled(self):
        return False

    def pushInfo(self, msg):
        sys.__stdout__.write(str(msg) + "\n")

    def pushWarning(self, msg):
        sys.__stdout__.write("WARNING: " + str(msg) + "\n")

    def setProgress(self, pct):
        pass


class _LogWriter(io.TextIOBase):
    """File-like object that forwards captured stdout lines to feedback.

    Parameters
    ----------
    feedback : object
        Feedback sink exposing ``pushInfo(str)``, e.g. a
        ``QgsProcessingFeedback`` or :class:`_NullFeedback`.
    """

    def __init__(self, feedback):
        self._fb = feedback
        self._buf = ""

    def write(self, s):
        self._buf += s
        while "\n" in self._buf:
            line, self._buf = self._buf.split("\n", 1)
            if line.strip():
                self._fb.pushInfo(line)
        return len(s)

    def flush(self):
        if self._buf.strip():
            self._fb.pushInfo(self._buf)
        self._buf = ""


def _check(feedback):
    if feedback.isCanceled():
        raise MeshCanceled()


class _SubProgress:
    """Feedback proxy mapping a sub-task's 0-100 progress into [lo, hi].

    Lets a stage function report absolute progress while a multi-stage
    caller (``generate_mesh``) keeps a monotonic overall bar.
    """

    def __init__(self, feedback, lo, hi):
        self._fb = feedback
        self._lo, self._hi = float(lo), float(hi)

    def isCanceled(self):
        return self._fb.isCanceled()

    def pushInfo(self, msg):
        self._fb.pushInfo(msg)

    def pushWarning(self, msg):
        self._fb.pushWarning(msg)

    def setProgress(self, pct):
        self._fb.setProgress(
            self._lo + (self._hi - self._lo) * float(pct) / 100.0)


def _available_ram_bytes():
    """Available system RAM in bytes, or ``None`` when it cannot be found."""
    try:
        import psutil
        return int(psutil.virtual_memory().available)
    except Exception:
        pass
    try:  # Linux fallback, no psutil needed
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except Exception:
        pass
    return None


def _warn_if_ram_risk(feedback, nbytes, what, hint=None):
    """Warn when an estimated allocation may exhaust the available RAM.

    Parameters
    ----------
    feedback : object
        Feedback sink exposing ``pushWarning``.
    nbytes : float
        Estimated memory need (bytes).
    what : str
        Short description of the allocation, used in the message.
    hint : str or None, optional
        Extra sentence suggesting how to reduce the memory need.

    Returns
    -------
    risky : bool
        ``True`` when a warning was emitted.
    """
    avail = _available_ram_bytes()
    if avail is None or nbytes < 0.5 * avail:
        return False
    msg = (f"{what} needs roughly {nbytes / 1e9:.1f} GB "
           f"(~{avail / 1e9:.1f} GB of RAM available). QGIS may become "
           "unresponsive or crash.")
    if hint:
        msg += " " + hint
    feedback.pushWarning(msg)
    return True


def _raster_crs(src):
    """Build a pyproj CRS from an open rasterio dataset, robustly.

    Prefer the EPSG code (a single, well-trodden PROJ database lookup) over
    parsing the full WKT: the WKT parser resolves the datum against the PROJ
    database and has been observed to hard-crash (access violation) in some
    QGIS/PROJ builds. Falls back to WKT only when there is no EPSG code.

    Parameters
    ----------
    src : rasterio.io.DatasetReader
        Open rasterio dataset.

    Returns
    -------
    crs : pyproj.CRS
        CRS of ``src``.
    """
    import pyproj

    try:
        epsg = src.crs.to_epsg()
    except Exception:
        epsg = None
    if epsg:
        return pyproj.CRS.from_epsg(epsg)
    return pyproj.CRS.from_wkt(src.crs.to_wkt())


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

def extract_water_polygon(raster_path, coast_zmax=2.0, domain_buffer=-0.05,
                          deep_zmax=None, extent_geom=None, keep_largest=True,
                          feedback=None):
    """Extract the water domain from a raster.

    The water region is ``z <= coast_zmax``; if `deep_zmax` is given (an
    elevation, e.g. -300), only the *band* between the two levels is kept
    (``deep_zmax < z <= coast_zmax``). The domain extent is `extent_geom`
    (a shapely polygon in the raster CRS) if provided -- the raster is then
    clipped to it *before* the contour extraction and `domain_buffer` is
    ignored -- otherwise the raster's data extent, grown (>0) or shrunk (<0)
    by `domain_buffer` with ``buffer_area``. With `keep_largest` (default),
    only the single largest water polygon is returned, dropping small
    disconnected water bodies.

    Parameters
    ----------
    raster_path : str
        Path to the bathymetry raster (elevation, positive up).
    coast_zmax : float, optional
        Wet threshold (m); the water region is ``z <= coast_zmax``. Default
        is 2.0.
    domain_buffer : float, optional
        Buffer factor applied to the raster's data extent (>0 grows, <0
        shrinks), ignored when `extent_geom` is given. Default is -0.05.
    deep_zmax : float or None, optional
        Deep elevation level (m); if given, only the band
        ``deep_zmax < z <= coast_zmax`` is kept. Default is ``None``.
    extent_geom : shapely.geometry.base.BaseGeometry or None, optional
        Domain extent polygon in the raster CRS; if given, the raster is
        clipped to it before contour extraction. Default is ``None``.
    keep_largest : bool, optional
        If ``True`` (default), keep only the single largest water polygon.
    feedback : object or None, optional
        Feedback sink exposing ``pushInfo``/``isCanceled`` (see
        :class:`_NullFeedback`); a no-op sink is used if ``None``.

    Returns
    -------
    poly : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Water domain polygon in the working CRS.
    utm_crs : pyproj.CRS
        Working CRS. Equal to `raster_crs` when the raster is already
        projected (e.g. UTM); otherwise a local UTM CRS.
    raster_crs : pyproj.CRS
        CRS of the input raster.
    """
    feedback = feedback or _NullFeedback()
    import numpy as np
    import pyproj
    import rasterio
    from shapely.geometry import MultiPolygon

    from bluemesh2d.geom_util.proj_util import get_local_utm_crs, reproject_geometry
    from bluemesh2d.geom_util.poly_util import buffer_area
    from bluemesh2d.geom_util.getiso import getiso_polygon

    feedback.pushInfo("Reading raster ...")
    with rasterio.open(raster_path) as src:
        # the raster is read whole and the contour extraction works on
        # float copies: budget ~4x the in-memory band size
        band_bytes = src.width * src.height * np.dtype(src.dtypes[0]).itemsize
        _warn_if_ram_risk(
            feedback, 4 * band_bytes,
            f"Reading and contouring this raster ({src.width} x {src.height} px)",
            hint="Clip the raster to the study area first (e.g. with GDAL "
                 "'Clip raster by extent'), or provide a smaller domain "
                 "extent polygon.")
        zdat = src.read(1)
        raster_crs = _raster_crs(src)
        w, h = src.width, src.height
        lon = (src.transform * (np.arange(w), np.zeros(w)))[0]
        lat = (src.transform * (np.zeros(h), np.arange(h)))[1]
    _check(feedback)
    feedback.setProgress(15)

    if extent_geom is not None:
        # clip the raster to the extent's bbox BEFORE contour extraction: the
        # contours are traced on far fewer pixels, and the polygon is clipped
        # exactly by the extent afterwards.
        exmin, eymin, exmax, eymax = extent_geom.bounds
        pad_x = 2.0 * abs(lon[1] - lon[0])
        pad_y = 2.0 * abs(lat[1] - lat[0])
        ix = np.flatnonzero((lon >= exmin - pad_x) & (lon <= exmax + pad_x))
        iy = np.flatnonzero((lat >= eymin - pad_y) & (lat <= eymax + pad_y))
        if ix.size < 2 or iy.size < 2:
            raise RuntimeError("The extent polygon does not overlap the raster.")
        lon = lon[ix[0]:ix[-1] + 1]
        lat = lat[iy[0]:iy[-1] + 1]
        zdat = zdat[iy[0]:iy[-1] + 1, ix[0]:ix[-1] + 1]
        feedback.pushInfo(
            f"Raster clipped to the extent polygon: {zdat.shape[0]} x "
            f"{zdat.shape[1]} px")
        if domain_buffer:
            feedback.pushInfo(
                "Domain buffer is ignored when an extent polygon is provided.")

    utm_crs = get_local_utm_crs(raster_crs, lon, lat)

    feedback.pushInfo("Extracting coastline ...")
    coast = MultiPolygon(getiso_polygon(lon, lat, zdat, zmax=coast_zmax))
    if deep_zmax is not None:
        # keep only the band between the two levels: deep_zmax < z <= coast_zmax
        if deep_zmax >= coast_zmax:
            raise RuntimeError(
                f"The deep level ({deep_zmax}) must be below the coastline "
                f"level ({coast_zmax}).")
        feedback.pushInfo(f"Clipping to depth band {deep_zmax} < z <= {coast_zmax} ...")
        deep = getiso_polygon(lon, lat, zdat, zmax=deep_zmax)
        if deep:
            coast = coast.difference(MultiPolygon(deep))
    _check(feedback)
    feedback.setProgress(60)

    feedback.pushInfo("Building domain extent ...")
    if extent_geom is not None:
        domain = extent_geom
    else:
        domain = MultiPolygon(
            getiso_polygon(lon, lat, zdat, zmax=float(np.max(zdat)) + 1.0))
        if domain_buffer:
            domain = buffer_area(domain, domain_buffer)
    _check(feedback)
    feedback.setProgress(85)

    if utm_crs == raster_crs:  # projected input: work natively in the tif CRS
        poly = coast.intersection(domain)
    else:
        poly = reproject_geometry(coast, raster_crs, utm_crs).intersection(
            reproject_geometry(domain, raster_crs, utm_crs))
    if poly.is_empty:
        raise RuntimeError(
            "No water polygon found: the coastline/domain intersection is empty. "
            "Check the raster and the coastline level.")

    # the intersection may be a Polygon, MultiPolygon or GeometryCollection
    if poly.geom_type == "Polygon":
        parts = [poly]
    else:
        parts = [g for g in poly.geoms if g.geom_type == "Polygon"]
    if not parts:
        raise RuntimeError("No polygonal water region in the intersection.")
    feedback.pushInfo(f"Water region(s): {len(parts)}")
    if keep_largest and len(parts) > 1:
        poly = max(parts, key=lambda g: g.area)
        feedback.pushInfo(
            f"Keeping only the largest region ({poly.area / 1e6:.0f} km2).")
    return poly, utm_crs, raster_crs


# ===========================================================================
# Stage 2: raster -> gradient-limited mesh-size (hfun) raster
# ===========================================================================

def _compile_custom_hfun(code):
    """Compile user Python into a callable size function.

    `code` is either a single expression using ``d`` (depth, m, >= 0), ``x``,
    ``y`` (UTM metres) and ``np`` -- e.g. ``np.sqrt(9.81 * d) * 30`` -- or a
    block that defines ``def hfun(d, x, y): ...``.

    Parameters
    ----------
    code : str
        Python expression or function definition, see above.

    Returns
    -------
    f : callable
        ``f(d, x, y) -> h``, arrays of element size in metres.
    """
    import numpy as np

    env = {"np": np}
    if "def hfun" in code:
        exec(compile(code, "<custom hfun>", "exec"), env)
        if "hfun" not in env or not callable(env["hfun"]):
            raise ValueError("Custom code must define `def hfun(d, x, y):`")
        return env["hfun"]
    expr = compile(code, "<custom hfun>", "eval")

    def f(d, x, y):
        return eval(expr, env, {"d": d, "x": x, "y": y, "depth": d})

    return f


def _make_depth_hfun(depth_field, method="polynomial",
                     a=0.14, b=28.0,
                     wave_period=12.0, cells_per_wavelength=20, zmin=1.0,
                     custom_code=None, h_const=1000.0,
                     hmin=100.0, hmax=10000.0, detail=None, detail_hmin=None,
                     slope_ncells=None, slope_step=500.0, slope_hmin=None):
    """Build a depth-based mesh-size function, one of three sizing laws.

    All methods are floored at `hmin` (`detail_hmin` inside the `detail`
    polygon) and capped at `hmax`.

    Parameters
    ----------
    depth_field : callable
        ``depth_field(xy) -> d``, depth (m) at query points ``xy`` of shape
        ``(N, 2)``. May expose a ``.bounds`` attribute, propagated to the
        returned function.
    method : {'polynomial', 'wavelength', 'custom', 'constant'}, optional
        Sizing law. ``'polynomial'``: ``h = a*d**2 + b*d``.
        ``'wavelength'``: ``h = L(T, d) / N`` (Hunt-1979 dispersion, see
        ``hfun_wavenumhunt``). ``'custom'``: user Python, see
        :func:`_compile_custom_hfun`. ``'constant'``: ``h = h_const``
        everywhere (see ``bluemesh2d.hfun_util.make_constant_hfun``).
        Default is ``'polynomial'``.
    a, b : float, optional
        Coefficients for the ``'polynomial'`` method. Defaults are 0.14 and
        28.0.
    wave_period : float, optional
        Wave period ``T`` (s) for the ``'wavelength'`` method. Default is
        12.0.
    cells_per_wavelength : int, optional
        Cells per wavelength ``N`` for the ``'wavelength'`` method. Default
        is 20.
    zmin : float, optional
        Minimum depth (m) used in the dispersion relation for the
        ``'wavelength'`` method. Default is 1.0.
    custom_code : str or None, optional
        Python code for the ``'custom'`` method, see
        :func:`_compile_custom_hfun`. Required when ``method='custom'``.
    h_const : float, optional
        Element size (m) for the ``'constant'`` method. Default is 1000.0.
    hmin, hmax : float, optional
        Element-size floor and cap (m). Defaults are 100.0 and 10000.0.
    detail : shapely.geometry.base.BaseGeometry or None, optional
        Detail-region polygon; inside it the floor is `detail_hmin` instead
        of `hmin`. Default is ``None``.
    detail_hmin : float or None, optional
        Element-size floor (m) inside `detail`. Default is ``None``.
    slope_ncells : float or None, optional
        If given, the size is also limited by the bathymetric-slope term
        ``h_slope = 2*pi*d / (slope_ncells * |grad d|)``, refining the mesh
        where the bathymetry is steep (shelf break) with ~`slope_ncells`
        cells across the slope feature. ``None`` (default) disables it.
    slope_step : float, optional
        Finite-difference step (m) used to estimate the depth gradient;
        use roughly the bathymetry raster resolution. Default is 500.0.
    slope_hmin : float or None, optional
        Independent floor (m) for the slope term: the slope refinement
        never asks for cells smaller than this, regardless of `hmin`.
        ``None`` (default) leaves the slope term floored at `hmin` only
        (via the final clip).

    Returns
    -------
    hfun : callable
        ``hfun(test) -> h``, element size (m) at query points ``test`` of
        shape ``(N, 2)``. Carries a ``.bounds`` attribute when `depth_field`
        has one.
    """
    import numpy as np
    import shapely

    if method not in ("polynomial", "wavelength", "custom", "constant"):
        raise ValueError(
            "method must be 'polynomial', 'wavelength', 'custom' or 'constant'")
    if method == "custom" and not custom_code:
        raise ValueError("method='custom' needs `custom_code`")
    custom = _compile_custom_hfun(custom_code) if method == "custom" else None
    if method == "constant":
        from bluemesh2d.hfun_util.make_constant_hfun import make_constant_hfun
        const_fun = make_constant_hfun(h_const)

    detail_mask = None
    if detail is not None and detail_hmin is not None:
        def detail_mask(xy):
            return shapely.contains_xy(detail, xy[:, 0], xy[:, 1])

    def grad_mag(xy):
        e = float(slope_step)
        dzdx = (np.asarray(depth_field(xy + [e, 0.0]), dtype=float).reshape(-1)
                - np.asarray(depth_field(xy - [e, 0.0]), dtype=float).reshape(-1)) / (2 * e)
        dzdy = (np.asarray(depth_field(xy + [0.0, e]), dtype=float).reshape(-1)
                - np.asarray(depth_field(xy - [0.0, e]), dtype=float).reshape(-1)) / (2 * e)
        return np.hypot(dzdx, dzdy)

    def hfun(test):
        xy = np.atleast_2d(np.asarray(test, dtype=float))
        d = np.asarray(depth_field(xy), dtype=float).reshape(-1)
        d = np.where(d < 0, 0.0, d)
        if method == "constant":
            values = const_fun(xy)
        elif method == "polynomial":
            values = a * d ** 2 + b * d
        elif method == "wavelength":
            from bluemesh2d.hfun_util.hfun_dispersion import hfun_wavenumhunt
            values = hfun_wavenumhunt(xy, d, wave_period,
                                      cells_per_wavelength, zmin, hmin)
        else:
            values = np.asarray(custom(d, xy[:, 0], xy[:, 1]),
                                dtype=float).reshape(-1)
            values = np.broadcast_to(values, d.shape).copy()
        if slope_ncells is not None:
            g = grad_mag(xy)
            h_slope = 2.0 * np.pi * d / (slope_ncells * np.maximum(g, 1e-12))
            if slope_hmin is not None:
                h_slope = np.maximum(h_slope, slope_hmin)
            values = np.minimum(values, h_slope)
        lo = np.full(xy.shape[0], hmin, dtype=float)
        if detail_mask is not None:
            lo[detail_mask(xy)] = detail_hmin
        return np.clip(values, lo, hmax)

    if hasattr(depth_field, "bounds"):
        hfun.bounds = depth_field.bounds
    return hfun


def build_hfun_raster(raster_path, out_path, method="polynomial",
                      a=0.14, b=28.0,
                      wave_period=12.0, cells_per_wavelength=20, zmin=1.0,
                      custom_code=None, h_const=1000.0,
                      hmin=100.0, hmax=10000.0,
                      detail_geom=None, detail_hmin=None,
                      domain_geom=None,
                      slope_ncells=None, slope_step=None, slope_hmin=None,
                      max_gradient=0.1, cell_size=None,
                      extent_buffer=None, feedback=None):
    """Build the gradient-limited element-size field and save it as a GeoTIFF.

    `domain_geom` (e.g. the stage-1 water polygon) restricts hfun computation
    to that polygon's extent instead of the whole raster -- both the
    gradient-limiting grid and the output raster are limited to it, which is
    much faster when only a small part of a large raster is meshed. The
    output raster is in the working CRS; its pixel values are element sizes
    in metres.

    Parameters
    ----------
    raster_path : str
        Path to the bathymetry raster (elevation, positive up).
    out_path : str
        Output GeoTIFF path.
    method : {'polynomial', 'wavelength', 'custom', 'constant'}, optional
        Sizing law, see :func:`_make_depth_hfun`. Default is
        ``'polynomial'``.
    a, b : float, optional
        Coefficients for the ``'polynomial'`` method. Defaults are 0.14 and
        28.0.
    wave_period : float, optional
        Wave period ``T`` (s) for the ``'wavelength'`` method. Default is
        12.0.
    cells_per_wavelength : int, optional
        Cells per wavelength ``N`` for the ``'wavelength'`` method. Default
        is 20.
    zmin : float, optional
        Minimum depth (m) for the ``'wavelength'`` method. Default is 1.0.
    custom_code : str or None, optional
        Python code for the ``'custom'`` method, see
        :func:`_compile_custom_hfun`. Default is ``None``.
    hmin, hmax : float, optional
        Element-size floor and cap (m). Defaults are 100.0 and 10000.0.
    detail_geom : shapely.geometry.base.BaseGeometry or None, optional
        Detail-region polygon in the *raster* CRS. Default is ``None``.
    detail_hmin : float or None, optional
        Element-size floor (m) inside `detail_geom`. Default is ``None``.
    domain_geom : shapely.geometry.base.BaseGeometry or None, optional
        Domain polygon in the *raster* CRS restricting the computed extent
        (e.g. the stage-1 water polygon). Default is ``None`` (whole raster).
    slope_ncells : float or None, optional
        Bathymetric-slope refinement, see :func:`_make_depth_hfun`.
        ``None`` (default) disables it.
    slope_step : float or None, optional
        Finite-difference step (m) for the depth gradient; ``None``
        (default) uses the raster pixel size in the working CRS.
    slope_hmin : float or None, optional
        Independent floor (m) for the slope term, see
        :func:`_make_depth_hfun`. Default is ``None``.
    max_gradient : float, optional
        Maximum allowed size gradient (m/m), see ``smooth_precomput_hfun``.
        Default is 0.1.
    cell_size : float or None, optional
        Sampling-grid resolution (m); auto-derived when ``None``. Default is
        ``None``.
    extent_buffer : float or None, optional
        Buffer (m) padding the computed extent. ``None`` or a negative value
        uses the automatic value
        ``min((hmax - hmin)/max_gradient, 0.25*max(width, height))`` -- the
        gradient-limiting influence radius. Default is ``None``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    out_path : str
        Path to the written GeoTIFF (same as the input `out_path`).
    utm_crs : pyproj.CRS
        Working CRS the output raster is written in.
    """
    feedback = feedback or _NullFeedback()
    import numpy as np
    import pyproj
    import rasterio
    from rasterio.transform import from_origin

    from bluemesh2d.geom_util.proj_util import get_local_utm_crs, reproject_geometry
    from bluemesh2d.geomesh_util.depth_field import depth_field_from_tif
    from bluemesh2d.hfun_util.smooth_and_precomput import smooth_precomput_hfun

    with rasterio.open(raster_path) as src:
        raster_crs = _raster_crs(src)
        w, h = src.width, src.height
        lon = (src.transform * (np.arange(w), np.zeros(w)))[0]
        lat = (src.transform * (np.zeros(h), np.arange(h)))[1]
    utm_crs = get_local_utm_crs(raster_crs, lon, lat)

    feedback.pushInfo(f"Building depth-based size function ({method}) ...")
    depth_field = depth_field_from_tif(raster_path, output_crs=utm_crs)

    def _to_utm(geom):
        return geom if utm_crs == raster_crs else \
            reproject_geometry(geom, raster_crs, utm_crs)

    detail_u = _to_utm(detail_geom) if detail_geom is not None else None

    if slope_ncells is not None and slope_step is None:
        # raster pixel size in metres (working CRS), measured at the centre
        cx = 0.5 * (lon[0] + lon[-1])
        cy = 0.5 * (lat[0] + lat[-1])
        px = abs(lon[1] - lon[0]) if len(lon) > 1 else abs(lat[1] - lat[0])
        tr = pyproj.Transformer.from_crs(raster_crs, utm_crs, always_xy=True)
        x0, y0 = tr.transform(cx, cy)
        x1, y1 = tr.transform(cx + px, cy)
        slope_step = float(np.hypot(x1 - x0, y1 - y0))
        feedback.pushInfo(
            f"Slope refinement on (N={slope_ncells:g}, "
            f"gradient step {slope_step:.0f} m from the raster resolution).")

    hfun = _make_depth_hfun(depth_field, method=method, a=a, b=b,
                            wave_period=wave_period,
                            cells_per_wavelength=cells_per_wavelength,
                            zmin=zmin, custom_code=custom_code,
                            h_const=h_const,
                            hmin=hmin, hmax=hmax,
                            detail=detail_u,
                            detail_hmin=(detail_hmin if detail_u is not None else None),
                            slope_ncells=slope_ncells,
                            slope_step=(slope_step if slope_step else 500.0),
                            slope_hmin=slope_hmin)
    _check(feedback)

    # region to compute hfun over: the domain (water) polygon if given, else
    # the whole raster extent
    if domain_geom is not None:
        region = tuple(_to_utm(domain_geom).bounds)
        feedback.pushInfo("Limiting hfun to the domain polygon extent.")
    else:
        region = tuple(depth_field.bounds)

    xmin, ymin, xmax, ymax = region
    dw, dh = xmax - xmin, ymax - ymin
    if cell_size is None:
        cell_size = max(max(dw, dh) / 1200.0, hmin / 2.0)
    # pad by the gradient-limiting influence radius so boundary queries near the
    # domain edge stay inside the raster (or by the user-set extent buffer)
    if extent_buffer is None or extent_buffer < 0:
        margin = min((hmax - hmin) / max_gradient, 0.25 * max(dw, dh))
    else:
        margin = float(extent_buffer)
    xs = np.arange(xmin - margin, xmax + margin + cell_size, cell_size)
    ys = np.arange(ymax + margin, ymin - margin - cell_size, -cell_size)  # top->down
    # the gradient-limiting grid and the output raster are both ~this size;
    # ~40 bytes/cell covers the working float64 arrays
    _warn_if_ram_risk(
        feedback, 40.0 * len(xs) * len(ys),
        f"The element-size grid ({len(ys)} x {len(xs)} cells "
        f"@ {cell_size:.0f} m)",
        hint="Increase Min element size, reduce the domain / extent buffer, "
             "or provide a water polygon to limit the computed area.")
    feedback.setProgress(10)

    feedback.pushInfo("Gradient-limiting the size function (this can take a moment) ...")
    hfuns = smooth_precomput_hfun(hfun, domain=region, max_gradient=max_gradient,
                                  cell_size=cell_size, plot=False)
    _check(feedback)
    feedback.setProgress(55)

    # ------------------------------- sample onto a regular UTM grid and save
    feedback.pushInfo(f"Sampling size raster {len(ys)} x {len(xs)} @ {cell_size:.0f} m ...")
    H = np.empty((len(ys), len(xs)), dtype=np.float32)
    # sample in row blocks: bounded peak memory, cancellable, real progress
    rows = max(1, int(2_000_000 // max(len(xs), 1)))
    for i0 in range(0, len(ys), rows):
        i1 = min(i0 + rows, len(ys))
        Xc, Yc = np.meshgrid(xs, ys[i0:i1])
        H[i0:i1] = np.asarray(
            hfuns(np.column_stack([Xc.ravel(), Yc.ravel()])),
            dtype=np.float32).reshape(i1 - i0, len(xs))
        _check(feedback)
        feedback.setProgress(55 + 40.0 * i1 / len(ys))

    transform = from_origin(xs[0] - cell_size / 2, ys[0] + cell_size / 2,
                            cell_size, cell_size)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=H.shape[0], width=H.shape[1],
        count=1, dtype="float32", crs=utm_crs.to_wkt(), transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(H, 1)
    feedback.pushInfo(f"Size raster written -> {out_path}")
    return out_path, utm_crs


def build_hfun_constant_raster(water_geom, out_path, h_domain,
                               detail_geom=None, h_detail=None,
                               max_gradient=0.1, extent_buffer=None,
                               layer_crs=None, feedback=None):
    """Uniform element-size raster from the water polygon (no bathymetry).

    The size is `h_domain` everywhere, `h_detail` inside `detail_geom`,
    gradient-limited so the transition between the two respects
    `max_gradient`. The computed extent is the water polygon's bounds plus
    a buffer. The output GeoTIFF is in the working CRS (the layer CRS when
    already projected, a local UTM otherwise), like
    :func:`build_hfun_raster`.

    Parameters
    ----------
    water_geom : shapely.geometry.base.BaseGeometry
        Water polygon (stage 1) in `layer_crs`; defines the computed extent.
    out_path : str
        Output GeoTIFF path.
    h_domain : float
        Element size (m) over the domain.
    detail_geom : shapely.geometry.base.BaseGeometry or None, optional
        Detail-region polygon in `layer_crs`. Default is ``None``.
    h_detail : float or None, optional
        Element size (m) inside `detail_geom`. Default is ``None`` (ignored).
    max_gradient : float, optional
        Maximum allowed size gradient (m/m). Default is 0.1.
    extent_buffer : float or None, optional
        Buffer (m) padding the computed extent; ``None`` or negative uses
        the automatic gradient-influence radius. Default is ``None``.
    layer_crs : pyproj.CRS or None, optional
        CRS of the input geometries. ``None`` assumes an already-metric CRS.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    out_path : str
        Path to the written GeoTIFF.
    utm_crs : pyproj.CRS
        Working CRS the output raster is written in.
    """
    feedback = feedback or _NullFeedback()
    import numpy as np
    import pyproj
    import rasterio
    import shapely
    from rasterio.transform import from_origin

    from bluemesh2d.geom_util.proj_util import get_local_utm_crs, reproject_geometry
    from bluemesh2d.hfun_util.make_constant_hfun import make_constant_hfun
    from bluemesh2d.hfun_util.smooth_and_precomput import smooth_precomput_hfun

    h_domain = float(h_domain)
    if not np.isfinite(h_domain) or h_domain <= 0:
        raise ValueError("The domain element size must be > 0.")
    use_detail = (detail_geom is not None and h_detail is not None
                  and float(h_detail) > 0)
    h_detail = float(h_detail) if use_detail else h_domain

    if layer_crs is None:
        layer_crs = pyproj.CRS.from_epsg(3857)  # assume metric
    xmin0, ymin0, xmax0, ymax0 = water_geom.bounds
    utm_crs = get_local_utm_crs(pyproj.CRS(layer_crs),
                                np.array([xmin0, xmax0]),
                                np.array([ymin0, ymax0]))
    if pyproj.CRS(layer_crs) != utm_crs:
        water_u = reproject_geometry(water_geom, layer_crs, utm_crs)
        detail_u = (reproject_geometry(detail_geom, layer_crs, utm_crs)
                    if use_detail else None)
    else:
        water_u = water_geom
        detail_u = detail_geom if use_detail else None

    base = make_constant_hfun(h_domain, bounds=water_u.bounds)
    if use_detail:
        def hfun(xy):
            xy = np.atleast_2d(np.asarray(xy, dtype=float))
            v = base(xy)
            v[shapely.contains_xy(detail_u, xy[:, 0], xy[:, 1])] = h_detail
            return v
        hfun.bounds = water_u.bounds
    else:
        hfun = base

    hmin = min(h_domain, h_detail)
    hmax = max(h_domain, h_detail)
    region = tuple(water_u.bounds)
    xmin, ymin, xmax, ymax = region
    dw, dh = xmax - xmin, ymax - ymin
    cell_size = max(max(dw, dh) / 1200.0, hmin / 2.0)
    if extent_buffer is None or extent_buffer < 0:
        margin = min((hmax - hmin) / max_gradient, 0.25 * max(dw, dh))
        margin = max(margin, 2.0 * cell_size)
    else:
        margin = float(extent_buffer)
    xs = np.arange(xmin - margin, xmax + margin + cell_size, cell_size)
    ys = np.arange(ymax + margin, ymin - margin - cell_size, -cell_size)
    _warn_if_ram_risk(
        feedback, 40.0 * len(xs) * len(ys),
        f"The element-size grid ({len(ys)} x {len(xs)} cells "
        f"@ {cell_size:.0f} m)",
        hint="Increase the element sizes or reduce the extent buffer.")
    feedback.setProgress(10)

    if hmax > hmin:
        feedback.pushInfo(
            "Gradient-limiting the size function (this can take a moment) ...")
        hfuns = smooth_precomput_hfun(hfun, domain=region,
                                      max_gradient=max_gradient,
                                      cell_size=cell_size, plot=False)
    else:
        feedback.pushInfo("Uniform size: no gradient limiting needed.")
        hfuns = hfun
    _check(feedback)
    feedback.setProgress(55)

    feedback.pushInfo(
        f"Sampling size raster {len(ys)} x {len(xs)} @ {cell_size:.0f} m ...")
    H = np.empty((len(ys), len(xs)), dtype=np.float32)
    rows = max(1, int(2_000_000 // max(len(xs), 1)))
    for i0 in range(0, len(ys), rows):
        i1 = min(i0 + rows, len(ys))
        Xc, Yc = np.meshgrid(xs, ys[i0:i1])
        H[i0:i1] = np.asarray(
            hfuns(np.column_stack([Xc.ravel(), Yc.ravel()])),
            dtype=np.float32).reshape(i1 - i0, len(xs))
        _check(feedback)
        feedback.setProgress(55 + 40.0 * i1 / len(ys))

    transform = from_origin(xs[0] - cell_size / 2, ys[0] + cell_size / 2,
                            cell_size, cell_size)
    with rasterio.open(
        out_path, "w", driver="GTiff", height=H.shape[0], width=H.shape[1],
        count=1, dtype="float32", crs=utm_crs.to_wkt(), transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(H, 1)
    feedback.pushInfo(f"Size raster written -> {out_path}")
    return out_path, utm_crs


def load_hfun_raster(hfun_path):
    """Load a size raster (stage 2 output) into a fast callable size function.

    Parameters
    ----------
    hfun_path : str
        Path to the GeoTIFF written by :func:`build_hfun_raster`.

    Returns
    -------
    hfuns : callable
        ``hfuns(xy) -> h``, element size (m) at query points ``xy`` of shape
        ``(N, 2)``, linearly interpolated from the raster. Exposes
        ``.bounds`` (raster bounds) and ``.crs_wkt`` (WKT of the raster's
        working CRS, or ``None``).
    """
    import numpy as np
    import rasterio
    from scipy.interpolate import RegularGridInterpolator

    with rasterio.open(hfun_path) as src:
        H = src.read(1).astype(float)
        t = src.transform
        xs = t.c + t.a * (np.arange(src.width) + 0.5)
        ys = t.f + t.e * (np.arange(src.height) + 0.5)
        bounds = tuple(src.bounds)
        crs_wkt = src.crs.to_wkt() if src.crs else None

    if ys[0] > ys[-1]:  # RegularGridInterpolator needs increasing axes
        ys, H = ys[::-1], H[::-1, :]
    interp = RegularGridInterpolator((ys, xs), H, method="linear",
                                     bounds_error=False, fill_value=None)
    h_lo, h_hi = float(H.min()), float(H.max())

    def hfuns(xy):
        xy = np.atleast_2d(np.asarray(xy, dtype=float))
        h = interp(np.column_stack([xy[:, 1], xy[:, 0]]))
        np.clip(h, h_lo, h_hi, out=h)
        return h

    hfuns.bounds = bounds
    hfuns.crs_wkt = crs_wkt  # the working (UTM) CRS the grid lives in
    return hfuns


# ===========================================================================
# Stage 3: polygon + hfun -> resampled boundary + PSLG
# ===========================================================================

def _fixed_part_from_z(poly):
    """Split a Z-flagged polygon into 2D geometry + resample ``part`` lists.

    Stage 1 marks vertices lying on the extent polygon boundary with Z=1
    (Z=0 elsewhere; the user can toggle any vertex in the Vertex Editor).
    Each arc of edges between two consecutive flagged vertices becomes one
    part, so every flagged vertex -- contiguous run or isolated -- is a part
    junction that ``resample_polygon_hfun`` keeps exactly. Rings holding
    flagged vertices are rotated to start at one, so the part junctions
    line up with the ring seam. Returns ``(poly, None)`` for 2D polygons.
    """
    import numpy as np
    from shapely.geometry import Polygon

    if not getattr(poly, "has_z", False):
        return poly, None

    rings = [np.asarray(poly.exterior.coords)]
    rings += [np.asarray(r.coords) for r in poly.interiors]
    part = []
    rings2d = []
    offset = 0
    for r in rings:
        pts = r[:-1] if len(r) > 1 and np.allclose(r[0, :2], r[-1, :2]) else r
        flag = pts[:, 2] > 0.5 if pts.shape[1] > 2 else np.zeros(len(pts), bool)
        n = len(pts)
        fidx = np.flatnonzero(flag)
        if fidx.size == 0:
            rings2d.append(pts[:, :2])
            offset += n
            continue
        # start the ring at a flagged vertex so arcs don't cross the seam
        k0 = int(fidx[0])
        pts = np.roll(pts, -k0, axis=0)
        fidx = np.flatnonzero(np.roll(flag, -k0))  # fidx[0] == 0
        bounds = list(fidx) + [n]
        for a, b in zip(bounds[:-1], bounds[1:]):
            part.append(np.arange(offset + a, offset + b))
        rings2d.append(pts[:, :2])
        offset += n
    poly2d = Polygon(rings2d[0], rings2d[1:])
    return poly2d, (part if part else None)


def resample_boundary(poly, hfuns, min_angle_deg=25.0, min_hole_vertices=15,
                      feedback=None):
    """Resample the water polygon to the size function and build the PSLG.

    `poly` may be a Polygon or MultiPolygon (each part is resampled
    independently).

    Parameters
    ----------
    poly : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Water domain polygon (working CRS), e.g. from
        :func:`extract_water_polygon`.
    hfuns : callable
        Element-size function, ``hfuns(xy) -> h``.
    min_angle_deg : float, optional
        Minimum interior angle (deg) enforced during resampling. Default is
        25.0.
    min_hole_vertices : int, optional
        Minimum vertex count for a hole to be kept. Default is 15.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    poly_comput : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Resampled polygon.
    node : ndarray of shape (N, 2)
        PSLG node coordinates.
    edge : ndarray of shape (E, 2)
        PSLG edges (0-based node indices).
    """
    feedback = feedback or _NullFeedback()
    from shapely.geometry import MultiPolygon

    from bluemesh2d.geom_util.poly_util import polygon_to_node_edge, resample_polygon_hfun

    if poly.geom_type == "Polygon":
        parts = [poly]
    else:
        parts = [g for g in poly.geoms if g.geom_type == "Polygon"]
    if not parts:
        raise RuntimeError("No polygon parts to resample.")
    feedback.pushInfo(f"Water region(s) to mesh: {len(parts)}")

    resampled = []
    n_fixed_total = 0
    for part in parts:
        part2d, fixed_part = _fixed_part_from_z(part)
        if fixed_part is not None:
            # each fixed edge is its own part: every flagged vertex is a
            # part junction and survives the resampling exactly
            n_fixed_total += len(fixed_part)
        rp = resample_polygon_hfun(part2d, hfuns,
                                   min_angle_deg=min_angle_deg,
                                   min_hole_vertices=min_hole_vertices,
                                   part=fixed_part)
        # drop parts that degenerate (smaller than the local element size)
        if rp is not None and not rp.is_empty and rp.geom_type == "Polygon" \
                and len(rp.exterior.coords) >= 4:
            resampled.append(rp)
        _check(feedback)
    if not resampled:
        raise RuntimeError(
            "All water polygons are smaller than the requested element size; "
            "decrease the minimum element size.")
    if len(resampled) < len(parts):
        feedback.pushInfo(
            f"Dropped {len(parts) - len(resampled)} water region(s) smaller "
            "than the local element size.")

    if n_fixed_total:
        feedback.pushInfo(
            f"Fixed vertices preserved exactly: {n_fixed_total}")

    poly_comput = resampled[0] if len(resampled) == 1 else MultiPolygon(resampled)
    node, edge = polygon_to_node_edge(poly_comput)
    feedback.pushInfo(f"Boundary: {len(node)} nodes, {len(edge)} edges")
    return poly_comput, node, edge


def pslg_from_segments(segments, tol=1e-3, close_rings=True):
    """Rebuild a PSLG (node, edge) from a boundary lines layer.

    Every consecutive vertex pair of each polyline becomes an edge. Vertices
    closer than `tol` are snapped to a single node, so nodes moved, added or
    deleted while editing in QGIS reconnect cleanly. Zero-length edges and
    duplicates are dropped.

    Parameters
    ----------
    segments : iterable of array_like of shape (N, 2)
        Boundary polylines (coordinates in the working CRS).
    tol : float, optional
        Snapping tolerance (m) for merging close vertices into one node.
        Default is 1e-3.
    close_rings : bool, optional
        If ``True`` (default), each polyline is treated as a closed boundary
        ring: when its two endpoints are distinct nodes, a closing edge is
        added. Already-closed rings (duplicate end vertex) and 2-point
        segments are unaffected, so both ring-style and segment-style layers
        work.

    Returns
    -------
    node : ndarray of shape (N, 2)
        PSLG node coordinates.
    edge : ndarray of shape (E, 2)
        PSLG edges (0-based node indices).

    Raises
    ------
    RuntimeError
        If fewer than 3 edges result, or if the boundary does not form closed
        loops (a node touches a number of edges other than two).
    """
    import numpy as np

    nodes = []
    index = {}

    def node_id(pt):
        key = (round(pt[0] / tol), round(pt[1] / tol))
        i = index.get(key)
        if i is None:
            i = len(nodes)
            index[key] = i
            nodes.append((float(pt[0]), float(pt[1])))
        return i

    edges = set()
    for seg in segments:
        seg = np.atleast_2d(np.asarray(seg, dtype=float))
        first = last = None
        for a, b in zip(seg[:-1], seg[1:]):
            i, j = node_id(a), node_id(b)
            if first is None:
                first = i
            last = j
            if i != j:
                edges.add((min(i, j), max(i, j)))
        if close_rings and first is not None and last is not None and first != last:
            edges.add((min(first, last), max(first, last)))

    if len(edges) < 3:
        raise RuntimeError("Boundary edges layer yields fewer than 3 edges.")
    node = np.asarray(nodes, dtype=float)
    edge = np.asarray(sorted(edges), dtype=int)

    # the mesher needs closed loops: every node on exactly 2 edges
    counts = np.bincount(edge.ravel(), minlength=len(node))
    bad = np.flatnonzero(counts != 2)
    if bad.size:
        sample = ", ".join(
            f"({node[b][0]:.1f}, {node[b][1]:.1f})" for b in bad[:5])
        raise RuntimeError(
            f"{bad.size} boundary node(s) are dangling ends or junctions "
            f"(not closed loops), e.g. near {sample} (working CRS, m). "
            "Fix the edges layer: every vertex must join exactly two segments.")
    return node, edge


# ===========================================================================
# Stage 4: PSLG + hfun -> mesh -> UGRID NetCDF
# ===========================================================================

def _warn_if_mesh_too_big(node, edge, hfuns, feedback):
    """Estimate the refined mesh size and warn when it looks RAM-risky.

    The expected triangle count is ``~(2/sqrt(3)) * integral(dA / h^2)``,
    evaluated by sampling `hfuns` on a coarse grid over the PSLG polygons.
    Estimation errors are fine here -- this only decides whether to warn.
    """
    try:
        import numpy as np
        import shapely
        from shapely.ops import polygonize

        rings = polygonize(
            [((node[a][0], node[a][1]), (node[b][0], node[b][1]))
             for a, b in edge])
        area_geom = shapely.unary_union(list(rings))
        if area_geom.is_empty:
            return
        xmin, ymin, xmax, ymax = area_geom.bounds
        n = 128
        xs = np.linspace(xmin, xmax, n)
        ys = np.linspace(ymin, ymax, n)
        X, Y = np.meshgrid(xs, ys)
        xy = np.column_stack([X.ravel(), Y.ravel()])
        inside = shapely.contains_xy(area_geom, xy[:, 0], xy[:, 1])
        if not inside.any():
            return
        h = np.asarray(hfuns(xy[inside]), dtype=float)
        cell_area = area_geom.area / inside.sum()
        n_tria = 2.0 / np.sqrt(3.0) * cell_area * np.sum(1.0 / h ** 2)
        if n_tria > 500_000:
            msg = (f"The size function implies roughly {n_tria / 1e6:.1f} "
                   "million triangles; refinement may take a long time.")
            est_bytes = 1000.0 * n_tria  # ~1 kB per triangle during refine/smooth
            avail = _available_ram_bytes()
            if avail is not None and est_bytes > 0.5 * avail:
                msg += (f" Estimated memory ~{est_bytes / 1e9:.1f} GB of "
                        f"~{avail / 1e9:.1f} GB available -- QGIS may become "
                        "unresponsive or crash.")
            msg += (" Consider increasing Min element size or reducing the "
                    "domain.")
            feedback.pushWarning(msg)
    except Exception:
        pass  # a failed estimate must never block the run


def _locate_fixed(vert, fixed_points, feedback, tol=1e-6):
    """Return indices in ``vert`` of each fixed point (nearest within tol)."""
    import numpy as np

    idx = []
    for p in np.asarray(fixed_points, dtype=float):
        d2 = np.sum((vert - p) ** 2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] <= tol * tol:
            idx.append(j)
        else:
            feedback.pushWarning(
                f"Fixed point ({p[0]:.3f}, {p[1]:.3f}) not found in mesh "
                f"(nearest node {np.sqrt(d2[j]):.3g} m away); skipped.")
    return np.asarray(idx, dtype=int)


def mesh_pslg(node, edge, hfuns, kind="delaunay", do_smooth=True,
              do_smood=False, smood_merge_small_links=False,
              fixed_points=None, feedback=None):
    """Refine a PSLG, then optionally smooth and/or smood it.

    Parameters
    ----------
    node : ndarray of shape (N, 2)
        PSLG node coordinates (working CRS).
    edge : ndarray of shape (E, 2)
        PSLG edges (0-based node indices).
    hfuns : callable
        Element-size function, ``hfuns(xy) -> h``.
    kind : {'delaunay', 'delfront'}, optional
        Refinement scheme passed to ``refine``. Default is ``'delaunay'``.
    do_smooth : bool, optional
        If ``True`` (default), run non-linear mesh optimisation (``smooth``)
        after refinement.
    do_smood : bool, optional
        If ``True``, additionally run orthogonalization (``smood``) after
        smoothing. Default is ``False``.
    smood_merge_small_links : bool, optional
        Enable the merge step inside smood's ortho-merge cycles (pairs of
        triangles whose circumcenters are too close are merged, then
        re-split). Use only when the default triangle-only smood cannot
        remove the remaining small flow links. Default is ``False``.
    fixed_points : ndarray of shape (K, 2), optional
        XY coordinates (working CRS) of points that must appear as mesh
        nodes at exactly these positions: they are inserted before
        refinement and pinned during smoothing and orthogonalization.
        Points outside the meshed domain or coincident with boundary
        nodes are ignored (with a warning).
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    vert : ndarray of shape (M, 2)
        Mesh vertex coordinates (working CRS).
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based vertex indices).
    """
    feedback = feedback or _NullFeedback()
    from bluemesh2d.refine import refine
    from bluemesh2d.smooth import smooth

    kind = str(kind).lower()
    if kind not in ("delaunay", "delfront"):
        raise ValueError("kind must be 'delaunay' or 'delfront'")

    import numpy as np

    if fixed_points is not None:
        fixed_points = np.asarray(fixed_points, dtype=float).reshape(-1, 2)
        if fixed_points.size:
            # drop fixed points (nearly) coincident with existing PSLG nodes:
            # a duplicate vertex would break the triangulation
            keep_fp = np.ones(fixed_points.shape[0], dtype=bool)
            for i, p in enumerate(fixed_points):
                if np.min(np.sum((node - p) ** 2, axis=1)) < 1e-6:
                    keep_fp[i] = False
                    feedback.pushWarning(
                        f"Fixed point ({p[0]:.3f}, {p[1]:.3f}) coincides with "
                        "a boundary node; skipped.")
            fixed_points = fixed_points[keep_fp]
        if fixed_points.size:
            feedback.pushInfo(f"Inserting {len(fixed_points)} fixed point(s)")
            node = np.vstack([node, fixed_points])
        else:
            fixed_points = None

    _warn_if_mesh_too_big(node, edge, hfuns, feedback)
    feedback.setProgress(5)

    feedback.pushInfo(f"Refining mesh ({len(node)} boundary nodes, kind={kind}) ...")
    with contextlib.redirect_stdout(_LogWriter(feedback)):
        vert, etri, tria, tnum = refine(node, edge, [], {"kind": kind}, hfuns)
    _check(feedback)
    feedback.pushInfo(f"Refined: {len(vert)} nodes, {len(tria)} triangles")
    feedback.setProgress(55)

    # refine never moves input nodes, so fixed points can be re-located by
    # coordinate after each stage (indices change with mesh compaction)
    fixed_idx = None
    if fixed_points is not None:
        fixed_idx = _locate_fixed(vert, fixed_points, feedback)
        # keep only the points actually present (e.g. outside the domain,
        # dropped by refine) so later stages don't re-warn about them
        fixed_points = vert[fixed_idx, :].copy()
        if fixed_points.size == 0:
            fixed_points = None
            fixed_idx = None

    if do_smooth:
        feedback.pushInfo("Smoothing mesh ...")
        with contextlib.redirect_stdout(_LogWriter(feedback)):
            vert, etri, tria, tnum = smooth(vert, etri, tria, tnum, {}, hfuns,
                                            fixed=fixed_idx)
        _check(feedback)
        if fixed_points is not None:
            fixed_idx = _locate_fixed(vert, fixed_points, feedback)
        feedback.setProgress(85)

    if do_smood:
        missing = smood_dependencies()
        if missing:
            raise RuntimeError(
                "smood (orthogonalization) requires: " + ", ".join(missing)
                + ". Install it, or disable the smood option.")
        feedback.pushInfo("Applying smood (orthogonalization) ...")
        from bluemesh2d.smood import smood
        smood_opts = {}
        if smood_merge_small_links:
            feedback.pushInfo("smood: small-link merging enabled")
            smood_opts["merge_small_links"] = True
        try:
            with contextlib.redirect_stdout(_LogWriter(feedback)):
                vert, etri, tria, tnum = smood(vert, etri, tria, tnum, smood_opts,
                                               fixed=fixed_idx)
        except ImportError as exc:
            raise RuntimeError(
                f"smood needs an optional package that is not installed: {exc}. "
                "Install it or disable the smood option.")
        _check(feedback)
        feedback.pushInfo(f"After smood: {len(vert)} nodes, {len(tria)} faces")

    return vert, tria


# UGRID fill values matching bluemesh2d.geomesh_util.grd_util
_UGRID_FILL = -999
_WGS84_FILL = -2147483647


def _write_ugrid_netcdf(ugrid: dict, path: str, crs=None):
    """Write a ``build_ugrid_arrays`` dict as a Delft3D-FM-style UGRID NetCDF.

    Mirrors ``grd_util._xr_dataset_from_ugrid_dict`` (variables, attributes,
    conventions) but uses ``netCDF4`` directly, so ``xarray`` is not required.

    Parameters
    ----------
    ugrid : dict
        UGRID arrays as returned by ``build_ugrid_arrays``.
    path : str
        Output NetCDF path.
    crs : pyproj.CRS or None, optional
        Controls the coordinate metadata: geographic (default) writes degree
        units and a ``wgs84`` grid mapping; a projected CRS writes metre
        units and a ``projected_coordinate_system`` grid mapping.
    """
    import numpy as np
    from datetime import datetime
    from netCDF4 import Dataset

    geographic = crs is None or getattr(crs, "is_geographic", True)
    if geographic:
        x_units, y_units = "degrees_east", "degrees_north"
        x_std, y_std = "longitude", "latitude"
        crs_name = "wgs84"
        epsg = 4326 if crs is None else (crs.to_epsg() or 4326)
        crs_attrs = {
            "name": "WGS 84", "epsg": np.int32(epsg),
            "grid_mapping_name": "latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245,
            "inverse_flattening": 298.257223563,
            "EPSG_code": "", "value": "value is equal to EPSG code",
            "proj_string": "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        }
    else:
        x_units, y_units = "m", "m"
        x_std, y_std = "projection_x_coordinate", "projection_y_coordinate"
        crs_name = "projected_coordinate_system"
        epsg = crs.to_epsg() or 0
        crs_attrs = {
            "name": crs.name, "epsg": np.int32(epsg),
            "grid_mapping_name": "Unknown projected",
            "EPSG_code": f"EPSG:{epsg}" if epsg else "",
            "value": "value is equal to EPSG code",
            "proj_string": crs.to_proj4() or "",
            "wkt": crs.to_wkt(),
        }

    face_nodes = np.asarray(ugrid["face_nodes"])
    n_max_face_nodes = int(face_nodes.shape[1])

    with Dataset(path, "w", format="NETCDF4") as nc:
        # ------------------------------------------------------------- dims
        nc.createDimension("mesh2d_nNodes", int(ugrid["num_nodes"]))
        nc.createDimension("mesh2d_nEdges", int(ugrid["num_edges"]))
        nc.createDimension("mesh2d_nFaces", int(ugrid["num_faces"]))
        nc.createDimension("mesh2d_nMax_face_nodes", n_max_face_nodes)
        nc.createDimension("Two", 2)

        def var(name, dtype, dims, data, attrs, fill_value=None):
            v = nc.createVariable(name, dtype, dims, fill_value=fill_value)
            for k, val in attrs.items():
                v.setncattr(k, val)
            v[...] = data
            return v

        i4 = np.int32
        # ------------------------------------------------------------ nodes
        var("mesh2d_node_x", "f8", ("mesh2d_nNodes",), ugrid["node_x"], {
            "standard_name": x_std,
            "long_name": "x-coordinate of mesh nodes",
            "units": x_units})
        var("mesh2d_node_y", "f8", ("mesh2d_nNodes",), ugrid["node_y"], {
            "standard_name": y_std,
            "long_name": "y-coordinate of mesh nodes",
            "units": y_units})
        var("mesh2d_node_z", "f8", ("mesh2d_nNodes",), -np.asarray(ugrid["node_z"]), {
            "mesh": "mesh2d", "location": "node", "units": "m",
            "standard_name": "altitude",
            "long_name": "z-coordinate of mesh nodes",
            "grid_mapping": crs_name,
            "coordinates": "mesh2d_node_x mesh2d_node_y"})
        # ------------------------------------------------------------ edges
        var("mesh2d_edge_x", "f8", ("mesh2d_nEdges",), ugrid["edge_x"], {
            "long_name": "characteristic x-coordinate of the mesh edge (e.g. midpoint)",
            "units": x_units, "standard_name": x_std})
        var("mesh2d_edge_y", "f8", ("mesh2d_nEdges",), ugrid["edge_y"], {
            "long_name": "characteristic y-coordinate of the mesh edge (e.g. midpoint)",
            "units": y_units, "standard_name": y_std})
        var("mesh2d_edge_nodes", "i4", ("mesh2d_nEdges", "Two"),
            np.asarray(ugrid["edge_nodes"], dtype=i4), {
            "cf_role": "edge_node_connectivity",
            "long_name": "Start and end nodes of mesh edges",
            "start_index": i4(1)})
        var("mesh2d_edge_faces", "i4", ("mesh2d_nEdges", "Two"),
            np.asarray(ugrid["edge_faces"], dtype=i4), {
            "cf_role": "edge_face_connectivity",
            "long_name": "Neighboring faces of mesh edges",
            "start_index": i4(1)}, fill_value=i4(_UGRID_FILL))
        # ------------------------------------------------------------ faces
        var("mesh2d_face_nodes", "i4", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
            np.asarray(face_nodes, dtype=i4), {
            "cf_role": "face_node_connectivity",
            "long_name": "Vertex nodes of mesh faces (counterclockwise)",
            "start_index": i4(1),
            "coordinates": "mesh2d_node_x mesh2d_node_y"},
            fill_value=i4(_UGRID_FILL))
        var("mesh2d_face_x", "f8", ("mesh2d_nFaces",), ugrid["face_x"], {
            "units": x_units, "standard_name": x_std,
            "long_name": "Characteristic x-coordinate of mesh face",
            "bounds": "mesh2d_face_x_bnd"})
        var("mesh2d_face_y", "f8", ("mesh2d_nFaces",), ugrid["face_y"], {
            "units": y_units, "standard_name": y_std,
            "long_name": "Characteristic y-coordinate of mesh face",
            "bounds": "mesh2d_face_y_bnd"})
        var("mesh2d_face_x_bnd", "f8", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
            ugrid["face_x_bnd"], {
            "long_name": "x-coordinate bounds of mesh faces (i.e. corner coordinates)",
            "units": x_units, "standard_name": x_std})
        var("mesh2d_face_y_bnd", "f8", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
            ugrid["face_y_bnd"], {
            "long_name": "y-coordinate bounds of mesh faces (i.e. corner coordinates)",
            "units": y_units, "standard_name": y_std})
        # -------------------------------------------------- CRS + topology
        var(crs_name, "i4", (), i4(epsg if epsg else _WGS84_FILL), crs_attrs,
            fill_value=i4(_WGS84_FILL))
        var("mesh2d", "i4", (), i4(_WGS84_FILL), {
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D mesh",
            "topology_dimension": i4(2),
            "node_coordinates": "mesh2d_node_x mesh2d_node_y",
            "node_dimension": "mesh2d_nNodes",
            "edge_node_connectivity": "mesh2d_edge_nodes",
            "edge_dimension": "mesh2d_nEdges",
            "edge_coordinates": "mesh2d_edge_x mesh2d_edge_y",
            "face_node_connectivity": "mesh2d_face_nodes",
            "face_dimension": "mesh2d_nFaces",
            "face_coordinates": "mesh2d_face_x mesh2d_face_y",
            "max_face_nodes_dimension": "mesh2d_nMax_face_nodes",
            "edge_face_connectivity": "mesh2d_edge_faces"})
        # ------------------------------------------------------ global attrs
        nc.institution = "GeoOcean"
        nc.references = "https://github.com/GeoOcean/BlueMesh2D"
        nc.source = f"BlueMesh2D QGIS plugin {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        nc.history = "Created with BlueMesh2D"
        nc.Conventions = "CF-1.8 UGRID-1.0 Deltares-0.10"


def export_ugrid(vert, tria, raster_path, utm_crs, output_path,
                 interp_order=3, feedback=None):
    """Sample bathymetry onto the mesh nodes and write the UGRID NetCDF.

    `vert` is in `utm_crs`; nodes are reprojected to the bathymetry raster's
    CRS for both the z-sampling and the output coordinates.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Mesh vertex coordinates in `utm_crs`.
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based vertex indices).
    raster_path : str
        Path to the bathymetry raster used for node-depth sampling.
    utm_crs : pyproj.CRS
        CRS `vert` is expressed in.
    output_path : str
        Output UGRID NetCDF path.
    interp_order : int, optional
        Interpolation order passed to ``interpolate_from_tiff``
        (0=nearest, 1=bilinear, 3=bicubic). Default is 3.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    output_path : str
        Path to the written NetCDF (same as the input `output_path`).
    """
    feedback = feedback or _NullFeedback()
    import numpy as np
    import pyproj
    import rasterio

    from bluemesh2d.geom_util.proj_util import reproject_node
    from bluemesh2d.geomesh_util.interpolation_mesh import interpolate_from_tiff
    from bluemesh2d.geomesh_util.grd_util import build_ugrid_arrays

    with rasterio.open(raster_path) as src:
        raster_crs = _raster_crs(src)

    feedback.pushInfo("Interpolating bathymetry onto mesh nodes ...")
    if utm_crs == raster_crs:
        vert_geo = vert  # projected input: mesh is already in the tif CRS
    else:
        vert_geo = reproject_node(vert, utm_crs, raster_crs)
    z = interpolate_from_tiff(raster_path, vert_geo, order=interp_order)
    _check(feedback)

    feedback.pushInfo(f"Writing UGRID NetCDF -> {output_path}")
    ugrid = build_ugrid_arrays(np.column_stack((vert_geo, z)),
                               np.asarray(tria, dtype=int))
    _write_ugrid_netcdf(ugrid, output_path, crs=raster_crs)
    return output_path


# ===========================================================================
# Stage 5: exports from a generated mesh (.nc)
# ===========================================================================

def read_ugrid_mesh(nc_path):
    """Read a stage-4 UGRID NetCDF back into vertex/triangle/depth arrays.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.

    Returns
    -------
    vert : ndarray of shape (N, 2)
        Node x/y coordinates.
    tria : ndarray of shape (M, 3)
        Triangle connectivity (0-based node indices).
    z_depth : ndarray of shape (N,)
        Node depth (positive down), matching the notebook's
        ``interpolate_from_tiff`` convention that ``identify_boundary``
        expects.

    Raises
    ------
    RuntimeError
        If the mesh contains quad faces (only triangle-only meshes are
        supported).
    """
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(nc_path) as nc:
        x = np.asarray(nc["mesh2d_node_x"][:], dtype=float)
        y = np.asarray(nc["mesh2d_node_y"][:], dtype=float)
        z_elev = np.asarray(nc["mesh2d_node_z"][:], dtype=float)
        fn_var = nc["mesh2d_face_nodes"]
        fn = np.ma.filled(fn_var[:], _UGRID_FILL).astype(int)
        start = int(getattr(fn_var, "start_index", 1))

    if fn.shape[1] > 3 and (fn[:, 3] >= start).any():
        raise RuntimeError(
            "The mesh contains quad faces; .grd / boundary export supports "
            "triangle-only meshes (disable quad-preserving smood output).")
    tria = fn[:, :3] - start
    vert = np.column_stack([x, y])
    return vert, tria, -z_elev  # file stores elevation; return depth


def export_boundary_conditions(nc_path, out_dir, zlim=20.0,
                               pli_name="Boundary01", bc_name="Riemann",
                               ext_name="FlowFM_bnd", feedback=None):
    """Write Delft3D-FM open-boundary files (.pli, .bc, .ext) for a mesh.

    Reproduces the notebook: boundary edges deeper than `zlim` are open; the
    **longest** open contour becomes the boundary polyline, with a Riemann
    time-series stanza per point.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    out_dir : str
        Output directory for the three files.
    zlim : float, optional
        Depth threshold (m); boundary edges deeper than `zlim` are
        classified open. Default is 20.0.
    pli_name, bc_name, ext_name : str, optional
        Base names (without extension) for the ``.pli``, ``.bc`` and ``.ext``
        files. Defaults are ``'Boundary01'``, ``'Riemann'`` and
        ``'FlowFM_bnd'``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    pli_path, bc_path, ext_path : str
        Paths to the three written files.

    Raises
    ------
    RuntimeError
        If no open boundary is found at the given threshold.
    """
    feedback = feedback or _NullFeedback()
    import os

    from bluemesh2d.geomesh_util.border_util import identify_boundary

    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    feedback.pushInfo(f"Identifying boundaries (open where depth > {zlim} m) ...")
    boundary = identify_boundary(vert, tria, z_depth, zlim=zlim)
    open_contours = boundary["open_contours"]
    if not open_contours:
        raise RuntimeError(
            f"No open boundary found with threshold {zlim} m; "
            "lower the threshold.")
    contour = max(open_contours, key=len)
    feedback.pushInfo(
        f"Open contours: {len(open_contours)}; using the longest "
        f"({len(contour)} points).")

    xb = vert[contour, 0]
    yb = vert[contour, 1]

    pli_path = os.path.join(out_dir, f"{pli_name}.pli")
    bc_path = os.path.join(out_dir, f"{bc_name}.bc")
    ext_path = os.path.join(out_dir, f"{ext_name}.ext")

    with open(pli_path, "w") as f_pli, open(bc_path, "w") as f_bc:
        f_pli.write(f"{pli_name}\n")
        f_pli.write(f"    {len(xb)}    2\n")
        for i, (xi, yi) in enumerate(zip(xb, yb)):
            boundary_id = f"{pli_name}_{i:04d}"
            f_pli.write(f"{xi:.15E}  {yi:.15E} {boundary_id}\n")
            f_bc.write("[forcing]\n")
            f_bc.write(f"Name = {boundary_id}\n")
            f_bc.write("Function = timeseries\n")
            f_bc.write("Time-interpolation = linear\n")
            f_bc.write("Quantity = time\n")
            f_bc.write("Unit = seconds since 2000-01-01 00:00:00\n")
            f_bc.write("Quantity = riemannbnd\n")
            f_bc.write("Unit = m\n")
            f_bc.write("0    0\n")
            f_bc.write("9999999999   0\n\n")

    with open(ext_path, "w") as f:
        f.write("[general]\n")
        f.write("fileVersion=2.01\n")
        f.write("fileType=extForce\n\n")
        f.write("[boundary]\n")
        f.write("quantity=riemannbnd\n")
        f.write(f"locationFile={pli_name}.pli\n")
        f.write(f"forcingFile={bc_name}.bc\n")

    return pli_path, bc_path, ext_path


def export_grd(nc_path, out_grd, zlim=20.0, crs="EPSG:4326", feedback=None):
    """Write an ADCIRC-style .grd with open/land boundary loops.

    Boundary edges deeper than `zlim` are tagged open, the rest land
    (``identify_boundary`` + ``export_to_grd``, as in the notebook).

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    out_grd : str
        Output ``.grd`` path.
    zlim : float, optional
        Depth threshold (m); boundary edges deeper than `zlim` are
        classified open. Default is 20.0.
    crs : str, optional
        CRS string written to the ``.grd`` header. Default is
        ``'EPSG:4326'``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    out_grd : str
        Path to the written file (same as the input `out_grd`).
    """
    feedback = feedback or _NullFeedback()
    from bluemesh2d.geomesh_util.border_util import identify_boundary
    from bluemesh2d.geomesh_util.grd_util import export_to_grd

    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    feedback.pushInfo(f"Identifying boundaries (open where depth > {zlim} m) ...")
    boundary = identify_boundary(vert, tria, z_depth, zlim=zlim)
    feedback.pushInfo(f"Writing .grd -> {out_grd}")
    export_to_grd(
        out_grd, vert=vert, tria=tria, z=z_depth, crs=crs,
        edge_tag=boundary["edge_tag"],
        edge_open=boundary["edge_open"],
        edge_land=boundary["edge_land"],
    )
    return out_grd


# ===========================================================================
# Boundary condition generation (editable open / closed / island lines)
# ===========================================================================

def _boundary_loops(vert, tria):
    """Assemble the mesh boundary (free) edges into ordered node loops.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Node coordinates.
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based).

    Returns
    -------
    loops : list of list of int
        One list of node indices per closed boundary loop (the first node is
        not repeated at the end).
    """
    import numpy as np
    from collections import defaultdict

    edges = np.vstack([tria[:, [0, 1]], tria[:, [1, 2]], tria[:, [2, 0]]])
    es = np.sort(edges, axis=1)
    uniq, counts = np.unique(es, axis=0, return_counts=True)
    free = uniq[counts == 1]

    adj = defaultdict(list)
    for a, b in free:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    def key(a, b):
        return (a, b) if a < b else (b, a)

    used = set()
    loops = []
    for a0, b0 in free:
        a0, b0 = int(a0), int(b0)
        if key(a0, b0) in used:
            continue
        used.add(key(a0, b0))
        loop = [a0]
        prev, cur = a0, b0
        while cur != a0:
            loop.append(cur)
            nbrs = [n for n in adj[cur] if key(cur, n) not in used]
            pref = [n for n in nbrs if n != prev]
            step = pref or nbrs
            if not step:
                break
            nxt = step[0]
            used.add(key(cur, nxt))
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


def classify_boundary_lines(vert, tria, z_depth, zlim=20.0):
    """Split the mesh boundary into open / closed / island polylines.

    The boundary free edges are assembled into loops. Loops contained inside
    another loop are *islands* (all their edges are coastline). On each outer
    loop, an edge is *open* where the mean node depth exceeds `zlim`, otherwise
    *closed* (land); consecutive edges of the same class form one continuous
    polyline.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Node coordinates (mesh CRS).
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based).
    z_depth : ndarray of shape (N,)
        Node depth (positive down), as returned by :func:`read_ugrid_mesh`.
    zlim : float, optional
        Depth threshold (m); outer-boundary edges deeper than `zlim` are open.
        Default is 20.0.

    Returns
    -------
    lines : dict of {str: list of ndarray}
        Keys ``'open'``, ``'closed'`` and ``'island'``; each value is a list
        of ``(M, 2)`` coordinate arrays (polylines in the mesh CRS).
    """
    import numpy as np
    from shapely.geometry import Polygon

    loops = _boundary_loops(vert, tria)
    polys = [Polygon(vert[lp]) if len(lp) >= 3 else None for lp in loops]

    # a loop is an island if it lies inside another (larger) loop
    is_island = [False] * len(loops)
    for i, pi in enumerate(polys):
        if pi is None or not pi.is_valid:
            continue
        for j, pj in enumerate(polys):
            if i == j or pj is None or not pj.is_valid:
                continue
            if pj.area > pi.area and pj.contains(pi.representative_point()):
                is_island[i] = True
                break

    out = {"open": [], "closed": [], "island": []}
    for lp, island in zip(loops, is_island):
        coords = vert[lp]
        ring = np.vstack([coords, coords[0]])  # close the ring for display
        if island:
            out["island"].append(ring)
            continue

        n = len(lp)
        tags = [0.5 * (z_depth[lp[k]] + z_depth[lp[(k + 1) % n]]) > zlim
                for k in range(n)]
        if all(tags):
            out["open"].append(ring)
            continue
        if not any(tags):
            out["closed"].append(ring)
            continue

        # rotate so the walk starts at a class transition (avoids wrap-around)
        start = next(k for k in range(n) if tags[k] != tags[k - 1])
        eord = [(start + k) % n for k in range(n)]
        runs = [[eord[0]]]
        for e in eord[1:]:
            if tags[e] == tags[runs[-1][-1]]:
                runs[-1].append(e)
            else:
                runs.append([e])
        for run in runs:
            node_seq = [lp[run[0]]] + [lp[(e + 1) % n] for e in run]
            line = vert[node_seq]
            out["open" if tags[run[0]] else "closed"].append(line)
    return out


def classify_boundary_points(vert, tria, z_depth, zlim=20.0):
    """Classify each mesh boundary node as open / closed / island.

    The boundary free edges are assembled into loops (see
    :func:`_boundary_loops`). Loops contained inside another loop are
    *islands* (every node tagged ``'island'``). On outer loops, a node is
    ``'open'`` where its depth exceeds `zlim`, otherwise ``'closed'``.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Node coordinates (mesh CRS).
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based).
    z_depth : ndarray of shape (N,)
        Node depth (positive down), as returned by :func:`read_ugrid_mesh`.
    zlim : float, optional
        Depth threshold (m); outer-boundary nodes deeper than `zlim` are
        open. Default is 20.0.

    Returns
    -------
    loops : list of dict
        One dict per boundary loop, with keys ``'coords'`` (``(n, 2)``
        node coordinates, in walk order, first node not repeated),
        ``'btype'`` (list of ``n`` strings), ``'depth'`` (list of ``n``
        floats) and ``'island'`` (bool).
    """
    import numpy as np
    from shapely.geometry import Polygon

    loops = _boundary_loops(vert, tria)
    polys = [Polygon(vert[lp]) if len(lp) >= 3 else None for lp in loops]

    # a loop is an island if it lies inside another (larger) loop
    is_island = [False] * len(loops)
    for i, pi in enumerate(polys):
        if pi is None or not pi.is_valid:
            continue
        for j, pj in enumerate(polys):
            if i == j or pj is None or not pj.is_valid:
                continue
            if pj.area > pi.area and pj.contains(pi.representative_point()):
                is_island[i] = True
                break

    out = []
    for lp, island in zip(loops, is_island):
        depths = [float(z_depth[k]) for k in lp]
        if island:
            btype = ["island"] * len(lp)
        else:
            btype = ["open" if d > zlim else "closed" for d in depths]
        out.append({"coords": np.asarray(vert[lp], dtype=float),
                    "btype": btype, "depth": depths, "island": island})
    return out


def boundary_lines_from_points(loops):
    """Rebuild open / closed / island polylines from per-node classifications.

    Inverse companion of :func:`classify_boundary_points`, applied after the
    user has edited node types: consecutive edges of the same class form one
    polyline. An edge takes the type of its two nodes when they agree; at an
    open/other transition the edge is not open (the ``.pli`` open boundary
    only spans fully-open stretches), otherwise it takes its first node's
    type.

    Parameters
    ----------
    loops : list of (ndarray of shape (n, 2), list of str)
        Per loop: node coordinates in walk order (first node not repeated)
        and one type string per node.

    Returns
    -------
    lines : dict of {str: list of ndarray}
        Type -> list of ``(M, 2)`` coordinate polylines, as in
        :func:`classify_boundary_lines`.
    """
    import numpy as np

    out = {}
    for coords, btype in loops:
        coords = np.asarray(coords, dtype=float)
        n = len(coords)
        if n < 2:
            continue

        def edge_type(k):
            a, b = btype[k], btype[(k + 1) % n]
            if a == b:
                return a
            if a == "open":
                return b
            if b == "open":
                return a
            return a

        tags = [edge_type(k) for k in range(n)]
        if all(t == tags[0] for t in tags):
            ring = np.vstack([coords, coords[:1]])
            out.setdefault(tags[0], []).append(ring)
            continue

        # rotate so the walk starts at a class transition (avoids wrap-around)
        start = next(k for k in range(n) if tags[k] != tags[k - 1])
        eord = [(start + k) % n for k in range(n)]
        runs = [[eord[0]]]
        for e in eord[1:]:
            if tags[e] == tags[runs[-1][-1]]:
                runs[-1].append(e)
            else:
                runs.append([e])
        for run in runs:
            node_seq = [run[0]] + [(e + 1) % n for e in run]
            out.setdefault(tags[run[0]], []).append(coords[node_seq])
    return out


def generate_boundary_condition_points(nc_path, zlim=20.0, feedback=None):
    """Classify each mesh boundary node as open / closed / island.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    zlim : float, optional
        Depth threshold (m) for the initial open/closed split. Default 20.0.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    loops : list of dict
        See :func:`classify_boundary_points`.
    """
    feedback = feedback or _NullFeedback()
    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    feedback.pushInfo(f"Classifying boundary (open where depth > {zlim} m) ...")
    loops = classify_boundary_points(vert, tria, z_depth, zlim=zlim)
    n_island = sum(1 for lp in loops if lp["island"])
    n_pts = sum(len(lp["btype"]) for lp in loops)
    feedback.pushInfo(
        f"Boundary points: {n_pts} on {len(loops)} loop(s) "
        f"({n_island} island).")
    return loops


def generate_boundary_conditions(nc_path, zlim=20.0, feedback=None):
    """Classify a mesh's boundary into open / closed / island polylines.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    zlim : float, optional
        Depth threshold (m) for the initial open/closed split. Default 20.0.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    lines : dict of {str: list of ndarray}
        See :func:`classify_boundary_lines`.
    """
    feedback = feedback or _NullFeedback()
    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    feedback.pushInfo(f"Classifying boundary (open where depth > {zlim} m) ...")
    lines = classify_boundary_lines(vert, tria, z_depth, zlim=zlim)
    feedback.pushInfo(
        f"Boundary lines: {len(lines['open'])} open, "
        f"{len(lines['closed'])} closed, {len(lines['island'])} island.")
    return lines


def write_open_boundary_pli(out_dir, open_lines, pli_name="Boundary01",
                            feedback=None):
    """Write a Delft3D-FM ``.pli`` polyline file from open boundary polylines.

    Parameters
    ----------
    out_dir : str
        Output directory.
    open_lines : list of array_like of shape (M, 2)
        Open-boundary polylines (coordinates), e.g. the ``'open'`` features of
        the stage-5 boundary-condition layer.
    pli_name : str, optional
        Base name for the ``.pli`` file. Default ``'Boundary01'``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    pli_path : str
        Path to the written file.
    boundary_ids : list of list of str
        Per-line lists of the boundary point ids written to the ``.pli``
        file, in the same order/nesting as `open_lines` -- for use e.g. when
        writing matching ``.bc`` forcing blocks.

    Raises
    ------
    RuntimeError
        If `open_lines` is empty.
    """
    import os
    import numpy as np

    feedback = feedback or _NullFeedback()
    lines = [np.atleast_2d(np.asarray(ln, dtype=float)) for ln in open_lines
             if len(ln) >= 2]
    if not lines:
        raise RuntimeError(
            "No open boundary polyline provided; classify one in "
            "'5 - Generate boundary conditions' (or lower the depth threshold).")

    pli_path = os.path.join(out_dir, f"{pli_name}.pli")
    idx = 0
    boundary_ids = []
    with open(pli_path, "w") as f_pli:
        for li, line in enumerate(lines):
            block = pli_name if len(lines) == 1 else f"{pli_name}_{li:03d}"
            f_pli.write(f"{block}\n")
            f_pli.write(f"    {len(line)}    2\n")
            ids = []
            for xi, yi in line:
                boundary_id = f"{pli_name}_{idx:04d}"
                idx += 1
                ids.append(boundary_id)
                f_pli.write(f"{xi:.15E}  {yi:.15E} {boundary_id}\n")
            boundary_ids.append(ids)

    feedback.pushInfo(f"Open boundary file: {pli_path}")
    return pli_path, boundary_ids


def write_open_boundary_files(out_dir, open_lines, pli_name="Boundary01",
                              bc_name="Riemann", ext_name="FlowFM_bnd",
                              feedback=None):
    """Write Delft3D-FM open-boundary files from open boundary polylines.

    Parameters
    ----------
    out_dir : str
        Output directory.
    open_lines : list of array_like of shape (M, 2)
        Open-boundary polylines (coordinates), e.g. the ``'open'`` features of
        the stage-5 boundary-condition layer.
    pli_name, bc_name, ext_name : str, optional
        Base names for the ``.pli``, ``.bc`` and ``.ext`` files. Defaults
        ``'Boundary01'``, ``'Riemann'``, ``'FlowFM_bnd'``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    pli_path, bc_path, ext_path : str
        Paths to the three written files.

    Raises
    ------
    RuntimeError
        If `open_lines` is empty.
    """
    import os

    feedback = feedback or _NullFeedback()
    pli_path, boundary_ids = write_open_boundary_pli(
        out_dir, open_lines, pli_name=pli_name, feedback=feedback)

    bc_path = os.path.join(out_dir, f"{bc_name}.bc")
    with open(bc_path, "w") as f_bc:
        for ids in boundary_ids:
            for boundary_id in ids:
                f_bc.write("[forcing]\n")
                f_bc.write(f"Name = {boundary_id}\n")
                f_bc.write("Function = timeseries\n")
                f_bc.write("Time-interpolation = linear\n")
                f_bc.write("Quantity = time\n")
                f_bc.write("Unit = seconds since 2000-01-01 00:00:00\n")
                f_bc.write("Quantity = riemannbnd\n")
                f_bc.write("Unit = m\n")
                f_bc.write("0    0\n")
                f_bc.write("9999999999   0\n\n")

    ext_path = os.path.join(out_dir, f"{ext_name}.ext")
    with open(ext_path, "w") as f:
        f.write("[general]\n")
        f.write("fileVersion=2.01\n")
        f.write("fileType=extForce\n\n")
        f.write("[boundary]\n")
        f.write("quantity=riemannbnd\n")
        f.write(f"locationFile={pli_name}.pli\n")
        f.write(f"forcingFile={bc_name}.bc\n")

    feedback.pushInfo(f"Open boundary files: {pli_path}, {bc_path}, {ext_path}")
    return pli_path, bc_path, ext_path


def export_grd_from_lines(nc_path, out_grd, open_lines, land_lines,
                          crs="EPSG:4326", snap_tol=None, feedback=None):
    """Write an ADCIRC ``.grd`` using an edited open/land boundary classification.

    Each polyline vertex is snapped to the nearest mesh node, so the (possibly
    edited) stage-5 lines are mapped back to mesh boundary edges and contours.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    out_grd : str
        Output ``.grd`` path.
    open_lines : list of array_like of shape (M, 2)
        Open-boundary polylines (the ``'open'`` features from stage 5).
    land_lines : list of array_like of shape (M, 2)
        Land-boundary polylines (the ``'closed'`` and ``'island'`` features).
    crs : str, optional
        CRS string written to the ``.grd`` header. Default ``'EPSG:4326'``.
    snap_tol : float or None, optional
        Maximum distance for snapping a vertex to a mesh node; auto-derived
        from the median boundary edge length when ``None``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    out_grd : str
        Path to the written file.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from bluemesh2d.geomesh_util.grd_util import export_to_grd

    feedback = feedback or _NullFeedback()
    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    tree = cKDTree(vert)
    if snap_tol is None:
        # median boundary edge length as a lenient default tolerance
        loops = _boundary_loops(vert, tria)
        d = [np.linalg.norm(vert[lp[k]] - vert[lp[(k + 1) % len(lp)]])
             for lp in loops for k in range(len(lp))]
        snap_tol = (np.median(d) if d else 1.0) * 0.75

    def lines_to_edges_contours(lines):
        edges, contours = [], []
        for ln in lines:
            ln = np.atleast_2d(np.asarray(ln, dtype=float))
            dist, idx = tree.query(ln)
            if np.any(dist > snap_tol):
                feedback.pushWarning(
                    "Some boundary vertices are far from any mesh node; "
                    "the classification may be imprecise (avoid moving "
                    "vertices when editing).")
            seq = [int(i) for i, _ in zip(idx, range(len(idx)))]
            # drop consecutive duplicates from snapping
            seq = [seq[0]] + [b for a, b in zip(seq[:-1], seq[1:]) if a != b]
            if len(seq) >= 2:
                contours.append(np.asarray(seq, dtype=int))
                edges.extend([seq[k], seq[k + 1]] for k in range(len(seq) - 1))
        return (np.asarray(edges, dtype=int) if edges
                else np.empty((0, 2), dtype=int)), contours

    edge_open, open_contours = lines_to_edges_contours(open_lines)
    edge_land, land_contours = lines_to_edges_contours(land_lines)

    tag_o = np.ones((edge_open.shape[0], 1), dtype=int)
    tag_l = np.full((edge_land.shape[0], 1), 2, dtype=int)
    parts = []
    if edge_open.shape[0]:
        parts.append(np.hstack([edge_open, tag_o]))
    if edge_land.shape[0]:
        parts.append(np.hstack([edge_land, tag_l]))
    edge_tag = np.vstack(parts) if parts else np.empty((0, 3), dtype=int)

    feedback.pushInfo(
        f"Writing .grd -> {out_grd} ({edge_open.shape[0]} open, "
        f"{edge_land.shape[0]} land edges)")
    export_to_grd(
        out_grd, vert=vert, tria=tria, z=z_depth, crs=crs,
        edge_tag=edge_tag, edge_open=edge_open, edge_land=edge_land,
        open_contours=open_contours, land_contours=land_contours,
    )
    return out_grd


# ===========================================================================
# All-in-one (composition of the four stages)
# ===========================================================================

@dataclass
class MeshConfig:
    """All tunable parameters of the full meshing pipeline (all-in-one run).

    Attributes
    ----------
    raster_path : str
        Path to the bathymetry raster (elevation, positive up).
    output_path : str
        Output UGRID NetCDF path.
    coast_zmax : float
        Wet threshold (m), see :func:`extract_water_polygon`. Default 2.0.
    domain_buffer : float
        Domain buffer factor, see :func:`extract_water_polygon`.
        Default -0.05.
    keep_largest : bool
        Keep only the largest water polygon. Default ``True``.
    detail_geom : shapely.geometry.base.BaseGeometry or None
        Detail-region polygon in the raster CRS. Default ``None``.
    detail_hmin : float
        Element-size floor (m) inside `detail_geom`. Default 30.0.
    a, b : float
        Depth-polynomial sizing coefficients, see
        :func:`_make_depth_hfun`. Defaults 0.14 and 28.0.
    hmin, hmax : float
        Element-size floor and cap (m). Defaults 100.0 and 10000.0.
    max_gradient : float
        Maximum allowed size gradient (m/m). Default 0.1.
    min_angle_deg : float
        Minimum boundary angle (deg), see :func:`resample_boundary`.
        Default 25.0.
    min_hole_vertices : int
        Minimum hole vertex count, see :func:`resample_boundary`. Default 15.
    kind : {'delaunay', 'delfront'}
        Refinement scheme. Default ``'delaunay'``.
    do_smooth : bool
        Run mesh smoothing after refinement. Default ``True``.
    do_smood : bool
        Run smood orthogonalization after smoothing. Default ``False``.
    smood_merge_small_links : bool
        Enable small-link merging inside smood, see :func:`mesh_pslg`.
        Default ``False``.
    interp_order : int
        Bathymetry sampling order (0=nearest, 1=bilinear, 3=bicubic).
        Default 3.
    """

    raster_path: str
    output_path: str

    coast_zmax: float = 2.0
    domain_buffer: float = -0.05
    keep_largest: bool = True

    detail_geom: object = None       # shapely polygon in the raster CRS
    detail_hmin: float = 30.0

    a: float = 0.14
    b: float = 28.0
    hmin: float = 100.0
    hmax: float = 10000.0
    max_gradient: float = 0.1

    min_angle_deg: float = 25.0
    min_hole_vertices: int = 15

    kind: str = "delaunay"           # refine: 'delaunay' | 'delfront'
    do_smooth: bool = True
    do_smood: bool = False
    smood_merge_small_links: bool = False

    interp_order: int = 3            # bathy sampling: 0=nearest,1=bilinear,3=bicubic


@dataclass
class MeshResult:
    """Result of a :func:`generate_mesh` run.

    Attributes
    ----------
    output_path : str
        Path to the written UGRID NetCDF.
    n_nodes : int
        Number of mesh nodes.
    n_triangles : int
        Number of mesh triangles.
    utm_crs : str
        String representation of the working CRS used.
    """

    output_path: str
    n_nodes: int
    n_triangles: int
    utm_crs: str


def generate_mesh(config: MeshConfig, feedback=None) -> MeshResult:
    """Run the full raster -> UGRID-NetCDF meshing pipeline headlessly.

    Composes stages 1-4 (:func:`extract_water_polygon`, an in-memory
    equivalent of :func:`build_hfun_raster`, :func:`resample_boundary`,
    :func:`mesh_pslg`) and :func:`export_ugrid` in one call.

    Parameters
    ----------
    config : MeshConfig
        Pipeline parameters.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    result : MeshResult
        Summary of the generated mesh.
    """
    feedback = feedback or _NullFeedback()

    missing = check_dependencies()
    if missing:
        raise RuntimeError(
            "Missing Python packages required by BlueMesh2D: "
            + ", ".join(missing)
            + ". Install them into this interpreter (see the plugin README).")
    for mod in optional_dependencies():
        feedback.pushInfo(
            f"Optional package '{mod}' not installed - using the built-in "
            "fallback (slower triangulation; install it for best performance).")

    import pyproj
    from bluemesh2d.geom_util.proj_util import reproject_geometry
    from bluemesh2d.geomesh_util.depth_field import depth_field_from_tif
    from bluemesh2d.hfun_util.smooth_and_precomput import smooth_precomput_hfun

    feedback.setProgress(2)
    poly, utm_crs, raster_crs = extract_water_polygon(
        config.raster_path, coast_zmax=config.coast_zmax,
        domain_buffer=config.domain_buffer,
        keep_largest=config.keep_largest,
        feedback=_SubProgress(feedback, 2, 25))
    feedback.pushInfo(f"Working CRS: {utm_crs.to_string() if hasattr(utm_crs, 'to_string') else utm_crs}")

    # In-memory hfun (no raster round-trip needed for the all-in-one run)
    feedback.pushInfo("Building depth-based size function ...")
    feedback.setProgress(25)
    depth_field = depth_field_from_tif(config.raster_path, output_crs=utm_crs)
    detail_u = None
    if config.detail_geom is not None:
        detail_u = reproject_geometry(config.detail_geom, raster_crs, utm_crs)
    hfun = _make_depth_hfun(
        depth_field, a=config.a, b=config.b,
        hmin=config.hmin, hmax=config.hmax,
        detail=detail_u,
        detail_hmin=(config.detail_hmin if detail_u is not None else None))

    feedback.pushInfo("Gradient-limiting the size function (this can take a moment) ...")
    feedback.setProgress(35)
    # limit the gradient-limiting grid to the water domain (poly is in utm_crs)
    hfuns = smooth_precomput_hfun(hfun, domain=poly, max_gradient=config.max_gradient,
                                  plot=False)
    _check(feedback)

    feedback.pushInfo("Resampling boundary to the size function ...")
    feedback.setProgress(45)
    poly_comput, node, edge = resample_boundary(
        poly, hfuns, config.min_angle_deg, config.min_hole_vertices, feedback)

    feedback.setProgress(55)
    vert, tria = mesh_pslg(node, edge, hfuns, kind=config.kind,
                           do_smooth=config.do_smooth,
                           do_smood=config.do_smood,
                           smood_merge_small_links=config.smood_merge_small_links,
                           feedback=_SubProgress(feedback, 55, 88))

    feedback.setProgress(88)
    export_ugrid(vert, tria, config.raster_path, utm_crs,
                 config.output_path, config.interp_order, feedback)
    feedback.setProgress(100)

    return MeshResult(
        output_path=config.output_path,
        n_nodes=int(len(vert)),
        n_triangles=int(len(tria)),
        utm_crs=utm_crs.to_string() if hasattr(utm_crs, "to_string") else str(utm_crs),
    )
