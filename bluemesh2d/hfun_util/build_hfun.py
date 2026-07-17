"""Stage 2: element-size (hfun) rasters -- build, save and load."""
from __future__ import annotations


from ..feedback import _NullFeedback, _check, _warn_if_ram_risk
from ..geom_util.proj_util import _raster_crs, bundled_raster_data_env


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

    with bundled_raster_data_env(), rasterio.open(raster_path) as src:
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
    with bundled_raster_data_env(), rasterio.open(
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
    with bundled_raster_data_env(), rasterio.open(
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

    with bundled_raster_data_env(), rasterio.open(hfun_path) as src:
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

