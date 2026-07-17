"""Stage 1: bathymetry raster -> water-domain polygon (working CRS)."""
from __future__ import annotations


from ..feedback import _NullFeedback, _check, _warn_if_ram_risk
from ..geom_util.proj_util import _raster_crs, bundled_raster_data_env


def _flag_fixed_vertices(p, ext_boundary, tol):
    """Return a PolygonZ copy of ``p`` with Z=1 on vertices on ``ext_boundary``.

    Vertices farther than ``tol`` from the extent boundary get Z=0. The Z
    flag travels inside the polygon geometry to stage 3, which keeps the
    flagged vertices fixed while resampling. The first and last vertex of
    each fixed run (the junctions with the coastline) are additionally
    projected exactly onto the extent boundary -- a sub-``tol`` move -- so
    the run ends at the true intersection; no other vertex is moved.
    """
    import numpy as np
    from shapely.geometry import Point, Polygon

    def ring3(coords):
        pts = np.asarray(coords, dtype=float)[:, :2]
        closed = len(pts) > 1 and np.allclose(pts[0], pts[-1])
        core = pts[:-1] if closed else pts
        n = len(core)
        flag = np.array(
            [ext_boundary.distance(Point(x, y)) <= tol for x, y in core])
        out = [(x, y, 1.0 if f else 0.0)
               for (x, y), f in zip(core, flag)]
        if n >= 2:
            for i in range(n):
                if flag[i] and (not flag[i - 1] or not flag[(i + 1) % n]):
                    q = ext_boundary.interpolate(
                        ext_boundary.project(Point(core[i])))
                    out[i] = (q.x, q.y, 1.0)
        if closed:
            out.append(out[0])
        return out

    return Polygon(ring3(p.exterior.coords),
                   [ring3(r.coords) for r in p.interiors])


def _corner_vertices(geom, min_deviation_deg=30.0):
    """Vertices of a polygon outline where the direction turns sharply.

    Used for the generated (buffered raster extent) clip domain, which has
    no user-drawn vertices: its geometric corners -- where the boundary
    direction deviates by at least ``min_deviation_deg`` -- play the role
    the extent polygon's original vertices play, and are kept fixed.
    """
    import numpy as np

    corners = []
    polys = geom.geoms if hasattr(geom, "geoms") else [geom]
    for p in polys:
        if p.geom_type != "Polygon":
            continue
        for ring in [p.exterior] + list(p.interiors):
            pts = np.asarray(ring.coords, dtype=float)[:, :2]
            if len(pts) > 1 and np.allclose(pts[0], pts[-1]):
                pts = pts[:-1]
            if len(pts) < 3:
                continue
            prev = np.roll(pts, 1, axis=0)
            nxt = np.roll(pts, -1, axis=0)
            a1 = np.arctan2(*(pts - prev).T[::-1])
            a2 = np.arctan2(*(nxt - pts).T[::-1])
            dev = np.abs((a2 - a1 + np.pi) % (2.0 * np.pi) - np.pi)
            corners.extend(pts[dev >= np.radians(min_deviation_deg)])
    return np.asarray(corners, dtype=float).reshape(-1, 2)


def _valid_parts(parts):
    """Repair invalid polygon parts (2D) and flatten any multi results.

    The raw ``coast ∩ domain`` intersection can be invalid (self-touching
    rings); it must be repaired BEFORE the Z flags are attached, because the
    usual repair (``buffer(0)``) drops Z coordinates.
    """
    out = []
    for p in parts:
        if not p.is_valid:
            p = p.buffer(0)
        if p.geom_type == "Polygon":
            if not p.is_empty:
                out.append(p)
        elif hasattr(p, "geoms"):
            out.extend(g for g in p.geoms
                       if g.geom_type == "Polygon" and not g.is_empty)
    return out


def _prune_nonoriginal_fixed(p, keep_xy, tol=1e-3, unflag_only=False):
    """Demote fixed (Z=1) vertices that are neither run endpoints nor listed.

    ``keep_xy`` holds the extent polygon's *original* vertex coordinates
    (working CRS). With ``unflag_only=False`` (extent-polygon case) the other
    fixed vertices -- densification points sitting on straight extent edges
    -- are deleted from the ring; the output keeps just the original extent
    vertices plus the coastline-junction points of each fixed run. With
    ``unflag_only=True`` (generated/buffered domain, whose shape lives in
    those vertices) they are kept geometrically but set free (Z=0), so only
    the coastline junctions stay fixed. Free (Z=0) vertices are never
    touched.
    """
    import numpy as np
    from shapely.geometry import Polygon

    keep_xy = np.asarray(keep_xy, dtype=float).reshape(-1, 2)

    def ring3(coords):
        pts = np.asarray(coords, dtype=float)
        closed = len(pts) > 1 and np.allclose(pts[0], pts[-1])
        core = pts[:-1] if closed else pts
        n = len(core)
        flag = core[:, 2] > 0.5
        keep = np.ones(n, dtype=bool)
        for i in range(n):
            if not flag[i]:
                continue
            if not flag[i - 1] or not flag[(i + 1) % n]:
                continue  # run endpoint: coastline junction, always kept
            if keep_xy.size:
                d2 = np.sum((keep_xy - core[i, :2]) ** 2, axis=1)
                if d2.min() <= tol * tol:
                    continue  # original extent vertex, kept
            if unflag_only:
                core[i, 2] = 0.0
            else:
                keep[i] = False
        core = core[keep]
        out = [tuple(c) for c in core]
        if closed and out:
            out.append(out[0])
        return out

    return Polygon(ring3(p.exterior.coords),
                   [ring3(r.coords) for r in p.interiors])


def extract_water_polygon(raster_path, coast_zmax=2.0, domain_buffer=-0.05,
                          deep_zmax=None, extent_geom=None, keep_largest=True,
                          return_domain=False, feedback=None):
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
    return_domain : bool, optional
        If ``True``, also return the domain polygon actually used for the
        clip (the extent polygon, or the buffered raster extent), in the
        working CRS -- exactly the geometry the water polygon's cut lies on,
        e.g. for flagging fixed vertices. Default is ``False``.
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
    domain_u : shapely geometry, only when ``return_domain=True``
        Clip domain in the working CRS.
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
    with bundled_raster_data_env(), rasterio.open(raster_path) as src:
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
        domain_u = domain
        poly = coast.intersection(domain_u)
    else:
        domain_u = reproject_geometry(domain, raster_crs, utm_crs)
        poly = reproject_geometry(coast, raster_crs, utm_crs).intersection(
            domain_u)
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
    if return_domain:
        return poly, utm_crs, raster_crs, domain_u
    return poly, utm_crs, raster_crs


# ===========================================================================
# Stage 2: raster -> gradient-limited mesh-size (hfun) raster
# ===========================================================================

