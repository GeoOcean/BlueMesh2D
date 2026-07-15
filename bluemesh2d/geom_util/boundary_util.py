"""Stage 3: water polygon + hfun -> resampled boundary and PSLG."""
from __future__ import annotations


from ..feedback import _NullFeedback, _check


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

