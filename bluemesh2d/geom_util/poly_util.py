import numpy as np
from shapely.geometry import Polygon, LineString

try:
    from scipy.spatial import cKDTree
except ImportError:
    cKDTree = None

def simplify_polygon_by_angle(
    polygon: Polygon,
    min_angle_deg: float = 3.0,
) -> Polygon:
    """Remove vertices whose interior angle is below a threshold.

    Applies to the exterior ring and all interior rings (holes).

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to simplify.
    min_angle_deg : float, optional
        Minimum interior angle in degrees. Vertices below this threshold are
        removed. Default is 3.0.

    Returns
    -------
    shapely.geometry.Polygon
        Simplified polygon.
    """

    def _calculate_interior_angle(p1, p2, p3):
        """Interior angle at point p2 between (p1-p2) and (p2-p3), in degrees (0 to 180)."""
        v1 = np.asarray(p1) - np.asarray(p2)
        v2 = np.asarray(p3) - np.asarray(p2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 180.0
        v1_norm = v1 / norm1
        v2_norm = v2 / norm2
        cos_angle = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def _simplify_ring_by_angle(coords, min_angle_deg):
        """Simplify a ring (exterior or interior) by removing only the small angles."""
        coords = np.asarray(coords)
        if len(coords) < 3:
            return coords
        if len(coords) > 0 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        if len(coords) < 3:
            return coords

        result = list(coords)
        max_iterations = len(coords) * 10

        for _ in range(max_iterations):
            if len(result) < 3:
                break
            n_before = len(result)
            n = len(result)
            i = 0
            while i < n:
                prev_idx = (i - 1) % n
                curr_idx = i
                next_idx = (i + 1) % n
                p1 = result[prev_idx]
                p2 = result[curr_idx]
                p3 = result[next_idx]
                interior_angle = _calculate_interior_angle(p1, p2, p3)
                if interior_angle < min_angle_deg:
                    result.pop(curr_idx)
                    n -= 1
                else:
                    i += 1
            if len(result) == n_before:
                break

        if len(result) < 3:
            return coords[:3] if len(coords) >= 3 else coords
        return np.array(result)

    if polygon.is_empty:
        return polygon
    if polygon.type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    # Exterior ring
    exterior_coords = np.array(polygon.exterior.coords[:-1])
    exterior_simplified = _simplify_ring_by_angle(exterior_coords, min_angle_deg)

    # Holes: same treatment as the exterior ring
    interiors_simplified = []
    for interior in polygon.interiors:
        interior_coords = np.array(interior.coords[:-1])
        ring_simplified = _simplify_ring_by_angle(interior_coords, min_angle_deg)
        if len(ring_simplified) >= 3:
            interiors_simplified.append(ring_simplified)

    return Polygon(exterior_simplified, interiors_simplified)

def _resample_ring_hfun(ring_coords, hfun, harg=()):
    """Resample one closed ring using a mesh-size function.

    Parameters
    ----------
    ring_coords : ndarray of shape (N, 2)
        Coordinates of the ring vertices.
    hfun : float or callable
        Mesh-size function.
    harg : tuple, optional
        Extra arguments passed to hfun when callable.
    
    Returns
    -------
    ndarray of shape (M, 2)
        Resampled ring coordinates (closed, first point repeated at end).
    """
    polygon = np.asarray(ring_coords, dtype=float)
    n = polygon.shape[0]
    if n < 2:
        if n == 0:
            # Empty ring: return empty array (will be skipped in resample_polygon_hfun)
            return np.array([]).reshape(0, 2)
        if n == 1:
            # Single point: create a minimal valid ring (4 points)
            p = polygon[0]
            eps_ring = max(np.linalg.norm(p) * 1e-6, 1e-6)
            return np.array([
                p,
                p + np.array([eps_ring, 0]),
                p + np.array([eps_ring, eps_ring]),
                p,  # closed
            ])
        # Two points: add a third point to form a valid ring
        p1, p2 = polygon[0], polygon[1]
        mid = (p1 + p2) / 2.0
        perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
        perp_norm = np.linalg.norm(perp)
        if perp_norm > 0:
            perp = perp / perp_norm * np.linalg.norm(p2 - p1) * 0.1
        else:
            perp = np.array([1e-6, 0])
        p3 = mid + perp
        return np.array([p1, p2, p3, p1])  # closed

    # Cumulative arc length for the closed contour
    nxt = np.roll(polygon, -1, axis=0)
    seg_len = np.linalg.norm(nxt - polygon, axis=1)
    eps = np.finfo(float).eps
    seg_len = np.maximum(seg_len, eps)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_total = s_cum[-1]
    if s_total <= 0:
        # Degenerate ring: ensure at least 3 distinct points
        if n < 3:
            # Create minimal valid ring from available points
            if n == 1:
                p = polygon[0]
                eps_ring = max(np.linalg.norm(p) * 1e-6, 1e-6)
                return np.array([
                    p,
                    p + np.array([eps_ring, 0]),
                    p + np.array([eps_ring, eps_ring]),
                    p,
                ])
            elif n == 2:
                p1, p2 = polygon[0], polygon[1]
                mid = (p1 + p2) / 2.0
                perp = np.array([-(p2[1] - p1[1]), p2[0] - p1[0]])
                perp_norm = np.linalg.norm(perp)
                if perp_norm > 0:
                    perp = perp / perp_norm * np.linalg.norm(p2 - p1) * 0.1
                else:
                    perp = np.array([1e-6, 0])
                p3 = mid + perp
                return np.array([p1, p2, p3, p1])
        return np.vstack([polygon[:3], polygon[0:1]])  # At least 3 points + closure

    def s_to_xy(s):
        """Map arc length(s) in [0, s_total] to (x, y) on the contour (vectorised)."""
        s = np.clip(np.asarray(s, dtype=float), 0.0, s_total)
        i = np.clip(np.searchsorted(s_cum, s, side="right") - 1, 0, n - 1)
        t = (s - s_cum[i]) / seg_len[i]
        return (1.0 - t)[:, None] * polygon[i] + t[:, None] * nxt[i]

    # hfun evaluator with NaN filled from the nearest input vertex
    eval_h = _make_hfun_evaluator(polygon, hfun, harg)

    # Build the mesh-size density integral I(s) = int_0^s ds' / h(s') on a fine
    # arc-length grid. Node count and positions come from this, so the input
    # vertex density does not influence the result.
    h_min = np.nanmin(eval_h(polygon))
    if not np.isfinite(h_min) or h_min <= 0:
        h_min = max(s_total * 0.01, eps)
    ds_fine = max(min(h_min / 4.0, s_total / 200.0), s_total / 20000.0)
    n_fine = int(np.ceil(s_total / ds_fine)) + 1
    s_fine = np.linspace(0.0, s_total, n_fine)
    h_fine = eval_h(s_to_xy(s_fine))
    h_fine = np.where(np.isfinite(h_fine) & (h_fine > 0), h_fine, h_min)
    inv_h = 1.0 / h_fine
    integral = np.concatenate(
        [[0.0], np.cumsum(0.5 * (inv_h[1:] + inv_h[:-1]) * np.diff(s_fine))]
    )
    total_units = integral[-1]

    # Equidistribute nodes: spacing is uniform (= 1 h-unit) in the normalised
    # coordinate, so every edge length is ~h(p). Because that coordinate wraps
    # exactly at total_units, the closing edge is ~h(p) as well -- no vertices
    # end up abnormally close, including at the seam.
    n_nodes = max(int(round(total_units)), 3)
    t_targets = np.arange(n_nodes) * (total_units / n_nodes)
    node = s_to_xy(np.interp(t_targets, integral, s_fine))

    # Close ring: first point repeated at end
    return np.vstack([node, node[0:1]])


def _resample_arc_hfun(arc_coords, hfun, harg=()):
    """Resample one open polyline (arc) using a mesh-size function.

    Both endpoints are preserved exactly; interior nodes are equidistributed
    in arc length weighted by ``1/h(p)``, like :func:`_resample_ring_hfun`.

    Parameters
    ----------
    arc_coords : ndarray of shape (N, 2)
        Coordinates of the arc vertices (open polyline, N >= 2).
    hfun : float or callable
        Mesh-size function.
    harg : tuple, optional
        Extra arguments passed to hfun when callable.

    Returns
    -------
    ndarray of shape (M, 2)
        Resampled arc coordinates (M >= 2, endpoints unchanged).
    """
    arc = np.asarray(arc_coords, dtype=float)
    if arc.shape[0] < 2:
        return arc

    seg = np.diff(arc, axis=0)
    eps = np.finfo(float).eps
    seg_len = np.maximum(np.linalg.norm(seg, axis=1), eps)
    s_cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_total = s_cum[-1]
    if s_total <= 0:
        return arc[[0, -1]]

    def s_to_xy(s):
        s = np.clip(np.asarray(s, dtype=float), 0.0, s_total)
        i = np.clip(np.searchsorted(s_cum, s, side="right") - 1, 0, len(seg_len) - 1)
        t = (s - s_cum[i]) / seg_len[i]
        return (1.0 - t)[:, None] * arc[i] + t[:, None] * arc[i + 1]

    eval_h = _make_hfun_evaluator(arc, hfun, harg)

    h_min = np.nanmin(eval_h(arc))
    if not np.isfinite(h_min) or h_min <= 0:
        h_min = max(s_total * 0.01, eps)
    ds_fine = max(min(h_min / 4.0, s_total / 200.0), s_total / 20000.0)
    n_fine = int(np.ceil(s_total / ds_fine)) + 1
    s_fine = np.linspace(0.0, s_total, n_fine)
    h_fine = eval_h(s_to_xy(s_fine))
    h_fine = np.where(np.isfinite(h_fine) & (h_fine > 0), h_fine, h_min)
    inv_h = 1.0 / h_fine
    integral = np.concatenate(
        [[0.0], np.cumsum(0.5 * (inv_h[1:] + inv_h[:-1]) * np.diff(s_fine))]
    )
    total_units = integral[-1]

    # Equidistribute with the endpoints pinned: n_seg edges of ~h(p) each.
    n_seg = max(int(round(total_units)), 1)
    t_targets = np.linspace(0.0, total_units, n_seg + 1)
    pts = s_to_xy(np.interp(t_targets, integral, s_fine))
    pts[0] = arc[0]
    pts[-1] = arc[-1]
    return pts


def _resample_ring_parts(ring_coords, labels, hfun, harg=()):
    """Resample a closed ring arc-by-arc between part transitions.

    Vertices where the edge part label changes are treated as fixed points:
    they are preserved exactly and each arc between two consecutive fixed
    points is resampled independently with :func:`_resample_arc_hfun`.

    Parameters
    ----------
    ring_coords : ndarray of shape (N, 2)
        Open ring coordinates (no duplicated closing point).
    labels : ndarray of shape (N,), dtype int
        Part label of edge ``i`` (joining vertex ``i`` to ``i+1 mod N``).
    hfun : float or callable
        Mesh-size function.
    harg : tuple, optional
        Extra arguments passed to hfun when callable.

    Returns
    -------
    ring : ndarray of shape (M, 2)
        Resampled ring (closed, first point repeated at end).
    protect : ndarray of shape (M-1,), dtype bool
        True at the fixed (part-junction) vertices of the open ring.
    """
    ring = np.asarray(ring_coords, dtype=float)
    labels = np.asarray(labels, dtype=int)
    n = ring.shape[0]

    # vertex i is fixed when the labels of its two incident edges differ
    breaks = [i for i in range(n) if labels[i - 1] != labels[i]]
    if len(breaks) < 2:
        # 0 breaks: uniform ring; 1 break cannot happen on a closed ring
        closed = _resample_ring_hfun(ring, hfun, harg)
        return closed, np.zeros(max(len(closed) - 1, 0), dtype=bool)

    pieces = []
    protect = []
    for k, i0 in enumerate(breaks):
        i1 = breaks[(k + 1) % len(breaks)]
        if i1 > i0:
            arc = ring[i0 : i1 + 1]
        else:  # wrap around the seam
            arc = np.vstack([ring[i0:], ring[: i1 + 1]])
        res = _resample_arc_hfun(arc, hfun, harg)
        # drop the shared endpoint (added by the next arc's first vertex)
        pieces.append(res[:-1])
        protect.append(
            np.concatenate([[True], np.zeros(len(res) - 2, dtype=bool)])
        )

    pts = np.vstack(pieces)
    protect = np.concatenate(protect)
    return np.vstack([pts, pts[0:1]]), protect


def _make_hfun_evaluator(reference_pts, hfun, harg=()):
    """Build a mesh-size evaluator with NaN filled from the nearest reference vertex."""

    def eval_h_raw(pts):
        pts = np.atleast_2d(pts)
        if np.isscalar(hfun) or isinstance(hfun, (int, float, np.number)):
            return np.full(pts.shape[0], float(hfun))
        return np.asarray(hfun(pts, *harg)).ravel()[: pts.shape[0]]

    reference_pts = np.asarray(reference_pts, dtype=float)
    h_verts = eval_h_raw(reference_pts).astype(float)
    nan_mask = np.isnan(h_verts)
    if np.any(nan_mask):
        tree_ref = cKDTree(reference_pts)
        ok = np.where(~nan_mask)[0]
        if len(ok) == 0:
            h_verts[:] = 1.0
        else:
            for i in np.where(nan_mask)[0]:
                _, j = tree_ref.query(reference_pts[i], k=1)
                j = j if np.isscalar(j) else j[0]
                if nan_mask[j]:
                    h_verts[i] = (
                        np.nanmean(h_verts[~nan_mask])
                        if np.any(~nan_mask)
                        else 1.0
                    )
                else:
                    h_verts[i] = h_verts[j]
    tree_ref = cKDTree(reference_pts)

    def eval_h(pts):
        pts = np.atleast_2d(pts)
        h = eval_h_raw(pts).astype(float)
        nan_pts = np.isnan(h)
        if np.any(nan_pts):
            idx_nan = np.where(nan_pts)[0]
            _, nearest = tree_ref.query(pts[idx_nan], k=1)
            if np.ndim(nearest) == 0:
                nearest = np.array([nearest])
            h[idx_nan] = h_verts[nearest]
        return h

    return eval_h


def _signed_area(ring):
    """Signed area (positive = CCW). ring: (N, 2), closed (last point = first)."""
    r = np.asarray(ring)
    if len(r) < 4:
        return 0.0
    r = r[:-1] if np.allclose(r[0], r[-1]) else r
    x, y = r[:, 0], r[:, 1]
    return 0.5 * (np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def _ensure_ring_orientation(ring_coords, want_ccw):
    """In-place: reverse ring if needed so that CCW == want_ccw."""
    r = np.asarray(ring_coords)
    if len(r) < 4:
        return r
    if np.allclose(r[0], r[-1]):
        r = r[:-1]
    area = _signed_area(np.vstack([r, r[0:1]]))
    is_ccw = area > 0
    if is_ccw != want_ccw:
        r = r[::-1]
    return np.vstack([r, r[0:1]])


def _ring_interior_angles(pts):
    """Interior angle (degrees, 0..180) at each vertex of an open ring (N, 2)."""
    prev = np.roll(pts, 1, axis=0)
    nxt = np.roll(pts, -1, axis=0)
    v1 = prev - pts
    v2 = nxt - pts
    n1 = np.linalg.norm(v1, axis=1)
    n2 = np.linalg.norm(v2, axis=1)
    denom = n1 * n2
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.einsum("ij,ij->i", v1, v2) / denom
    cos = np.clip(cos, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    ang[denom < 1e-20] = 180.0  # degenerate (repeated point) -> not sharp
    return ang


def _prune_ring_by_angle(ring_coords, min_angle_deg, protect=None):
    """Remove vertices at sharp interior angles from a closed ring.

    ``protect`` is an optional boolean mask over the open ring's vertices
    (``ring_coords[:-1]``); protected vertices are never removed.
    """
    if min_angle_deg <= 0:
        return ring_coords
    ring = np.asarray(ring_coords, dtype=float)
    if len(ring) < 4:
        return ring
    pts = ring[:-1].copy() if np.allclose(ring[0], ring[-1]) else ring.copy()
    if len(pts) < 3:
        return ring
    if protect is None:
        prot = np.zeros(len(pts), dtype=bool)
    else:
        prot = np.asarray(protect, dtype=bool)[: len(pts)].copy()

    for _ in range(len(pts) * 2):
        if len(pts) <= 3:
            break
        ang = _ring_interior_angles(pts)
        sharp = (ang < min_angle_deg) & ~prot
        if not sharp.any():
            break
        keep = ~sharp
        if keep.sum() < 3:  # never drop below a valid triangle
            keep = prot.copy()
            order = np.argsort(ang)[::-1]
            for j in order:
                if keep.sum() >= 3:
                    break
                keep[j] = True
        pts = pts[keep]
        prot = prot[keep]

    if len(pts) < 3:
        return ring
    return np.vstack([pts, pts[0:1]])


def resample_polygon_hfun(
    polygon,
    hfun,
    harg=(),
    min_angle_deg=25,
    min_hole_vertices=15,
    part=None,
):
    """Resample a polygon boundary at approximately ``h(p)`` spacing.

    Nodes are equidistributed in arc length weighted by the mesh-size function.
    NaN values from ``hfun`` are replaced by the value at the nearest polygon
    vertex. Optional angle pruning and hole filtering are applied per ring.

    When ``part`` is given, the boundary is resampled iteratively between
    fixed points: every vertex where two different parts meet is preserved
    exactly, and each arc between two consecutive fixed points is resampled
    independently (fixed points are also protected from angle pruning).

    Parameters
    ----------
    polygon : shapely.geometry.Polygon or ndarray of shape (N, 2)
        Polygon to resample. Either a Shapely Polygon (exterior and interiors
        are resampled) or an array of vertices (x, y) in order along the boundary.
    hfun : float or callable
        Mesh-size function. If callable, must have signature hfun(pts, *harg)
        with pts of shape (M, 2), returning mesh-size values (M,) or scalar.
        If float, a constant spacing is used.
    harg : tuple, optional
        Extra arguments passed to hfun when callable.
    min_angle_deg : float, optional
        After resampling, remove vertices whose interior angle is below this
        threshold (sharp spikes / needles that are poor for meshing). Applied to
        the exterior and every hole. Default 0.0 (disabled).
    min_hole_vertices : int, optional
        Drop holes (interiors) that end up with fewer than this many vertices
        after resampling and angle pruning. Default 4 (the minimum for a valid
        ring); raise it to discard small gaps.
    part : list of ndarray, optional
        Boundary partition, same form as ``refine``'s ``part`` argument: each
        entry is an array of 0-based boundary-edge indices defining one part.
        Edges are numbered like :func:`polygon_to_node_edge` builds them:
        edge ``i`` joins boundary vertex ``i`` to vertex ``i+1`` around each
        ring (exterior ring first, then each hole in order). Edges listed in
        no part form an implicit extra part. Vertices where two parts meet
        are kept fixed by the resampling.

    Returns
    -------
    shapely.geometry.Polygon
        Resampled polygon with exterior and holes (interiors) preserved.
        Invalid geometries are fixed with buffer(0).

    Raises
    ------
    ValueError
        If polygon is not a Shapely Polygon or an (N, 2) array.
    ImportError
        If scipy is not available (required for nearest-neighbor NaN fill).
    """
    if cKDTree is None:
        raise ImportError("resample_polygon_hfun requires scipy (scipy.spatial.cKDTree)")

    # Check if input is a Shapely Polygon with holes
    has_interiors = False
    interiors = []
    if hasattr(polygon, "exterior") and hasattr(polygon.exterior, "coords"):
        # Extract exterior
        coords = np.array(polygon.exterior.coords)
        if len(coords) > 1 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        exterior_coords = np.asarray(coords, dtype=float)
        
        # Extract interiors (holes) if present
        if hasattr(polygon, "interiors") and len(polygon.interiors) > 0:
            has_interiors = True
            for interior in polygon.interiors:
                interior_coords = np.array(interior.coords)
                if len(interior_coords) > 1 and np.allclose(interior_coords[0], interior_coords[-1]):
                    interior_coords = interior_coords[:-1]
                interiors.append(np.asarray(interior_coords, dtype=float))
        
        polygon = exterior_coords
    else:
        polygon = np.asarray(polygon, dtype=float)

    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError("polygon must be a Shapely Polygon or an (N, 2) array")

    # Per-ring edge part labels (edge i joins ring vertex i to i+1 mod N;
    # global numbering: exterior edges first, then each hole's, in order)
    ring_sizes = [len(polygon)] + [len(r) for r in interiors]
    nedge_total = sum(ring_sizes)
    ring_labels = [None] * len(ring_sizes)
    if part is not None:
        labels_global = np.full(nedge_total, -1, dtype=int)
        for k, p in enumerate(part):
            p = np.asarray(p, dtype=int).ravel()
            if p.size and (p.min() < 0 or p.max() >= nedge_total):
                raise ValueError(
                    "resample_polygon_hfun: invalid PART edge indices")
            labels_global[p] = k
        offset = 0
        for r, sz in enumerate(ring_sizes):
            lab = labels_global[offset : offset + sz]
            if np.unique(lab).size > 1:  # ring crosses part boundaries
                ring_labels[r] = lab
            offset += sz

    # Resample exterior, then remove sharp "closed angle" spikes
    if ring_labels[0] is not None:
        exterior_ring, prot = _resample_ring_parts(
            polygon, ring_labels[0], hfun, harg)
        exterior_ring = _prune_ring_by_angle(exterior_ring, min_angle_deg, prot)
    else:
        exterior_ring = _resample_ring_hfun(polygon, hfun, harg)
        exterior_ring = _prune_ring_by_angle(exterior_ring, min_angle_deg)
    exterior_ring = _ensure_ring_orientation(exterior_ring, want_ccw=True)

    # Ensure exterior has at least 4 points (required for LinearRing)
    if len(exterior_ring) < 4:
        # If exterior is invalid, return empty polygon
        return Polygon()

    # Resample interiors (holes) if present
    min_hole_vertices = max(int(min_hole_vertices), 4)
    resampled_interiors = []
    if has_interiors:
        for hidx, interior_coords in enumerate(interiors):
            lab = ring_labels[1 + hidx]
            if lab is not None:
                resampled_interior, prot = _resample_ring_parts(
                    interior_coords, lab, hfun, harg)
                resampled_interior = _prune_ring_by_angle(
                    resampled_interior, min_angle_deg, prot)
            else:
                resampled_interior = _resample_ring_hfun(interior_coords, hfun, harg)
                resampled_interior = _prune_ring_by_angle(resampled_interior, min_angle_deg)
            # Drop holes ("gaps") with too few vertices (need >= 4 for a ring)
            n_distinct = len(resampled_interior)
            if n_distinct > 1 and np.allclose(resampled_interior[0], resampled_interior[-1]):
                n_distinct -= 1
            if n_distinct >= min_hole_vertices:
                resampled_interiors.append(resampled_interior)
    
    # Create Polygon with exterior and holes
    if len(resampled_interiors) > 0:
        poly_resampled = Polygon(exterior_ring, resampled_interiors)
    else:
        poly_resampled = Polygon(exterior_ring)
    
    if not poly_resampled.is_valid:
        poly_resampled = poly_resampled.buffer(0)
    return poly_resampled


def _resample_ring_by_spacing(xy: np.ndarray, spacing: float) -> np.ndarray:
    """Resample a closed ring at uniform arc-length spacing."""
    xy = np.asarray(xy, dtype=float)
    if len(xy) < 2:
        return xy

    # Close the ring for length/interpolation purposes
    if not np.allclose(xy[0], xy[-1]):
        xy = np.vstack([xy, xy[0:1]])
    n = len(xy)

    # Cumulative distance along the contour (includes the closing segment)
    d = np.cumsum(
        np.r_[0, np.sqrt(((np.diff(xy, axis=0)) ** 2).sum(axis=1))]
    )
    total_length = d[-1]
    if total_length <= 0:
        return xy[:1]

    # Number of points to get segments >= spacing
    n_pts = max(4, int(np.floor(total_length / spacing)))
    n_pts = min(n_pts, max(4, n - 1))

    # Regularly spaced curvilinear abscissas (without duplicating the closing point)
    d_sampled = np.linspace(0, total_length, n_pts, endpoint=False)

    # Interpolate x and y
    x_new = np.interp(d_sampled, d, xy[:, 0])
    y_new = np.interp(d_sampled, d, xy[:, 1])
    xy_interp = np.column_stack([x_new, y_new])

    # Closed ring for Shapely (first point repeated at the end)
    return np.vstack([xy_interp, xy_interp[0:1]])


def resample_polygon(
    polygon: Polygon,
    spacing: float,
) -> Polygon:
    """Resample polygon boundaries at uniform arc-length spacing.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to resample (exterior and interior rings).
    spacing : float
        Target spacing between consecutive vertices along each ring.

    Returns
    -------
    shapely.geometry.Polygon
        Resampled polygon.

    Raises
    ------
    ValueError
        If ``spacing`` is not positive.
    """
    if spacing <= 0:
        raise ValueError("spacing must be positive")
    if polygon.is_empty:
        return polygon
    if polygon.geom_type == "MultiPolygon":
        polygon = max(polygon.geoms, key=lambda p: p.area)

    # Exterior
    exterior_coords = np.asarray(polygon.exterior.coords)
    exterior_ring = _resample_ring_by_spacing(exterior_coords, spacing)
    if len(exterior_ring) < 4:
        return polygon

    # Interiors
    interiors_rings = []
    for interior in polygon.interiors:
        ring = _resample_ring_by_spacing(np.asarray(interior.coords), spacing)
        if len(ring) >= 4:
            interiors_rings.append(ring)

    poly_new = Polygon(exterior_ring, interiors_rings)
    if not poly_new.is_valid:
        poly_new = poly_new.buffer(0)
    return poly_new


def buffer_area(polygon: Polygon, area_factor: float) -> Polygon:
    """Buffer a polygon by a distance proportional to its area-to-perimeter ratio.

    Parameters
    ----------
    polygon : shapely.geometry.Polygon
        Polygon to buffer.
    area_factor : float
        Multiplier applied to ``polygon.area / polygon.length``.

    Returns
    -------
    shapely.geometry.Polygon
        Buffered polygon.
    """

    return polygon.buffer(area_factor * polygon.area / polygon.length)


def polygon_to_node_edge(poly):
    """Extract PSLG node and edge arrays from a Shapely polygon.

    Parameters
    ----------
    poly : shapely.geometry.Polygon or shapely.geometry.MultiPolygon
        Input polygon geometry.

    Returns
    -------
    node : ndarray of shape (N, 2)
        Vertex coordinates ``(x, y)``.
    edge : ndarray of shape (E, 2), dtype int
        Edge connectivity (0-based vertex indices).

    Raises
    ------
    ValueError
        If any vertex has odd connectivity (open contour).
    """
    # Handle MultiPolygon recursively
    if poly.geom_type == "MultiPolygon":
        nodes_all, edges_all = [], []
        offset = 0
        for p in poly.geoms:
            node, edge = polygon_to_node_edge(p)
            edges_all.append(edge + offset)
            nodes_all.append(node)
            offset += len(node)
        return np.vstack(nodes_all), np.vstack(edges_all)

    ext = np.array(poly.exterior.coords)
    node = [ext[:-1]]  # remove duplicate closing point
    edge = [np.column_stack([np.arange(len(ext) - 1), np.arange(1, len(ext))])]
    edge[-1][-1, 1] = 0  # close loop explicitly

    for hole in poly.interiors:
        pts = np.array(hole.coords)
        n0 = len(np.vstack(node))
        node.append(pts[:-1])  # skip duplicate closure
        e = np.column_stack(
            [np.arange(n0, n0 + len(pts) - 1), np.arange(n0 + 1, n0 + len(pts))]
        )
        e[-1, 1] = n0
        edge.append(e)

    node = np.vstack(node)
    edge = np.vstack(edge).astype(int)

    nnod = node.shape[0]
    nadj = np.bincount(edge.ravel(), minlength=nnod)
    if np.any(nadj % 2 != 0):
        raise ValueError(
            "Invalid topology: some nodes are not closed (odd connectivity)."
        )

    return node, edge
