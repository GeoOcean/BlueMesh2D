"""
Mesh orthogonalization and small-flow-link handling on in-memory arrays.

Ported from the Delft3D MeshKernel orthogonalizer, working directly on node
coordinates and triangle/mixed-face connectivity (no file I/O). Orthogonality
is measured by ``|cos(phi)|`` on flow links (edge vs. circumcenter line);
small flow links are those whose circumcenter separation is short relative to
the adjacent face sizes. Both are cleared by a zone smoother that combines
Laplacian smoothing, orthogonality-oriented node moves, quality-guarded edge
flips, and circumcenter-separation moves.

Geometry works in lon/lat degrees (``jsferic=1``, spherical formulas) or in a
projected planar CRS (``jsferic=0``). All internal indexing is 0-based, with
-1 marking an invalid entry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple
from collections import deque

import numpy as np

from .constants import (
    EARTH_RADIUS,
    DEG2RAD,
    RAD2DEG,
    EARTH_RADIUS_DEG2RAD,
    EARTH_RADIUS_SQ,
    DTOL_POLE,
    DEFAULT_ORTHO_ALPHA,
)
from .geometry import build_edges_from_tria as _build_edges_from_tria


def _getdx_vec(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, jsferic: int = 1
) -> np.ndarray:
    """Vectorized version of Delft's _getdx. x1, y1, x2, y2 same shape."""
    if jsferic != 1:
        return (x2 - x1).astype(np.float64)
    pole1 = np.abs(np.abs(y1) - 90.0) <= DTOL_POLE
    pole2 = np.abs(np.abs(y2) - 90.0) <= DTOL_POLE
    different_poles = pole1 != pole2
    xx1 = x1.astype(np.float64)
    mask_hi = (xx1 - x2) > 180.0
    mask_lo = (xx1 - x2) < -180.0
    xx1 = np.where(mask_hi, xx1 - 360.0, xx1)
    xx1 = np.where(mask_lo, xx1 + 360.0, xx1)
    c = np.cos(0.5 * (y1 + y2) * DEG2RAD)
    out = EARTH_RADIUS_DEG2RAD * c * (x2 - xx1)
    out = np.where(different_poles, 0.0, out)
    return out


def _getdy_vec(
    x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray, jsferic: int = 1
) -> np.ndarray:
    """Vectorized version of Delft's _getdy."""
    if jsferic != 1:
        return (y2 - y1).astype(np.float64)
    return (EARTH_RADIUS_DEG2RAD * (y2 - y1)).astype(np.float64)


def _gather_face_node_coords(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    faces: np.ndarray,
    n: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Coordinates of the valid (1-based, > 0) nodes of `faces`, which must all
    have exactly `n` valid nodes. Returns (xv, yv), each (len(faces), n),
    with the valid nodes of each row in their original order.
    """
    sub = face_nodes[faces, :]
    order = np.argsort(sub <= 0, axis=1, kind="stable")
    idx = np.take_along_axis(sub, order[:, :n], axis=1) - 1
    return node_x[idx].astype(np.float64), node_y[idx].astype(np.float64)


def _face_centers(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    jsferic: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Barycentric cell centers (comp_masscenter2D; jsferic=0 for planar x/y)."""
    n_faces = face_nodes.shape[0]
    if face_mask is not None:
        face_x = np.full(n_faces, np.nan, dtype=np.float64)
        face_y = np.full(n_faces, np.nan, dtype=np.float64)
        faces_to_do = np.where(face_mask)[0]
    else:
        face_x = np.zeros(n_faces, dtype=np.float64)
        face_y = np.zeros(n_faces, dtype=np.float64)
        faces_to_do = np.arange(n_faces)
    if faces_to_do.size == 0:
        return face_x, face_y
    counts = np.sum(face_nodes[faces_to_do, :] > 0, axis=1)
    # Vectorized per group of equal node count (faces with 0 nodes keep the
    # initial fill value, as in the historical per-face loop).
    for n in np.unique(counts):
        if n == 0:
            continue
        faces = faces_to_do[counts == n]
        xin, yin = _gather_face_node_coords(node_x, node_y, face_nodes, faces, int(n))
        y0 = yin[np.arange(faces.size), np.argmin(np.abs(yin), axis=1)]
        x = xin.copy()
        if jsferic == 1:
            xmax = np.max(x, axis=1)
            wrap = (xmax - np.min(x, axis=1)) > 180.0
            if np.any(wrap):
                x = np.where(
                    wrap[:, None] & (x < (xmax - 180.0)[:, None]), x + 360.0, x
                )
        x0 = np.min(x, axis=1)
        dxs = np.empty_like(x)
        dys = np.empty_like(x)
        for i in range(n):
            dxs[:, i] = _getdx_vec(x0, y0, x[:, i], yin[:, i], jsferic)
            dys[:, i] = _getdy_vec(x0, y0, x[:, i], yin[:, i], jsferic)
        area = np.zeros(faces.size, dtype=np.float64)
        xcg = np.zeros(faces.size, dtype=np.float64)
        ycg = np.zeros(faces.size, dtype=np.float64)
        for i in range(n):
            ip1 = (i + 1) % n
            xc = 0.5 * (dxs[:, i] + dxs[:, ip1])
            yc = 0.5 * (dys[:, i] + dys[:, ip1])
            dxe = _getdx_vec(x[:, i], yin[:, i], x[:, ip1], yin[:, ip1], jsferic)
            dye = _getdy_vec(x[:, i], yin[:, i], x[:, ip1], yin[:, ip1], jsferic)
            dsx, dsy = dye, -dxe
            xds = xc * dsx + yc * dsy
            area += 0.5 * xds
            xcg += xds * xc
            ycg += xds * yc
        degenerate = np.abs(area) < 1e-8
        area_safe = np.where(
            degenerate, 1.0, np.sign(area) * np.maximum(np.abs(area), 1e-8)
        )
        fac = 1.0 / (3.0 * area_safe)
        if jsferic == 1:
            fy = y0 + (ycg * fac) / EARTH_RADIUS_DEG2RAD
            fx = x0 + (xcg * fac) / (EARTH_RADIUS_DEG2RAD * np.cos(fy * DEG2RAD))
        else:
            fy = y0 + ycg * fac
            fx = x0 + xcg * fac
        face_x[faces] = np.where(degenerate, np.mean(xin, axis=1), fx)
        face_y[faces] = np.where(degenerate, np.mean(yin, axis=1), fy)
    return face_x, face_y


def _sphertocart3d_vec(lon_deg: np.ndarray, lat_deg: np.ndarray) -> np.ndarray:
    """Vectorized: (N,) -> (N, 3) in Cartesian meters."""
    lon_deg = np.asarray(lon_deg, dtype=np.float64)
    lat_deg = np.asarray(lat_deg, dtype=np.float64)
    r = EARTH_RADIUS * np.cos(lat_deg * DEG2RAD)
    xx = r * np.cos(lon_deg * DEG2RAD)
    yy = r * np.sin(lon_deg * DEG2RAD)
    zz = EARTH_RADIUS * np.sin(lat_deg * DEG2RAD)
    return np.column_stack([xx, yy, zz])


def _cart3dtospher(
    xx: float, yy: float, zz: float, xref_deg: float
) -> Tuple[float, float]:
    """Cartesian (meters) -> spherical (deg)."""
    raddeg = 180.0 / np.pi
    x1 = np.arctan2(yy, xx) * raddeg
    y1 = np.arctan2(zz, np.sqrt(xx * xx + yy * yy)) * raddeg
    x1 = x1 + np.round((xref_deg - x1) / 360.0) * 360.0
    return float(x1), float(y1)


def _cart3dtospher_vec(
    xx: np.ndarray, yy: np.ndarray, zz: np.ndarray, xref_deg: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized `_cart3dtospher`: all args (N,) -> (lon_deg, lat_deg)."""
    raddeg = 180.0 / np.pi
    x1 = np.arctan2(yy, xx) * raddeg
    y1 = np.arctan2(zz, np.sqrt(xx * xx + yy * yy)) * raddeg
    x1 = x1 + np.round((xref_deg - x1) / 360.0) * 360.0
    return x1, y1


def _getdx(x1: float, y1: float, x2: float, y2: float, jsferic: int = 1) -> float:
    """Scalar version of Delft _getdx (copied from meshkernel_orthogonality)."""
    if jsferic != 1:
        return float(x2 - x1)
    if (abs(abs(y1) - 90.0) <= DTOL_POLE) != (abs(abs(y2) - 90.0) <= DTOL_POLE):
        return 0.0
    xx1, xx2 = x1, x2
    if xx1 - xx2 > 180.0:
        xx1 -= 360.0
    elif xx1 - xx2 < -180.0:
        xx1 += 360.0
    c = np.cos(0.5 * (y1 + y2) * DEG2RAD)
    return float(EARTH_RADIUS_DEG2RAD * c * (xx2 - xx1))


def _getdy(x1: float, y1: float, x2: float, y2: float, jsferic: int = 1) -> float:
    """Scalar version of Delft _getdy (copied from meshkernel_orthogonality)."""
    if jsferic != 1:
        return float(y2 - y1)
    return float(EARTH_RADIUS_DEG2RAD * (y2 - y1))


# ---------------------------------------------------------------------------
# Small flow links (Delft3D): circumcenters in lon/lat, dxlink < 0.9*thresh*0.5*(sqrt(ba1)+sqrt(ba2))
# ---------------------------------------------------------------------------


def _lonlat_to_local_xy(
    node_x: np.ndarray, node_y: np.ndarray, jsferic: int = 1
) -> Tuple[np.ndarray, float, float]:
    """Lon/lat (deg) -> local planar (m) around reference (x0,y0) using _getdx/_getdy."""
    node_x = np.asarray(node_x, dtype=np.float64)
    node_y = np.asarray(node_y, dtype=np.float64)
    x0 = float(np.nanmean(node_x))
    y0 = float(np.nanmean(node_y))
    x0_arr = np.full_like(node_x, x0)
    y0_arr = np.full_like(node_y, y0)
    dx = _getdx_vec(x0_arr, y0_arr, node_x, node_y, jsferic)
    dy = _getdy_vec(x0_arr, y0_arr, node_x, node_y, jsferic)
    vert_xy = np.column_stack([dx, dy])
    return vert_xy, x0, y0


def _triarea_2d(pp: np.ndarray, tt: np.ndarray) -> np.ndarray:
    """Signed triangle areas in 2D (pp: nnode x 2, tt: nface x 3, 0-based indices)."""
    ev12 = pp[tt[:, 1], :] - pp[tt[:, 0], :]
    ev13 = pp[tt[:, 2], :] - pp[tt[:, 0], :]
    area = 0.5 * (ev12[:, 0] * ev13[:, 1] - ev12[:, 1] * ev13[:, 0])
    return area


def _circumcenter_of_triangle_lonlat(
    n0: np.ndarray, n1: np.ndarray, n2: np.ndarray
) -> np.ndarray:
    """Circumcenter of triangle in lon/lat (deg), MeshKernel spherical formula."""
    x1, y1 = float(n0[0]), float(n0[1])
    x2, y2 = float(n1[0]), float(n1[1])
    x3, y3 = float(n2[0]), float(n2[1])
    dx2 = _getdx(x1, y1, x2, y2, 1)
    dy2 = _getdy(x1, y1, x2, y2, 1)
    dx3 = _getdx(x1, y1, x3, y3, 1)
    dy3 = _getdy(x1, y1, x3, y3, 1)
    den = dy2 * dx3 - dy3 * dx2
    z = (dx2 * (dx2 - dx3) + dy2 * (dy2 - dy3)) / den if abs(den) > 1e-20 else 0.0
    phi = (y1 + y2 + y3) / 3.0
    xf = 1.0 / np.cos(phi * DEG2RAD)
    cx = x1 + xf * 0.5 * (dx3 - z * dy3) * RAD2DEG / EARTH_RADIUS
    cy = y1 + 0.5 * (dy3 + z * dx3) * RAD2DEG / EARTH_RADIUS
    return np.array([cx, cy], dtype=np.float64)


def _cross_product_cartesian2d(
    seg_a: np.ndarray, seg_b: np.ndarray, point: np.ndarray
) -> float:
    """Cross product (seg_b - seg_a) x (point - seg_a) in (x,y) plane."""
    return float(
        (seg_b[0] - seg_a[0]) * (point[1] - seg_a[1])
        - (seg_b[1] - seg_a[1]) * (point[0] - seg_a[0])
    )


def _point_in_triangle_winding_lonlat(
    point: np.ndarray, v0: np.ndarray, v1: np.ndarray, v2: np.ndarray
) -> bool:
    """Point-in-triangle via winding in (lon, lat), MeshKernel-style."""
    tol = 1e-12
    winding = 0
    for va, vb in ((v0, v1), (v1, v2), (v2, v0)):
        cp = _cross_product_cartesian2d(va, vb, point)
        if abs(cp) <= tol:
            return True
        if va[1] <= point[1]:
            if vb[1] > point[1] and cp > 0:
                winding += 1
        else:
            if vb[1] <= point[1] and cp < 0:
                winding -= 1
    return winding != 0


def _segments_crossing_ratio_intersection_lonlat(
    p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, p4: np.ndarray
) -> Optional[Tuple[float, np.ndarray]]:
    """If (p1,p2) and (p3,p4) cross, return (ratio_first, intersection) in lon/lat."""
    x21 = _getdx(p1[0], p1[1], p2[0], p2[1], 1)
    y21 = _getdy(p1[0], p1[1], p2[0], p2[1], 1)
    x43 = _getdx(p3[0], p3[1], p4[0], p4[1], 1)
    y43 = _getdy(p3[0], p3[1], p4[0], p4[1], 1)
    x31 = _getdx(p1[0], p1[1], p3[0], p3[1], 1)
    y31 = _getdy(p1[0], p1[1], p3[0], p3[1], 1)
    det = x43 * y21 - y43 * x21
    max_val = max(abs(x21), abs(y21), abs(x43), abs(y43), 1e-30)
    if abs(det) < max(1e-10 * max_val, 1e-15):
        return None
    ratio_second = (y31 * x21 - x31 * y21) / det
    ratio_first = (y31 * x43 - x31 * y43) / det
    if not (0.0 <= ratio_first <= 1.0 and 0.0 <= ratio_second <= 1.0):
        return None
    inter = p1 + ratio_first * (p2 - p1)
    return (float(ratio_first), np.asarray(inter, dtype=np.float64))


def _point_in_triangle_winding_lonlat_vec(
    px: np.ndarray,
    py: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> np.ndarray:
    """Vectorized `_point_in_triangle_winding_lonlat`: (F,) points vs (F,2) vertices."""
    tol = 1e-12
    winding = np.zeros(px.shape[0], dtype=np.int64)
    on_edge = np.zeros(px.shape[0], dtype=bool)
    for va, vb in ((v0, v1), (v1, v2), (v2, v0)):
        cp = (vb[:, 0] - va[:, 0]) * (py - va[:, 1]) - (vb[:, 1] - va[:, 1]) * (
            px - va[:, 0]
        )
        on_edge |= np.abs(cp) <= tol
        up = va[:, 1] <= py
        winding += (up & (vb[:, 1] > py) & (cp > 0)).astype(np.int64)
        winding -= (~up & (vb[:, 1] <= py) & (cp < 0)).astype(np.int64)
    return on_edge | (winding != 0)


def _circumcenters_lonlat_compact(
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
    num_interior: np.ndarray,
    jsferic: int = 1,
) -> np.ndarray:
    """
    Circumcenters in lon/lat (or planar x/y for jsferic=0) for triangles given
    by vertex coordinates (each (F, 2), all valid). Faces with
    num_interior == 0 (boundary) use the mass center; circumcenters outside
    their triangle are pulled inside. Returns (F, 2).
    """
    out = np.empty((v0.shape[0], 2), dtype=np.float64)
    mass = (v0 + v1 + v2) / 3.0

    boundary = num_interior == 0
    out[boundary] = mass[boundary]

    keep = np.where(~boundary)[0]
    if keep.size == 0:
        return out
    v0, v1, v2, mass = v0[keep], v1[keep], v2[keep], mass[keep]

    # Vectorized `_circumcenter_of_triangle_lonlat` (MeshKernel formula; the
    # planar variant is the same construction without the metric conversions).
    x1, y1 = v0[:, 0], v0[:, 1]
    dx2 = _getdx_vec(x1, y1, v1[:, 0], v1[:, 1], jsferic)
    dy2 = _getdy_vec(x1, y1, v1[:, 0], v1[:, 1], jsferic)
    dx3 = _getdx_vec(x1, y1, v2[:, 0], v2[:, 1], jsferic)
    dy3 = _getdy_vec(x1, y1, v2[:, 0], v2[:, 1], jsferic)
    den = dy2 * dx3 - dy3 * dx2
    den_ok = np.abs(den) > 1e-20
    z = np.where(
        den_ok,
        (dx2 * (dx2 - dx3) + dy2 * (dy2 - dy3)) / np.where(den_ok, den, 1.0),
        0.0,
    )
    if jsferic == 1:
        phi = (y1 + v1[:, 1] + v2[:, 1]) / 3.0
        xf = 1.0 / np.cos(phi * DEG2RAD)
        circum = np.column_stack(
            [
                x1 + xf * 0.5 * (dx3 - z * dy3) * RAD2DEG / EARTH_RADIUS,
                y1 + 0.5 * (dy3 + z * dx3) * RAD2DEG / EARTH_RADIUS,
            ]
        )
    else:
        circum = np.column_stack(
            [
                x1 + 0.5 * (dx3 - z * dy3),
                y1 + 0.5 * (dy3 + z * dx3),
            ]
        )

    inside = _point_in_triangle_winding_lonlat_vec(
        circum[:, 0], circum[:, 1], v0, v1, v2
    )
    out[keep[inside]] = circum[inside]

    # Pull-inside for circumcenters outside their triangle: first crossing of
    # the mass-center -> circumcenter segment with a triangle edge wins (as
    # `_segments_crossing_ratio_intersection_lonlat`), else the mass center.
    o = np.where(~inside)[0]
    if o.size > 0:
        p1 = mass[o]
        p2 = circum[o]
        x21 = _getdx_vec(p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1], jsferic)
        y21 = _getdy_vec(p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1], jsferic)
        vs = (v0[o], v1[o], v2[o])
        result = p1.copy()
        found = np.zeros(o.size, dtype=bool)
        for n in range(3):
            p3 = vs[n]
            p4 = vs[(n + 1) % 3]
            x43 = _getdx_vec(p3[:, 0], p3[:, 1], p4[:, 0], p4[:, 1], jsferic)
            y43 = _getdy_vec(p3[:, 0], p3[:, 1], p4[:, 0], p4[:, 1], jsferic)
            x31 = _getdx_vec(p1[:, 0], p1[:, 1], p3[:, 0], p3[:, 1], jsferic)
            y31 = _getdy_vec(p1[:, 0], p1[:, 1], p3[:, 0], p3[:, 1], jsferic)
            det = x43 * y21 - y43 * x21
            max_val = np.maximum.reduce(
                [np.abs(x21), np.abs(y21), np.abs(x43), np.abs(y43)]
            )
            max_val = np.maximum(max_val, 1e-30)
            hit = np.abs(det) >= np.maximum(1e-10 * max_val, 1e-15)
            det_safe = np.where(hit, det, 1.0)
            ratio_second = (y31 * x21 - x31 * y21) / det_safe
            ratio_first = (y31 * x43 - x31 * y43) / det_safe
            hit &= (
                (0.0 <= ratio_first)
                & (ratio_first <= 1.0)
                & (0.0 <= ratio_second)
                & (ratio_second <= 1.0)
            )
            take = hit & ~found
            if np.any(take):
                inter = p1 + ratio_first[:, None] * (p2 - p1)
                result[take] = inter[take]
                found |= take
        out[keep[o]] = result

    return out


def _num_interior_edges_per_face(edge_faces: np.ndarray, nface: int) -> np.ndarray:
    """Number of interior edges (edge with two distinct valid faces) per face."""
    interior = (
        (edge_faces[:, 0] >= 0)
        & (edge_faces[:, 1] >= 0)
        & (edge_faces[:, 0] != edge_faces[:, 1])
    )
    counts = np.bincount(
        np.concatenate([edge_faces[interior, 0], edge_faces[interior, 1]]).astype(
            np.intp
        ),
        minlength=nface,
    )
    return counts.astype(np.int32)


def _circumcenters_lonlat_ugrid(
    vert_deg: np.ndarray,
    face_nodes: np.ndarray,
    edge_faces: np.ndarray,
    face_mask: Optional[np.ndarray] = None,
    jsferic: int = 1,
    num_interior: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Circumcenters in lon/lat per face (UGRID). Boundary faces use mass center.
    numberOfInteriorEdges is derived from edge_faces (or passed precomputed).
    If face_mask is provided, only compute for faces where face_mask is True (faster for zones).
    """
    nface = face_nodes.shape[0]
    if num_interior is None:
        num_interior = _num_interior_edges_per_face(edge_faces, nface)

    out = (
        np.full((nface, 2), np.nan, dtype=np.float64)
        if face_mask is not None
        else np.zeros((nface, 2), dtype=np.float64)
    )
    tria = face_nodes[:, :3]
    indices_to_compute = (
        np.where(face_mask)[0]
        if face_mask is not None
        else np.arange(nface, dtype=np.int64)
    )
    if indices_to_compute.size == 0:
        return out

    t_nodes = tria[indices_to_compute]
    invalid = np.any(t_nodes < 0, axis=1)
    out[indices_to_compute[invalid]] = np.nan

    faces = indices_to_compute[~invalid]
    if faces.size == 0:
        return out
    out[faces] = _circumcenters_lonlat_compact(
        vert_deg[tria[faces, 0]],
        vert_deg[tria[faces, 1]],
        vert_deg[tria[faces, 2]],
        num_interior[faces],
        jsferic=jsferic,
    )
    return out


def compute_small_links_from_arrays(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    edge_nodes: np.ndarray,
    edge_faces: np.ndarray,
    removesmalllinkstrsh: float = 0.11,
    edge_indices: Optional[np.ndarray] = None,
    jsferic: int = 1,
    num_interior: Optional[np.ndarray] = None,
) -> Tuple[int, np.ndarray]:
    """
    Small flow links (Delft3D): dxlink < 0.9*removesmalllinkstrsh*0.5*(sqrt(ba1)+sqrt(ba2)).
    Inputs 0-based (invalid = -1). Returns (n_small, edge_indices_of_small_links).
    If edge_indices is provided, only those edges are tested (returned small_edges are a subset).
    Coordinates are lon/lat degrees for jsferic=1, planar x/y for jsferic=0.
    `num_interior` may pass the precomputed `_num_interior_edges_per_face`
    (topology-only) to avoid the O(n_edges) recount in tight trial loops.
    """
    node_x = np.asarray(node_x, dtype=np.float64).ravel()
    node_y = np.asarray(node_y, dtype=np.float64).ravel()
    nface = face_nodes.shape[0]
    n_edges = edge_faces.shape[0]
    if nface == 0 or n_edges == 0:
        return 0, np.array([], dtype=np.int64)

    no_small = (0, np.array([], dtype=np.int64))
    edges_to_test = (
        np.asarray(edge_indices, dtype=np.int64).ravel()
        if edge_indices is not None
        else np.arange(n_edges, dtype=np.int64)
    )
    ee = edges_to_test[(edges_to_test >= 0) & (edges_to_test < n_edges)]
    if ee.size == 0:
        return no_small
    f1 = edge_faces[ee, 0].astype(np.int64)
    f2 = edge_faces[ee, 1].astype(np.int64)
    ok = (f1 >= 0) & (f2 >= 0) & (f1 != f2) & (f1 < nface) & (f2 < nface)
    if not np.any(ok):
        return no_small

    # Work only on the faces adjacent to the tested edges (big speedup when
    # testing a small subset).
    tria = face_nodes[:, :3]
    cand = np.unique(np.concatenate([f1[ok], f2[ok]]))
    valid_cand = (tria[cand] >= 0).all(axis=1)
    rows = np.where(ok)[0]
    g1 = np.searchsorted(cand, f1[rows])
    g2 = np.searchsorted(cand, f2[rows])
    keep = valid_cand[g1] & valid_cand[g2]
    rows = rows[keep]
    if rows.size == 0:
        return no_small

    faces_needed = cand[valid_cand]
    pos_in_needed = np.cumsum(valid_cand) - 1
    g1 = pos_in_needed[g1[keep]]
    g2 = pos_in_needed[g2[keep]]

    # Circumcenters (lon/lat) of the needed faces.
    if num_interior is None:
        num_interior = _num_interior_edges_per_face(edge_faces, nface)
    t0, t1, t2 = tria[faces_needed, 0], tria[faces_needed, 1], tria[faces_needed, 2]
    circ = _circumcenters_lonlat_compact(
        np.column_stack([node_x[t0], node_y[t0]]),
        np.column_stack([node_x[t1], node_y[t1]]),
        np.column_stack([node_x[t2], node_y[t2]]),
        num_interior[faces_needed],
        jsferic=jsferic,
    )

    # Face areas in local planar coordinates (as `_lonlat_to_local_xy` +
    # `_triarea_2d`, computed only for the nodes of the needed faces; the
    # reference point stays the full-mesh mean).
    x0 = float(np.nanmean(node_x))
    y0 = float(np.nanmean(node_y))
    nodes_used = np.unique(np.concatenate([t0, t1, t2]))
    x0a = np.full(nodes_used.size, x0)
    y0a = np.full(nodes_used.size, y0)
    ux = _getdx_vec(x0a, y0a, node_x[nodes_used], node_y[nodes_used], jsferic)
    uy = _getdy_vec(x0a, y0a, node_x[nodes_used], node_y[nodes_used], jsferic)
    q0 = np.searchsorted(nodes_used, t0)
    q1 = np.searchsorted(nodes_used, t1)
    q2 = np.searchsorted(nodes_used, t2)
    ev12x, ev12y = ux[q1] - ux[q0], uy[q1] - uy[q0]
    ev13x, ev13y = ux[q2] - ux[q0], uy[q2] - uy[q0]
    ba = np.abs(0.5 * (ev12x * ev13y - ev12y * ev13x))

    c1 = circ[g1]
    c2 = circ[g2]
    good = ~np.any(np.isnan(c1), axis=1) & ~np.any(np.isnan(c2), axis=1)
    dx = _getdx_vec(c1[:, 0], c1[:, 1], c2[:, 0], c2[:, 1], jsferic)
    dy = _getdy_vec(c1[:, 0], c1[:, 1], c2[:, 0], c2[:, 1], jsferic)
    dxlink = np.sqrt(dx * dx + dy * dy)
    sqrt_ba1 = np.sqrt(np.maximum(ba[g1], 1e-20))
    sqrt_ba2 = np.sqrt(np.maximum(ba[g2], 1e-20))
    dxlim = 0.9 * removesmalllinkstrsh * 0.5 * (sqrt_ba1 + sqrt_ba2)
    small_edges = ee[rows[good & (dxlink < dxlim)]]

    return int(small_edges.size), small_edges.astype(np.int64)


def _signed_area_tri_deg(
    node_x: np.ndarray, node_y: np.ndarray, i: int, j: int, k: int
) -> float:
    """Signed area (doubled) of triangle (i,j,k) in lon/lat degrees. Positive = CCW."""
    xi, yi = node_x[i], node_y[i]
    xj, yj = node_x[j], node_y[j]
    xk, yk = node_x[k], node_y[k]
    return (xj - xi) * (yk - yi) - (xk - xi) * (yj - yi)


def try_flip_small_flow_edge_ugrid(
    mesh: "MeshData",
    edge_index: int,
) -> bool:
    """
    Try to fix a small flow link by flipping the shared edge (swap diagonal of the quad).
    The two triangles must form a strictly convex quadrilateral. Modifies mesh.face_nodes in place.
    Returns True if the edge was flipped.
    """
    face_nodes = mesh.face_nodes
    edge_nodes = mesh.edge_nodes
    edge_faces = mesh.edge_faces
    node_x = mesh.node_x
    node_y = mesh.node_y
    n_face = face_nodes.shape[0]
    if edge_index < 0 or edge_index >= edge_faces.shape[0]:
        return False
    k3, k4 = int(edge_nodes[edge_index, 0]), int(edge_nodes[edge_index, 1])
    f1, f2 = int(edge_faces[edge_index, 0]), int(edge_faces[edge_index, 1])
    if k3 < 0 or k4 < 0 or f1 < 0 or f2 < 0 or f1 == f2 or f1 >= n_face or f2 >= n_face:
        return False
    tri1 = face_nodes[f1, :3].copy()
    tri2 = face_nodes[f2, :3].copy()
    opp1 = None
    for v in tri1:
        v = int(v)
        if v >= 0 and v != k3 and v != k4:
            opp1 = v
            break
    opp2 = None
    for v in tri2:
        v = int(v)
        if v >= 0 and v != k3 and v != k4:
            opp2 = v
            break
    if opp1 is None or opp2 is None or opp1 == opp2:
        return False
    # Convex quad: k3 and k4 on opposite sides of line (opp1, opp2)
    sa1 = _signed_area_tri_deg(node_x, node_y, opp1, opp2, k3)
    sa2 = _signed_area_tri_deg(node_x, node_y, opp1, opp2, k4)
    if sa1 * sa2 >= 0:
        return False
    # Both new triangles must have positive area (CCW). The sign of sa1 only
    # reflects the arbitrary storage order of the apexes: if k3 is on the
    # negative side of (opp1 -> opp2), swap the apexes instead of refusing a
    # geometrically valid flip.
    if sa1 <= 0:
        opp1, opp2 = opp2, opp1
    # New triangles: (opp1, opp2, k3) and (opp2, opp1, k4)
    face_nodes[f1, 0], face_nodes[f1, 1], face_nodes[f1, 2] = opp1, opp2, k3
    face_nodes[f2, 0], face_nodes[f2, 1], face_nodes[f2, 2] = opp2, opp1, k4
    return True


def _edges_among_nodes(edge_nodes: np.ndarray, nodes: np.ndarray) -> np.ndarray:
    """Indices of edges whose both endpoints are in `nodes`."""
    m = np.isin(edge_nodes[:, 0], nodes) & np.isin(edge_nodes[:, 1], nodes)
    return np.where(m)[0]


def try_flip_candidate_edges_ugrid(
    mesh: "MeshData",
    candidate_edges: np.ndarray,
    removesmalllinkstrsh: float,
    max_flip_iter: int = 20,
    max_cosphi_allowed: Optional[float] = None,
    jsferic: int = 1,
) -> int:
    """
    Try to flip a set of candidate edges.

    If ``max_cosphi_allowed`` is given, each flip is quality-checked on the
    edges of its convex quad: the flip is reverted unless the local max
    |cosphi| stays within ``max(max_cosphi_allowed, value before the flip)``.
    Without this, a geometrically valid flip can create very obtuse triangles
    whose (pulled-inside) circumcenters give |cosphi| ~ 1.0 on a neighbouring
    edge — an unrecoverable degradation that stalls the outer/recovery cycles.
    With ``max_cosphi_allowed=None`` only geometric validity is verified and
    the caller is responsible for the global quality check after the batch.
    """
    candidate_edges = np.asarray(candidate_edges, dtype=np.int64).ravel()
    if candidate_edges.size == 0:
        return 0

    total_flipped = 0

    def _quad_max_cosphi(quad_edges: np.ndarray) -> float:
        cos = _cosphi_abs_for_edges(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            quad_edges,
            use_circumcenter_3d=True,
            jsferic=jsferic,
        )
        vals = cos[quad_edges]
        vals = vals[np.isfinite(vals)]
        return float(np.max(vals)) if vals.size else 0.0

    for _ in range(max_flip_iter):
        flipped_any = False
        for ei in range(candidate_edges.size):
            e = int(candidate_edges[ei])
            quad_nodes = None
            if max_cosphi_allowed is not None and 0 <= e < mesh.edge_faces.shape[0]:
                f1, f2 = int(mesh.edge_faces[e, 0]), int(mesh.edge_faces[e, 1])
                if f1 >= 0 and f2 >= 0:
                    quad_nodes = np.unique(
                        np.concatenate(
                            [mesh.face_nodes[f1, :3], mesh.face_nodes[f2, :3]]
                        )
                    )
                    quad_nodes = quad_nodes[quad_nodes >= 0]
                    # The flip only rewires the quad: its 5 edges keep their
                    # slots (the new diagonal reuses slot `e`), so snapshot
                    # them and update surgically instead of rebuilding the
                    # full edge arrays twice per rejected attempt.
                    quad_edges = _edges_among_nodes(mesh.edge_nodes, quad_nodes)
                    rows_before = (
                        f1,
                        mesh.face_nodes[f1].copy(),
                        f2,
                        mesh.face_nodes[f2].copy(),
                    )
                    edge_nodes_before = mesh.edge_nodes[e].copy()
                    edge_faces_before = mesh.edge_faces[quad_edges].copy()
                    max_before = _quad_max_cosphi(quad_edges)

            if not try_flip_small_flow_edge_ugrid(mesh, e):
                continue

            if quad_nodes is None:
                mesh.edge_nodes, mesh.edge_faces = _build_edges_from_tria(
                    mesh.face_nodes[:, :3]
                )
                total_flipped += 1
                flipped_any = True
                break

            # Surgical topology update: slot `e` becomes the new diagonal
            # (the node pair shared by the two rewritten faces) and each quad
            # edge's face pair is recomputed against the new rows.
            r1 = set(int(x) for x in mesh.face_nodes[f1, :3])
            r2 = set(int(x) for x in mesh.face_nodes[f2, :3])
            diag = sorted(r1 & r2)
            mesh.edge_nodes[e, 0] = diag[0]
            mesh.edge_nodes[e, 1] = diag[1]
            for eidx in quad_edges:
                a, b = int(mesh.edge_nodes[eidx, 0]), int(mesh.edge_nodes[eidx, 1])
                fa, fb = int(mesh.edge_faces[eidx, 0]), int(mesh.edge_faces[eidx, 1])
                adj = []
                if a in r1 and b in r1:
                    adj.append(f1)
                if a in r2 and b in r2:
                    adj.append(f2)
                adj.extend(f for f in (fa, fb) if f >= 0 and f != f1 and f != f2)
                mesh.edge_faces[eidx, 0] = adj[0] if len(adj) >= 1 else -1
                mesh.edge_faces[eidx, 1] = adj[1] if len(adj) >= 2 else -1

            max_after = _quad_max_cosphi(quad_edges)
            if max_after > max(float(max_cosphi_allowed), max_before) + 1.0e-12:
                # Revert: restore face rows and the snapshotted edge entries.
                f1r, row1, f2r, row2 = rows_before
                mesh.face_nodes[f1r] = row1
                mesh.face_nodes[f2r] = row2
                mesh.edge_nodes[e] = edge_nodes_before
                mesh.edge_faces[quad_edges] = edge_faces_before
                continue

            # Accepted: the surgically updated arrays are already consistent
            # (the caller re-canonicalizes edge numbering after the batch).
            total_flipped += 1
            flipped_any = True
            break

        if not flipped_any:
            break

        _, candidate_edges = compute_small_links_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            removesmalllinkstrsh=removesmalllinkstrsh,
            jsferic=jsferic,
        )

    return total_flipped


def _point_in_polygon(
    px: float, py: float, xv: np.ndarray, yv: np.ndarray, x0: float, y0: float
) -> bool:
    """Exact copy from meshkernel_orthogonality: ray casting in projected coords."""
    n = len(xv)
    if n < 3:
        return False
    dxp = _getdx(x0, y0, px, py, 1)
    dyp = _getdy(x0, y0, px, py, 1)
    count = 0
    for i in range(n):
        ip1 = (i + 1) % n
        dx_i = _getdx(x0, y0, xv[i], yv[i], 1)
        dy_i = _getdy(x0, y0, xv[i], yv[i], 1)
        dx_ip1 = _getdx(x0, y0, xv[ip1], yv[ip1], 1)
        dy_ip1 = _getdy(x0, y0, xv[ip1], yv[ip1], 1)
        if (dy_i <= dyp < dy_ip1) or (dy_ip1 <= dyp < dy_i):
            if abs(dy_ip1 - dy_i) < 1e-12:
                continue
            t = (dyp - dy_i) / (dy_ip1 - dy_i)
            x_cross = dx_i + t * (dx_ip1 - dx_i)
            if x_cross > dxp:
                count += 1
    return (count % 2) == 1


def _segment_edge_intersect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    xa: float,
    ya: float,
    xb: float,
    yb: float,
    x0: float,
    y0: float,
):
    """Exact copy from meshkernel_orthogonality: segment–edge intersection in projected coords."""
    dx1 = _getdx(x0, y0, x1, y1, 1)
    dy1 = _getdy(x0, y0, x1, y1, 1)
    dx2 = _getdx(x0, y0, x2, y2, 1)
    dy2 = _getdy(x0, y0, x2, y2, 1)
    dxa = _getdx(x0, y0, xa, ya, 1)
    dya = _getdy(x0, y0, xa, ya, 1)
    dxb = _getdx(x0, y0, xb, yb, 1)
    dyb = _getdy(x0, y0, xb, yb, 1)
    den = (dx2 - dx1) * (dyb - dya) - (dy2 - dy1) * (dxb - dxa)
    if abs(den) < 1e-15:
        return None
    t = ((dxa - dx1) * (dyb - dya) - (dya - dy1) * (dxb - dxa)) / den
    s = ((dxa - dx1) * (dy2 - dy1) - (dya - dy1) * (dx2 - dx1)) / den
    if not (0 <= t <= 1 and 0 <= s <= 1):
        return None
    xcr_l = dx1 + t * (dx2 - dx1)
    ycr_l = dy1 + t * (dy2 - dy1)
    ycr_deg = y0 + ycr_l / EARTH_RADIUS_DEG2RAD
    xcr_deg = x0 + xcr_l / (EARTH_RADIUS_DEG2RAD * np.cos(ycr_deg * DEG2RAD))
    return (xcr_deg, ycr_deg, t)


def _circumcenter3d(xv: np.ndarray, yv: np.ndarray) -> Tuple[float, float]:
    """Spherical 3D circumcenter (comp_circumcenter3D). xv, yv in deg -> (xz, yz) deg."""
    N = len(xv)
    if N < 2:
        return float(xv[0]), float(yv[0])
    xx = _sphertocart3d_vec(
        np.asarray(xv, dtype=np.float64), np.asarray(yv, dtype=np.float64)
    )
    xxc, yyc, zzc = np.mean(xx[:, 0]), np.mean(xx[:, 1]), np.mean(xx[:, 2])
    dtol, deps, maxiter = 1e-8, 1e-8, 100
    ip1 = np.arange(N)
    ip1 = (ip1 + 1) % N
    ttx = xx[ip1, 0] - xx[:, 0]
    tty = xx[ip1, 1] - xx[:, 1]
    ttz = xx[ip1, 2] - xx[:, 2]
    ds = np.sqrt(ttx * ttx + tty * tty + ttz * ttz)
    valid = ds >= dtol
    dsi = np.where(valid, 1.0 / ds, 0.0)
    ttx = ttx * dsi
    tty = tty * dsi
    ttz = ttz * dsi
    xxe = 0.5 * (xx[:, 0] + xx[ip1, 0])
    yye = 0.5 * (xx[:, 1] + xx[ip1, 1])
    zze = 0.5 * (xx[:, 2] + xx[ip1, 2])
    lam = 0.0
    for _ in range(maxiter):
        A = np.zeros((4, 4))
        rhs = np.zeros(4)
        for i in range(N):
            if ds[i] < dtol:
                continue
            A[0, 0] += ttx[i] * ttx[i]
            A[0, 1] += ttx[i] * tty[i]
            A[0, 2] += ttx[i] * ttz[i]
            A[1, 1] += tty[i] * tty[i]
            A[1, 2] += tty[i] * ttz[i]
            A[2, 2] += ttz[i] * ttz[i]
            dinpr = (
                (xxc - xxe[i]) * ttx[i]
                + (yyc - yye[i]) * tty[i]
                + (zzc - zze[i]) * ttz[i]
            )
            rhs[0] -= dinpr * ttx[i]
            rhs[1] -= dinpr * tty[i]
            rhs[2] -= dinpr * ttz[i]
        A[0, 0] -= 2 * lam
        A[1, 1] -= 2 * lam
        A[2, 2] -= 2 * lam
        A[0, 3], A[1, 3], A[2, 3] = -2 * xxc, -2 * yyc, -2 * zzc
        A[3, 3] = 0.0
        rhs[0] += 2 * lam * xxc
        rhs[1] += 2 * lam * yyc
        rhs[2] += 2 * lam * zzc
        rhs[3] = xxc * xxc + yyc * yyc + zzc * zzc - EARTH_RADIUS_SQ
        A[1, 0], A[2, 0], A[2, 1] = A[0, 1], A[0, 2], A[1, 2]
        A[3, 0], A[3, 1], A[3, 2] = A[0, 3], A[1, 3], A[2, 3]
        try:
            sol = np.linalg.solve(A, rhs)
        except np.linalg.LinAlgError:
            break
        xxc += sol[0]
        yyc += sol[1]
        zzc += sol[2]
        lam += sol[3]
        if sol[0] ** 2 + sol[1] ** 2 + sol[2] ** 2 < deps:
            break
    return _cart3dtospher(xxc, yyc, zzc, float(np.max(xv)))


def _circumcenters3d_batch(
    xv: np.ndarray, yv: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Batched `_circumcenter3d` for faces with an equal node count.
    xv, yv: (F, N) in deg -> (xz, yz), each (F,) in deg.

    Runs the same constrained Newton iteration as the scalar version, with a
    per-face active mask so each face performs the identical update sequence.
    """
    F, N = xv.shape
    if F == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if N < 2:
        return xv[:, 0].astype(np.float64), yv[:, 0].astype(np.float64)
    xx = _sphertocart3d_vec(xv.ravel(), yv.ravel()).reshape(F, N, 3)
    xxc = np.mean(xx[:, :, 0], axis=1)
    yyc = np.mean(xx[:, :, 1], axis=1)
    zzc = np.mean(xx[:, :, 2], axis=1)
    dtol, deps, maxiter = 1e-8, 1e-8, 100
    ip1 = (np.arange(N) + 1) % N
    ttx = xx[:, ip1, 0] - xx[:, :, 0]
    tty = xx[:, ip1, 1] - xx[:, :, 1]
    ttz = xx[:, ip1, 2] - xx[:, :, 2]
    ds = np.sqrt(ttx * ttx + tty * tty + ttz * ttz)
    valid = ds >= dtol
    dsi = np.where(valid, 1.0 / np.where(valid, ds, 1.0), 0.0)
    ttx = ttx * dsi
    tty = tty * dsi
    ttz = ttz * dsi
    xxe = 0.5 * (xx[:, :, 0] + xx[:, ip1, 0])
    yye = 0.5 * (xx[:, :, 1] + xx[:, ip1, 1])
    zze = 0.5 * (xx[:, :, 2] + xx[:, ip1, 2])
    # The A-matrix tangent terms are invariant across Newton iterations:
    # sequential accumulation over nodes, same summation order as the scalar
    # loop (invalid edges have tt == 0, so their terms are 0).
    a00_full = np.zeros(F)
    a01_full = np.zeros(F)
    a02_full = np.zeros(F)
    a11_full = np.zeros(F)
    a12_full = np.zeros(F)
    a22_full = np.zeros(F)
    for i in range(N):
        a00_full += ttx[:, i] * ttx[:, i]
        a01_full += ttx[:, i] * tty[:, i]
        a02_full += ttx[:, i] * ttz[:, i]
        a11_full += tty[:, i] * tty[:, i]
        a12_full += tty[:, i] * ttz[:, i]
        a22_full += ttz[:, i] * ttz[:, i]

    lam = np.zeros(F, dtype=np.float64)
    active = np.ones(F, dtype=bool)
    # For small batches, gathering the active lanes costs more than computing
    # discarded updates for converged ones; updates are masked either way.
    gather = F >= 64
    for _ in range(maxiter):
        if not np.any(active):
            break
        if gather:
            act = np.where(active)[0]
            tx, ty, tz = ttx[act], tty[act], ttz[act]
            xe, ye, ze = xxe[act], yye[act], zze[act]
            cx, cy, cz = xxc[act], yyc[act], zzc[act]
            lam_a = lam[act]
            a00, a01, a02 = a00_full[act], a01_full[act], a02_full[act]
            a11, a12, a22 = a11_full[act], a12_full[act], a22_full[act]
        else:
            act = None
            tx, ty, tz = ttx, tty, ttz
            xe, ye, ze = xxe, yye, zze
            cx, cy, cz = xxc, yyc, zzc
            lam_a = lam
            a00, a01, a02 = a00_full, a01_full, a02_full
            a11, a12, a22 = a11_full, a12_full, a22_full
        na = cx.shape[0]
        r0 = np.zeros(na)
        r1 = np.zeros(na)
        r2 = np.zeros(na)
        for i in range(N):
            dinpr = (
                (cx - xe[:, i]) * tx[:, i]
                + (cy - ye[:, i]) * ty[:, i]
                + (cz - ze[:, i]) * tz[:, i]
            )
            r0 -= dinpr * tx[:, i]
            r1 -= dinpr * ty[:, i]
            r2 -= dinpr * tz[:, i]
        A = np.zeros((na, 4, 4))
        A[:, 0, 0] = a00 - 2 * lam_a
        A[:, 1, 1] = a11 - 2 * lam_a
        A[:, 2, 2] = a22 - 2 * lam_a
        A[:, 0, 1] = A[:, 1, 0] = a01
        A[:, 0, 2] = A[:, 2, 0] = a02
        A[:, 1, 2] = A[:, 2, 1] = a12
        A[:, 0, 3] = A[:, 3, 0] = -2 * cx
        A[:, 1, 3] = A[:, 3, 1] = -2 * cy
        A[:, 2, 3] = A[:, 3, 2] = -2 * cz
        rhs = np.empty((na, 4))
        rhs[:, 0] = r0 + 2 * lam_a * cx
        rhs[:, 1] = r1 + 2 * lam_a * cy
        rhs[:, 2] = r2 + 2 * lam_a * cz
        rhs[:, 3] = cx * cx + cy * cy + cz * cz - EARTH_RADIUS_SQ
        solved = np.ones(na, dtype=bool)
        try:
            sol = np.linalg.solve(A, rhs[:, :, None])[:, :, 0]
        except np.linalg.LinAlgError:
            # Some face has a singular system: solve per face; a singular face
            # stops iterating with its current center (as in the scalar code).
            sol = np.zeros((na, 4))
            for j in range(na):
                try:
                    sol[j] = np.linalg.solve(A[j], rhs[j])
                except np.linalg.LinAlgError:
                    solved[j] = False
        conv = sol[:, 0] ** 2 + sol[:, 1] ** 2 + sol[:, 2] ** 2 < deps
        if act is not None:
            upd = act[solved]
            xxc[upd] += sol[solved, 0]
            yyc[upd] += sol[solved, 1]
            zzc[upd] += sol[solved, 2]
            lam[upd] += sol[solved, 3]
            active[act[~solved]] = False
            active[act[solved & conv]] = False
        else:
            upd = active & solved
            xxc = np.where(upd, xxc + sol[:, 0], xxc)
            yyc = np.where(upd, yyc + sol[:, 1], yyc)
            zzc = np.where(upd, zzc + sol[:, 2], zzc)
            lam = np.where(upd, lam + sol[:, 3], lam)
            active &= solved & ~conv
    return _cart3dtospher_vec(xxc, yyc, zzc, np.max(xv, axis=1))


def _circumcenters2d_batch(
    xv: np.ndarray, yv: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Planar (jsferic=0) counterpart of `_circumcenters3d_batch` for faces with
    an equal node count. xv, yv: (F, N) in x/y -> (xz, yz), each (F,).

    The circumcenter is the least-squares intersection of the edge
    perpendicular bisectors: minimize sum_i ((c - e_i) . t_i)^2 over edge
    midpoints e_i and unit tangents t_i (exact circumcenter for triangles).
    Degenerate faces (collinear or too-short edges) fall back to the vertex
    mean, like the spherical version's singular-system fallback.
    """
    F, N = xv.shape
    if F == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
    if N < 2:
        return xv[:, 0].astype(np.float64), yv[:, 0].astype(np.float64)
    dtol = 1e-8
    ip1 = (np.arange(N) + 1) % N
    ttx = xv[:, ip1] - xv
    tty = yv[:, ip1] - yv
    ds = np.sqrt(ttx * ttx + tty * tty)
    valid = ds >= dtol
    dsi = np.where(valid, 1.0 / np.where(valid, ds, 1.0), 0.0)
    ttx = ttx * dsi
    tty = tty * dsi
    xxe = 0.5 * (xv + xv[:, ip1])
    yye = 0.5 * (yv + yv[:, ip1])
    a00 = np.zeros(F)
    a01 = np.zeros(F)
    a11 = np.zeros(F)
    b0 = np.zeros(F)
    b1 = np.zeros(F)
    for i in range(N):
        a00 += ttx[:, i] * ttx[:, i]
        a01 += ttx[:, i] * tty[:, i]
        a11 += tty[:, i] * tty[:, i]
        einpr = xxe[:, i] * ttx[:, i] + yye[:, i] * tty[:, i]
        b0 += einpr * ttx[:, i]
        b1 += einpr * tty[:, i]
    det = a00 * a11 - a01 * a01
    ok = np.abs(det) >= 1e-12
    det_safe = np.where(ok, det, 1.0)
    cx = (b0 * a11 - b1 * a01) / det_safe
    cy = (a00 * b1 - a01 * b0) / det_safe
    mean_x = np.mean(xv, axis=1)
    mean_y = np.mean(yv, axis=1)
    return np.where(ok, cx, mean_x), np.where(ok, cy, mean_y)


def _point_in_polygon_vec(
    px: np.ndarray,
    py: np.ndarray,
    xv: np.ndarray,
    yv: np.ndarray,
    x0: np.ndarray,
    y0: np.ndarray,
    jsferic: int = 1,
) -> np.ndarray:
    """Vectorized `_point_in_polygon`: (F,) points vs (F, N) polygons."""
    F, N = xv.shape
    if N < 3:
        return np.zeros(F, dtype=bool)
    dxp = _getdx_vec(x0, y0, px, py, jsferic)
    dyp = _getdy_vec(x0, y0, px, py, jsferic)
    dxs = np.empty_like(xv)
    dys = np.empty_like(yv)
    for i in range(N):
        dxs[:, i] = _getdx_vec(x0, y0, xv[:, i], yv[:, i], jsferic)
        dys[:, i] = _getdy_vec(x0, y0, xv[:, i], yv[:, i], jsferic)
    count = np.zeros(F, dtype=np.int64)
    for i in range(N):
        ip1 = (i + 1) % N
        dy_i = dys[:, i]
        dy_ip1 = dys[:, ip1]
        crosses = ((dy_i <= dyp) & (dyp < dy_ip1)) | ((dy_ip1 <= dyp) & (dyp < dy_i))
        crosses &= np.abs(dy_ip1 - dy_i) >= 1e-12
        denom = np.where(crosses, dy_ip1 - dy_i, 1.0)
        t = (dyp - dy_i) / denom
        x_cross = dxs[:, i] + t * (dxs[:, ip1] - dxs[:, i])
        count += (crosses & (x_cross > dxp)).astype(np.int64)
    return (count % 2) == 1


def _face_centers_circumcenter3d(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    dcenterinside: float = 1.0,
    face_mask: Optional[np.ndarray] = None,
    jsferic: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Centers = 3D circumcenter (planar 2D circumcenter for jsferic=0), with
    pull-inside when outside cell (as in Delft).
    If face_mask is provided, only compute for faces where face_mask[f] is True;
    others are left as nan.
    """
    mass_x, mass_y = _face_centers(
        node_x, node_y, face_nodes, face_mask=face_mask, jsferic=jsferic
    )
    n_faces = face_nodes.shape[0]
    if face_mask is not None:
        face_x = np.full(n_faces, np.nan, dtype=np.float64)
        face_y = np.full(n_faces, np.nan, dtype=np.float64)
        faces_to_do = np.where(face_mask)[0]
    else:
        face_x = np.zeros(n_faces, dtype=np.float64)
        face_y = np.zeros(n_faces, dtype=np.float64)
        faces_to_do = np.arange(n_faces)
    if faces_to_do.size == 0:
        return face_x, face_y
    counts = np.sum(face_nodes[faces_to_do, :] > 0, axis=1)

    for n in np.unique(counts):
        faces = faces_to_do[counts == n]
        if n == 0:
            face_x[faces] = mass_x[faces]
            face_y[faces] = mass_y[faces]
            continue
        if n == 1:
            xv1, yv1 = _gather_face_node_coords(node_x, node_y, face_nodes, faces, 1)
            face_x[faces] = xv1[:, 0]
            face_y[faces] = yv1[:, 0]
            continue
        n = int(n)
        xv, yv = _gather_face_node_coords(node_x, node_y, face_nodes, faces, n)
        if jsferic == 1:
            xz, yz = _circumcenters3d_batch(xv, yv)
        else:
            xz, yz = _circumcenters2d_batch(xv, yv)
        if n == 3:
            # Axis-aligned right triangles: use the circumcenter of the
            # bounding rectangle (as in Delft).
            has_v = np.zeros(faces.size, dtype=bool)
            has_h = np.zeros(faces.size, dtype=bool)
            for k in range(3):
                k2 = (k + 1) % 3
                has_v |= np.abs(xv[:, k] - xv[:, k2]) < 1e-10
                has_h |= np.abs(yv[:, k] - yv[:, k2]) < 1e-10
            rect = has_v & has_h
            if np.any(rect):
                xmin = xv[rect].min(axis=1)
                xmax = xv[rect].max(axis=1)
                ymin = yv[rect].min(axis=1)
                ymax = yv[rect].max(axis=1)
                xh = np.column_stack([xmin, xmax, xmax, xmin])
                yh = np.column_stack([ymin, ymin, ymax, ymax])
                if jsferic == 1:
                    xz[rect], yz[rect] = _circumcenters3d_batch(xh, yh)
                else:
                    xz[rect], yz[rect] = _circumcenters2d_batch(xh, yh)
        if 0 <= dcenterinside <= 1:
            x0 = np.min(xv, axis=1)
            y0 = yv[np.arange(faces.size), np.argmin(np.abs(yv), axis=1)]
            inside = _point_in_polygon_vec(xz, yz, xv, yv, x0, y0, jsferic=jsferic)
            # Pull-inside for the centers outside their face: intersect the
            # mass-center -> circumcenter segment with each face edge and keep
            # the crossing with the smallest ratio (as `_segment_edge_intersect`).
            out = np.where(~inside)[0]
            if out.size > 0:
                x0o, y0o = x0[out], y0[out]
                dx1 = _getdx_vec(
                    x0o, y0o, mass_x[faces[out]], mass_y[faces[out]], jsferic
                )
                dy1 = _getdy_vec(
                    x0o, y0o, mass_x[faces[out]], mass_y[faces[out]], jsferic
                )
                dx2 = _getdx_vec(x0o, y0o, xz[out], yz[out], jsferic)
                dy2 = _getdy_vec(x0o, y0o, xz[out], yz[out], jsferic)
                dxs = np.empty((out.size, n))
                dys = np.empty((out.size, n))
                for i in range(n):
                    dxs[:, i] = _getdx_vec(x0o, y0o, xv[out, i], yv[out, i], jsferic)
                    dys[:, i] = _getdy_vec(x0o, y0o, xv[out, i], yv[out, i], jsferic)
                best_t = np.full(out.size, 2.0)
                xcr = xz[out].copy()
                ycr = yz[out].copy()
                for i in range(n):
                    ip1 = (i + 1) % n
                    dxa, dya = dxs[:, i], dys[:, i]
                    dxb, dyb = dxs[:, ip1], dys[:, ip1]
                    den = (dx2 - dx1) * (dyb - dya) - (dy2 - dy1) * (dxb - dxa)
                    hit = np.abs(den) >= 1e-15
                    den_safe = np.where(hit, den, 1.0)
                    t = (
                        (dxa - dx1) * (dyb - dya) - (dya - dy1) * (dxb - dxa)
                    ) / den_safe
                    s = (
                        (dxa - dx1) * (dy2 - dy1) - (dya - dy1) * (dx2 - dx1)
                    ) / den_safe
                    hit &= (0 <= t) & (t <= 1) & (0 <= s) & (s <= 1)
                    xcr_l = dx1 + t * (dx2 - dx1)
                    ycr_l = dy1 + t * (dy2 - dy1)
                    if jsferic == 1:
                        ycr_deg = y0o + ycr_l / EARTH_RADIUS_DEG2RAD
                        xcr_deg = x0o + xcr_l / (
                            EARTH_RADIUS_DEG2RAD * np.cos(ycr_deg * DEG2RAD)
                        )
                    else:
                        ycr_deg = y0o + ycr_l
                        xcr_deg = x0o + xcr_l
                    upd = hit & (t < best_t)
                    best_t = np.where(upd, t, best_t)
                    xcr = np.where(upd, xcr_deg, xcr)
                    ycr = np.where(upd, ycr_deg, ycr)
                xz[out] = xcr
                yz[out] = ycr
        face_x[faces] = xz
        face_y[faces] = yz
    return face_x, face_y


def _dcosphi_sph_vec(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    x3: np.ndarray,
    y3: np.ndarray,
    x4: np.ndarray,
    y4: np.ndarray,
) -> np.ndarray:
    """Vectorized: all args (n_edges,) -> (n_edges,) |cos(phi)|."""
    p1 = _sphertocart3d_vec(x1, y1)
    p2 = _sphertocart3d_vec(x2, y2)
    p3 = _sphertocart3d_vec(x3, y3)
    p4 = _sphertocart3d_vec(x4, y4)
    d1 = p2 - p1
    d2 = p4 - p3
    r1 = np.sqrt(np.sum(d1 * d1, axis=1))
    r2 = np.sqrt(np.sum(d2 * d2, axis=1))
    dot = np.sum(d1 * d2, axis=1)
    cosphi = np.where(
        (r1 > 0) & (r2 > 0),
        dot / (r1 * r2),
        0.0,
    )
    cosphi = np.clip(cosphi, -1.0, 1.0)
    return np.abs(cosphi)


def _dcosphi_flat_vec(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
    x3: np.ndarray,
    y3: np.ndarray,
    x4: np.ndarray,
    y4: np.ndarray,
) -> np.ndarray:
    """Planar (jsferic=0) counterpart of `_dcosphi_sph_vec`: |cos(phi)| in 2D."""
    d1x = x2 - x1
    d1y = y2 - y1
    d2x = x4 - x3
    d2y = y4 - y3
    r1 = np.sqrt(d1x * d1x + d1y * d1y)
    r2 = np.sqrt(d2x * d2x + d2y * d2y)
    dot = d1x * d2x + d1y * d2y
    cosphi = np.where(
        (r1 > 0) & (r2 > 0),
        dot / np.where((r1 > 0) & (r2 > 0), r1 * r2, 1.0),
        0.0,
    )
    cosphi = np.clip(cosphi, -1.0, 1.0)
    return np.abs(cosphi)


def _opposite_sides_vec(
    xk3: np.ndarray,
    yk3: np.ndarray,
    xk4: np.ndarray,
    yk4: np.ndarray,
    xc1: np.ndarray,
    yc1: np.ndarray,
    xc2: np.ndarray,
    yc2: np.ndarray,
    jsferic: int = 1,
) -> np.ndarray:
    """Vectorized version of _opposite_sides. Returns bool (n_edges,)."""
    ex = _getdx_vec(xk3, yk3, xk4, yk4, jsferic)
    ey = _getdy_vec(xk3, yk3, xk4, yk4, jsferic)
    c1 = ex * _getdy_vec(xk3, yk3, xc1, yc1, jsferic) - ey * _getdx_vec(
        xk3, yk3, xc1, yc1, jsferic
    )
    c2 = ex * _getdy_vec(xk3, yk3, xc2, yc2, jsferic) - ey * _getdx_vec(
        xk3, yk3, xc2, yc2, jsferic
    )
    return c1 * c2 < 0.0


def _edge_faces_from_faces_edges(
    face_nodes: np.ndarray, edge_nodes: np.ndarray
) -> np.ndarray:
    """Rebuild edge_faces (1-based) from faces/edges connectivity."""
    n_edges = edge_nodes.shape[0]
    n_faces = face_nodes.shape[0]
    if n_faces == 0 or n_edges == 0:
        return np.zeros((n_edges, 2), dtype=np.int64)
    max_node = int(face_nodes.max())
    node_to_faces: List[List[int]] = [[] for _ in range(max_node + 1)]
    for f in range(n_faces):
        for n in face_nodes[f, :]:
            if n > 0:
                node_to_faces[int(n)].append(f)
    edge_faces = np.zeros((n_edges, 2), dtype=np.int64)
    for e in range(n_edges):
        n1, n2 = edge_nodes[e, :]
        if n1 <= 0 or n2 <= 0:
            continue
        common = list(set(node_to_faces[int(n1)]) & set(node_to_faces[int(n2)]))
        if len(common) >= 1:
            edge_faces[e, 0] = common[0] + 1
        if len(common) >= 2:
            edge_faces[e, 1] = common[1] + 1
    return edge_faces


# ---------------------------------------------------------------------------
# Index conversion: UGRID/orthogonality use 1-based; we use 0-based internally.
# ---------------------------------------------------------------------------


def _to_0b(arr: np.ndarray) -> np.ndarray:
    """Convert 1-based indices to 0-based (valid: 0..n-1, invalid: -1)."""
    out = np.where(arr > 0, arr - 1, -1)
    return out.astype(arr.dtype)


def _to_1b(arr: np.ndarray) -> np.ndarray:
    """Convert 0-based indices to 1-based for meshkernel_orthogonality (invalid -1 -> 0)."""
    out = np.where(arr >= 0, arr + 1, 0)
    return out.astype(arr.dtype)


# ---------------------------------------------------------------------------
# Orthogonality: |cosphi| on in-memory arrays (0-based in, cosphi_abs out)
# ---------------------------------------------------------------------------


def compute_cosphi_abs_from_arrays(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    edge_nodes: np.ndarray,
    edge_faces: Optional[np.ndarray],
    use_file_centers: bool = False,
    use_circumcenter_3d: bool = True,
    edge_indices: Optional[np.ndarray] = None,
    jsferic: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute |cosphi| for edges. Inputs 0-based (invalid = -1).
    If edge_indices is provided (0-based), only compute for those edges (and only
    the face centers needed). Returns (edge_nodes, edge_faces, cosphi_abs).
    Coordinates are lon/lat degrees for jsferic=1, planar x/y for jsferic=0.
    """
    if use_file_centers:
        raise ValueError(
            "use_file_centers=True is not supported in this in-memory variant."
        )
    if edge_indices is not None and edge_faces is not None:
        # Fast path: compute face centers/cosphi only for the requested edges
        # (0-based throughout; same formulas as the full path).
        cosphi_abs = _cosphi_abs_for_edges(
            node_x,
            node_y,
            face_nodes,
            edge_nodes,
            edge_faces,
            edge_indices,
            use_circumcenter_3d=use_circumcenter_3d,
            jsferic=jsferic,
        )
        return _to_1b(edge_nodes), _to_1b(edge_faces), cosphi_abs

    # Convert to 1-based for orthogonality helpers (unchanged formulas)
    face_nodes = _to_1b(face_nodes)
    edge_nodes = _to_1b(edge_nodes)
    if edge_faces is not None:
        edge_faces = _to_1b(edge_faces)

    if edge_faces is None:
        edge_faces = _edge_faces_from_faces_edges(face_nodes, edge_nodes)

    n_faces = face_nodes.shape[0]
    n_edges = edge_nodes.shape[0]

    # When edge_indices provided, only compute face centers for faces adjacent to those edges
    face_mask: Optional[np.ndarray] = None
    if edge_indices is not None:
        edge_indices = np.asarray(edge_indices, dtype=np.int64).ravel()
        face_mask = np.zeros(n_faces, dtype=bool)
        for e in edge_indices:
            if e < 0 or e >= n_edges:
                continue
            for j in (0, 1):
                f1b = edge_faces[e, j]
                if f1b > 0:
                    face_mask[int(f1b) - 1] = True

    if use_circumcenter_3d:
        face_x, face_y = _face_centers_circumcenter3d(
            node_x, node_y, face_nodes, face_mask=face_mask, jsferic=jsferic
        )
    else:
        face_x, face_y = _face_centers(
            node_x, node_y, face_nodes, face_mask=face_mask, jsferic=jsferic
        )

    cosphi_abs = np.full(n_edges, np.nan, dtype=np.float64)

    k3 = edge_nodes[:, 0]
    k4 = edge_nodes[:, 1]
    f1 = edge_faces[:, 0]
    f2 = edge_faces[:, 1]
    valid = (k3 > 0) & (k4 > 0) & (f1 > 0) & (f2 > 0) & (f1 != f2)
    idx = np.where(valid)[0]
    if edge_indices is not None:
        idx = np.intersect1d(idx, edge_indices, assume_unique=True)
    if idx.size == 0:
        return edge_nodes, edge_faces, cosphi_abs

    k3i = k3[idx] - 1
    k4i = k4[idx] - 1
    f1i = f1[idx] - 1
    f2i = f2[idx] - 1

    if not use_circumcenter_3d:
        opp = _opposite_sides_vec(
            node_x[k3i],
            node_y[k3i],
            node_x[k4i],
            node_y[k4i],
            face_x[f1i],
            face_y[f1i],
            face_x[f2i],
            face_y[f2i],
            jsferic=jsferic,
        )
        idx = idx[opp]
        k3i, k4i, f1i, f2i = k3i[opp], k4i[opp], f1i[opp], f2i[opp]
        if idx.size == 0:
            return edge_nodes, edge_faces, cosphi_abs

    dx_edge = _getdx_vec(node_x[k3i], node_y[k3i], node_x[k4i], node_y[k4i], jsferic)
    dy_edge = _getdy_vec(node_x[k3i], node_y[k3i], node_x[k4i], node_y[k4i], jsferic)
    d = np.hypot(dx_edge, dy_edge)
    valid_d = d >= 1.0e-6
    idx = idx[valid_d]
    k3i, k4i, f1i, f2i = k3i[valid_d], k4i[valid_d], f1i[valid_d], f2i[valid_d]
    if idx.size == 0:
        return edge_nodes, edge_faces, cosphi_abs

    dcosphi = _dcosphi_sph_vec if jsferic == 1 else _dcosphi_flat_vec
    cosphi_abs[idx] = dcosphi(
        face_x[f1i],
        face_y[f1i],
        face_x[f2i],
        face_y[f2i],
        node_x[k3i],
        node_y[k3i],
        node_x[k4i],
        node_y[k4i],
    )
    return edge_nodes, edge_faces, cosphi_abs


def _cosphi_abs_for_edges(
    node_x: np.ndarray,
    node_y: np.ndarray,
    face_nodes: np.ndarray,
    edge_nodes: np.ndarray,
    edge_faces: np.ndarray,
    edge_indices: np.ndarray,
    use_circumcenter_3d: bool = True,
    jsferic: int = 1,
) -> np.ndarray:
    """
    |cosphi| restricted to `edge_indices` (0-based inputs, invalid = -1),
    computing face centers only for the adjacent faces. Returns a full-length
    (n_edges,) array, NaN outside the requested edges — the same values the
    full `compute_cosphi_abs_from_arrays` produces for those edges.
    """
    n_faces = face_nodes.shape[0]
    n_edges = edge_nodes.shape[0]
    cosphi_abs = np.full(n_edges, np.nan, dtype=np.float64)

    edge_indices = np.asarray(edge_indices, dtype=np.int64).ravel()
    idx = np.unique(edge_indices[(edge_indices >= 0) & (edge_indices < n_edges)])
    if idx.size == 0:
        return cosphi_abs
    k3 = edge_nodes[idx, 0]
    k4 = edge_nodes[idx, 1]
    f1 = edge_faces[idx, 0]
    f2 = edge_faces[idx, 1]
    valid = (
        (k3 >= 0)
        & (k4 >= 0)
        & (f1 >= 0)
        & (f2 >= 0)
        & (f1 != f2)
        & (f1 < n_faces)
        & (f2 < n_faces)
    )
    idx, k3, k4, f1, f2 = idx[valid], k3[valid], k4[valid], f1[valid], f2[valid]
    if idx.size == 0:
        return cosphi_abs

    faces_needed = np.unique(np.concatenate([f1, f2]))
    sub_face_nodes_1b = _to_1b(np.asarray(face_nodes)[faces_needed])
    if use_circumcenter_3d:
        fx, fy = _face_centers_circumcenter3d(
            node_x, node_y, sub_face_nodes_1b, jsferic=jsferic
        )
    else:
        fx, fy = _face_centers(node_x, node_y, sub_face_nodes_1b, jsferic=jsferic)
    p1 = np.searchsorted(faces_needed, f1)
    p2 = np.searchsorted(faces_needed, f2)

    if not use_circumcenter_3d:
        opp = _opposite_sides_vec(
            node_x[k3],
            node_y[k3],
            node_x[k4],
            node_y[k4],
            fx[p1],
            fy[p1],
            fx[p2],
            fy[p2],
            jsferic=jsferic,
        )
        idx, k3, k4, p1, p2 = idx[opp], k3[opp], k4[opp], p1[opp], p2[opp]
        if idx.size == 0:
            return cosphi_abs

    dx_edge = _getdx_vec(node_x[k3], node_y[k3], node_x[k4], node_y[k4], jsferic)
    dy_edge = _getdy_vec(node_x[k3], node_y[k3], node_x[k4], node_y[k4], jsferic)
    d = np.hypot(dx_edge, dy_edge)
    valid_d = d >= 1.0e-6
    idx, k3, k4, p1, p2 = (
        idx[valid_d],
        k3[valid_d],
        k4[valid_d],
        p1[valid_d],
        p2[valid_d],
    )
    if idx.size == 0:
        return cosphi_abs

    dcosphi = _dcosphi_sph_vec if jsferic == 1 else _dcosphi_flat_vec
    cosphi_abs[idx] = dcosphi(
        fx[p1],
        fy[p1],
        fx[p2],
        fy[p2],
        node_x[k3],
        node_y[k3],
        node_x[k4],
        node_y[k4],
    )
    return cosphi_abs


# ---------------------------------------------------------------------------
# Utility structures for zones
# ---------------------------------------------------------------------------


@dataclass
class MeshData:
    node_x: np.ndarray
    node_y: np.ndarray
    face_nodes: np.ndarray
    edge_nodes: np.ndarray
    edge_faces: np.ndarray


def _classify_zone_nodes(
    face_nodes: np.ndarray,
    edge_nodes: np.ndarray,
    faces_zone: Set[int],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify internal vs boundary nodes for a given zone.

    - Internal node: belongs only to faces in the zone AND has no edge
      to a node outside the zone.
    - Boundary node: any zone node that is not internal.
    """
    zone_nodes = zone_nodes_from_faces(face_nodes, faces_zone)
    if zone_nodes.size == 0:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
        )

    max_node = int(
        max(int(zone_nodes.max()), int(edge_nodes.max()), int(face_nodes.max()))
    )
    in_zone = np.zeros(max_node + 2, dtype=bool)
    in_zone[zone_nodes] = True

    # Node appears in a face outside the zone -> boundary.
    zone_face_mask = np.zeros(face_nodes.shape[0], dtype=bool)
    zone_face_mask[np.fromiter(faces_zone, dtype=np.int64, count=len(faces_zone))] = (
        True
    )
    out_nodes = face_nodes[~zone_face_mask, :].ravel()
    out_nodes = out_nodes[out_nodes >= 0]
    appears_outside = np.zeros(max_node + 2, dtype=bool)
    appears_outside[out_nodes] = True

    # Node connected by an edge to a node outside the zone -> boundary.
    n1 = edge_nodes[:, 0]
    n2 = edge_nodes[:, 1]
    ok = (n1 >= 0) & (n2 >= 0)
    n1c = np.where(ok, n1, max_node + 1)
    n2c = np.where(ok, n2, max_node + 1)
    edge_to_outside = np.zeros(max_node + 2, dtype=bool)
    m1 = ok & in_zone[n1c] & ~in_zone[n2c]
    m2 = ok & in_zone[n2c] & ~in_zone[n1c]
    edge_to_outside[n1c[m1]] = True
    edge_to_outside[n2c[m2]] = True

    zone_sorted = np.sort(np.asarray(zone_nodes, dtype=np.int64))
    is_boundary_sorted = (appears_outside | edge_to_outside)[zone_sorted]
    internal = zone_sorted[~is_boundary_sorted]
    boundary = zone_sorted[is_boundary_sorted]
    return internal, boundary


# ---------------------------------------------------------------------------
# Zone graph (face adjacency, BFS, zone nodes)
# ---------------------------------------------------------------------------


def build_face_adjacency(edge_faces: np.ndarray, n_faces: int) -> List[List[int]]:
    """Adjacency graph (neighboring faces via a shared edge; sorted, unique)."""
    ef = np.asarray(edge_faces)
    m = (ef[:, 0] >= 0) & (ef[:, 1] >= 0) & (ef[:, 0] != ef[:, 1])
    a = ef[m, 0].astype(np.int64)
    b = ef[m, 1].astype(np.int64)
    src = np.concatenate([a, b])
    dst = np.concatenate([b, a])
    order = np.lexsort((dst, src))
    src, dst = src[order], dst[order]
    keep = (
        np.r_[True, (src[1:] != src[:-1]) | (dst[1:] != dst[:-1])]
        if src.size
        else np.zeros(0, dtype=bool)
    )
    src, dst = src[keep], dst[keep]
    counts = (
        np.bincount(src, minlength=n_faces)
        if src.size
        else np.zeros(n_faces, dtype=np.int64)
    )
    dst_list = dst.tolist()
    neigh: List[List[int]] = []
    idx = 0
    for f in range(n_faces):
        c = int(counts[f])
        neigh.append(dst_list[idx : idx + c])
        idx += c
    return neigh


def bfs_faces(
    start_faces: Iterable[int],
    neighbors: List[List[int]],
    max_depth: int,
) -> Set[int]:
    """Return the set of faces at topological distance <= max_depth."""
    visited: Set[int] = set()
    frontier: Set[int] = set(int(f) for f in start_faces)
    depth = 0
    while frontier and depth <= max_depth:
        visited.update(frontier)
        next_frontier: Set[int] = set()
        for f in frontier:
            for g in neighbors[f]:
                if g not in visited:
                    next_frontier.add(g)
        frontier = next_frontier
        depth += 1
    return visited


def zone_nodes_from_faces(face_nodes: np.ndarray, faces_zone: Set[int]) -> np.ndarray:
    """Nodes used by a subset of faces (0-based indices)."""
    if not faces_zone:
        return np.empty(0, dtype=np.int64)
    f_idx = np.fromiter(faces_zone, dtype=np.int64)
    nodes = face_nodes[f_idx, :].ravel()
    nodes = nodes[nodes >= 0]
    return np.unique(nodes.astype(np.int64))


def apply_combined_ortho_smoother_to_zone(
    mesh: MeshData,
    faces_zone: Set[int],
    cosphi_abs: np.ndarray,
    cosphi_threshold: float,
    it: int,
    max_global_iter: int,
    n_inner: int = 2,
    mu_max: float = 0.4,
    relax: float = 0.2,
    small_edges_global: Optional[np.ndarray] = None,
    removesmalllinkstrsh: float = 0.1,
    verbose: bool = True,
    jsferic: int = 1,
    smalllink_priority: bool = False,
) -> Tuple[bool, bool]:
    """
    Combine simple (Laplacian) smoothing and orthogonality-oriented displacement,
    with a mu(it) factor increasing as in smood:

        Δx = (1 - mu) * Δx_smooth + mu * (Δx_ortho + beta * Δx_small)

    where Δx_ortho reduces |cosphi| and Δx_small pushes circumcenters apart on small-link edges.
    Returns (improved, zone_was_good) for stats; zone_was_good is False on early exit.
    """
    if not faces_zone:
        return (False, False)

    # Zone nodes (0-based indices)
    zone_nodes = zone_nodes_from_faces(mesh.face_nodes, faces_zone)
    if zone_nodes.size == 0:
        return (False, False)

    # Internal / boundary classification
    internal_global, boundary_global = _classify_zone_nodes(
        mesh.face_nodes, mesh.edge_nodes, faces_zone
    )
    internal_set: Set[int] = set(int(n) for n in internal_global.tolist())

    # Neighbors in the zone (for smooth term); also all neighbors (incl. out-of-zone) for boundary
    zone_set: Set[int] = set(int(n) for n in zone_nodes.tolist())
    neighbors: Dict[int, Set[int]] = {int(n): set() for n in zone_nodes.tolist()}
    all_neighbors: Dict[int, Set[int]] = {int(n): set() for n in zone_nodes.tolist()}
    en1 = mesh.edge_nodes[:, 0]
    en2 = mesh.edge_nodes[:, 1]
    ok_e = (en1 >= 0) & (en2 >= 0)
    zmask = np.zeros(
        int(max(zone_nodes.max(), en1.max(initial=0), en2.max(initial=0))) + 2,
        dtype=bool,
    )
    zmask[zone_nodes] = True
    in1 = ok_e & zmask[np.where(ok_e, en1, -1)]
    in2 = ok_e & zmask[np.where(ok_e, en2, -1)]
    edges_in_zone: List[int] = np.where(in1 & in2)[0].tolist()
    ring_edges: List[int] = np.where(in1 ^ in2)[0].tolist()
    for e in edges_in_zone:
        g1 = int(en1[e])
        g2 = int(en2[e])
        neighbors[g1].add(g2)
        neighbors[g2].add(g1)
    for e in np.where(in1 | in2)[0]:
        g1 = int(en1[e])
        g2 = int(en2[e])
        if in1[e]:
            all_neighbors[g1].add(g2)
        if in2[e]:
            all_neighbors[g2].add(g1)
    # Out-of-zone neighbors for boundary nodes (for weighted Laplacian)
    neighbors_out: Dict[int, Set[int]] = {}
    for gid in zone_nodes:
        gid = int(gid)
        out_set = all_neighbors[gid] - zone_set
        if out_set:
            neighbors_out[gid] = out_set
    w_in, w_out = 1.0, 0.3

    # If no internal edge in the zone, nothing to do
    if not edges_in_zone:
        return (False, False)

    edges_in_zone_arr = np.array(edges_in_zone, dtype=int)
    # For acceptance/rollback tests, also include a "crown" of edges crossing the zone boundary
    if ring_edges:
        eval_edges_arr = np.array(edges_in_zone + ring_edges, dtype=int)
    else:
        eval_edges_arr = edges_in_zone_arr
    # Small-link edges in this zone (when global small list is provided)
    small_edges_set = (
        set(small_edges_global.tolist())
        if small_edges_global is not None and small_edges_global.size > 0
        else set()
    )
    small_in_zone = [e for e in edges_in_zone if e in small_edges_set]
    n_small_zone_before = 0
    # Topology is fixed during this zone call: precompute the per-face interior
    # edge counts used by every small-link evaluation below (recomputing them is
    # an O(n_edges) scan per trial and dominated the runtime).
    num_interior_pre: Optional[np.ndarray] = None
    if small_edges_global is not None and small_edges_global.size > 0:
        num_interior_pre = _num_interior_edges_per_face(
            mesh.edge_faces, mesh.face_nodes.shape[0]
        )
        _, small_zone_list = compute_small_links_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            removesmalllinkstrsh=removesmalllinkstrsh,
            edge_indices=edges_in_zone_arr,
            jsferic=jsferic,
            num_interior=num_interior_pre,
        )
        n_small_zone_before = len(small_zone_list)

    # Local quality before displacement (max |cosphi| on zone edges)
    cos_zone_before = np.abs(cosphi_abs[edges_in_zone_arr])
    mask_before = np.isfinite(cos_zone_before)
    if not np.any(mask_before):
        return (False, False)
    max_cosphi_before = float(np.max(cos_zone_before[mask_before]))
    min_cosphi_before = float(np.min(cos_zone_before[mask_before]))

    # Build a distance-based weight in the zone: nodes near the worst edges get full ortho;
    # nodes near the buffer boundary get a reduced ortho weight to avoid degrading good areas.
    core_nodes: Set[int] = set()
    for e in edges_in_zone:
        if abs(cosphi_abs[e]) > cosphi_threshold:
            n1, n2 = mesh.edge_nodes[e, :]
            if n1 >= 0:
                core_nodes.add(int(n1))
            if n2 >= 0:
                core_nodes.add(int(n2))
    dist_weight: Dict[int, float] = {}
    if core_nodes:
        # BFS on zone graph (neighbors) starting from core_nodes
        INF = 10**9
        dist: Dict[int, int] = {int(n): INF for n in zone_nodes.tolist()}
        q: deque[int] = deque()
        for n in core_nodes:
            if n in dist:
                dist[n] = 0
                q.append(n)
        while q:
            u = q.popleft()
            du = dist[u]
            for v in neighbors.get(u, set()):
                if v in dist and dist[v] == INF:
                    dist[v] = du + 1
                    q.append(v)
        # Convert distances to weights in [0.2, 1.0]
        finite_d = [d for d in dist.values() if d < INF]
        if finite_d:
            max_d = max(finite_d)
            decay = max(1.0, max_d / 2.0)
            for n, d in dist.items():
                if d >= INF:
                    w = 0.5
                else:
                    w = 1.0 - float(d) / (decay + 1e-9)
                    w = max(0.2, min(1.0, w))
                dist_weight[n] = w
    else:
        # No clearly bad edges in this zone: uniform weight
        for n in zone_nodes.tolist():
            dist_weight[int(n)] = 1.0

    # When zone is already below cosphi threshold (e.g. only small-link), use smaller steps
    # and no Laplacian (scale_smooth=0) to avoid degrading orthogonality at zone boundaries
    zone_already_good = max_cosphi_before <= cosphi_threshold
    relax_zone = 0.12 if zone_already_good else relax
    scale_smooth = 0.0 if zone_already_good else 1.0
    # Very-good zones (max_cosphi already very low, only small-link): cap relax further
    cosphi_very_good = 0.05
    if (
        zone_already_good
        and max_cosphi_before < cosphi_very_good
        and n_small_zone_before > 0
    ):
        relax_zone = min(relax_zone, 0.08)

    # Skip [good] zones with no small links: nothing to do, avoid useless rollbacks
    if zone_already_good and n_small_zone_before == 0:
        if verbose:
            print(
                f"    [ZONE] it={it} faces={len(faces_zone)} [good] skip (n_small_zone=0, nothing to do)"
            )
        return (False, True)

    # Snapshot of zone node coordinates (for possible rollback)
    zone_idx = zone_nodes.astype(np.int64)  # 0-based
    x_old = mesh.node_x[zone_idx].copy()
    y_old = mesh.node_y[zone_idx].copy()

    # Factor mu(it) increasing from 0 to mu_max
    if max_global_iter > 0:
        mu_it = mu_max * float(it + 1) / float(max_global_iter)
        mu_it = max(0.0, min(mu_max, mu_it))
    else:
        mu_it = mu_max

    alpha = DEFAULT_ORTHO_ALPHA  # Base amplitude of ortho displacement per edge
    # Cumulative targeted ortho displacement (kept separate so the line search
    # can retry it alone when the combined delta fails acceptance).
    ortho_cum_x = np.zeros_like(mesh.node_x)
    ortho_cum_y = np.zeros_like(mesh.node_y)
    tag = "good" if zone_already_good else "bad"
    # Log start for this zone (relax_zone, scale_smooth help tune when many rollbacks)
    if verbose:
        print(
            f"    [ZONE] it={it} faces={len(faces_zone)} [{tag}] "
            f"max|cosphi|_before={max_cosphi_before:.6f} min|cosphi|_before={min_cosphi_before:.6f} "
            f"relax_zone={relax_zone:.2f} scale_smooth={scale_smooth:.2f} mu={mu_it:.3f} n_small_zone={n_small_zone_before}"
        )

    for inner_i in range(max(1, int(n_inner))):
        # Smooth term (local Laplacian); boundary: include out-of-zone neighbors with weight w_out
        dx_s = np.zeros_like(mesh.node_x)
        dy_s = np.zeros_like(mesh.node_y)
        for g in zone_nodes:
            gid = int(g)
            if gid not in internal_set:
                continue
            neigh = neighbors.get(gid, set())
            if not neigh:
                continue
            out_neigh = neighbors_out.get(gid, set())
            if not out_neigh:
                idxs = np.array(list(neigh), dtype=np.int64)
                dx_s[gid] = mesh.node_x[idxs].mean() - mesh.node_x[gid]
                dy_s[gid] = mesh.node_y[idxs].mean() - mesh.node_y[gid]
            else:
                idxs_in = np.array(list(neigh), dtype=np.int64)
                idxs_out = np.array(list(out_neigh), dtype=np.int64)
                bary_in_x = mesh.node_x[idxs_in].mean()
                bary_in_y = mesh.node_y[idxs_in].mean()
                bary_out_x = mesh.node_x[idxs_out].mean()
                bary_out_y = mesh.node_y[idxs_out].mean()
                bary_x = (bary_in_x * w_in + bary_out_x * w_out) / (w_in + w_out)
                bary_y = (bary_in_y * w_in + bary_out_y * w_out) / (w_in + w_out)
                dx_s[gid] = bary_x - mesh.node_x[gid]
                dy_s[gid] = bary_y - mesh.node_y[gid]

        # Ortho term: small corrections on the worst edges in the zone.
        # Skip when zone is already below threshold (e.g. zone only for small-link) to save cost.
        dx_o = np.zeros_like(mesh.node_x)
        dy_o = np.zeros_like(mesh.node_y)
        if edges_in_zone and max_cosphi_before > cosphi_threshold:
            cos_vals = np.abs(cosphi_abs[edges_in_zone])
            bad_mask = cos_vals > cosphi_threshold
            if np.any(bad_mask):
                bad_idx = np.where(bad_mask)[0]
                # Limit number of edges to stay local
                sort_loc = np.argsort(cos_vals[bad_idx])[::-1]
                top_loc = bad_idx[sort_loc[: min(3, sort_loc.size)]]
                for li in top_loc:
                    e = edges_in_zone[li]
                    k3, k4 = mesh.edge_nodes[e, :]
                    if k3 < 0 or k4 < 0:
                        continue
                    g3 = int(k3)
                    g4 = int(k4)
                    move3 = g3 in internal_set
                    move4 = g4 in internal_set
                    # Movable apex (opposite) vertices of the two adjacent
                    # faces: for obtuse pairs (pulled-inside circumcenters)
                    # moving the apex is often the only effective correction.
                    opps: List[int] = []
                    for fe in (int(mesh.edge_faces[e, 0]), int(mesh.edge_faces[e, 1])):
                        if fe >= 0:
                            for vtx in mesh.face_nodes[fe, :3]:
                                vtx = int(vtx)
                                if (
                                    vtx >= 0
                                    and vtx != g3
                                    and vtx != g4
                                    and vtx in internal_set
                                ):
                                    opps.append(vtx)
                    if not (move3 or move4 or opps):
                        continue
                    x3, y3 = mesh.node_x[g3], mesh.node_y[g3]
                    x4, y4 = mesh.node_x[g4], mesh.node_y[g4]
                    ex = x4 - x3
                    ey = y4 - y3
                    norm_e = np.hypot(ex, ey)
                    if norm_e < 1.0e-8:
                        continue
                    # Fresh |cosphi| at the current positions: the global
                    # cosphi_abs snapshot goes stale as nodes move, and a stale
                    # base makes the improvement test accept no-ops forever.
                    base_arr = _cosphi_abs_for_edges(
                        mesh.node_x,
                        mesh.node_y,
                        mesh.face_nodes,
                        mesh.edge_nodes,
                        mesh.edge_faces,
                        np.array([e]),
                        use_circumcenter_3d=True,
                        jsferic=jsferic,
                    )
                    base_val = float(np.abs(base_arr[e]))
                    if not np.isfinite(base_val):
                        continue
                    w_excess = base_val - cosphi_threshold
                    if w_excess <= 0.0:
                        continue
                    px = -ey / norm_e
                    py = ex / norm_e
                    midx = 0.5 * (x3 + x4)
                    midy = 0.5 * (y3 + y4)
                    # Candidate move sets: endpoints perpendicular to the edge
                    # (both senses) and each movable apex radially from the
                    # edge midpoint (both senses).
                    candidates: List[List[Tuple[int, float, float]]] = []
                    for s in (1.0, -1.0):
                        moves: List[Tuple[int, float, float]] = []
                        if move3:
                            moves.append((g3, -s * px, -s * py))
                        if move4:
                            moves.append((g4, s * px, s * py))
                        if moves:
                            candidates.append(moves)
                    for vo in opps:
                        rx = float(mesh.node_x[vo]) - midx
                        ry = float(mesh.node_y[vo]) - midy
                        rn = np.hypot(rx, ry)
                        if rn < 1.0e-12:
                            continue
                        for s in (1.0, -1.0):
                            candidates.append([(vo, s * rx / rn, s * ry / rn)])
                    # Line search over fractions of the local edge length: the
                    # zone-level acceptance (strict improvement + rollback)
                    # still gates the combined displacement, so trying larger
                    # steps here is safe. An absolute step (as before) is
                    # metres-invisible on projected meshes and kilometres-huge
                    # on lon/lat ones.
                    best_improve = 0.0
                    best_moves: Optional[List[Tuple[int, float, float]]] = None
                    done = False
                    for moves in candidates:
                        for frac in (0.3, 0.1, 0.03):
                            step = frac * norm_e
                            trial_x = mesh.node_x.copy()
                            trial_y = mesh.node_y.copy()
                            for nd, udx, udy in moves:
                                trial_x[nd] += step * udx
                                trial_y[nd] += step * udy
                            cosphi_trial = _cosphi_abs_for_edges(
                                trial_x,
                                trial_y,
                                mesh.face_nodes,
                                mesh.edge_nodes,
                                mesh.edge_faces,
                                np.array([e]),
                                use_circumcenter_3d=True,
                                jsferic=jsferic,
                            )
                            new_val = float(np.abs(cosphi_trial[e]))
                            if not np.isfinite(new_val):
                                continue
                            improve = base_val - new_val
                            if improve > best_improve and new_val < base_val:
                                best_improve = improve
                                best_moves = [
                                    (nd, step * udx, step * udy)
                                    for nd, udx, udy in moves
                                ]
                                if new_val <= cosphi_threshold:
                                    done = True
                                    break
                        if done:
                            break
                    if best_moves is not None:
                        for nd, ddx, ddy in best_moves:
                            dx_o[nd] += ddx
                            dy_o[nd] += ddy

        # Small-link term: move opposite vertices along circumcenter separation
        dx_small = np.zeros_like(mesh.node_x)
        dy_small = np.zeros_like(mesh.node_y)
        alpha_small = 0.05
        beta_small = 0.5
        max_small_edges_per_zone = 6
        step_small = min(alpha_small * 0.12, 0.004)
        # Much less aggressive: only in fairly good zones, and smaller steps
        # - Only apply when the zone is already reasonably orthogonal (max_cosphi_before <= 0.30)
        # - Softer aggressive_factor
        # - Clamp step_meters tightly to avoid overshoot and cosphi blow-up
        if n_small_zone_before <= 1:
            aggressive_factor = 1.5
        elif n_small_zone_before <= 3:
            aggressive_factor = 1.2
        else:
            aggressive_factor = 1.0
        # In small-link priority mode (triangles-only pipeline: no quad merge
        # available) the term must fire for every good zone and every inner
        # iteration — the 0.30 gate and parity skip were tuned for the merge
        # pipeline where remaining links were merge's job.
        if small_in_zone and (
            (smalllink_priority and zone_already_good)
            or (zone_already_good and max_cosphi_before <= 0.30 and (inner_i % 2 == 0))
        ):
            _, small_zone_current = compute_small_links_from_arrays(
                mesh.node_x,
                mesh.node_y,
                mesh.face_nodes,
                mesh.edge_nodes,
                mesh.edge_faces,
                removesmalllinkstrsh=removesmalllinkstrsh,
                edge_indices=edges_in_zone_arr,
                jsferic=jsferic,
                num_interior=num_interior_pre,
            )
            n_small_zone_current = len(small_zone_current)
            nface = mesh.face_nodes.shape[0]
            # Circumcenters/areas are only read for the faces adjacent to the
            # zone's small links; computing them mesh-wide per inner iteration
            # dominated the runtime on large meshes.
            link_faces = sorted(
                {
                    int(ff)
                    for e2 in small_in_zone[:max_small_edges_per_zone]
                    for ff in mesh.edge_faces[e2]
                    if ff >= 0 and ff < nface
                }
            )
            link_faces_arr = np.asarray(link_faces, dtype=np.int64)
            face_mask_links = np.zeros(nface, dtype=bool)
            face_mask_links[link_faces_arr] = True
            vert_deg = np.column_stack([mesh.node_x, mesh.node_y])
            circum_ll = _circumcenters_lonlat_ugrid(
                vert_deg,
                mesh.face_nodes,
                mesh.edge_faces,
                face_mask=face_mask_links,
                jsferic=jsferic,
                num_interior=num_interior_pre,
            )
            tria = mesh.face_nodes[:, :3]
            # Face areas in the same local frame `_lonlat_to_local_xy` uses
            # (full-mesh mean reference), computed only for the link faces.
            x0m = float(np.nanmean(mesh.node_x))
            y0m = float(np.nanmean(mesh.node_y))
            ba = np.zeros(nface, dtype=np.float64)
            if link_faces_arr.size > 0:
                tlf = tria[link_faces_arr]
                valid_lf = (tlf >= 0).all(axis=1)
                tv = tlf[valid_lf]
                nodes_lf = np.unique(tv)
                x0a = np.full(nodes_lf.size, x0m)
                y0a = np.full(nodes_lf.size, y0m)
                ux = _getdx_vec(
                    x0a, y0a, mesh.node_x[nodes_lf], mesh.node_y[nodes_lf], jsferic
                )
                uy = _getdy_vec(
                    x0a, y0a, mesh.node_x[nodes_lf], mesh.node_y[nodes_lf], jsferic
                )
                q0 = np.searchsorted(nodes_lf, tv[:, 0])
                q1 = np.searchsorted(nodes_lf, tv[:, 1])
                q2 = np.searchsorted(nodes_lf, tv[:, 2])
                ev12x, ev12y = ux[q1] - ux[q0], uy[q1] - uy[q0]
                ev13x, ev13y = ux[q2] - ux[q0], uy[q2] - uy[q0]
                ba[link_faces_arr[valid_lf]] = np.abs(
                    0.5 * (ev12x * ev13y - ev12y * ev13x)
                )
            for li, e in enumerate(small_in_zone[:max_small_edges_per_zone]):
                f1, f2 = mesh.edge_faces[e, 0], mesh.edge_faces[e, 1]
                k3, k4 = mesh.edge_nodes[e, 0], mesh.edge_nodes[e, 1]
                if k3 < 0 or k4 < 0 or f1 < 0 or f2 < 0 or f1 >= nface or f2 >= nface:
                    continue
                f1, f2, k3, k4 = int(f1), int(f2), int(k3), int(k4)
                tri1 = mesh.face_nodes[f1, :3]
                tri2 = mesh.face_nodes[f2, :3]
                opp1 = next(
                    (int(v) for v in tri1 if v >= 0 and v != k3 and v != k4), None
                )
                opp2 = next(
                    (int(v) for v in tri2 if v >= 0 and v != k3 and v != k4), None
                )
                if opp1 is None or opp2 is None or opp1 == opp2:
                    continue
                move_opp1 = opp1 in internal_set
                move_opp2 = opp2 in internal_set
                if not (move_opp1 or move_opp2):
                    if smalllink_priority and (
                        k3 in internal_set or k4 in internal_set
                    ):
                        # Boundary-locked apexes: fall back to searching moves
                        # of the shared edge's endpoints instead.
                        opp1, opp2 = k3, k4
                        move_opp1 = k3 in internal_set
                        move_opp2 = k4 in internal_set
                    else:
                        continue
                cc1, cc2 = circum_ll[f1], circum_ll[f2]
                if np.any(np.isnan(cc1)) or np.any(np.isnan(cc2)):
                    continue
                dx_m = _getdx(cc1[0], cc1[1], cc2[0], cc2[1], jsferic)
                dy_m = _getdy(cc1[0], cc1[1], cc2[0], cc2[1], jsferic)
                dxlink = np.sqrt(dx_m * dx_m + dy_m * dy_m)
                if not smalllink_priority and dxlink < 1e-12:
                    continue
                sqrt_ba1 = np.sqrt(max(ba[f1], 1e-20))
                sqrt_ba2 = np.sqrt(max(ba[f2], 1e-20))
                dxlim = 0.9 * removesmalllinkstrsh * 0.5 * (sqrt_ba1 + sqrt_ba2)
                if dxlink >= dxlim:
                    continue
                max_step_meters = (
                    min(np.sqrt(max(ba[f1], 1e-20)), np.sqrt(max(ba[f2], 1e-20))) * 0.5
                )

                if smalllink_priority:
                    # Probe-style search: move the apexes along the edge's
                    # perpendicular bisector (both circumcenters live on that
                    # axis; the cc-difference direction is numerical noise for
                    # near-cocircular links), trying independent per-apex sign
                    # combinations and ascending step sizes. Accept the first
                    # candidate that reduces the zone's link count without
                    # pushing the local |cosphi| above the threshold.
                    exk = float(mesh.node_x[k4] - mesh.node_x[k3])
                    eyk = float(mesh.node_y[k4] - mesh.node_y[k3])
                    eln = np.hypot(exk, eyk)
                    if eln < 1.0e-12:
                        continue
                    axx = -eyk / eln
                    axy = exk / eln
                    quad_nodes = np.unique(np.concatenate([tri1, tri2]))
                    quad_nodes = quad_nodes[quad_nodes >= 0]
                    quad_edges = _edges_among_nodes(mesh.edge_nodes, quad_nodes)
                    s1_opts = (1.0, -1.0) if move_opp1 else (0.0,)
                    s2_opts = (1.0, -1.0) if move_opp2 else (0.0,)
                    found = False
                    # Trial by in-place displace/restore of the two apexes:
                    # copying the full coordinate arrays per candidate is the
                    # dominant allocation cost of this search.
                    keep1 = (float(mesh.node_x[opp1]), float(mesh.node_y[opp1]))
                    keep2 = (float(mesh.node_x[opp2]), float(mesh.node_y[opp2]))
                    for frac in (0.25, 0.5, 1.0, 2.0):
                        step = frac * max_step_meters
                        for s1 in s1_opts:
                            for s2 in s2_opts:
                                if s1 == 0.0 and s2 == 0.0:
                                    continue
                                if move_opp1:
                                    mesh.node_x[opp1] = keep1[0] + s1 * step * axx
                                    mesh.node_y[opp1] = keep1[1] + s1 * step * axy
                                if move_opp2:
                                    mesh.node_x[opp2] = keep2[0] + s2 * step * axx
                                    mesh.node_y[opp2] = keep2[1] + s2 * step * axy
                                n_tr, _ = compute_small_links_from_arrays(
                                    mesh.node_x,
                                    mesh.node_y,
                                    mesh.face_nodes,
                                    mesh.edge_nodes,
                                    mesh.edge_faces,
                                    removesmalllinkstrsh=removesmalllinkstrsh,
                                    edge_indices=edges_in_zone_arr,
                                    jsferic=jsferic,
                                    num_interior=num_interior_pre,
                                )
                                ok_trial = int(n_tr) < n_small_zone_current
                                if ok_trial:
                                    cq = _cosphi_abs_for_edges(
                                        mesh.node_x,
                                        mesh.node_y,
                                        mesh.face_nodes,
                                        mesh.edge_nodes,
                                        mesh.edge_faces,
                                        quad_edges,
                                        use_circumcenter_3d=True,
                                        jsferic=jsferic,
                                    )
                                    vals = cq[quad_edges]
                                    vals = vals[np.isfinite(vals)]
                                    if (
                                        vals.size
                                        and float(np.max(vals)) > cosphi_threshold
                                    ):
                                        ok_trial = False
                                mesh.node_x[opp1], mesh.node_y[opp1] = keep1
                                mesh.node_x[opp2], mesh.node_y[opp2] = keep2
                                if not ok_trial:
                                    continue
                                if move_opp1:
                                    dx_small[opp1] += s1 * step * axx
                                    dy_small[opp1] += s1 * step * axy
                                if move_opp2:
                                    dx_small[opp2] += s2 * step * axx
                                    dy_small[opp2] += s2 * step * axy
                                found = True
                                break
                            if found:
                                break
                        if found:
                            break
                    continue

                needed_distance = (dxlim - dxlink) * aggressive_factor
                cc_diff_deg = np.array(
                    [cc2[0] - cc1[0], cc2[1] - cc1[1]], dtype=np.float64
                )
                # Conservative step: at most 25% of needed distance, and at most half the current link length
                step_meters = min(needed_distance * 0.25, max_step_meters, 0.5 * dxlink)
                if step_meters < 1e-12:
                    continue
                disp_deg = cc_diff_deg * (step_meters / dxlink)
                best_n_small = n_small_zone_current
                best_dopp1 = np.zeros(2)
                best_dopp2 = np.zeros(2)
                for sign in (1, -1):
                    trial_x = mesh.node_x.copy()
                    trial_y = mesh.node_y.copy()
                    d1 = (-sign * 0.5 * disp_deg) if move_opp1 else np.zeros(2)
                    d2 = (sign * 0.5 * disp_deg) if move_opp2 else np.zeros(2)
                    if move_opp1:
                        trial_x[opp1] = mesh.node_x[opp1] + d1[0]
                        trial_y[opp1] = mesh.node_y[opp1] + d1[1]
                    if move_opp2:
                        trial_x[opp2] = mesh.node_x[opp2] + d2[0]
                        trial_y[opp2] = mesh.node_y[opp2] + d2[1]
                    _, small_trial = compute_small_links_from_arrays(
                        trial_x,
                        trial_y,
                        mesh.face_nodes,
                        mesh.edge_nodes,
                        mesh.edge_faces,
                        removesmalllinkstrsh=removesmalllinkstrsh,
                        edge_indices=edges_in_zone_arr,
                        jsferic=jsferic,
                        num_interior=num_interior_pre,
                    )
                    if len(small_trial) < best_n_small:
                        best_n_small = len(small_trial)
                        best_dopp1 = d1
                        best_dopp2 = d2
                if best_n_small < n_small_zone_current:
                    if move_opp1:
                        dx_small[opp1] += best_dopp1[0]
                        dy_small[opp1] += best_dopp1[1]
                    if move_opp2:
                        dx_small[opp2] += best_dopp2[0]
                        dy_small[opp2] += best_dopp2[1]

        # Combine and update: smooth + ortho + small_link (scale_smooth dampens Laplacian for "good" zones)
        new_x = mesh.node_x.copy()
        new_y = mesh.node_y.copy()
        for g in zone_nodes:
            gid = int(g)
            if gid not in internal_set:
                continue
            w_node = dist_weight.get(gid, 1.0)
            if smalllink_priority:
                # The circumcenter-separation push was validated by the sign
                # trial at this exact magnitude; apply it at full weight like
                # the ortho displacement (the mu*relax damping below shrinks
                # it ~20x, which cannot clear a link).
                full_dx = w_node * (dx_o[gid] + dx_small[gid])
                full_dy = w_node * (dy_o[gid] + dy_small[gid])
                dx = (1.0 - mu_it) * scale_smooth * dx_s[gid]
                dy = (1.0 - mu_it) * scale_smooth * dy_s[gid]
            else:
                full_dx = w_node * dx_o[gid]
                full_dy = w_node * dy_o[gid]
                dx_small_w = w_node * beta_small * dx_small[gid]
                dy_small_w = w_node * beta_small * dy_small[gid]
                dx = (1.0 - mu_it) * scale_smooth * dx_s[gid] + mu_it * dx_small_w
                dy = (1.0 - mu_it) * scale_smooth * dy_s[gid] + mu_it * dy_small_w
            # The ortho displacement was validated by a per-edge line search at
            # this exact magnitude; damping it by relax*mu makes it ineffective
            # (a fraction of a percent of the local edge length). Apply it at
            # full weight — the zone-level acceptance/rollback still gates it.
            new_x[gid] += relax_zone * dx + full_dx
            new_y[gid] += relax_zone * dy + full_dy
            ortho_cum_x[gid] += full_dx
            ortho_cum_y[gid] += full_dy
        mesh.node_x[:] = new_x
        mesh.node_y[:] = new_y

    # Line search: try factors 1.0, 0.5, 0.25 on total displacement to avoid rollback
    # For [good] zones: accept "no degradation" (max_ca <= threshold, n_small_za <= n_small_before)
    delta_x = mesh.node_x[zone_idx].copy() - x_old
    delta_y = mesh.node_y[zone_idx].copy() - y_old
    mesh.node_x[zone_idx] = x_old
    mesh.node_y[zone_idx] = y_old
    # Candidate displacements: the combined delta first; if it fails, the
    # accumulated targeted ortho moves alone. On strongly graded meshes the
    # Laplacian part of the combined delta can degrade the crown and veto a
    # valid per-edge repair bundled in the same displacement.
    delta_candidates = [(delta_x, delta_y)]
    delta_ox = ortho_cum_x[zone_idx].copy()
    delta_oy = ortho_cum_y[zone_idx].copy()
    if np.any(delta_ox != 0.0) or np.any(delta_oy != 0.0):
        delta_candidates.append((delta_ox, delta_oy))
    improved = False
    best_factor = 0.0
    best_delta = delta_candidates[0]
    max_cosphi_after = float("inf")
    min_cosphi_after = 0.0
    n_small_zone_after = n_small_zone_before
    for cand_dx, cand_dy in delta_candidates:
        if improved:
            break
        for factor in (1.0, 0.5, 0.25):
            mesh.node_x[zone_idx] = x_old + factor * cand_dx
            mesh.node_y[zone_idx] = y_old + factor * cand_dy
            cosphi_after = _cosphi_abs_for_edges(
                mesh.node_x,
                mesh.node_y,
                mesh.face_nodes,
                mesh.edge_nodes,
                mesh.edge_faces,
                eval_edges_arr,
                use_circumcenter_3d=True,
                jsferic=jsferic,
            )
            cos_zone_after = np.abs(cosphi_after[eval_edges_arr])
            mask_after = np.isfinite(cos_zone_after)
            if not np.any(mask_after):
                continue
            max_ca = float(np.max(cos_zone_after[mask_after]))
            min_ca = float(np.min(cos_zone_after[mask_after]))
            n_small_za = 0
            if small_edges_global is not None and small_edges_global.size > 0:
                _, small_zone_after_list = compute_small_links_from_arrays(
                    mesh.node_x,
                    mesh.node_y,
                    mesh.face_nodes,
                    mesh.edge_nodes,
                    mesh.edge_faces,
                    removesmalllinkstrsh=removesmalllinkstrsh,
                    edge_indices=edges_in_zone_arr,
                    jsferic=jsferic,
                    num_interior=num_interior_pre,
                )
                n_small_za = len(small_zone_after_list)
            is_strict_improved = max_ca < max_cosphi_before or (
                small_edges_global is not None
                and small_edges_global.size > 0
                and n_small_za < n_small_zone_before
                and max_ca <= cosphi_threshold
            )
            if zone_already_good:
                # For [good]: accept only if we don't degrade orthogonality near the zone boundary too much.
                # This prevents "small-link only" actions from quietly increasing max|cosphi| in the crown.
                ortho_slack = 0.02
                if smalllink_priority:
                    # Clearing a link may legitimately trade |cosphi| slack below
                    # the criteria threshold; capping at max_before+0.02 vetoes
                    # valid repairs in already-very-orthogonal zones.
                    ortho_cap = cosphi_threshold
                else:
                    ortho_cap = min(cosphi_threshold, max_cosphi_before + ortho_slack)
                is_acceptable = (max_ca <= ortho_cap) and (
                    n_small_za <= n_small_zone_before
                )
                if is_acceptable and not improved:
                    improved = True
                    best_factor = factor
                    best_delta = (cand_dx, cand_dy)
                    max_cosphi_after = max_ca
                    min_cosphi_after = min_ca
                    n_small_zone_after = n_small_za
                elif is_acceptable and improved:
                    if smalllink_priority:
                        # Prefer clearing links first, then smaller max_ca, then larger factor
                        better = n_small_za < n_small_zone_after or (
                            n_small_za == n_small_zone_after
                            and (
                                max_ca < max_cosphi_after
                                or (max_ca == max_cosphi_after and factor > best_factor)
                            )
                        )
                    else:
                        # Prefer smaller max_ca, then smaller n_small_za, then larger factor
                        better = max_ca < max_cosphi_after or (
                            max_ca == max_cosphi_after
                            and (
                                n_small_za < n_small_zone_after
                                or (
                                    n_small_za == n_small_zone_after
                                    and factor > best_factor
                                )
                            )
                        )
                    if better:
                        best_factor = factor
                        best_delta = (cand_dx, cand_dy)
                        max_cosphi_after = max_ca
                        min_cosphi_after = min_ca
                        n_small_zone_after = n_small_za
            else:
                # [bad] zone: first strict improvement wins
                if is_strict_improved:
                    improved = True
                    best_factor = factor
                    best_delta = (cand_dx, cand_dy)
                    max_cosphi_after = max_ca
                    min_cosphi_after = min_ca
                    n_small_zone_after = n_small_za
                    break
    if improved and zone_already_good:
        # Apply best factor (we may have tried several)
        mesh.node_x[zone_idx] = x_old + best_factor * best_delta[0]
        mesh.node_y[zone_idx] = y_old + best_factor * best_delta[1]
    elif improved and not zone_already_good:
        mesh.node_x[zone_idx] = x_old + best_factor * best_delta[0]
        mesh.node_y[zone_idx] = y_old + best_factor * best_delta[1]
    if not improved:
        mesh.node_x[zone_idx] = x_old
        mesh.node_y[zone_idx] = y_old
        if verbose:
            print(
                f"    [ZONE] rollback [{tag}]: max_before={max_cosphi_before:.6f} "
                f"(line_search: all factors 1.0, 0.5, 0.25 failed)"
            )
    else:
        log_small = ""
        if small_edges_global is not None and small_edges_global.size > 0:
            log_small = f" n_small_zone={n_small_zone_before}->{n_small_zone_after}"
        if verbose:
            print(
                f"    [ZONE] accept [{tag}]: max_before={max_cosphi_before:.6f} "
                f"max_after={max_cosphi_after:.6f} "
                f"min_after={min_cosphi_after:.6f} factor={best_factor:.2f}{log_small}"
            )
    return (improved, zone_already_good)


# ---------------------------------------------------------------------------
# Triangle-mesh entry point: orthogonalize (vert, tria) directly.
# ---------------------------------------------------------------------------


@dataclass
class TriaOrthoResult:
    vert: np.ndarray  # (N,2) float64
    tria: np.ndarray  # (T,3) int64 (topology unchanged unless edge flips are enabled)
    max_cosphi: float
    n_small_flow_links: int
    n_zones_orthogonalized: int


def orthogonalize_tria_mesh(
    vert: np.ndarray,
    tria: np.ndarray,
    *,
    cosphi_threshold: float = 0.49,
    removesmalllinkstrsh: float = 0.1,
    buffer_layers: int = 2,
    max_global_iter: int = 8,
    smooth_iter: int = 16,
    enable_edge_flips: bool = True,
    verbose: bool = True,
    jsferic: int = 1,
    smalllink_priority: bool = False,
) -> TriaOrthoResult:
    """
    Orthogonalize a pure triangle mesh, working directly on ``(vert, tria)``.

    Parameters
    ----------
    vert : (N, 2) float array
        Node coordinates: lon/lat degrees for ``jsferic=1``, projected x/y for
        ``jsferic=0``.
    tria : (T, 3) int array
        0-based triangle connectivity.

    Notes
    -----
    - Topology changes only when ``enable_edge_flips=True``, and only via local
      edge flips inside convex quads (the mesh stays triangles).
    - No merging into quads is performed here.
    """
    vert = np.asarray(vert, dtype=np.float64)
    if vert.ndim != 2 or vert.shape[1] != 2:
        raise ValueError("vert must be an array of shape (N,2)")
    tria = np.asarray(tria, dtype=np.int64)

    face_nodes = tria.copy()
    edge_nodes, edge_faces = _build_edges_from_tria(tria)

    mesh = MeshData(
        node_x=vert[:, 0].copy(),
        node_y=vert[:, 1].copy(),
        face_nodes=face_nodes,
        edge_nodes=edge_nodes,
        edge_faces=edge_faces,
    )

    n_faces = face_nodes.shape[0]
    face_neighbors = build_face_adjacency(mesh.edge_faces, n_faces)

    # Number of zone passes run (reported for the pipeline's progress log).
    n_zones_orthogonalized = 0

    for it in range(int(max_global_iter)):
        _, _, cosphi_abs = compute_cosphi_abs_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            use_circumcenter_3d=True,
            jsferic=jsferic,
        )
        mask = ~np.isnan(cosphi_abs)
        if not np.any(mask):
            break

        max_cosphi = float(np.nanmax(cosphi_abs[mask]))
        bad_edges = np.where((mask) & (cosphi_abs > cosphi_threshold))[0]
        n_small, small_edges_arr = compute_small_links_from_arrays(
            mesh.node_x,
            mesh.node_y,
            mesh.face_nodes,
            mesh.edge_nodes,
            mesh.edge_faces,
            removesmalllinkstrsh=removesmalllinkstrsh,
            jsferic=jsferic,
        )

        # Edge-flip pre-pass (still triangles). If no small links remain but
        # orthogonality is still bad, let the problematic edges participate too.
        if enable_edge_flips:
            flip_candidates = small_edges_arr if n_small > 0 else bad_edges
            # Always cap the local |cosphi| a flip may introduce: an unguarded
            # flip can create very obtuse triangles with |cosphi| ~ 1.0 that no
            # later node movement can repair.
            try_flip_candidate_edges_ugrid(
                mesh,
                flip_candidates,
                removesmalllinkstrsh,
                max_cosphi_allowed=cosphi_threshold,
                jsferic=jsferic,
            )
            # Topology changed: rebuild edges/faces and re-measure.
            mesh.edge_nodes, mesh.edge_faces = _build_edges_from_tria(
                mesh.face_nodes[:, :3]
            )
            face_neighbors = build_face_adjacency(mesh.edge_faces, n_faces)
            _, _, cosphi_abs = compute_cosphi_abs_from_arrays(
                mesh.node_x,
                mesh.node_y,
                mesh.face_nodes,
                mesh.edge_nodes,
                mesh.edge_faces,
                use_circumcenter_3d=True,
                jsferic=jsferic,
            )
            mask = ~np.isnan(cosphi_abs)
            max_cosphi = (
                float(np.nanmax(cosphi_abs[mask])) if np.any(mask) else max_cosphi
            )
            n_small, small_edges_arr = compute_small_links_from_arrays(
                mesh.node_x,
                mesh.node_y,
                mesh.face_nodes,
                mesh.edge_nodes,
                mesh.edge_faces,
                removesmalllinkstrsh=removesmalllinkstrsh,
                jsferic=jsferic,
            )

        if max_cosphi <= cosphi_threshold and n_small == 0:
            break

        bad_edges = np.where((mask) & (cosphi_abs > cosphi_threshold))[0]
        bad_set = set(int(e) for e in bad_edges.tolist())
        sort_idx = (
            np.argsort(cosphi_abs[bad_edges])[::-1]
            if bad_edges.size > 0
            else np.array([], dtype=np.int64)
        )
        bad_edges_sorted = (
            bad_edges[sort_idx] if bad_edges.size > 0 else np.array([], dtype=np.int64)
        )
        small_only = np.array(
            [e for e in small_edges_arr.tolist() if e not in bad_set], dtype=np.int64
        )
        problematic_edges = (
            np.concatenate([bad_edges_sorted, small_only])
            if bad_edges_sorted.size > 0
            else small_only
        )
        if problematic_edges.size == 0:
            break

        visited_faces_global: Set[int] = set()
        for e in problematic_edges:
            f1, f2 = mesh.edge_faces[e, :]
            start_faces: List[int] = []
            if f1 >= 0:
                start_faces.append(int(f1))
            if f2 >= 0:
                start_faces.append(int(f2))
            if not start_faces:
                continue

            this_buffer = buffer_layers + (1 if int(e) in bad_set else 0)
            faces_zone = bfs_faces(start_faces, face_neighbors, this_buffer)
            if faces_zone.issubset(visited_faces_global):
                continue

            apply_combined_ortho_smoother_to_zone(
                mesh=mesh,
                faces_zone=faces_zone,
                cosphi_abs=cosphi_abs,
                cosphi_threshold=cosphi_threshold,
                it=it,
                max_global_iter=max_global_iter,
                n_inner=max(1, int(smooth_iter)),
                small_edges_global=small_edges_arr,
                removesmalllinkstrsh=removesmalllinkstrsh,
                verbose=verbose,
                jsferic=jsferic,
                smalllink_priority=smalllink_priority,
            )
            n_zones_orthogonalized += 1
            visited_faces_global.update(faces_zone)

    # Final metrics.
    _, _, cosphi_abs_final = compute_cosphi_abs_from_arrays(
        mesh.node_x,
        mesh.node_y,
        mesh.face_nodes,
        mesh.edge_nodes,
        mesh.edge_faces,
        use_circumcenter_3d=True,
        jsferic=jsferic,
    )
    mask_final = ~np.isnan(cosphi_abs_final)
    max_final = (
        float(np.nanmax(cosphi_abs_final[mask_final]))
        if np.any(mask_final)
        else float("nan")
    )
    n_small_final, _ = compute_small_links_from_arrays(
        mesh.node_x,
        mesh.node_y,
        mesh.face_nodes,
        mesh.edge_nodes,
        mesh.edge_faces,
        removesmalllinkstrsh=removesmalllinkstrsh,
        jsferic=jsferic,
    )

    vert_out = np.column_stack([mesh.node_x, mesh.node_y]).astype(
        np.float64, copy=False
    )
    tria_out = np.asarray(mesh.face_nodes[:, :3], dtype=np.int64)
    return TriaOrthoResult(
        vert=vert_out,
        tria=tria_out,
        max_cosphi=max_final,
        n_small_flow_links=int(n_small_final),
        n_zones_orthogonalized=int(n_zones_orthogonalized),
    )
