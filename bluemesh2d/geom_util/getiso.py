"""Iso-contour extraction from structured 2D scalar fields.

Utilities for extracting contour polylines and polygon regions from gridded
data when building mesh domains from bathymetry or other scalar fields."""
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata
from shapely.geometry import Polygon


def getiso(xpos, ypos, zdat, ilev, filt=0.0):
    """An iso-contour from a structured 2D dataset.

    Parameters
    ----------
    xpos, ypos : ndarray of shape (N, M)
        Grid coordinates (same shape as ``zdat``).
    zdat : ndarray of shape (N, M)
        Scalar field values.
    ilev : float
        Iso-contour level.
    filt : float, optional
        Minimum bounding-box extent in x and y for a contour segment to be
        kept. Segments smaller than ``filt`` in either direction are dropped.
        Default is 0 (keep all).

    Returns
    -------
    node : ndarray of shape (K, 2)
        Vertex coordinates of contour polylines.
    edge : ndarray of shape (E, 2), dtype int
        Edge list (vertex indices) defining the contour PSLG.

    References
    ----------
    Translation of the MESH2D function ``getiso``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    fig, ax = plt.subplots()
    cs = ax.contour(xpos, ypos, zdat, levels=[ilev])

    # Matplotlib < 3.9 exposes line contours via cs.collections; >= 3.9 uses
    # cs.allsegs (collections was removed in 3.10).
    try:
        collections = cs.collections
    except AttributeError:
        class DummyCollection:
            def __init__(self, segs):
                self.paths = [type("PathLike", (), {"vertices": seg})() for seg in segs]

            def get_paths(self):
                return self.paths

        collections = [DummyCollection(segs) for segs in cs.allsegs if segs]

    plt.close(fig)

    node = []
    edge = []
    for collection in collections:
        for path in collection.get_paths():
            ppts = path.vertices
            numc = ppts.shape[0]

            pmin = ppts.min(axis=0)
            pmax = ppts.max(axis=0)
            pdel = pmax - pmin

            if np.min(pdel) >= filt:
                if np.allclose(ppts[0], ppts[-1]):
                    enew = np.vstack(
                        [
                            np.column_stack(
                                [np.arange(0, numc - 1), np.arange(1, numc)]
                            ),
                            [numc - 1, 0],
                        ]
                    )
                else:
                    enew = np.column_stack([np.arange(0, numc - 1), np.arange(1, numc)])

                offset = len(node)
                enew = enew + offset

                node.extend(ppts.tolist())
                edge.extend(enew.tolist())

    node = np.array(node)
    edge = np.array(edge, dtype=int)

    return node, edge


def getiso_polygon(x, y, z, zmax=None, grid_res=None):
    """Polygon regions from a 2D scalar field by thresholding.

    Builds filled contours for regions where ``z <= zmax`` (or the zero level
    when ``zmax`` is ``None``) and returns valid Shapely polygons with holes
    resolved.

    Parameters
    ----------
    x, y : ndarray
        1D or 2D arrays defining grid coordinates (must match ``z``).
    z : ndarray
        Scalar field values.
    zmax : float, optional
        Threshold value. Polygons enclose regions where ``z <= zmax``. If
        ``None``, the zero level is used.
    grid_res : int, optional
        Grid resolution used when ``x`` and ``y`` are 1D scattered points.
        If ``None``, inferred from ``len(z)``.

    Returns
    -------
    polygons : list of shapely.geometry.Polygon or None
        Polygons sorted by area with hole geometry attached. ``None`` if no
        valid region is found.

    Notes
    -----
    Filled-contour paths are compound: one path holds the outer boundary plus
    hole rings as separate closed sub-rings. ``Path.to_polygons()`` splits
    those sub-rings; using ``path.vertices`` alone would concatenate rings
    into an invalid polygon.

    References
    ----------
    Translation of the MESH2D function ``getiso``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
        if grid_res is None:
            grid_res = int(np.sqrt(len(z)))

        dx = (np.max(x) - np.min(x)) / (grid_res - 1)
        dy = (np.max(y) - np.min(y)) / (grid_res - 1)

        xi = np.arange(np.min(x) - dx / 2, np.max(x) + dx / 2 + dx, dx)
        yi = np.arange(np.min(y) - dy / 2, np.max(y) + dy / 2 + dy, dy)

        X, Y = np.meshgrid(xi, yi)

        Z = griddata((x, y), z, (X, Y), method="linear")

        z = np.nan_to_num(Z, nan=np.nanmedian(z))
    else:
        X, Y = x, y

    if z.ndim == 2 and X.ndim == 1 and Y.ndim == 1:
        X, Y = np.meshgrid(X, Y)

    if X.shape != Y.shape or X.shape != z.shape:
        raise ValueError("Inconsistent array shapes between x, y, z.")

    new_mask = np.full(z.shape, 1)
    if zmax is not None:
        new_mask[z > zmax] = -1

    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, new_mask, levels=[0, 1])

    # Matplotlib < 3.9: one PathCollection per level (cs.collections, removed
    # in 3.10). >= 3.9: cs.get_paths() returns one compound path per level.
    try:
        paths = [p for c in cs.collections for p in c.get_paths()]
    except AttributeError:
        paths = list(cs.get_paths())

    plt.close(fig)

    polygons = []
    for path in paths:
        for pts in path.to_polygons():
            pts = np.asarray(pts)
            if pts.shape[0] < 4:
                continue
            poly = Polygon(pts)
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                polygons.append(poly)

    if not polygons:
        return None

    polygons.sort(key=lambda p: p.area, reverse=True)

    final_polys = []
    while polygons:
        outer = polygons.pop(0)
        holes = [p.exterior.coords for p in polygons if p.within(outer)]
        polygons = [p for p in polygons if not p.within(outer)]
        final_polys.append(Polygon(outer.exterior.coords, holes))

    return final_polys
