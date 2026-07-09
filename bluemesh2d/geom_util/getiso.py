import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import Polygon
from scipy.interpolate import griddata


def getiso(xpos, ypos, zdat, ilev, filt=0.0):
    """
    getiso extract an iso-contour from a structured 2D dataset.

    Parameters
    ----------
    xpos, ypos : ndarray (N, M)
        Grid coordinates (must be same shape as zdat).
    zdat : ndarray (N, M)
        Scalar field values.
    ilev : float
        Isocontour level.
    filt : float, optional
        Minimum length scale filter (default 0).

    Returns
    -------
    node : ndarray (K, 2)
        Coordinates of contour vertices.
    edge : ndarray (E, 2)
        PSLG edges between contour vertices.

    References
    ----------
    Translation of the MESH2D function `getiso`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ----------------------------------- compute the isocontour
    fig, ax = plt.subplots()
    cs = ax.contour(xpos, ypos, zdat, levels=[ilev])

    try:
        # ------------------------------------ Matplotlib <3.9
        collections = cs.collections
    except AttributeError:
        # ------------------------------------ Matplotlib >=3.9
        class DummyCollection:
            def __init__(self, segs):
                self.paths = [type("PathLike", (), {"vertices": seg})() for seg in segs]

            def get_paths(self):
                return self.paths

        collections = [DummyCollection(segs) for segs in cs.allsegs if segs]

    plt.close(fig)

    node = []
    edge = []
    # ------------------------------------ "walk" contour segment
    for collection in collections:
        for path in collection.get_paths():
            ppts = path.vertices
            numc = ppts.shape[0]

            pmin = ppts.min(axis=0)
            pmax = ppts.max(axis=0)
            pdel = pmax - pmin

            if np.min(pdel) >= filt:
                if np.allclose(ppts[0], ppts[-1]):
                    # -------------------------------- closed - back to start
                    enew = np.vstack(
                        [
                            np.column_stack(
                                [np.arange(0, numc - 1), np.arange(1, numc)]
                            ),
                            [numc - 1, 0],
                        ]
                    )
                else:
                    # -------------------------------- open - dangling endpts
                    enew = np.column_stack([np.arange(0, numc - 1), np.arange(1, numc)])

                offset = len(node)
                enew = enew + offset

                node.extend(ppts.tolist())
                edge.extend(enew.tolist())

    node = np.array(node)
    edge = np.array(edge, dtype=int)

    return node, edge


def getiso_polygon(x, y, z, zmax=None, grid_res=None) -> Polygon:
    """
    Extract a MultiPolygon from a 2D scalar field by thresholding (similar to getiso logic).

    Parameters
    ----------
    x, y : ndarray
        1D or 2D arrays defining the grid coordinates (must match z).
    z : ndarray
        2D scalar field.
    zmax : float, optional
        Threshold value. Polygons will enclose regions where z <= zmax.
        If None, the 0-level is used.
    grid_res : int, optional
        If x and y are 1D arrays, this defines the grid resolution for interpolation. If None, it is inferred from the length of z.

    Returns
    -------
    multipolygon : shapely.geometry.MultiPolygon
        Extracted polygons with holes (if any).
    """

    # -----------------------ensure arrays are 2D and consistent
    if x.ndim == 1 and y.ndim == 1 and z.ndim == 1:
        if grid_res is None:
            grid_res = int(np.sqrt(len(z)))

        dx = (np.max(x) - np.min(x)) / (grid_res - 1)
        dy = (np.max(y) - np.min(y)) / (grid_res - 1)

        xi = np.arange(np.min(x) - dx/2, np.max(x) + dx/2 + dx, dx)
        yi = np.arange(np.min(y) - dy/2, np.max(y) + dy/2 + dy, dy)

        X, Y = np.meshgrid(xi, yi)

        Z = griddata((x, y), z, (X, Y), method='linear')

        z = np.nan_to_num(Z, nan=np.nanmedian(z))
    else:
        X, Y = x, y
        
    if z.ndim == 2 and X.ndim == 1 and  Y.ndim == 1:
        X, Y = np.meshgrid(X, Y)

    if X.shape != Y.shape or X.shape != z.shape:
        raise ValueError("Inconsistent array shapes between x, y, z.")

    # -----------------------select contour level
    new_mask = np.full(z.shape, 1)
    if zmax is not None:
        new_mask[z > zmax] = -1

    # -----------------------compute contour lines
    fig, ax = plt.subplots()
    cs = ax.contourf(X, Y, new_mask, levels=[0, 1])

    # Collect the filled-contour paths across matplotlib versions:
    #   < 3.9 : one PathCollection per level (cs.collections, removed in 3.10)
    #  >= 3.9 : cs.get_paths() returns one compound path per level
    try:
        paths = [p for c in cs.collections for p in c.get_paths()]
    except AttributeError:
        paths = list(cs.get_paths())

    plt.close(fig)

    polygons = []

    # -----------------------extract closed rings as polygons
    # Filled-contour paths are *compound*: one path holds the outer boundary
    # plus all its holes as separate closed sub-rings. to_polygons() splits a
    # path into those sub-rings (identically in every matplotlib version);
    # treating path.vertices as a single ring would concatenate the rings into
    # a self-intersecting (invalid) polygon and silently drop the region.
    for path in paths:
        for pts in path.to_polygons():
            pts = np.asarray(pts)
            if pts.shape[0] < 4:
                continue
            poly = Polygon(pts)
            if poly.is_valid and not poly.is_empty and poly.area > 0:
                polygons.append(poly)

    # -----------------------no valid polygons
    if not polygons:
        return None

    # -----------------------sort polygons by area (largest first)
    polygons.sort(key=lambda p: p.area, reverse=True)

    # -----------------------build hierarchy (holes)
    final_polys = []
    while polygons:
        outer = polygons.pop(0)
        holes = [p.exterior.coords for p in polygons if p.within(outer)]
        polygons = [p for p in polygons if not p.within(outer)]
        final_polys.append(Polygon(outer.exterior.coords, holes))

    #final_polys = max(final_polys, key=lambda p: p.area)

    return final_polys
