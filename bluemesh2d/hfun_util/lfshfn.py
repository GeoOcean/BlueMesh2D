import numpy as np

from ..refine import refine
from .limhfn import limhfn


def lfshfn(node=None, PSLG=None, part=None, opts=None):
    """
    Compute a discrete local-feature-size (LFS) estimate for a 2D polygonal domain.

    This function estimates the local feature size (LFS) field for a polygonal
    geometry embedded in 2D space, returning a triangulated mesh representation
    of the feature-size distribution. The LFS is a measure of the local spacing
    between nearby features (edges, corners, boundaries) and is often used to
    define mesh-size constraints in adaptive meshing.

    Parameters
    ----------
    NODE : ndarray of shape (N, 2)
        XY-coordinates of the polygon vertices.
    EDGE : ndarray of shape (E, 2), optional
        Array of polygon edge indices. Each row defines one edge as
        `[start_vertex, end_vertex]`.
        If omitted, vertices in `NODE` are assumed to be connected in order.
    PART : list of lists or list of ndarrays, optional
        For multi-connected geometries, specifies multiple polygonal regions.
        Each `PART[k]` contains edge indices into `EDGE` defining one subregion.

    Returns
    -------
    VERT : ndarray of shape (V, 2)
        XY-coordinates of the vertices in the generated triangulation.
    TRIA : ndarray of shape (T, 3)
        Array of triangle vertex indices.
    HFUN : ndarray of shape (V,)
        Estimated local feature-size (mesh-size) values at each vertex.

    Notes
    -----
    - The local feature size quantifies the minimum distance to nearby boundaries
      or sharp geometric features.
    - The resulting field can be used to guide mesh refinement or smoothing
      algorithms (e.g., `refine` or `smooth`).
    - For multi-part geometries, the computation is performed separately on each part.

    References
    ----------
    Translation of the MESH2D function `LFSHFN2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    See also
    --------
    trihfn : Compute a mesh-size function based on triangle areas.
    limhfn : Limit or smooth a mesh-size field.
    idxtri : Index-based triangle utilities.
    """

    # ---------------------------------------------- extract args
    if node is None:
        node = np.empty((0, 2))
    if PSLG is None:
        PSLG = np.empty((0, 2), dtype=int)
    if part is None:
        part = []
    if opts is None:
        opts = {}

    # ------------------------------ build coarse background grid
    opts = makeopt(opts)

    vert, conn, tria, tnum = refine(node, PSLG, part, opts)

    # ------------------------------ estimate local-feature-size
    hlfs = np.full(vert.shape[0], np.inf)

    # ------------------------------ calc. LFS based on edge-len.
    evec = vert[conn[:, 1], :] - vert[conn[:, 0], :]
    elen = np.sqrt(np.sum(evec**2, axis=1))
    hlen = elen.copy()

    for epos in range(conn.shape[0]):
        ivrt = conn[epos, 0]
        jvrt = conn[epos, 1]

        hlfs[ivrt] = min(hlfs[ivrt], hlen[epos])
        hlfs[jvrt] = min(hlfs[jvrt], hlen[epos])

    # ------------------------------ push gradient limits on HFUN
    DHDX = opts["dhdx"]

    hlfs = limhfn(vert, tria, hlfs, DHDX)

    return vert, tria, hlfs


def makeopt(opts):
    """
    Setup the options dictionary for lfshfn.
    """
    # clone to avoid side-effects
    opts = dict(opts)

    if "kind" not in opts:
        opts["kind"] = "delaunay"
    else:
        if opts["kind"].lower() not in ("delfront", "delaunay"):
            raise ValueError("lfshfn:invalidOption: Invalid refinement KIND.")

    if "rho2" not in opts:
        opts["rho2"] = np.sqrt(2.0)
    else:
        if not np.isscalar(opts["rho2"]):
            raise ValueError("lfshfn:incorrectDimensions")
        if opts["rho2"] < 1.0:
            raise ValueError("lfshfn:invalidOptionValues: rho2 must be >= 1.")

    if "dhdx" not in opts:
        opts["dhdx"] = 0.25
    else:
        if not np.isscalar(opts["dhdx"]):
            raise ValueError("lfshfn:incorrectDimensions")
        if opts["dhdx"] <= 0.0:
            raise ValueError("lfshfn:invalidOptionValues: dhdx must be > 0.")

    return opts
