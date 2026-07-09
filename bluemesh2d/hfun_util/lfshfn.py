import numpy as np

from ..refine import refine
from .limhfn import limhfn


def lfshfn(node=None, PSLG=None, part=None, opts=None):
    """Estimate a local feature-size field for a 2D polygonal domain.

    Parameters
    ----------
    node : ndarray of shape (N, 2), optional
        Polygon vertex coordinates.
    PSLG : ndarray of shape (E, 2), optional
        Edge connectivity as indices into ``node``. When omitted, vertices
        in ``node`` are connected in order.
    part : list of ndarray, optional
        For multiply-connected domains, edge-index lists into ``PSLG`` for
        each subregion.
    opts : dict, optional
        Refinement options forwarded to :func:`refine`; defaults are filled
        by :func:`makeopt`.

    Returns
    -------
    vert : ndarray of shape (V, 2)
        Vertex coordinates of the generated triangulation.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.
    hlfs : ndarray of shape (V,)
        Local feature-size estimate at each vertex.

    References
    ----------
    Translation of the MESH2D function ``LFSHFN2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if node is None:
        node = np.empty((0, 2))
    if PSLG is None:
        PSLG = np.empty((0, 2), dtype=int)
    if part is None:
        part = []
    if opts is None:
        opts = {}

    opts = makeopt(opts)

    vert, conn, tria, tnum = refine(node, PSLG, part, opts)

    hlfs = np.full(vert.shape[0], np.inf)

    evec = vert[conn[:, 1], :] - vert[conn[:, 0], :]
    elen = np.sqrt(np.sum(evec**2, axis=1))
    hlen = elen.copy()

    for epos in range(conn.shape[0]):
        ivrt = conn[epos, 0]
        jvrt = conn[epos, 1]

        hlfs[ivrt] = min(hlfs[ivrt], hlen[epos])
        hlfs[jvrt] = min(hlfs[jvrt], hlen[epos])

    DHDX = opts["dhdx"]

    hlfs = limhfn(vert, tria, hlfs, DHDX)

    return vert, tria, hlfs


def makeopt(opts):
    """Fill default options for :func:`lfshfn`.

    Parameters
    ----------
    opts : dict
        User options; missing keys receive defaults.

    Returns
    -------
    opts : dict
        Options with defaults for ``"kind"``, ``"rho2"``, and ``"dhdx"``.
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
