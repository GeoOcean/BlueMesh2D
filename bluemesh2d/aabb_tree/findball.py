import numpy as np

from .maketree import maketree
from .mapvert import mapvert
from .queryset import queryset


def findball(bb, pp, tr=None, op=None):
    """
    Perform spatial queries for collections of d-dimensional balls.

    This routine identifies which d-dimensional balls intersect a set of
    query points. Balls are defined by their centers and squared radii.

    Specifically, for each query point in `PI`, the function returns the
    subset of balls from `BB` that enclose or intersect the point. This
    operation is typically used in geometric and meshing algorithms to
    detect neighborhood relations or spatial overlaps.

    Parameters
    ----------
    BB : ndarray of shape (M, ND + 1)
        Array of ball definitions, where the first `ND` columns correspond
        to the center coordinates, and the last column gives the **squared radius**.
        For example, in 2D: `BB[:, :2] = centers`, `BB[:, 2] = radii²`.
    PI : ndarray of shape (P, ND)
        Query points for which intersecting balls are to be found.
    TR : object, optional
        Precomputed AABB tree structure used to accelerate the search.
        If provided, it is reused for faster queries, assuming that the
        set of balls in `BB` has not changed.
    OP : dict, optional
        Additional parameters controlling the creation or behavior of the
        underlying AABB tree.

    Returns
    -------
    BP : ndarray of shape (P, 2)
        Index array defining, for each query point, the start and end
        positions in `BI` that list the intersecting balls.
        Points not enclosed by any ball have `BP[i, 0] == 0`.
    BI : ndarray of shape (K,)
        Flattened list of ball indices that intersect the query points.
        For query `i`, intersecting balls are `BI[BP[i,0]:BP[i,1]]`.
    TR : object, optional
        AABB tree structure used internally to compute the query.
        Can be reused in future calls for efficiency.

    Notes
    -----
    If the same collection of balls is queried multiple times, reusing
    the returned `TR` structure can significantly improve performance.
    However, the ball set `BB` **must not be modified** between queries.

    References
    ----------
    Translation of the MESH2D function `FINDBALL`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    Additional reference:
    D. Engwirda, *"Locally-optimal Delaunay-refinement & optimisation-based mesh generation"*,
    Ph.D. Thesis, School of Mathematics and Statistics, University of Sydney, 2014.
    http://hdl.handle.net/2123/13148
    """

    bp, bj = np.array([]), np.array([])

    # ------------------------------ basic checks
    if bb is None or pp is None:
        raise ValueError("findball:incorrectNumInputs (need at least bb, pp)")

    bb = np.asarray(bb, dtype=float)
    pp = np.asarray(pp, dtype=float)

    if bb.ndim != 2 or bb.shape[1] < 3:
        raise ValueError("findball:incorrectDimensions (bb must be (B,ND+1))")
    if pp.ndim != 2 or bb.shape[1] != pp.shape[1] + 1:
        raise ValueError("findball:incorrectDimensions (pp must be (P,ND))")

    if tr is not None and not isinstance(tr, dict):
        raise TypeError("findball:incorrectInputClass (tr must be struct/dict)")
    if op is not None and not isinstance(op, dict):
        raise TypeError("findball:incorrectInputClass (op must be struct/dict)")

    if bb.size == 0:
        return bp, bj, tr

    # ------------------------------ build tree if not given
    if tr is None:
        nd = pp.shape[1]
        rs = np.sqrt(bb[:, nd])[:, None]  # radii
        rs = np.tile(rs, (1, nd))
        ab = np.hstack([bb[:, :nd] - rs, bb[:, :nd] + rs])  # aabb
        tr = maketree(ab, op)

    # ------------------------------ map query vertices
    tm, _ = mapvert(tr, pp)

    # ------------------------------ run query
    bi, ip, bj = queryset(tr, tm, ballkern, pp, bb)

    # ------------------------------ reindex onto full list
    bp = np.zeros((pp.shape[0], 2), dtype=int)
    bp[:, 1] = -1
    if bi.size > 0:
        bp[bi, :] = ip

    return bp, bj, tr


def ballkern(pk, bk, pp, bb):
    """
    BALLKERN: d-dimensional ball-vertex intersection kernel.

    Parameters
    ----------
    pk : (M,) array
        Indices of query points.
    bk : (N,) array
        Indices of balls.
    pp : (P,ND) array
        Query points.
    bb : (B,ND+1) array
        Ball centers + squared radii.

    Returns
    -------
    ip : array
        Query point indices where intersection occurs.
    ib : array
        Ball indices intersecting.

    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Translation of the MESH2D function `BALLKERN`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    """
    mp = len(pk)
    mb = len(bk)
    nd = pp.shape[1]

    # -------------------------- push ball/vert onto n*m tile
    bk_tiled = np.tile(bk, mp)
    pk_tiled = np.repeat(pk, mb)

    # -------------------------- compute O(n*m) loc. distance
    diff = pp[pk_tiled, :] - bb[bk_tiled, :nd]
    dd = np.sum(diff**2, axis=1)

    inside = dd <= bb[bk_tiled, nd]

    ip = pk_tiled[inside]
    ib = bk_tiled[inside]

    return ip, ib
