import numpy as np

from .maketree import maketree
from .mapvert import mapvert
from .queryset import queryset


def findball(bb, pp, tr=None, op=None):
    """Find d-dimensional balls intersecting query points.

    Parameters
    ----------
    bb : ndarray of shape (M, ND + 1)
        Ball definitions: center coordinates in the first ``ND`` columns and
        squared radius in the last column.
    pp : ndarray of shape (P, ND)
        Query points.
    tr : dict, optional
        Precomputed AABB tree from a prior call; reuse only if ``bb`` is
        unchanged.
    op : dict, optional
        Options passed to :func:`maketree` when building the tree.

    Returns
    -------
    bp : ndarray of shape (P, 2)
        Pointer ranges into ``bj`` for each query point; ``bp[i, 0] == 0``
        when point ``i`` lies outside all balls.
    bj : ndarray
        Flattened list of intersecting ball indices.
    tr : dict
        AABB tree used for the query; may be reused on later calls.

    References
    ----------
    Translation of the MESH2D function ``FINDBALL``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    bp, bj = np.array([]), np.array([])

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

    if tr is None:
        nd = pp.shape[1]
        rs = np.sqrt(bb[:, nd])[:, None]  # radii
        rs = np.tile(rs, (1, nd))
        ab = np.hstack([bb[:, :nd] - rs, bb[:, :nd] + rs])  # aabb
        tr = maketree(ab, op)

    tm, _ = mapvert(tr, pp)

    bi, ip, bj = queryset(tr, tm, ballkern, pp, bb)

    bp = np.zeros((pp.shape[0], 2), dtype=int)
    bp[:, 1] = -1
    if bi.size > 0:
        bp[bi, :] = ip

    return bp, bj, tr


def ballkern(pk, bk, pp, bb):
    """Test ball-point intersections within an AABB tree tile.

    Parameters
    ----------
    pk : ndarray
        Indices of query points in the tile.
    bk : ndarray
        Indices of candidate balls in the tile.
    pp : ndarray of shape (P, ND)
        Query point coordinates.
    bb : ndarray of shape (B, ND + 1)
        Ball centers and squared radii.

    Returns
    -------
    ip : ndarray
        Query point indices with an intersection.
    ib : ndarray
        Matching ball indices.

    References
    ----------
    Translation of the MESH2D function ``BALLKERN``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    mp = len(pk)
    mb = len(bk)
    nd = pp.shape[1]

    bk_tiled = np.tile(bk, mp)
    pk_tiled = np.repeat(pk, mb)

    diff = pp[pk_tiled, :] - bb[bk_tiled, :nd]
    dd = np.sum(diff**2, axis=1)

    inside = dd <= bb[bk_tiled, nd]

    ip = pk_tiled[inside]
    ib = bk_tiled[inside]

    return ip, ib
