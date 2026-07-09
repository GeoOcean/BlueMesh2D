import numpy as np


def queryset(tr, tm, fn, *args):
    """Perform spatial queries on AABB-indexed collections.

    Parameters
    ----------
    tr : dict
        AABB tree from :func:`maketree`, with keys ``"xx"``, ``"ii"``, and
        ``"ll"``.
    tm : dict
        Query-to-tree mapping (typically from :func:`mapvert`), with keys
        ``"ii"`` (tree node indices) and ``"ll"`` (query indices per node).
    fn : callable
        Intersection kernel called as ``pk, ck = fn(pj, cj, *args)``, where
        ``pj`` and ``cj`` are query and object indices within a node and
        ``pk``/``ck`` are matching pairs.
    *args
        Extra arguments forwarded to ``fn``.

    Returns
    -------
    qi : ndarray
        Indices of query items with at least one intersection.
    qp : ndarray of shape (N, 2)
        Pointer ranges into ``qj`` for each entry in ``qi``; intersections
        for ``qi[i]`` lie in ``qj[qp[i, 0]:qp[i, 1]]``.
    qj : ndarray
        Flattened list of intersecting object indices.

    References
    ----------
    Translation of the MESH2D function ``QUERYSET``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if tr is None or len(tr) == 0:
        return np.array([]), np.array([]), np.array([])
    if not isinstance(tr, dict) or not isinstance(tm, dict):
        raise TypeError("queryset: incorrect input class.")
    if not all(k in tm for k in ("ii", "ll")):
        raise ValueError("queryset: invalid aabb-maps obj.")
    if not all(k in tr for k in ("xx", "ii", "ll")):
        raise ValueError("queryset: invalid aabb-tree obj.")
    ic = []
    jc = []

    for ip in range(len(tm["ii"])):
        ni = tm["ii"][ip] #  node (in tree)

        qi, qj = fn(
            tm["ll"][ip],  # query in tile
            tr["ll"][ni],  # items in tile
            *args,
        )
        ic.append(np.atleast_1d(qi))
        jc.append(np.atleast_1d(qj))

    if len(ic) == 0 or len(jc) == 0:
        return np.array([]), np.array([]), np.array([])
    qi = np.concatenate(ic)
    qj = np.concatenate(jc)

    if qj.size == 0:
        return np.array([]), np.array([]), np.array([])

    sort_idx = np.argsort(qi)
    qi = qi[sort_idx]
    qj = qj[sort_idx]

    diff_idx = np.nonzero(np.diff(qi) != 0)[0]
    ni = len(qi)

    qi_unique = np.concatenate((qi[diff_idx], [qi[-1]]))
    nj = len(qj)
    ni = len(qi_unique)

    qp = np.zeros((ni, 2), dtype=int)
    qp[:, 0] = np.concatenate(([0], diff_idx + 1))
    qp[:, 1] = np.concatenate((diff_idx, [nj - 1]))

    return qi_unique, qp, qj
