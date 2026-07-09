import numpy as np

from .maketree import maketree
from .mapvert import mapvert
from .queryset import queryset


def findtria(pp, tt, pj, tree=None, options=None):
    """Find d-dimensional simplexes intersecting query points.

    Parameters
    ----------
    pp : ndarray of shape (N, ND)
        Vertex coordinates.
    tt : ndarray of shape (NS, M)
        Simplex connectivity; ``M=3`` for triangles, ``M=4`` for tetrahedra.
    pj : ndarray of shape (NP, ND)
        Query points.
    tree : dict, optional
        Precomputed AABB tree from a prior call; reuse only if ``pp`` and
        ``tt`` are unchanged.
    options : dict, optional
        Options passed to :func:`maketree` when building the tree.

    Returns
    -------
    tp : ndarray of shape (NP, 2)
        Pointer ranges into ``tj`` for each query point; ``tp[i, 0] == 0``
        when point ``i`` has no match.
    tj : ndarray
        Flattened list of intersecting simplex indices.
    tree : dict
        AABB tree used for the query; may be reused on later calls.

    Notes
    -----
    The simplex collection need not form a conforming triangulation. To pick
    one simplex per query point, take ``tj[tp[mask, 0]]`` where
    ``mask = tp[:, 0] > 0``.

    References
    ----------
    Translation of the MESH2D function ``FINDTRIA``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not (
        isinstance(pp, np.ndarray)
        and isinstance(tt, np.ndarray)
        and isinstance(pj, np.ndarray)
    ):
        raise TypeError("pp, tt, and pj must be numpy arrays.")

    if pj.size == 0:
        return np.zeros((0, 2), dtype=int), np.array([], dtype=int), tree

    if pp.ndim != 2 or tt.ndim != 2 or pj.ndim != 2:
        raise ValueError("Inputs must be 2D arrays.")

    if pp.shape[1] < 2 or pp.shape[1] > tt.shape[1]:
        raise ValueError("Incorrect input dimensions.")

    if tt.shape[1] < 3:
        raise ValueError("Triangles must have at least 3 vertices.")

    if pj.shape[1] != pp.shape[1]:
        raise ValueError("pj and pp must have the same dimensionality.")

    if tree is None:
        bi = pp[tt[:, 0], :].copy()
        bj = pp[tt[:, 0], :].copy()
        for ii in range(1, tt.shape[1]):
            bi = np.minimum(bi, pp[tt[:, ii], :])
            bj = np.maximum(bj, pp[tt[:, ii], :])
        bb = np.hstack([bi, bj])
        tree = maketree(bb, options)  # compute aabb-tree

    tm, _ = mapvert(tree, pj)

    x0 = np.min(pp, axis=0)
    x1 = np.max(pp, axis=0)
    rt = np.prod(x1 - x0) * np.finfo(float).eps ** 0.8

    ti, ip, tj = queryset(tree, tm, triakern, pj, pp, tt, rt)

    tp = np.zeros((pj.shape[0], 2), dtype=int)
    tp[:, 1] = -1
    if ti.size == 0:
        return tp, tj, tree
    tp[ti, :] = ip

    return tp, tj, tree


def triakern(pk, tk, pi, pp, tt, rt):
    """Test point/simplex intersections within an AABB tree tile.

    Parameters
    ----------
    pk : ndarray
        Indices of query points in the tile.
    tk : ndarray
        Indices of candidate simplexes in the tile.
    pi : ndarray of shape (P, ND)
        Query point coordinates.
    pp : ndarray of shape (V, ND)
        Vertex coordinates.
    tt : ndarray of shape (T, nv)
        Simplex connectivity (0-based vertex indices).
    rt : float
        Relative tolerance for point-in-simplex tests.

    Returns
    -------
    ip : ndarray
        Query point indices with an intersection.
    it : ndarray
        Matching simplex indices.

    References
    ----------
    Translation of the MESH2D function ``TRIAKERN``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    mp = len(pk)
    mt = len(tk)

    pk = np.repeat(pk, mt)
    tk = np.tile(tk, mp)

    n_vertices = tt.shape[1]

    if n_vertices == 3:
        inside = intria2(pp, tt[tk, :], pi[pk, :], rt)
    elif n_vertices == 4:
        inside = intria3(pp, tt[tk, :], pi[pk, :], rt)
    else:
        ii, jj = intrian(pp, tt[tk, :], pi[pk, :])
        ip = pk[ii]
        it = tk[jj]
        return ip, it

    ip = pk[inside]
    it = tk[inside]
    return ip, it


def intria2(pp, tt, pi, rt):
    """Test whether query points lie inside 2-simplexes (triangles).

    Parameters
    ----------
    pp : ndarray of shape (V, 2)
        Vertex coordinates.
    tt : ndarray of shape (T, 3)
        Triangle connectivity (0-based vertex indices).
    pi : ndarray of shape (P, 2)
        Query points (one per triangle row in ``tt``).
    rt : float
        Relative tolerance for point-in-triangle tests.

    Returns
    -------
    inside : ndarray of bool, shape (T,)
        ``True`` where the corresponding query point lies inside the triangle.

    References
    ----------
    Translation of the MESH2D function ``INTRIA2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    t1, t2, t3 = tt[:, 0], tt[:, 1], tt[:, 2]
    vi = pp[t1, :] - pi
    vj = pp[t2, :] - pi
    vk = pp[t3, :] - pi
    aa = np.zeros((tt.shape[0], 3))
    aa[:, 0] = vi[:, 0] * vj[:, 1] - vj[:, 0] * vi[:, 1]
    aa[:, 1] = vj[:, 0] * vk[:, 1] - vk[:, 0] * vj[:, 1]
    aa[:, 2] = vk[:, 0] * vi[:, 1] - vi[:, 0] * vk[:, 1]
    rt2 = rt**2
    inside = (
        (aa[:, 0] * aa[:, 1] >= -rt2)
        & (aa[:, 1] * aa[:, 2] >= -rt2)
        & (aa[:, 2] * aa[:, 0] >= -rt2)
    )

    return inside


def intria3(pp, tt, pi, rt):
    """Test whether query points lie inside 3-simplexes (tetrahedra).

    Parameters
    ----------
    pp : ndarray of shape (V, 3)
        Vertex coordinates.
    tt : ndarray of shape (T, 4)
        Tetrahedron connectivity (0-based vertex indices).
    pi : ndarray of shape (P, 3)
        Query points (one per tetrahedron row in ``tt``).
    rt : float
        Relative tolerance for point-in-tetrahedron tests.

    Returns
    -------
    inside : ndarray of bool, shape (T,)
        ``True`` where the corresponding query point lies inside the tetrahedron.

    References
    ----------
    Translation of the MESH2D function ``INTRIA3``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    t1, t2, t3, t4 = tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3]
    v1 = pi - pp[t1, :]
    v2 = pi - pp[t2, :]
    v3 = pi - pp[t3, :]
    v4 = pi - pp[t4, :]
    aa = np.zeros((tt.shape[0], 4))
    aa[:, 0] = (
        v1[:, 0] * (v2[:, 1] * v3[:, 2] - v2[:, 2] * v3[:, 1])
        - v1[:, 1] * (v2[:, 0] * v3[:, 2] - v2[:, 2] * v3[:, 0])
        + v1[:, 2] * (v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0])
    )
    aa[:, 1] = (
        v1[:, 0] * (v4[:, 1] * v2[:, 2] - v4[:, 2] * v2[:, 1])
        - v1[:, 1] * (v4[:, 0] * v2[:, 2] - v4[:, 2] * v2[:, 0])
        + v1[:, 2] * (v4[:, 0] * v2[:, 1] - v4[:, 1] * v2[:, 0])
    )
    aa[:, 2] = (
        v2[:, 0] * (v4[:, 1] * v3[:, 2] - v4[:, 2] * v3[:, 1])
        - v2[:, 1] * (v4[:, 0] * v3[:, 2] - v4[:, 2] * v3[:, 0])
        + v2[:, 2] * (v4[:, 0] * v3[:, 1] - v4[:, 1] * v3[:, 0])
    )
    aa[:, 3] = (
        v3[:, 0] * (v4[:, 1] * v1[:, 2] - v4[:, 2] * v1[:, 1])
        - v3[:, 1] * (v4[:, 0] * v1[:, 2] - v4[:, 2] * v1[:, 0])
        + v3[:, 2] * (v4[:, 0] * v1[:, 1] - v4[:, 1] * v1[:, 0])
    )
    rt2 = rt**2
    inside = (
        (aa[:, 0] * aa[:, 1] >= -rt2)
        & (aa[:, 0] * aa[:, 2] >= -rt2)
        & (aa[:, 0] * aa[:, 3] >= -rt2)
        & (aa[:, 1] * aa[:, 2] >= -rt2)
        & (aa[:, 1] * aa[:, 3] >= -rt2)
        & (aa[:, 2] * aa[:, 3] >= -rt2)
    )

    return inside


def intrian(pp, tt, pi):
    """Locate query points in general n-simplexes via barycentric coordinates.

    Parameters
    ----------
    pp : ndarray of shape (V, ND)
        Vertex coordinates.
    tt : ndarray of shape (T, nv)
        Simplex connectivity (0-based vertex indices).
    pi : ndarray of shape (P, ND)
        Query points.

    Returns
    -------
    ii : ndarray
        Indices of query points inside a simplex.
    jj : ndarray
        Indices of simplexes containing those points.

    References
    ----------
    Translation of the MESH2D function ``INTRIAN``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    np_, pd = pi.shape
    nt, td = tt.shape
    # Coefficient matrices for barycentric coord.
    mm = np.zeros((pd, pd, nt))
    for id_ in range(pd):
        for jd in range(pd):
            mm[id_, jd, :] = pp[tt[:, jd], id_] - pp[tt[:, td - 1], id_]
    # Solve linear systems for barycentric coord.
    xx = np.zeros((pd, np_, nt))
    vp = np.zeros((pd, np_))

    for ti in range(nt):
        for id_ in range(pd):
            vp[id_, :] = pi[:, id_] - pp[tt[ti, td - 1], id_]
        # Solve linear systems (LU equivalent)
        xx[:, :, ti] = np.linalg.solve(mm[:, :, ti], vp)
    # PI is internal if coord. have same sign
    in_mask = np.all(xx >= -(np.finfo(float).eps ** 0.8), axis=0) & (
        np.sum(xx, axis=0) <= 1.0 + np.finfo(float).eps ** 0.8
    )
    # Find lists of matching points/simplexes
    ii, jj = np.where(in_mask.T)
    return ii, jj
