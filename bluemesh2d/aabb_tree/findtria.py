import numpy as np

from .maketree import maketree
from .mapvert import mapvert
from .queryset import queryset


def findtria(pp, tt, pj, tree=None, options=None):
    """
    Perform spatial queries for collections of d-dimensional simplexes.

    This function finds the set of d-dimensional simplexes that intersect
    a given set of query points. Simplexes are specified by their vertex
    coordinates and indexing arrays, and do not need to form a conforming
    triangulation — non-Delaunay, non-convex, and even overlapping
    configurations are supported.

    For each query point, the function returns the subset of simplexes that
    contain or intersect it. Multiple matches may occur in overlapping meshes.

    Parameters
    ----------
    PP : ndarray of shape (N, ND)
        Array of vertex coordinates, where `ND` is the number of dimensions.
    TT : ndarray of shape (NS, M)
        Indexing array defining the vertices of each simplex.
        For example, in 2D, `M=3` for triangles, in 3D `M=4` for tetrahedra.
    PJ : ndarray of shape (NP, ND)
        Query points for which intersecting simplexes are sought.
    TR : object, optional
        Precomputed AABB tree structure used to accelerate the search.
        If provided, it will be reused for faster subsequent queries.
    OP : dict, optional
        Additional parameters to control the creation or behavior of the
        underlying AABB tree.

    Returns
    -------
    TP : ndarray of shape (NP, 2)
        Index pairs defining, for each query point, the start and end
        positions in `TI` listing the intersecting simplexes.
        Points with no matches have `TP[i, 0] == 0`.
    TI : ndarray of shape (K,)
        Flattened list of simplex indices intersecting the query points.
        For query `i`, intersecting simplexes are `TI[TP[i,0]:TP[i,1]]`.
    TR : object, optional
        AABB tree structure used internally to compute the query.
        It can be reused to improve performance in later calls.

    Notes
    -----
    - The collection of simplexes `[PP, TT]` does not need to be a
      conforming triangulation.
    - If the same simplex collection is queried multiple times,
      reusing the returned `TR` can significantly improve performance,
      but the simplexes **must remain unchanged** between calls.
    - To obtain a single consistent index array (similar to MATLAB’s
      `pointLocation`), one can postprocess the output as:

        ```python
        tp, tj = findtria(pp, tt, pj)
        ti = np.full(tp.shape[0], np.nan)
        mask = tp[:, 0] > 0
        ti[mask] = tj[tp[mask, 0]]
        ```

    References
    ----------
    Translation of the MESH2D function `FINDTRIA`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    Additional reference:
    D. Engwirda, *"Locally-optimal Delaunay-refinement & optimisation-based mesh generation"*,
    Ph.D. Thesis, School of Mathematics and Statistics, University of Sydney, 2014.
    http://hdl.handle.net/2123/13148
    """

    # ---------------------------------------------- basic checks
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

    # ----------------------------- compute aabb's for triangles
    if tree is None:
        bi = pp[tt[:, 0], :].copy()
        bj = pp[tt[:, 0], :].copy()
        for ii in range(1, tt.shape[1]):
            bi = np.minimum(bi, pp[tt[:, ii], :])
            bj = np.maximum(bj, pp[tt[:, ii], :])
        bb = np.hstack([bi, bj])
        tree = maketree(bb, options)  # compute aabb-tree

    # ----------------------------- compute tree-to-vert mapping
    tm, _ = mapvert(tree, pj)

    # ------------------------------ compute vert-to-tria queries
    x0 = np.min(pp, axis=0)
    x1 = np.max(pp, axis=0)
    rt = np.prod(x1 - x0) * np.finfo(float).eps ** 0.8

    ti, ip, tj = queryset(tree, tm, triakern, pj, pp, tt, rt)

    # ------------------------------ re-index onto full obj. list
    tp = np.zeros((pj.shape[0], 2), dtype=int)
    tp[:, 1] = -1
    if ti.size == 0:
        return tp, tj, tree
    tp[ti, :] = ip

    return tp, tj, tree


def triakern(pk, tk, pi, pp, tt, rt):
    """
    TRIAKERN - Compute point/simplex intersections within a tile.

    Parameters
    ----------
    pk : (M,) array
        Indices of query points.
    tk : (N,) array
        Indices of simplexes.
    pi : (P,ND) array
        Query points.
    pp : (V,ND) array
        Vertex coordinates.
    tt : (T,nv) array
        Simplex connectivity (0-based indices).
    rt : float
        Relative tolerance for point-in-simplex tests.

    Returns
    -------
    ip : array
        Query point indices where intersection occurs.
    it : array
        Simplex indices where intersection occurs.

    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Translation of the MESH2D function `TRIAKERN`.
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
    """
    INTRIA2 - Returns True for points enclosed by 2-simplexes (triangles).

    Parameters
    ----------
    pp : (V,2) array
        Vertex coordinates.
    tt : (T,3) array
        Triangle connectivity (0-based indices).
    pi : (P,2) array
        Query points.
    rt : float
        Relative tolerance for point-in-triangle tests.

    Returns
    -------
    inside : (T,) boolean array
        True for triangles containing the query point.

    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Translation of the MESH2D function `INTRIA2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    t1, t2, t3 = tt[:, 0], tt[:, 1], tt[:, 2]
    vi = pp[t1, :] - pi
    vj = pp[t2, :] - pi
    vk = pp[t3, :] - pi
    # ------------------------------- compute sub-volume about PI
    aa = np.zeros((tt.shape[0], 3))
    aa[:, 0] = vi[:, 0] * vj[:, 1] - vj[:, 0] * vi[:, 1]
    aa[:, 1] = vj[:, 0] * vk[:, 1] - vk[:, 0] * vj[:, 1]
    aa[:, 2] = vk[:, 0] * vi[:, 1] - vi[:, 0] * vk[:, 1]
    # ------------------------------- PI is internal if same sign
    rt2 = rt**2
    inside = (
        (aa[:, 0] * aa[:, 1] >= -rt2)
        & (aa[:, 1] * aa[:, 2] >= -rt2)
        & (aa[:, 2] * aa[:, 0] >= -rt2)
    )

    return inside


def intria3(pp, tt, pi, rt):
    """
    INTRIA3 - Returns True for points enclosed by 3-simplexes (tetrahedra).

    Parameters
    ----------
    pp : (V,3) array
        Vertex coordinates.
    tt : (T,4) array
        Tetrahedron connectivity (0-based indices).
    pi : (P,3) array
        Query points.
    rt : float
        Relative tolerance for point-in-tetrahedron tests.

    Returns
    -------
    inside : (T,) boolean array
        True for tetrahedra containing the query point.

    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Translation of the MESH2D function `TRIAKERN`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    t1, t2, t3, t4 = tt[:, 0], tt[:, 1], tt[:, 2], tt[:, 3]
    v1 = pi - pp[t1, :]
    v2 = pi - pp[t2, :]
    v3 = pi - pp[t3, :]
    v4 = pi - pp[t4, :]
    # ------------------------------- compute sub-volume about PI
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
    # ------------------------------- PI is internal if same sign
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
    """
    INTRIAN - General n-simplex point-location using barycentric coordinates.

    Parameters
    ----------
    pp : (V,ND) array
        Vertex coordinates.
    tt : (T,nv) array
        Simplex connectivity (0-based indices).
    pi : (P,ND) array
        Query points.

    Returns
    -------
    ii : array
        Indices of query points inside simplexes.
    jj : array
        Indices of simplexes containing the query points.

    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Translation of the MESH2D function `INTRIAN`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    np_, pd = pi.shape
    nt, td = tt.shape
    # ---------------- coefficient matrices for barycentric coord.
    mm = np.zeros((pd, pd, nt))
    for id_ in range(pd):
        for jd in range(pd):
            mm[id_, jd, :] = pp[tt[:, jd], id_] - pp[tt[:, td - 1], id_]
    # ---------------- solve linear systems for barycentric coord.
    xx = np.zeros((pd, np_, nt))
    vp = np.zeros((pd, np_))

    for ti in range(nt):
        # ------------------------------------- form rhs coeff.
        for id_ in range(pd):
            vp[id_, :] = pi[:, id_] - pp[tt[ti, td - 1], id_]
        # -------------- solve linear systems (LU equivalent)
        xx[:, :, ti] = np.linalg.solve(mm[:, :, ti], vp)
    # -------------------- PI is internal if coord. have same sign
    in_mask = np.all(xx >= -(np.finfo(float).eps ** 0.8), axis=0) & (
        np.sum(xx, axis=0) <= 1.0 + np.finfo(float).eps ** 0.8
    )
    # -------------------- find lists of matching points/simplexes
    ii, jj = np.where(in_mask.T)
    return ii, jj
