import numpy as np

from .inpoly_mat import inpoly_mat


def inpoly(vert, node, edge=None, ftol=None):
    """
    INPOLY : perform "point-in-polygon" queries.

    [stat] = inpoly(vert, node, edge) determines whether each point in
    `vert` lies inside or outside a polygon defined by {node, edge} in the
    2D plane. Supports general non-convex and multiply-connected polygons.

    Parameters
    ----------
    vert : ndarray (N, 2)
        Coordinates of query points to test.
    node : ndarray (M, 2)
        Coordinates of polygon vertices.
    edge : ndarray (P, 2), optional
        Edge connectivity array, where each row defines a polygon edge.
        If omitted, vertices in `node` are assumed to be connected in order.
    ftol : float, optional
        Floating-point tolerance for boundary comparisons. Default: `eps**0.85`.

    Returns
    -------
    stat : ndarray (N,)
        Boolean array, where `True` indicates points lying inside the polygon.
    bnds : ndarray (N,)
        Boolean array, where `True` indicates points lying on a polygon edge.

    Notes
    -----
    - Implements a robust **crossing-number algorithm** that counts the number
      of times a ray extending from each test point intersects polygon edges.
      Points with an odd number of crossings are classified as "inside".
    - Uses sorting and binary-search techniques to accelerate the process:
        * Sorts query points by y-value.
        * Uses bounding-box filters to avoid unnecessary edge intersection checks.
    - The resulting complexity scales as:
        O(M * H + M * log(N) + N * log(N)),
      where:
        - N is the number of query points,
        - M is the number of polygon edges,
        - H is the average point–edge overlap (typically H ≪ N).

    References
    ----------
    Translation of the MESH2D function `INPOLY2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ---------------------------------------------- extract args
    node = np.asarray(node, dtype=float)
    vert = np.asarray(vert, dtype=float)
    # ---------------------------------------------- default args
    if edge is None:
        nnod = node.shape[0]
        edge = np.vstack(
            [np.column_stack([np.arange(nnod - 1), np.arange(1, nnod)]), [nnod - 1, 0]]
        )
    else:
        edge = np.asarray(edge, dtype=int)

    if ftol is None:
        ftol = np.finfo(float).eps ** 0.85

    nnod = node.shape[0]
    nvrt = vert.shape[0]

    # ---------------------------------------------- basic checks
    if edge.min() < 0 or edge.max() > nnod:
        raise ValueError("inpoly: invalid EDGE input array.")

    STAT = np.zeros(nvrt, dtype=bool)
    BNDS = np.zeros(nvrt, dtype=bool)

    # ----------------------------------- prune points using bbox
    nmin = node.min(axis=0)
    nmax = node.max(axis=0)
    ddxy = nmax - nmin
    lbar = ddxy.sum() / 2.0
    veps = ftol * lbar

    mask = (
        (vert[:, 0] >= nmin[0] - veps)
        & (vert[:, 0] <= nmax[0] + veps)
        & (vert[:, 1] >= nmin[1] - veps)
        & (vert[:, 1] <= nmax[1] + veps)
    )

    if not np.any(mask):
        return STAT, BNDS

    vmask = np.where(mask)[0]
    vsub = vert[mask, :].copy()
    nsub = node.copy()

    # -------------- flip to ensure the y-axis is the "long" axis
    vmin = vsub.min(axis=0)
    vmax = vsub.max(axis=0)
    ddxy = vmax - vmin
    if ddxy[0] > ddxy[1]:
        vsub = vsub[:, [1, 0]]
        nsub = nsub[:, [1, 0]]

    # ----------------------------------- sort points via y-value
    swap = nsub[edge[:, 1], 1] < nsub[edge[:, 0], 1]
    edge[swap] = edge[swap][:, [1, 0]]

    ivec = np.lexsort((vsub[:, 0], vsub[:, 1]))
    vsub = vsub[ivec, :]

    stat, bnds = inpoly_mat(vsub, nsub, edge, ftol, lbar)

    inv = np.argsort(ivec)
    stat = stat[inv]
    bnds = bnds[inv]

    STAT[vmask] = stat
    BNDS[vmask] = bnds

    return STAT, BNDS
