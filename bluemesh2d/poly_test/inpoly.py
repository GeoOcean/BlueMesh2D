import numpy as np

from .inpoly_mat import inpoly_mat


def inpoly(vert, node, edge=None, ftol=None):
    """Test whether points lie inside a 2D polygon.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Query point coordinates.
    node : ndarray of shape (M, 2)
        Polygon vertex coordinates.
    edge : ndarray of shape (P, 2), optional
        Edge connectivity as vertex-index pairs. When omitted, vertices in
        ``node`` are connected in order.
    ftol : float, optional
        Floating-point tolerance for boundary tests; default is
        ``eps**0.85``.

    Returns
    -------
    stat : ndarray of bool, shape (N,)
        ``True`` for points classified as inside the polygon.
    bnds : ndarray of bool, shape (N,)
        ``True`` for points lying on a polygon edge.

    Notes
    -----
    Uses a crossing-number algorithm with sorted query points and binary
    search over edge y-ranges.

    References
    ----------
    Translation of the MESH2D function ``INPOLY2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    node = np.asarray(node, dtype=float)
    vert = np.asarray(vert, dtype=float)
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

    if edge.min() < 0 or edge.max() > nnod:
        raise ValueError("inpoly: invalid EDGE input array.")

    STAT = np.zeros(nvrt, dtype=bool)
    BNDS = np.zeros(nvrt, dtype=bool)

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

    # Flip to ensure the y-axis is the "long" axis
    vmin = vsub.min(axis=0)
    vmax = vsub.max(axis=0)
    ddxy = vmax - vmin
    if ddxy[0] > ddxy[1]:
        vsub = vsub[:, [1, 0]]
        nsub = nsub[:, [1, 0]]

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
