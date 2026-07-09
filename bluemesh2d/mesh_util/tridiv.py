import numpy as np

from .setset import setset
from .tricon import tricon


def tridiv(vert=None, conn=None, tria=None, tnum=None, tdiv=None):
    """Refine a triangular mesh by edge bisection.

    Subdivide marked triangles into four sub-triangles and propagate a
    conforming halo of one- and two-edge bisections to avoid hanging nodes.

    Parameters
    ----------
    vert : ndarray of shape (V, 2), optional
        Vertex coordinates.
    conn : ndarray of shape (C, 2), optional
        Constraining edges as vertex-index pairs.
    tria : ndarray of shape (T, 3), optional
        Triangle connectivity.
    tnum : ndarray of shape (T, 1), optional
        Part index for each triangle.
    tdiv : ndarray of shape (T,), dtype bool, optional
        ``True`` marks triangles to refine. Defaults to refining all triangles.

    Returns
    -------
    vert : ndarray of shape (V', 2)
        Updated vertex coordinates including edge midpoints.
    conn : ndarray of shape (C', 2)
        Updated constraining edges.
    tria : ndarray of shape (T', 3)
        Updated triangle connectivity.
    tnum : ndarray of shape (T', 1)
        Part indices for the refined triangles.

    References
    ----------
    Translation of the MESH2D function ``TRIDIV2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if vert is None:
        vert = np.empty((0, 2))
    if conn is None:
        conn = np.empty((0, 2), dtype=int)
    if tria is None:
        tria = np.empty((0, 3), dtype=int)
    if tnum is None or len(tnum) == 0:
        tnum = np.ones((tria.shape[0], 1), dtype=int)
    if tdiv is None or len(tdiv) == 0:
        tdiv = np.ones((tria.shape[0],), dtype=bool)

    nvrt = vert.shape[0]
    if conn.size and (np.min(conn[:, :2]) < 0 or np.max(conn[:, :2]) >= nvrt):
        raise ValueError("tridiv: invalid EDGE input array.")
    if tria.size and (np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) >= nvrt):
        raise ValueError("tridiv: invalid TRIA input array.")

    edge, tria = tricon(tria, conn)

    ediv = np.zeros(edge.shape[0], dtype=bool)
    ediv[tria[tdiv, 3:6].ravel()] = True

    snum = np.count_nonzero(ediv)
    while True:
        div3 = np.sum(ediv[tria[:, 3:6]], axis=1) >= 2
        ediv[tria[div3, 3:6].ravel()] = True
        snew = np.count_nonzero(ediv)
        if snew == snum:
            break
        snum = snew
    div1 = np.sum(ediv[tria[:, 3:6]], axis=1) == 1

    ivec = np.zeros(edge.shape[0], dtype=int)
    ivec[ediv] = np.arange(snum, dtype=int) + vert.shape[0]
    emid = (vert[edge[ediv, 0], :] + vert[edge[ediv, 1], :]) * 0.5
    vert = np.vstack([vert, emid])

    cvec, eloc = setset(conn, edge[ediv, 0:2])
    epos = ivec[ediv]
    epos = epos[eloc[eloc >= 0]]

    conn = np.vstack(
        [
            conn[~cvec, :],
            np.column_stack([conn[cvec, 0], epos]),
            np.column_stack([conn[cvec, 1], epos]),
        ]
    )

    div3_idx = np.where(div3)[0]
    tr41 = np.column_stack(
        [tria[div3_idx, 0], ivec[tria[div3_idx, 3]], ivec[tria[div3_idx, 5]]]
    )
    tr42 = np.column_stack(
        [ivec[tria[div3_idx, 3]], tria[div3_idx, 1], ivec[tria[div3_idx, 4]]]
    )
    tr43 = np.column_stack(
        [ivec[tria[div3_idx, 5]], ivec[tria[div3_idx, 4]], tria[div3_idx, 2]]
    )
    tr44 = np.column_stack(
        [ivec[tria[div3_idx, 5]], ivec[tria[div3_idx, 3]], ivec[tria[div3_idx, 4]]]
    )

    tn41 = tnum[div3_idx]
    tn42 = tnum[div3_idx]
    tn43 = tnum[div3_idx]
    tn44 = tnum[div3_idx]

    tvec = ediv[tria[:, 3]] & div1
    tr21 = (
        np.column_stack([ivec[tria[tvec, 3]], tria[tvec, 2], tria[tvec, 0]])
        if np.any(tvec)
        else np.empty((0, 3), int)
    )
    tr22 = (
        np.column_stack([ivec[tria[tvec, 3]], tria[tvec, 1], tria[tvec, 2]])
        if np.any(tvec)
        else np.empty((0, 3), int)
    )
    tn21 = tnum[tvec]
    tn22 = tnum[tvec]

    tvec = ediv[tria[:, 4]] & div1
    tr23 = (
        np.column_stack([ivec[tria[tvec, 4]], tria[tvec, 0], tria[tvec, 1]])
        if np.any(tvec)
        else np.empty((0, 3), int)
    )
    tr24 = (
        np.column_stack([ivec[tria[tvec, 4]], tria[tvec, 2], tria[tvec, 0]])
        if np.any(tvec)
        else np.empty((0, 3), int)
    )
    tn23 = tnum[tvec]
    tn24 = tnum[tvec]

    tvec = ediv[tria[:, 5]] & div1
    tr25 = (
        np.column_stack([ivec[tria[tvec, 5]], tria[tvec, 1], tria[tvec, 2]])
        if np.any(tvec)
        else np.empty((0, 3), int)
    )
    tr26 = (
        np.column_stack([ivec[tria[tvec, 5]], tria[tvec, 0], tria[tvec, 1]])
        if np.any(tvec)
        else np.empty((0, 3), int)
    )
    tn25 = tnum[tvec]
    tn26 = tnum[tvec]

    # Update triangulation (safe concat)
    keep = ~div1 & ~div3
    tri_blocks = [
        tria[keep, 0:3],
        tr41,
        tr42,
        tr43,
        tr44,
        tr21,
        tr22,
        tr23,
        tr24,
        tr25,
        tr26,
    ]
    tri_blocks = [b for b in tri_blocks if b.size > 0]
    tria = np.vstack(tri_blocks) if tri_blocks else np.empty((0, 3), int)

    tnum_blocks = [
        tnum[keep],
        tn41,
        tn42,
        tn43,
        tn44,
        tn21,
        tn22,
        tn23,
        tn24,
        tn25,
        tn26,
    ]
    tnum_blocks = [np.atleast_2d(b) for b in tnum_blocks if b.size > 0]
    tnum = np.vstack(tnum_blocks) if tnum_blocks else np.empty((0, 1), int)

    return vert, conn, tria, tnum
