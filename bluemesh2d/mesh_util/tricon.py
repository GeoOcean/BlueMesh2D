import numpy as np


def tricon(tt, cc=None):
    """
    TRICON : compute edge-centered connectivity for a conforming 2-simplex
    (triangular) mesh in the 2D plane.

    [ee, tt] = tricon(tt, cc) returns the edge-based adjacency for a
    triangulated mesh. The output `ee` contains the set of unique edges,
    and `tt` contains triangle-to-edge connectivity information.

    Parameters
    ----------
    tt : ndarray (T, 3)
        Array of vertex indices defining each triangle.
    cc : ndarray (C, 2)
        Array of constrained edges (optional). Each row defines a boundary edge.

    Returns
    -------
    ee : ndarray (E, 5)
        Edge connectivity array [V1, V2, T1, T2, CE], where:
            - V1, V2 : vertex indices forming the edge
            - T1, T2 : indices of adjacent triangles (T2 = 0 for boundary edges)
            - CE : index of the matching constraint in `cc`, or 0 if none
    tt : ndarray (T, 6)
        Triangle-to-edge mapping [V1, V2, V3, E1, E2, E3], where E1–E3 are
        the indices of the edges forming each triangle.

    Notes
    -----
    This routine builds explicit edge-triangle connectivity for conforming
    triangulations, identifying adjacency relationships and constrained edges.
    It is a key step in mesh refinement and optimization algorithms.

    References
    ----------
    Translation of the MESH2D function `TRICON2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ---------------------------------------------- extract args
    if cc is None:
        cc = np.empty((0, 2), dtype=int)

    # ---------------------------------------------- basic checks
    if not isinstance(tt, np.ndarray) or tt.ndim != 2 or tt.shape[1] != 3:
        raise ValueError("tricon:incorrectDimensions - tt must be (n,3) int array")

    if tt.min() < 0:
        raise ValueError("tricon:invalidInputs - indices must be >= 0 (0-based)")

    if cc.size > 0 and (
        not isinstance(cc, np.ndarray) or cc.ndim != 2 or cc.shape[1] != 2
    ):
        raise ValueError("tricon:incorrectDimensions - cc must be (m,2) int array")

    nt = tt.shape[0]
    _nc = cc.shape[0]

    # ------------------------------ assemble non-unique edge set
    ee = np.zeros((nt * 3, 2), dtype=int)
    ee[0 * nt : 1 * nt, :] = tt[:, [0, 1]]
    ee[1 * nt : 2 * nt, :] = tt[:, [1, 2]]
    ee[2 * nt : 3 * nt, :] = tt[:, [2, 0]]

    # ------------------------------ unique edges and re-indexing
    # [ee, iv, jv] = ...
    #     unique(sort(ee, 2), 'rows');

    # as a (much) faster alternative to the 'ROWS' based call
    # to UNIQUE above, the edge list (i.e. pairs of UINT32 va-
    # lues) can be cast to DOUBLE, and the sorted comparisons
    # performed on vector inputs!
    ee_sorted = np.sort(ee, axis=1)
    ed = ee_sorted[:, 0] * (2**31) + ee_sorted[:, 1]
    _, iv, jv = np.unique(ed, return_index=True, return_inverse=True)
    ee_unique = ee_sorted[iv, :]

    # ------------------- tria-to-edge indexing: 3 edges per tria
    tt_full = np.zeros((nt, 6), dtype=int)
    tt_full[:, :3] = tt
    tt_full[:, 3] = jv[0 * nt : 1 * nt]
    tt_full[:, 4] = jv[1 * nt : 2 * nt]
    tt_full[:, 5] = jv[2 * nt : 3 * nt]

    # ------------------- edge-to-tria indexing: 2 trias per edge
    ne = ee_unique.shape[0]
    ee_full = np.zeros((ne, 5), dtype=int)
    ee_full[:, :2] = ee_unique

    for ti in range(nt):
        for ei in tt_full[ti, 3:6]:
            if ee_full[ei, 2] == 0:
                ee_full[ei, 2] = ti
            elif ee_full[ei, 3] == 0:
                ee_full[ei, 3] = ti

    ee_full[ee_full[:, 3] == 0, 3] = -1

    # ------------------------------------ find constrained edges
    if cc.size > 0:
        cc_sorted = np.sort(cc, axis=1)
        cd = cc_sorted[:, 0] * (2**31) + cc_sorted[:, 1]
        constraint_flag = np.isin(ed[iv], cd).astype(int)
        # ----------------------------------- mark constrained edges
        ee_full[:, 4] = constraint_flag

    return ee_full, tt_full
