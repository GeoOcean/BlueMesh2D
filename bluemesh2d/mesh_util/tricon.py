import numpy as np


def tricon(tt, cc=None):
    """Build edge-centered connectivity for a conforming triangular mesh.

    Parameters
    ----------
    tt : ndarray of shape (T, 3)
        Triangle vertex indices.
    cc : ndarray of shape (C, 2), optional
        Constraining edges as vertex-index pairs.

    Returns
    -------
    ee : ndarray of shape (E, 5)
        Unique edges with columns ``[v1, v2, tri1, tri2, constraint]``.
        ``tri2`` is ``-1`` on boundary edges; ``constraint`` is ``1`` when the
        edge matches an entry in ``cc``, else ``0``.
    tt : ndarray of shape (T, 6)
        Triangle-to-edge map with columns ``[v1, v2, v3, e1, e2, e3]``.

    References
    ----------
    Translation of the MESH2D function ``TRICON2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if cc is None:
        cc = np.empty((0, 2), dtype=int)

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

    ee = np.zeros((nt * 3, 2), dtype=int)
    ee[0 * nt : 1 * nt, :] = tt[:, [0, 1]]
    ee[1 * nt : 2 * nt, :] = tt[:, [1, 2]]
    ee[2 * nt : 3 * nt, :] = tt[:, [2, 0]]

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

    # Tria-to-edge indexing: 3 edges per tria
    tt_full = np.zeros((nt, 6), dtype=int)
    tt_full[:, :3] = tt
    tt_full[:, 3] = jv[0 * nt : 1 * nt]
    tt_full[:, 4] = jv[1 * nt : 2 * nt]
    tt_full[:, 5] = jv[2 * nt : 3 * nt]

    # Edge-to-tria indexing: 2 trias per edge
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

    if cc.size > 0:
        cc_sorted = np.sort(cc, axis=1)
        cd = cc_sorted[:, 0] * (2**31) + cc_sorted[:, 1]
        constraint_flag = np.isin(ed[iv], cd).astype(int)
        ee_full[:, 4] = constraint_flag

    return ee_full, tt_full
