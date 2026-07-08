import numpy as np


def isfeat(pp, ee, tt):
    """
    ISFEAT : identify "feature" triangles in a 2D constrained triangulation.

    [stat] = isfeat(vert, edge, tria) returns a boolean array indicating which
    triangles contain sufficiently "sharp" angles located at the intersection of
    two constrained edges. Triangles with angles sharper than `acos(+0.8)` degrees
    are flagged as features.

    Parameters
    ----------
    vert : ndarray (V, 2)
        Array of XY coordinates of the triangulation vertices.
    edge : ndarray (E, 2)
        Array of constrained edges, where each row defines an edge by vertex indices.
    tria : ndarray (T, 3)
        Array of triangles, where each row defines a triangle by vertex indices.
    stat : ndarray (T,), bool
        Boolean array where `True` indicates the presence of a sharp feature.

    Returns
    -------
    stat : ndarray of bool
        `True` for triangles that include a sharp feature (angle < acos(+0.8)).

    Notes
    -----
    Sharp features typically correspond to corners or narrow regions in the
    constrained polygonal geometry.

    References
    ----------
    Translation of the MESH2D function `ISFEAT2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # --------------------------------------------- basic checks
    if not (
        isinstance(pp, np.ndarray)
        and isinstance(ee, np.ndarray)
        and isinstance(tt, np.ndarray)
    ):
        raise TypeError("isfeat:incorrectInputClass")

    if pp.ndim != 2 or ee.ndim != 2 or tt.ndim != 2:
        raise ValueError("isfeat:incorrectDimensions")
    if pp.shape[1] != 2 or ee.shape[1] < 5 or tt.shape[1] < 6:
        raise ValueError("isfeat:incorrectDimensions")

    nnod = pp.shape[0]
    nedg = ee.shape[0]
    ntri = tt.shape[0]

    if np.min(tt[:, :3]) < 0 or np.max(tt[:, :3]) > nnod:
        raise ValueError("isfeat:invalidInputs")
    if np.min(tt[:, 3:6]) < 0 or np.max(tt[:, 3:6]) > nedg:
        raise ValueError("isfeat:invalidInputs")
    if np.min(ee[:, :2]) < 0 or np.max(ee[:, :2]) > nnod:
        raise ValueError("isfeat:invalidInputs")
    if np.min(ee[:, 2:4]) < -1 or np.max(ee[:, 2:4]) > ntri:  ###0
        raise ValueError("isfeat:invalidInputs")

    # ----------------------------- compute "feature"
    isf = np.zeros((tt.shape[0],), dtype=bool)
    bv = np.zeros((tt.shape[0], 3), dtype=bool)

    EI = [2, 0, 1]
    EJ = [0, 1, 2]
    NI = [2, 0, 1]
    NJ = [0, 1, 2]
    NK = [1, 2, 0]

    for ii in range(3):
        # ------------------------------------- common edge index
        ei = tt[:, EI[ii] + 3]
        ej = tt[:, EJ[ii] + 3]
        # ------------------------------------ is boundary edge?
        bi = ee[ei, 4] >= 1
        bj = ee[ej, 4] >= 1

        ok = bi & bj
        if not np.any(ok):
            continue

        ni = tt[ok, NI[ii]]
        nj = tt[ok, NJ[ii]]
        nk = tt[ok, NK[ii]]
        # ------------------------------------- adj. edge vectors
        vi = pp[ni, :] - pp[nj, :]
        vj = pp[nk, :] - pp[nj, :]
        # ------------------------------------- adj. edge lengths
        li = np.sqrt(np.sum(vi**2, axis=1))
        lj = np.sqrt(np.sum(vj**2, axis=1))
        ll = li * lj
        # ------------------------------------- adj. dot-product!
        aa = np.sum(vi * vj, axis=1) / ll

        bv[ok, ii] = aa >= 0.80
        isf[ok] = isf[ok] | bv[ok, ii]

    return isf, bv
