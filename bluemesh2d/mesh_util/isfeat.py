import numpy as np


def isfeat(pp, ee, tt):
    """Identify feature triangles in a constrained triangulation.

    Flag triangles whose corners lie on two constrained edges and form an angle
    sharper than ``acos(0.8)``.

    Parameters
    ----------
    pp : ndarray of shape (V, 2)
        Vertex coordinates.
    ee : ndarray of shape (E, 5)
        Edge connectivity from :func:`tricon` (vertex pair, adjacent triangles,
        constraint flag).
    tt : ndarray of shape (T, 6)
        Triangle connectivity from :func:`tricon` (vertices and edge indices).

    Returns
    -------
    isf : ndarray of shape (T,), dtype bool
        ``True`` for triangles containing a sharp feature.
    bv : ndarray of shape (T, 3), dtype bool
        ``True`` at local corners where the sharp-angle test passed.

    References
    ----------
    Translation of the MESH2D function ``ISFEAT2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

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

    isf = np.zeros((tt.shape[0],), dtype=bool)
    bv = np.zeros((tt.shape[0], 3), dtype=bool)

    EI = [2, 0, 1]
    EJ = [0, 1, 2]
    NI = [2, 0, 1]
    NJ = [0, 1, 2]
    NK = [1, 2, 0]

    for ii in range(3):
        ei = tt[:, EI[ii] + 3]
        ej = tt[:, EJ[ii] + 3]
        bi = ee[ei, 4] >= 1
        bj = ee[ej, 4] >= 1

        ok = bi & bj
        if not np.any(ok):
            continue

        ni = tt[ok, NI[ii]]
        nj = tt[ok, NJ[ii]]
        nk = tt[ok, NK[ii]]
        vi = pp[ni, :] - pp[nj, :]
        vj = pp[nk, :] - pp[nj, :]
        li = np.sqrt(np.sum(vi**2, axis=1))
        lj = np.sqrt(np.sum(vj**2, axis=1))
        ll = li * lj
        aa = np.sum(vi * vj, axis=1) / ll

        bv[ok, ii] = aa >= 0.80
        isf[ok] = isf[ok] | bv[ok, ii]

    return isf, bv
