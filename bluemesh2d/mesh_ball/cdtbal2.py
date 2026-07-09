import numpy as np

from .tribal2 import tribal2


def cdtbal2(pp, ee, tt):
    """Compute clipped circumballs for a constrained Delaunay triangulation.

    Start from triangle circumballs and replace any ball that extends outside
    the domain with a smaller edge-centered diametric ball on a constrained
    face.

    Parameters
    ----------
    pp : ndarray of shape (N, 2)
        Vertex coordinates.
    ee : ndarray of shape (E, 5)
        Edge connectivity from :func:`tricon`.
    tt : ndarray of shape (T, 6)
        Triangle connectivity from :func:`tricon`.

    Returns
    -------
    cc : ndarray of shape (T, 3)
        Clipped circumball parameters ``[xc, yc, r²]`` for each triangle.

    References
    ----------
    Translation of the MESH2D function ``CDTBAL2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not (
        isinstance(pp, np.ndarray)
        and isinstance(ee, np.ndarray)
        and isinstance(tt, np.ndarray)
    ):
        raise TypeError("cdtbal2:incorrectInputClass")

    if pp.ndim != 2 or ee.ndim != 2 or tt.ndim != 2:
        raise ValueError("cdtbal2:incorrectDimensions")

    if pp.shape[1] != 2 or ee.shape[1] < 5 or tt.shape[1] < 6:
        raise ValueError("cdtbal2:incorrectDimensions")

    cc = tribal2(pp, tt)

    # Replace with face-balls if smaller
    cc = minfac2(cc, pp, ee, tt, 0, 1, 2)
    cc = minfac2(cc, pp, ee, tt, 1, 2, 0)
    cc = minfac2(cc, pp, ee, tt, 2, 0, 1)

    return cc


def minfac2(cc, pp, ee, tt, ni, nj, nk):
    """Clip circumballs to constrained faces of a CDT.

    Replace circumballs that extend beyond a constrained edge with the
    diametric ball centered on that edge.

    Parameters
    ----------
    cc : ndarray of shape (T, 3)
        Circumball parameters ``[xc, yc, r²]``.
    pp : ndarray of shape (N, 2)
        Vertex coordinates.
    ee : ndarray of shape (E, 5)
        Edge connectivity from :func:`tricon`.
    tt : ndarray of shape (T, 6)
        Triangle connectivity from :func:`tricon`.
    ni, nj, nk : int
        Local vertex indices (0, 1, or 2) defining the edge ``[ni, nj]`` and
        opposite vertex ``nk`` for the current face pass.

    Returns
    -------
    cc : ndarray of shape (T, 3)
        Updated circumball parameters.

    References
    ----------
    Translation of the MESH2D function ``MINFAC2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    EF = ee[tt[:, ni + 3], 4] > 0

    bc = 0.5 * (pp[tt[EF, ni], :] + pp[tt[EF, nj], :])

    br = np.sum((bc - pp[tt[EF, ni], :]) ** 2, axis=1) + np.sum(
        (bc - pp[tt[EF, nj], :]) ** 2, axis=1
    )
    br = br * 0.5

    ll = np.sum((bc - pp[tt[EF, nk], :]) ** 2, axis=1)

    bi = (br >= ll) & (br <= cc[EF, 2])
    ei = np.where(EF)[0]
    ti = ei[bi]
    cc[ti, 0:2] = bc[bi, :]
    cc[ti, 2] = br[bi]

    return cc
