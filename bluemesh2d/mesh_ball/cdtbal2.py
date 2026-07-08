import numpy as np

from .tribal2 import tribal2


def cdtbal2(pp, ee, tt):
    """
    Compute the modified circumballs for a constrained 2-simplex Delaunay triangulation in 2D.

    This function computes the smallest enclosing balls associated with the
    triangles of a constrained Delaunay triangulation (CDT) in 2D.
    These circumballs are guaranteed to remain inside the boundaries of the CDT.

    Parameters
    ----------
    PP : ndarray of shape (N, 2)
        XY-coordinates of the mesh vertices.
    EE : ndarray of shape (E, 2)
        Array of constrained edges in the triangulation.
    TT : ndarray of shape (T, 3)
        Triangle connectivity array, where each row defines a triangle by the indices
        of its three vertices.

    Returns
    -------
    CC : ndarray of shape (T, 3)
        Array containing the parameters of the smallest enclosing balls for each triangle:
        `[XC, YC, RC²]`, where `(XC, YC)` is the center and `RC²` is the squared radius.

    Notes
    -----
    - These circumballs are "clipped" to ensure they remain fully contained within
      the domain boundaries defined by the constrained edges `EE`.
    - For details about the edge array format, see the `tricon` or `tricon2` routines.

    References
    ----------
    Translation of the MESH2D function `CDTBAL2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ------------------- basic checks
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

    # ---------------------------------------- calc. circumballs
    cc = tribal2(pp, tt)

    # ------------------------ replace with face-balls if smaller
    cc = minfac2(cc, pp, ee, tt, 0, 1, 2)
    cc = minfac2(cc, pp, ee, tt, 1, 2, 0)
    cc = minfac2(cc, pp, ee, tt, 2, 0, 1)

    return cc


def minfac2(cc, pp, ee, tt, ni, nj, nk):
    """
    Constrain circumball centers to the boundaries of a constrained Delaunay triangulation (CDT).

    This function modifies a set of circumballs so that any ball lying outside
    the boundaries of a CDT is replaced by an edge-centered diametric ball.

    Parameters
    ----------
    CC : ndarray of shape (T, 3)
        Original circumballs, where each row is `[XC, YC, RC²]`
        — the center coordinates and squared radius.
    PP : ndarray of shape (N, 2)
        XY-coordinates of the mesh vertices.
    EE : ndarray of shape (E, 2)
        Constrained edges of the triangulation.
    TT : ndarray of shape (T, 3)
        Triangle connectivity array (indices of vertices per triangle).
    NI : ndarray or int
        Local index (or indices) of the first vertex of the edge being tested.
    NJ : ndarray or int
        Local index (or indices) of the second vertex of the edge being tested.
    NK : ndarray or int
        Local index (or indices) of the vertex opposite to the edge.

    Returns
    -------
    CM : ndarray of shape (T, 3)
        Modified circumballs, where any out-of-domain center is replaced by
        the diametric ball centered on the edge `[NI, NJ]`.

    Notes
    -----
    - The procedure ensures that the CDT circumballs remain consistent
      with domain boundaries.
    - Each invalid ball is projected onto the domain by constructing the
      smallest edge-centered replacement sphere.

    References
    ----------
    Translation of the MESH2D function `MINFAC2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ------------------------------------------------ outer edge
    EF = ee[tt[:, ni + 3], 4] > 0

    # ------------------------------------------------ edge balls
    bc = 0.5 * (pp[tt[EF, ni], :] + pp[tt[EF, nj], :])

    # ----------------------------------------------- edge radii
    br = np.sum((bc - pp[tt[EF, ni], :]) ** 2, axis=1) + np.sum(
        (bc - pp[tt[EF, nj], :]) ** 2, axis=1
    )
    br = br * 0.5

    # ------------------------------------------- enclosing radii
    ll = np.sum((bc - pp[tt[EF, nk], :]) ** 2, axis=1)

    # ------------------------------------------- replace if min.
    bi = (br >= ll) & (br <= cc[EF, 2])
    ei = np.where(EF)[0]
    ti = ei[bi]
    # ------------------------------------------- replace is min.
    cc[ti, 0:2] = bc[bi, :]
    cc[ti, 2] = br[bi]

    return cc
