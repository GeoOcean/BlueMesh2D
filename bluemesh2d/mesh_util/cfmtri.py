import numpy as np
from scipy.spatial import Delaunay

from .setset import setset


def cfmtri(vert, econ):
    """Compute a conforming 2-simplex Delaunay triangulation in the plane.

    Insert Steiner vertices by bisecting constraining edges until every edge
    in ``econ`` appears in the Delaunay triangulation.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        Vertex coordinates to triangulate.
    econ : ndarray of shape (C, 2)
        Constraining edges as vertex-index pairs.

    Returns
    -------
    vert : ndarray of shape (V, 2)
        Vertex coordinates (including bisection Steiner points).
    econ : ndarray of shape (C, 2)
        Updated constraining edges after bisection.
    tria : ndarray of shape (T, 3)
        Triangle connectivity (CCW-oriented).

    References
    ----------
    Translation of the MESH2D function ``CFMTRI2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not isinstance(vert, np.ndarray) or not isinstance(econ, np.ndarray):
        raise TypeError("cfmtri:incorrectInputClass")

    if vert.ndim != 2 or econ.ndim != 2:
        raise ValueError("cfmtri:incorrectDimensions")

    if vert.shape[1] != 2 or econ.shape[1] != 2:
        raise ValueError("cfmtri:incorrectDimensions")

    # the DELAUNAYN routine is *not* well-behaved numerically,
    # so explicitly re-scale the problem about [-1,-1; +1,+1].
    vmax = np.max(vert, axis=0)
    vmin = np.min(vert, axis=0)

    vdel = np.mean(vmax - vmin) * 0.5
    vmid = (vmax + vmin) * 0.5

    vert = (vert - vmid) / vdel

    #  keep bisecting edge constraints until they are all recovered!
    while True:
        # Un-constrained delaunay triangulation
        tria = delaunay2(vert)

        nv = vert.shape[0]
        nt = tria.shape[0]

        ee = np.zeros((nt * 3, 2), dtype=int)
        ee[0:nt, :] = tria[:, [0, 1]]
        ee[nt : 2 * nt, :] = tria[:, [1, 2]]
        ee[2 * nt : 3 * nt, :] = tria[:, [2, 0]]

        # Find constraints within tria-edge set
        in_mask, _ = setset(econ, ee)
        if np.all(in_mask):
            break

        vm = (vert[econ[~in_mask, 0], :] + vert[econ[~in_mask, 1], :]) * 0.5

        ev = np.arange(nv, nv + vm.shape[0])
        en = np.vstack(
            [
                np.column_stack([econ[~in_mask, 0], ev]),
                np.column_stack([econ[~in_mask, 1], ev]),
            ]
        )

        vert = np.vstack([vert, vm])
        econ = np.vstack([econ[in_mask, :], en])

    vert = vert * vdel + vmid

    return vert, econ, tria


def delaunay2(points, options=None):
    """Compute a 2-simplex Delaunay triangulation in the plane.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Vertex coordinates to triangulate.
    options : str, optional
        Qhull options passed to :class:`scipy.spatial.Delaunay`. Default is
        ``'Qt Qbb Qc'`` in 2D and ``'Qt Qbb Qc Qx'`` for higher dimension.

    Returns
    -------
    t : ndarray of shape (T, 3)
        Triangle connectivity.
    """
    n, d = points.shape

    if options is None:
        if d >= 4:
            options = "Qt Qbb Qc Qx"
        else:
            options = "Qt Qbb Qc"

    tri = Delaunay(points, qhull_options=options)

    t = tri.simplices

    return t
