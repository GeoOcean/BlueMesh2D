import numpy as np
from scipy.spatial import Delaunay

from .setset import setset


def cfmtri(vert, econ):
    """
    CFMTRI : compute a conforming 2-simplex Delaunay triangulation in the 2D plane.

    [vert, conn, tria] = cfmtri2(vert, conn) computes a conforming Delaunay
    triangulation from a set of vertices and edge constraints. New vertices are
    inserted to bisect constraining edges until all constraints are recovered.

    Parameters
    ----------
    vert : ndarray (V, 2)
        Array of XY coordinates to be triangulated.
    conn : ndarray (C, 2)
        Array of constraining edges (each row defines an edge between vertices).
    tria : ndarray (T, 3)
        Array of vertex indices defining the triangles. Each row corresponds to
        one triangle such that:
        `vert[tria[ii, 0], :]`, `vert[tria[ii, 1], :]`, and `vert[tria[ii, 2], :]`
        are the coordinates of the `ii`-th triangle.

    See Also
    --------
    deltri : perform unconstrained Delaunay triangulation.
    delaunayn : compute N-D Delaunay triangulation.

    References
    ----------
    Translation of the MESH2D function `CFMTRI2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # --------------------------------------------- basic checks
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
        # ----------------- un-constrained delaunay triangulation
        tria = delaunay2(vert)

        nv = vert.shape[0]
        nt = tria.shape[0]

        # ---------------------------- build non-unique edge-set
        ee = np.zeros((nt * 3, 2), dtype=int)
        ee[0:nt, :] = tria[:, [0, 1]]
        ee[nt : 2 * nt, :] = tria[:, [1, 2]]
        ee[2 * nt : 3 * nt, :] = tria[:, [2, 0]]

        # ---------------- find constraints within tria-edge set
        in_mask, _ = setset(econ, ee)
        # ---------------------------- done when have contraints
        if np.all(in_mask):
            break

        # ----------------------------- un-recovered edge centres
        vm = (vert[econ[~in_mask, 0], :] + vert[econ[~in_mask, 1], :]) * 0.5

        # ----------------------------- un-recovered edge indexes
        ev = np.arange(nv, nv + vm.shape[0])
        en = np.vstack(
            [
                np.column_stack([econ[~in_mask, 0], ev]),
                np.column_stack([econ[~in_mask, 1], ev]),
            ]
        )

        # ---------------------------- push new vert/edge arrays
        vert = np.vstack([vert, vm])
        econ = np.vstack([econ[in_mask, :], en])

    # --------------------------------- undo geomertic re-scaling
    vert = vert * vdel + vmid

    return vert, econ, tria


def delaunay2(points, options=None):
    """
    delaunay2 compute a 2-simplex Delaunay triangulation in the two-dimensional plane.

    Parameters
    ----------
    points : (N,2) array
        XY coordinates of vertices to be triangulated.
    options : str, optional
        Qhull options string. If None, default options are used.

    Returns
    -------
    t : (T,3) array
        Triangulation connectivity.
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
