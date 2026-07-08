import numpy as np

from ..aabb_tree.findtria import findtria


def trihfn(test, vert, tria, tree, hfun):
    """
    Interpolate a discrete mesh-size function over a 2D triangulation.

    This function evaluates a mesh-size field (`HFUN`) defined on a 2-simplex
    triangulation `{VERT, TRIA}` at arbitrary query points `TEST`.
    The interpolation is performed using barycentric coordinates within
    the containing triangle, efficiently located using a spatial index.

    Parameters
    ----------
    TEST : ndarray of shape (Q, 2)
        XY-coordinates of the query points where the mesh-size field will be evaluated.
    VERT : ndarray of shape (V, 2)
        XY-coordinates of the mesh vertices.
    TRIA : ndarray of shape (T, 3)
        Triangle connectivity array, where each row defines a triangle by the indices
        of its three vertices.
    TREE : object
        Spatial search structure (AABB tree) built for `{VERT, TRIA}`, as returned by `idxtri`.
        Used to efficiently locate which triangle each query point belongs to.
    HFUN : ndarray of shape (V,)
        Discrete mesh-size values defined at the vertices of the triangulation.

    Returns
    -------
    HVAL : ndarray of shape (Q,)
        Interpolated mesh-size values evaluated at each query point in `TEST`.

    Notes
    -----
    - The interpolation is linear within each triangle, based on barycentric coordinates.
    - Points located outside the triangulated domain may return `NaN` or extrapolated values.
    - A spatial index (e.g., an AABB tree) greatly accelerates point-location queries.

    References
    ----------
    Translation of the MESH2D function `TRIHFN2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    See also
    --------
    lfshfn : Estimate local feature size fields for polygonal domains.
    limhfn : Apply gradient limiting to mesh-size functions.
    idxtri : Build spatial search trees for triangle meshes.
    """

    # -------------------------- basic checks
    if not (
        isinstance(test, np.ndarray)
        and isinstance(vert, np.ndarray)
        and isinstance(tria, np.ndarray)
        and isinstance(hfun, np.ndarray)
    ):
        raise TypeError("trihfn:incorrectInputClass")

    if test.ndim != 2 or vert.ndim != 2 or tria.ndim != 2 or hfun.ndim != 1:
        raise ValueError("trihfn:incorrectDimensions")

    if test.shape[1] != 2 or vert.shape[1] != 2 or tria.shape[1] < 3:
        raise ValueError("trihfn:incorrectDimensions")

    if vert.shape[0] != hfun.shape[0]:
        raise ValueError("trihfn:incorrectDimensions")

    nvrt = vert.shape[0]
    if tria.min() < 0 or tria.max() >= nvrt:
        raise ValueError("trihfn:invalidInputs: invalid TRIA array")

    # ---------------------- test-to-tria queries
    tp, tj, _ = findtria(vert, tria, test, tree)

    if tp is None or len(tp) == 0:
        in_mask = np.zeros(test.shape[0], dtype=bool)
        ti = np.array([], dtype=int)
    else:
        in_mask = tp[:, 0] > 0
        ti = tj[tp[in_mask, 0]]

    # ------------------------------------- calc. linear interp.
    hval = np.full(test.shape[0], np.max(hfun))

    if np.any(in_mask):
        d1 = test[in_mask, :] - vert[tria[ti, 0], :]
        d2 = test[in_mask, :] - vert[tria[ti, 1], :]
        d3 = test[in_mask, :] - vert[tria[ti, 2], :]

        a3 = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
        a2 = np.abs(d1[:, 0] * d3[:, 1] - d1[:, 1] * d3[:, 0])
        a1 = np.abs(d3[:, 0] * d2[:, 1] - d3[:, 1] * d2[:, 0])

        hval[in_mask] = (
            a1 * hfun[tria[ti, 0]] + a2 * hfun[tria[ti, 1]] + a3 * hfun[tria[ti, 2]]
        ) / (a1 + a2 + a3)

    return hval
