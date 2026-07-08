import numpy as np

from ..hjac_util.limgrad import limgrad
from ..mesh_util.tricon import tricon


def limhfn(vert, tria, hfun, dhdx):
    """
    Apply gradient limiting to a discrete mesh-size function over a 2D triangulation.

    This function enforces a maximum gradient constraint on a mesh-size field (`HFUN`)
    defined over a 2-simplex triangulation `{VERT, TRIA}`. The goal is to ensure that
    mesh-size variations between adjacent vertices do not exceed a prescribed limit,
    improving smoothness and stability in mesh generation.

    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        XY-coordinates of the mesh vertices.
    TRIA : ndarray of shape (T, 3)
        Triangle connectivity array.
        Each row defines one triangle by the indices of its three vertices.
    HFUN : ndarray of shape (V,)
        Input mesh-size field to be gradient-limited.
    DHDX : float
        Maximum allowed relative gradient of the mesh-size function.
        The constraint `(HFUN[v2] - HFUN[v1]) / L <= DHDX` is enforced
        for all triangle edges `{v1, v2}`, where `L` is the edge length.

    Returns
    -------
    HFUN : ndarray of shape (V,)
        Gradient-limited mesh-size field satisfying the imposed slope constraint.

    Notes
    -----
    - The algorithm iteratively enforces the gradient limit across all mesh edges.
    - Useful for ensuring that mesh-size transitions remain smooth and gradual,
      preventing abrupt local refinements.
    - Based on the concept of PDE-based gradient limiting for implicit geometries.

    References
    ----------
    Translation of the MESH2D function `LIMHFN2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d

    Adapted from:
        Persson, P.-O. (2006).
        *"Mesh size functions for implicit geometries and PDE-based gradient limiting."*
        Engineering with Computers, 22: 95–109.

    See also
    --------
    trihfn : Compute mesh-size fields from triangle geometry.
    lfshfn : Estimate local feature size fields for polygonal domains.
    """

    # -------------------------- basic checks
    if not (
        isinstance(vert, np.ndarray)
        and isinstance(tria, np.ndarray)
        and isinstance(hfun, np.ndarray)
        and np.isscalar(dhdx)
    ):
        raise TypeError("limhfn:incorrectInputClass")

    if (
        vert.ndim != 2
        or tria.ndim != 2
        or hfun.ndim != 1
        or vert.shape[1] != 2
        or tria.shape[1] < 3
        or vert.shape[0] != hfun.shape[0]
    ):
        raise ValueError("limhfn:incorrectDimensions")

    nvrt = vert.shape[0]

    if tria.min() < 0 or tria.max() >= nvrt:
        raise ValueError("limhfn:invalidInputArgument: invalid TRIA array")

    # -------------------- impose gradient limits over mesh edges
    edge, tria = tricon(tria)

    evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
    elen = np.sqrt(np.sum(evec**2, axis=1))

    # -------------------- impose gradient limits over edge graph
    hfun, _ = limgrad(edge, elen, hfun, dhdx, np.sqrt(nvrt))

    return hfun
