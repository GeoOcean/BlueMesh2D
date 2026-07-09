import numpy as np

from ..hjac_util.limgrad import limgrad
from ..mesh_util.tricon import tricon


def limhfn(vert, tria, hfun, dhdx):
    """Apply gradient limiting to a mesh-size function on a 2D triangulation.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        Mesh vertex coordinates.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.
    hfun : ndarray of shape (V,)
        Input mesh-size field at vertices.
    dhdx : float
        Maximum allowed relative gradient; for each edge ``(v1, v2)`` with
        length ``L``, enforce ``(hfun[v2] - hfun[v1]) / L <= dhdx``.

    Returns
    -------
    hfun : ndarray of shape (V,)
        Gradient-limited mesh-size field.

    References
    ----------
    Translation of the MESH2D function ``LIMHFN2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

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

    # Impose gradient limits over mesh edges
    edge, tria = tricon(tria)

    evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
    elen = np.sqrt(np.sum(evec**2, axis=1))

    # Impose gradient limits over edge graph
    hfun, _ = limgrad(edge, elen, hfun, dhdx, np.sqrt(nvrt))

    return hfun
