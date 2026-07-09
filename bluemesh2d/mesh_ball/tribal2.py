import numpy as np
from .pwrbal2 import pwrbal2

def tribal2(pp, tt):
    """Compute circumballs for triangles in R² or R³.

    Equivalent to :func:`pwrbal2` with zero vertex weights.

    Parameters
    ----------
    pp : ndarray of shape (N, 2) or (N, 3)
        Vertex coordinates.
    tt : ndarray of shape (T, 3)
        Triangle connectivity.

    Returns
    -------
    bb : ndarray of shape (T, 3) or (T, 4)
        Circumball parameters ``[xc, yc, r²]`` in 2D or ``[xc, yc, zc, r²]`` in 3D.

    References
    ----------
    Translation of the MESH2D function ``TRIBAL2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    return pwrbal2(pp, np.zeros((pp.shape[0], 1)), tt)
