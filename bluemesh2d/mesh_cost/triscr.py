import numpy as np
from .triarea import triarea

def triscr(pp, tt):
    """Compute scaled area-to-length ratios for mesh quality.

    Higher values indicate more equilateral triangles. The scale factor
    ``4*sqrt(3)/3`` normalizes an equilateral triangle to unity.

    Parameters
    ----------
    pp : ndarray of shape (V, 2)
        Vertex coordinates.
    tt : ndarray of shape (T, 3)
        Triangle connectivity.

    Returns
    -------
    tscr : ndarray of shape (T,)
        Area-to-mean-squared-edge-length ratio for each triangle.

    References
    ----------
    Translation of the MESH2D function ``TRISCR2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # Compute signed area-len. ratios
    scal = 4.0 * np.sqrt(3.0) / 3.0

    area = triarea(pp, tt) # also error checks!

    lrms = (
        np.sum((pp[tt[:, 1], :] - pp[tt[:, 0], :])**2, axis=1) +
        np.sum((pp[tt[:, 2], :] - pp[tt[:, 1], :])**2, axis=1) +
        np.sum((pp[tt[:, 2], :] - pp[tt[:, 0], :])**2, axis=1)
    )

    lrms = (lrms / 3.0) ** 1.0

    tscr = scal * area / lrms

    return tscr
