import numpy as np


def cdtbal1(pp, ee):
    """Compute circumballs for 1-simplices (edges) in the plane.

    Parameters
    ----------
    pp : ndarray of shape (N, 2)
        Vertex coordinates.
    ee : ndarray of shape (E, 2)
        Edge connectivity as vertex-index pairs.

    Returns
    -------
    bb : ndarray of shape (E, 3)
        Circumball parameters ``[xc, yc, r²]`` for each edge.

    References
    ----------
    Translation of the MESH2D function ``CDTBAL1``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    if not (isinstance(pp, np.ndarray) and isinstance(ee, np.ndarray)):
        raise TypeError("cdtbal1:incorrectInputClass")

    if pp.ndim != 2 or ee.ndim != 2:
        raise ValueError("cdtbal1:incorrectDimensions")

    if pp.shape[1] != 2 or ee.shape[1] < 2:
        raise ValueError("cdtbal1:incorrectDimensions")

    bb = np.zeros((ee.shape[0], 3))

    bb[:, 0:2] = 0.5 * (pp[ee[:, 0], :] + pp[ee[:, 1], :])
    bb[:, 2] = 0.25 * np.sum((pp[ee[:, 0], :] - pp[ee[:, 1], :]) ** 2, axis=1)

    return bb
