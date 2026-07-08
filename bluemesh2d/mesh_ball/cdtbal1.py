import numpy as np


def cdtbal1(pp, ee):
    """
    Compute the circumballs associated with a 1-simplex triangulation in 2D.

    This function computes the circumscribing balls (center and squared radius)
    for a set of 1-simplexes (edges) defined in a 2D space.

    Parameters
    ----------
    PP : ndarray of shape (N, 2)
        XY-coordinates of the mesh vertices.
    EE : ndarray of shape (E, 2)
        Edge connectivity array, where each row defines a line segment between
        two vertex indices.

    Returns
    -------
    BB : ndarray of shape (E, 3)
        Array containing the circumball parameters for each edge:
        `[XC, YC, RC²]`, where `(XC, YC)` is the ball center and `RC²`
        is the squared radius.

    References
    ----------
    Translation of the MESH2D function `CDTBAL1`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    # ----------------------- basic checks
    if not (isinstance(pp, np.ndarray) and isinstance(ee, np.ndarray)):
        raise TypeError("cdtbal1:incorrectInputClass")

    if pp.ndim != 2 or ee.ndim != 2:
        raise ValueError("cdtbal1:incorrectDimensions")

    if pp.shape[1] != 2 or ee.shape[1] < 2:
        raise ValueError("cdtbal1:incorrectDimensions")

    # ----------------------- compute circumballs
    bb = np.zeros((ee.shape[0], 3))

    bb[:, 0:2] = 0.5 * (pp[ee[:, 0], :] + pp[ee[:, 1], :])
    bb[:, 2] = 0.25 * np.sum((pp[ee[:, 0], :] - pp[ee[:, 1], :]) ** 2, axis=1)

    return bb
