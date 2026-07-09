import numpy as np

def minlen(pp, tt):
    """Compute the minimum squared edge length per triangle.

    Parameters
    ----------
    pp : ndarray of shape (V, 2)
        Vertex coordinates.
    tt : ndarray of shape (T, 3)
        Triangle connectivity.

    Returns
    -------
    ll : ndarray of shape (T,)
        Minimum squared edge length for each triangle.
    ei : ndarray of shape (T,), dtype int
        Local edge index (0, 1, or 2) of the shortest edge.

    References
    ----------
    Translation of the MESH2D function ``MINLEN2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not (isinstance(pp, np.ndarray) and isinstance(tt, np.ndarray)):
        raise TypeError("minlen:incorrectInputClass")

    if pp.ndim != 2 or tt.ndim != 2:
        raise ValueError("minlen:incorrectDimensions")
    if pp.shape[1] != 2 or tt.shape[1] < 3:
        raise ValueError("minlen:incorrectDimensions")

    nnod = pp.shape[0]
    if tt[:, :3].min() < 0 or tt[:, :3].max() >= nnod:
        raise ValueError("minlen:invalidInputs")

    l1 = np.sum((pp[tt[:, 1], :] - pp[tt[:, 0], :])**2, axis=1)
    l2 = np.sum((pp[tt[:, 2], :] - pp[tt[:, 1], :])**2, axis=1)
    l3 = np.sum((pp[tt[:, 0], :] - pp[tt[:, 2], :])**2, axis=1)

    lengths = np.vstack([l1, l2, l3]).T
    ei = np.argmin(lengths, axis=1)
    ll = lengths[np.arange(lengths.shape[0]), ei]

    return ll, ei