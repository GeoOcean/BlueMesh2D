import numpy as np

def minlen(pp, tt):
    """
    MINLEN2 : compute the minimum edge length for each triangle in a 2D triangulation.

    [elen, imin] = minlen2(vert, tria) returns the shortest edge length `elen`
    and its corresponding local edge index `imin` for each triangle in the
    triangulation {vert, tria}.

    Parameters
    ----------
    vert : ndarray (V, 2)
        Array of XY coordinates of the triangulation vertices.
    tria : ndarray (T, 3)
        Array of vertex indices defining each triangle.
    elen : ndarray (T,)
        Array of minimum edge lengths for each triangle.
    imin : ndarray (T,)
        Array of local edge indices (1–3) corresponding to the minimum edge.

    Returns
    -------
    elen : ndarray
        Minimum edge length for each triangle.
    imin : ndarray
        Local edge index (1–3) of the minimum-length edge in each triangle.

    Notes
    -----
    Useful for mesh-quality control and identifying small elements in
    Delaunay-based or constrained triangulations.

    References
    ----------
    Translation of the MESH2D function `MINLEN2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # --------------------------------------------- basic checks
    if not (isinstance(pp, np.ndarray) and isinstance(tt, np.ndarray)):
        raise TypeError("minlen:incorrectInputClass")

    if pp.ndim != 2 or tt.ndim != 2:
        raise ValueError("minlen:incorrectDimensions")
    if pp.shape[1] != 2 or tt.shape[1] < 3:
        raise ValueError("minlen:incorrectDimensions")

    nnod = pp.shape[0]
    if tt[:, :3].min() < 0 or tt[:, :3].max() >= nnod:
        raise ValueError("minlen:invalidInputs")

    # ------------------------------------------ compute edge-len
    l1 = np.sum((pp[tt[:, 1], :] - pp[tt[:, 0], :])**2, axis=1)
    l2 = np.sum((pp[tt[:, 2], :] - pp[tt[:, 1], :])**2, axis=1)
    l3 = np.sum((pp[tt[:, 0], :] - pp[tt[:, 2], :])**2, axis=1)

    # ------------------------------------------ compute min.-len
    lengths = np.vstack([l1, l2, l3]).T
    ei = np.argmin(lengths, axis=1)
    ll = lengths[np.arange(lengths.shape[0]), ei]

    return ll, ei