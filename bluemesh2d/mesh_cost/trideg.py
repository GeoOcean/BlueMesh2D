import numpy as np


def trideg(pp, tt):
    """Count incident triangles at each vertex.

    Parameters
    ----------
    pp : ndarray of shape (V, D)
        Vertex coordinates (``D >= 2``).
    tt : ndarray of shape (T, 3)
        Triangle connectivity.

    Returns
    -------
    vdeg : ndarray of shape (V,), dtype int
        Number of triangles incident to each vertex.

    References
    ----------
    Translation of the MESH2D function ``TRIDEG2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not (isinstance(pp, np.ndarray) and isinstance(tt, np.ndarray)):
        raise TypeError("trideg:incorrectInputClass")

    if pp.ndim != 2 or tt.ndim != 2:
        raise ValueError("trideg:incorrectDimensions")
    if pp.shape[1] < 2 or tt.shape[1] < 3:
        raise ValueError("trideg:incorrectDimensions")

    nvrt = pp.shape[0]
    _ntri = tt.shape[0]

    if np.min(tt[:, :3]) < 0 or np.max(tt[:, :3]) >= nvrt:
        raise ValueError("trideg:invalidInputs")

    vdeg = np.zeros(nvrt, dtype=int)

    for tri in tt[:, :3]:
        vdeg[tri] += 1

    return vdeg
