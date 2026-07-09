import numpy as np


def triarea(pp, tt):
    """Compute signed triangle areas.

    Positive area indicates counter-clockwise vertex ordering in 2D.

    Parameters
    ----------
    pp : ndarray of shape (V, 2) or (V, 3)
        Vertex coordinates.
    tt : ndarray of shape (T, 3)
        Triangle connectivity.

    Returns
    -------
    area : ndarray of shape (T,)
        Signed triangle areas (2D) or Euclidean areas (3D).

    References
    ----------
    Translation of the MESH2D function ``TRIAREA``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not (isinstance(pp, np.ndarray) and isinstance(tt, np.ndarray)):
        raise TypeError("triarea:incorrectInputClass")

    if pp.ndim != 2 or tt.ndim != 2:
        raise ValueError("triarea:incorrectDimensions")
    if pp.shape[1] not in (2, 3) or tt.shape[1] < 3:
        raise ValueError("triarea:incorrectDimensions")

    nnod = pp.shape[0]
    if np.min(tt[:, :3]) < 0 or np.max(tt[:, :3]) >= nnod:
        raise ValueError("triarea:invalidInputs")
    ev12 = pp[tt[:, 1], :] - pp[tt[:, 0], :]
    ev13 = pp[tt[:, 2], :] - pp[tt[:, 0], :]

    if pp.shape[1] == 2:
        area = ev12[:, 0] * ev13[:, 1] - ev12[:, 1] * ev13[:, 0]
        area = 0.5 * area
    elif pp.shape[1] == 3:
        avec = np.cross(ev12, ev13)
        area = np.sqrt(np.sum(avec**2, axis=1))
        area = 0.5 * area
    else:
        raise ValueError("triarea:Unsupported dimension")

    return area
