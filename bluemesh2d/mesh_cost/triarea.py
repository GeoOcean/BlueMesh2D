import numpy as np


def triarea(pp, tt):
    """
    Compute the signed areas of triangles for a 2-simplex triangulation
    embedded in the two-dimensional plane.

    This function calculates the signed area of each triangle in the triangulation.
    The sign of the area indicates the orientation of the triangle
    (positive for counter-clockwise vertex ordering, negative for clockwise).

    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        Coordinates of the vertices in the triangulation.
    TRIA : ndarray of shape (T, 3)
        Array of triangle vertex indices defining the 2-simplexes.

    Returns
    -------
    AREA : ndarray of shape (T,)
        Signed areas of each triangle. Positive values correspond to
        counter-clockwise orientation.

    Notes
    -----
    - The computed area is equal to half the magnitude of the cross product
      between two triangle edges.
    - Useful for verifying mesh orientation and computing geometric properties.

    References
    ----------
    Translation of the MESH2D function `TRIAREA`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ---------------------------------------------- basic checks
    if not (isinstance(pp, np.ndarray) and isinstance(tt, np.ndarray)):
        raise TypeError("triarea:incorrectInputClass")

    if pp.ndim != 2 or tt.ndim != 2:
        raise ValueError("triarea:incorrectDimensions")
    if pp.shape[1] not in (2, 3) or tt.shape[1] < 3:
        raise ValueError("triarea:incorrectDimensions")

    nnod = pp.shape[0]
    if np.min(tt[:, :3]) < 0 or np.max(tt[:, :3]) >= nnod:
        raise ValueError("triarea:invalidInputs")
    # --------------------------------------- compute signed area
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
