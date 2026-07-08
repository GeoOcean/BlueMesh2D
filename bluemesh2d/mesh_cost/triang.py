import numpy as np


def triang(pp, tt):
    """
    Compute the internal angles for a 2-simplex triangulation
    embedded in the two-dimensional plane.

    This function calculates the internal (enclosed) angles of each triangle
    in a given triangulation.

    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        Coordinates of the vertices in the triangulation.
    TRIA : ndarray of shape (T, 3)
        Array of triangle vertex indices defining the 2-simplexes.

    Returns
    -------
    ADEG : ndarray of shape (T, 3)
        Internal angles (in degrees) for each triangle, where each row
        contains the three vertex angles of a given triangle.

    Notes
    -----
    - Angles are computed from the vertex coordinates and returned in degrees.
    - Useful for evaluating triangle shape quality and detecting sliver elements.

    References
    ----------
    Translation of the MESH2D function `TRIANG2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # --------------------------------------------- basic checks
    if not (isinstance(pp, np.ndarray) and isinstance(tt, np.ndarray)):
        raise TypeError("triang:incorrectInputClass")

    if pp.ndim != 2 or tt.ndim != 2:
        raise ValueError("triang:incorrectDimensions")
    if pp.shape[1] != 2 or tt.shape[1] < 3:
        raise ValueError("triang:incorrectDimensions")

    nnod = pp.shape[0]
    if np.min(tt[:, :3]) < 0 or np.max(tt[:, :3]) >= nnod:
        raise ValueError("triang:invalidInputs")

    # ----------------------------------- compute enclosed angles
    dcos = np.zeros((tt.shape[0], 3))

    ev12 = pp[tt[:, 1], :] - pp[tt[:, 0], :]
    ev23 = pp[tt[:, 2], :] - pp[tt[:, 1], :]
    ev31 = pp[tt[:, 0], :] - pp[tt[:, 2], :]

    lv11 = np.sqrt(np.sum(ev12**2, axis=1))
    lv22 = np.sqrt(np.sum(ev23**2, axis=1))
    lv33 = np.sqrt(np.sum(ev31**2, axis=1))

    ev12 = ev12 / lv11[:, None]
    ev23 = ev23 / lv22[:, None]
    ev31 = ev31 / lv33[:, None]

    dcos[:, 0] = np.sum(-ev12 * ev23, axis=1)
    dcos[:, 1] = np.sum(-ev23 * ev31, axis=1)
    dcos[:, 2] = np.sum(-ev31 * ev12, axis=1)

    dcos = np.clip(dcos, -1.0, 1.0)

    dcos = np.arccos(dcos) * 180.0 / np.pi

    return dcos
