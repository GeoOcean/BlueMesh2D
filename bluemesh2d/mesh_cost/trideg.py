import numpy as np


def trideg(pp, tt):
    """
    Compute the topological degree of vertices in a 2-simplex triangulation.

    This function calculates the number of triangles incident to each vertex
    in a 2D triangulation, also known as the vertex degree.

    Parameters
    ----------
    VERT : ndarray of shape (V, D)
        Coordinates of the vertices in the triangulation, where `D` = 2.
    TRIA : ndarray of shape (T, 3)
        Array of triangle vertex indices defining the 2-simplexes.

    Returns
    -------
    VDEG : ndarray of shape (V,)
        Number of triangles incident to each vertex (vertex degree).

    Notes
    -----
    - The vertex degree indicates local mesh connectivity and can be used
      to detect irregular or boundary vertices.
    - Typically, well-shaped interior vertices in a regular mesh have a degree of ~6.

    References
    ----------
    Translation of the MESH2D function `TRIDEG2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ---------------------------------------------- basic checks
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

    # ------------------------------------- compute vertex degree
    vdeg = np.zeros(nvrt, dtype=int)

    for tri in tt[:, :3]:
        vdeg[tri] += 1

    return vdeg
