import numpy as np
from .triarea import triarea

def triscr(pp, tt):
    """
    Compute area-to-length ratios for triangles in a 2-simplex triangulation 
    embedded in the two-dimensional plane.

    This function calculates the ratio between the area and the squared 
    perimeter (or characteristic edge length) of each triangle. 
    This ratio is a measure of triangle quality — higher values typically 
    indicate more equilateral and well-shaped elements.

    Parameters
    ----------
    VERT : ndarray of shape (V, 2)
        Coordinates of the vertices in the triangulation.
    TRIA : ndarray of shape (T, 3)
        Array of triangle vertex indices defining the 2-simplexes.

    Returns
    -------
    SCR2 : ndarray of shape (T,)
        Area-to-length ratios for each triangle. 
        Values closer to the theoretical maximum correspond to better-shaped triangles.

    Notes
    -----
    - The area-to-length ratio provides a simple geometric quality measure.
    - It is invariant to scaling and can be used to identify distorted or 
      elongated triangles.

    References
    ----------
    Translation of the MESH2D function `TRISCR2`.  
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # --------------------------- compute signed area-len. ratios
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
