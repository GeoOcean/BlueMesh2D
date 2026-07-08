import numpy as np
from .pwrbal2 import pwrbal2

def tribal2(pp, tt):
    """
    Compute the circumballs associated with a 2-simplex triangulation 
    embedded in R² or R³.

    This function calculates the circumscribing balls for the set of 
    2-simplexes (triangles) in a triangulation.

    Parameters
    ----------
    PP : ndarray of shape (N, D)
        Coordinates of the vertices in the triangulation, where `D` = 2 or 3.
    TT : ndarray of shape (T, 3)
        Array of triangle vertex indices defining the 2-simplexes.

    Returns
    -------
    BB : ndarray of shape (T, 3)
        Circumballs associated with each triangle, where each row is `[XC, YC, RC²]`
        — the center coordinates and squared radius of the circumball.

    Notes
    -----
    - Each circumball is the unique sphere (or circle in 2D) passing through 
      the three vertices of a triangle.
    - Useful for Delaunay triangulation, mesh refinement, and quality metrics.

    References
    ----------
    Translation of the MESH2D function `TRIBAL2`.  
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    return pwrbal2(pp, np.zeros((pp.shape[0], 1)), tt)
