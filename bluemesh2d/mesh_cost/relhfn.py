import numpy as np


def relhfn(vert, tria, hvrt):
    """
    Compute the relative edge-length for a 2-simplex triangulation
    embedded in Euclidean space.

    This function calculates the normalized edge lengths for each triangle edge,
    indicating how well the triangulation conforms to the prescribed mesh-spacing
    constraints.

    Parameters
    ----------
    VERT : ndarray of shape (V, D)
        Coordinates of the vertices in the triangulation, where `D` = 2 or 3.
    TRIA : ndarray of shape (T, 3)
        Array of triangle vertex indices defining the 2-simplexes.
    HVRT : ndarray of shape (V,)
        Mesh-spacing values evaluated at the triangulation vertices.

    Returns
    -------
    HREL : ndarray of shape (E,)
        Relative edge-lengths for all edges in the triangulation, defined as
        the ratio between actual edge length and the average of the local
        mesh-spacing values at the edge endpoints.

    Notes
    -----
    - A relative length `HREL ≈ 1` indicates good conformance to the mesh-size
      function, while values significantly greater than 1 suggest under-refinement.
    - Commonly used in mesh-quality evaluation and adaptive refinement schemes.

    References
    ----------
    Translation of the MESH2D function `RELHFN2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ---------------------------------------------- basic checks
    if not (
        isinstance(vert, np.ndarray)
        and isinstance(tria, np.ndarray)
        and isinstance(hvrt, np.ndarray)
    ):
        raise TypeError("relhfn:incorrectInputClass")

    if vert.ndim != 2 or tria.ndim != 2:
        raise ValueError("relhfn:incorrectDimensions")
    if vert.shape[1] != 2 or tria.shape[1] < 3:
        raise ValueError("relhfn:incorrectDimensions")
    if len(hvrt.shape) != 1 or hvrt.shape[0] != vert.shape[0]:
        raise ValueError("relhfn:incorrectDimensions")

    nnod = vert.shape[0]

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) >= nnod:
        raise ValueError("relhfn:invalidInputs")

    # ----------------------------------- compute rel. mesh-sizes
    eset = np.vstack([tria[:, [0, 1]], tria[:, [1, 2]], tria[:, [2, 0]]])

    eset = np.sort(eset, axis=1)
    eset = np.unique(eset, axis=0)

    evec = vert[eset[:, 1], :] - vert[eset[:, 0], :]

    elen = np.sqrt(np.sum(evec**2, axis=1))

    hmid = hvrt[eset[:, 1]] + hvrt[eset[:, 0]]
    hmid = 0.5 * hmid
    hrel = elen / hmid

    return hrel
