import numpy as np

from ..aabb_tree.maketree import maketree


def idxtri(vert, tria):
    """
    IDXTRI : create a spatial-indexing structure for a 2-simplex triangulation in 2D.

    [tree] = idxtri(vert, tria) builds an Axis-Aligned Bounding Box (AABB) tree
    designed to accelerate spatial queries within a triangulation defined by
    {vert, tria}.

    Parameters
    ----------
    vert : ndarray (V, 2)
        Array of XY coordinates of the triangulation vertices.
    tria : ndarray (T, 3)
        Array of vertex indices defining each triangle. Each row defines one
        triangle such that:
        `vert[tria[ii, 0], :]`, `vert[tria[ii, 1], :]`, and `vert[tria[ii, 2], :]`
        are the coordinates of the ii-th triangle.

    Returns
    -------
    tree : dict or object
        A spatial AABB-tree indexing the triangles of the mesh, useful for
        efficient point-location and intersection queries.

    See Also
    --------
    trihfn : evaluate a mesh-size function on a triangulation.
    lfshfn : compute local feature-size estimates.
    maketree : build an AABB tree for general rectangular bounds.

    References
    ----------
    Translation of the MESH2D function `IDXTRI2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # --------------------------------------------- basic checks
    if not (isinstance(vert, np.ndarray) and isinstance(tria, np.ndarray)):
        raise TypeError("idxtri:incorrectInputClass")

    if vert.ndim != 2 or tria.ndim != 2:
        raise ValueError("idxtri:incorrectDimensions")
    if vert.shape[1] != 2 or tria.shape[1] < 3:
        raise ValueError("idxtri:incorrectDimensions")

    nvrt = vert.shape[0]

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) >= nvrt:
        raise ValueError("idxtri:invalidInputs")

    # ------------------------------ calc. AABB indexing for TRIA
    bmin = vert[tria[:, 0], :].copy()
    bmax = vert[tria[:, 0], :].copy()

    for ii in range(tria.shape[1]):
        bmin = np.minimum(bmin, vert[tria[:, ii], :])
        bmax = np.maximum(bmax, vert[tria[:, ii], :])

    # ------------------------------ opts (MATLAB/Octave specific, we mimic)
    opts = {}
    opts["nobj"] = 16

    # ------------------------------ build tree (dummy version)
    tree = maketree(np.hstack([bmin, bmax]), opts)

    return tree
