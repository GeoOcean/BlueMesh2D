import numpy as np

from ..aabb_tree.maketree import maketree


def idxtri(vert, tria):
    """Build an AABB tree over the triangles of a 2D triangulation.

    Parameters
    ----------
    vert : ndarray of shape (V, 2)
        Vertex coordinates.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.

    Returns
    -------
    tree
        AABB tree from :func:`maketree` indexing triangle bounding boxes.

    References
    ----------
    Translation of the MESH2D function ``IDXTRI2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not (isinstance(vert, np.ndarray) and isinstance(tria, np.ndarray)):
        raise TypeError("idxtri:incorrectInputClass")

    if vert.ndim != 2 or tria.ndim != 2:
        raise ValueError("idxtri:incorrectDimensions")
    if vert.shape[1] != 2 or tria.shape[1] < 3:
        raise ValueError("idxtri:incorrectDimensions")

    nvrt = vert.shape[0]

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) >= nvrt:
        raise ValueError("idxtri:invalidInputs")

    bmin = vert[tria[:, 0], :].copy()
    bmax = vert[tria[:, 0], :].copy()

    for ii in range(tria.shape[1]):
        bmin = np.minimum(bmin, vert[tria[:, ii], :])
        bmax = np.maximum(bmax, vert[tria[:, ii], :])

    # Opts (MATLAB/Octave specific, we mimic)
    opts = {}
    opts["nobj"] = 16

    tree = maketree(np.hstack([bmin, bmax]), opts)

    return tree
