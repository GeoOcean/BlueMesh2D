import numpy as np

from .scantree import scantree


def mapvert(tr, pi):
    """Compute tree-to-vertex and vertex-to-tree mappings for an AABB tree.

    Parameters
    ----------
    tr : dict
        AABB tree from :func:`maketree`, with keys ``"xx"``, ``"ii"``, and
        ``"ll"``.
    pi : ndarray of shape (NP, NDIM)
        Vertex coordinates.

    Returns
    -------
    tm : dict
        Tree-to-vertex mapping with keys ``"ii"`` (node indices) and ``"ll"``
        (vertex indices per node).
    im : dict
        Vertex-to-tree mapping with keys ``"ii"`` (vertex indices) and ``"ll"``
        (tree node indices per vertex).

    References
    ----------
    Translation of the MESH2D function ``MAPVERT``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """


    # Call SCANTREE to do the actual work
    results = scantree(tr, pi, partvert)
    if len(results) == 1:
        return results[0], None
    else:
        return results


def partvert(pi, b1, b2):
    """Partition points between two bounding boxes for :func:`scantree`.

    Parameters
    ----------
    pi : ndarray of shape (N, ND)
        Query points.
    b1, b2 : ndarray of shape (2*ND,)
        Bounding boxes as ``[pmin, pmax]``.

    Returns
    -------
    j1, j2 : ndarray of bool, shape (N,)
        Masks of points inside ``b1`` and ``b2``, respectively.

    References
    ----------
    Translation of the MESH2D function ``PARTVERT``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """
    nd = b1.shape[0] // 2

    j1 = np.ones(pi.shape[0], dtype=bool)
    j2 = np.ones(pi.shape[0], dtype=bool)

    for ax in range(nd):
        # Remains TRUE if inside bounds along axis AX
        j1 &= (pi[:, ax] >= b1[ax]) & (pi[:, ax] <= b1[ax + nd])
        # Remains TRUE if inside bounds along axis AX
        j2 &= (pi[:, ax] >= b2[ax]) & (pi[:, ax] <= b2[ax + nd])

    return j1, j2
