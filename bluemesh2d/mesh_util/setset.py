import numpy as np


def setset(iset, jset):
    """Test edge membership between two edge lists.

    Fast row-wise membership for undirected vertex-index edge pairs.

    Parameters
    ----------
    iset : ndarray of shape (I, 2)
        Query edges to test.
    jset : ndarray of shape (J, 2)
        Reference edges to search.

    Returns
    -------
    same : ndarray of shape (I,), dtype bool
        ``True`` when the corresponding row of ``iset`` appears in ``jset``.
    sloc : ndarray of shape (I,), dtype int
        Index into ``jset`` for matched edges, or ``-1`` when not found.

    References
    ----------
    Translation of the MESH2D function ``SETSET2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if not (isinstance(iset, np.ndarray) and isinstance(jset, np.ndarray)):
        raise TypeError("setset: inputs must be numpy arrays")

    if iset.ndim != 2 or jset.ndim != 2:
        raise ValueError("setset: inputs must be 2D arrays")

    if iset.shape[1] != 2 or jset.shape[1] != 2:
        raise ValueError("setset: each row must define an edge (2 columns)")

    iset = np.sort(iset, axis=1)
    jset = np.sort(jset, axis=1)

    iset_keys = iset[:, 0] * (2**31) + iset[:, 1]
    jset_keys = jset[:, 0] * (2**31) + jset[:, 1]

    # equivalent to: [same, sloc] = ismember(iset, jset, 'rows')
    jdict = {val: idx for idx, val in enumerate(jset_keys)}
    sloc = np.array([jdict.get(val, -1) for val in iset_keys], dtype=int)
    same = sloc >= 0

    return same, sloc
