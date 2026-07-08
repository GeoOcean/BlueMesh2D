import numpy as np


def setset(iset, jset):
    """
    SETSET2 : a fast variant of `ismember` for edge lists.

    [in_set] = setset2(iset, jset) returns a boolean array where each entry
    indicates whether the corresponding edge in `iset` is present in `jset`.
    This function provides an optimized alternative to MATLAB’s `ismember`,
    specifically designed for comparing lists of edge indices.

    Parameters
    ----------
    iset : ndarray (I, 2)
        Array of "query" edges to test.
    jset : ndarray (J, 2)
        Array of reference edges to compare against.

    Returns
    -------
    in_set : ndarray (I,), bool
        Boolean array where `True` indicates that the corresponding edge in
        `iset` exists in `jset`.

    Notes
    -----
    This function is useful for geometric algorithms that require fast set
    membership tests over large collections of edge pairs, such as in mesh
    refinement or boundary detection.

    References
    ----------
    Translation of the MESH2D function `SETSET2`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # ---------------------------------------------- basic checks
    if not (isinstance(iset, np.ndarray) and isinstance(jset, np.ndarray)):
        raise TypeError("setset: inputs must be numpy arrays")

    if iset.ndim != 2 or jset.ndim != 2:
        raise ValueError("setset: inputs must be 2D arrays")

    if iset.shape[1] != 2 or jset.shape[1] != 2:
        raise ValueError("setset: each row must define an edge (2 columns)")

    # ---------------------------------------------- ensure v1 <= v2
    iset = np.sort(iset, axis=1)
    jset = np.sort(jset, axis=1)

    # ---------------------------------------------- encode edges as 1D keys
    iset_keys = iset[:, 0] * (2**31) + iset[:, 1]
    jset_keys = jset[:, 0] * (2**31) + jset[:, 1]

    # ---------------------------------------------- fast membership
    # equivalent to: [same, sloc] = ismember(iset, jset, 'rows')
    jdict = {val: idx for idx, val in enumerate(jset_keys)}
    sloc = np.array([jdict.get(val, -1) for val in iset_keys], dtype=int)
    same = sloc >= 0

    return same, sloc
