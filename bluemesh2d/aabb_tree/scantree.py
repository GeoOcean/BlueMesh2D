import numpy as np


def scantree(tr, pi, fn):
    """Compute tree-to-item and item-to-tree mappings for an AABB tree.

    Parameters
    ----------
    tr : dict
        AABB tree from :func:`maketree`, with keys ``"xx"``, ``"ii"``, and
        ``"ll"``. ``tr["xx"]`` holds node bounding boxes as ``[pmin, pmax]``;
        ``tr["ii"]`` holds parent–child indices; ``tr["ll"]`` lists item
        indices per node.
    pi : ndarray
        Query items (e.g. vertices or bounding boxes) to map against the tree.
    fn : callable
        Partition function called as ``ki, kj = fn(pj, ni, nj)``, where ``pj``
        is a subset of items in the current node, ``ni`` and ``nj`` are child
        node bounding boxes, and ``ki``/``kj`` are boolean masks of items
        intersecting each child.

    Returns
    -------
    tm : dict
        Tree-to-item mapping with keys ``"ii"`` (node indices) and ``"ll"``
        (lists of item indices per node).
    im : dict
        Item-to-tree mapping with keys ``"ii"`` (item indices) and ``"ll"``
        (lists of tree node indices per item).

    References
    ----------
    Translation of the MESH2D function ``SCANTREE``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    tm = {"ii": [], "ll": []}
    im = {"ii": [], "ll": []}

    if pi is None or len(pi) == 0:
        return tm, im
    if tr is None or len(tr) == 0:
        return tm, im
    if not isinstance(tr, dict) or not isinstance(pi, np.ndarray) or not callable(fn):
        raise TypeError("scantree: incorrect input class.")

    if not all(k in tr for k in ("xx", "ii", "ll")):
        raise ValueError("scantree: incorrect AABB struct.")

    n_nodes = tr["ii"].shape[0]
    tm["ii"] = np.zeros(n_nodes, dtype=int)
    tm["ll"] = [None] * n_nodes

    ss = np.zeros(n_nodes, dtype=int)
    sl = [None] * n_nodes
    sl[0] = np.arange(pi.shape[0])

    tf = np.array([len(x) > 0 for x in tr["ll"]])
    # Descend tree from root, push items amongst nodes
    ss[0] = 0
    ns = 1
    no = 0

    while ns > 0:
        ns -= 1
        ni = ss[ns]  # pop

        if tf[ni]:
            # push onto tree-item mapping -- non-empty node NI contains items LL
            tm["ii"][no] = ni
            tm["ll"][no] = sl[ns]
            no += 1

        if tr["ii"][ni, 1] != 0:
            c1 = tr["ii"][ni, 1]
            c2 = tr["ii"][ni, 1] + 1
            j1, j2 = fn(pi[sl[ns], :], tr["xx"][c1, :], tr["xx"][c2, :])
            l1 = sl[ns][j1]
            l2 = sl[ns][j2]

            if l1.size > 0:
                ss[ns] = c1
                sl[ns] = l1
                ns += 1
            if l2.size > 0:
                ss[ns] = c2
                sl[ns] = l2
                ns += 1

    tm["ii"] = tm["ii"][:no]
    tm["ll"] = tm["ll"][:no]

    # Compute inverse map only if desired
    if tm and im is None:
        return tm

    # Accumulate pair'd tree-item matches
    ic = []
    jc = tm["ll"]

    for ip in range(no):
        ni = tm["ii"][ip]
        ic.append(np.full(len(jc[ip]), ni, dtype=int))

    if len(ic) == 0:
        return tm, im

    ii = np.concatenate(ic)
    jj = np.concatenate(jc)

    im["ll"] = [None] * pi.shape[0]

    jx = np.argsort(jj)
    jj = jj[jx]
    ii = ii[jx]

    diff_idx = np.nonzero(np.diff(jj) != 0)[0]
    im["ii"] = np.concatenate((jj[diff_idx], [jj[-1]]))
    bounds = np.concatenate(([0], diff_idx + 1, [len(ii)]))
    # Distribute single item-tree matches
    for ip in range(len(im["ii"])):
        im["ll"][ip] = ii[bounds[ip] : bounds[ip + 1]]

    return tm, im
