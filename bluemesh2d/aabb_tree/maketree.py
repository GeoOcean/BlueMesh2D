import numpy as np


def maketree(rp, op=None):
    """
    Assemble an AABB search tree for a collection of (hyper-)rectangles.

    Parameters
    ----------
    rp : ndarray of shape (NR, 2*NDIM)
        Rectangles defined as [PMIN, PMAX].
    op : dict, optional
        User-defined options:
        - 'nobj': max number of rectangles per tree-node (default=32)
        - 'long': relative length tolerance (default=0.75)
        - 'vtol': volume tolerance (default=0.55)

    Returns
    -------
    tr : dict
        Tree structure with:
        - 'xx': node coordinates [PMIN, PMAX]
        - 'ii': parent/child indexing
        - 'll': list of rectangles per node

    Notes
    -----
    Traduced from MATLAB aabb-tree repository

    References
    ----------
    Darren Engwirda, Locally-optimal Delaunay-refinement
    and optimisation-based mesh generation, Ph.D. Thesis,
    School of Mathematics and Statistics, The University
    of Sydney, September 2014.
    """

    tr = {"xx": [], "ii": [], "ll": []}

    # ------------------------------ quick return on empty inputs
    if rp is None or len(rp) == 0:
        return tr

    # ---------------------------------------------- basic checks
    if not isinstance(rp, np.ndarray):
        raise TypeError("maketree: incorrect input class")

    if rp.ndim != 2 or rp.shape[1] % 2 != 0:
        raise ValueError("maketree: incorrect input dimensions")

    # -------------------- options
    NOBJ = 32
    if op is None:
        op = {"nobj": NOBJ, "long": 0.75, "vtol": 0.55}
    else:
        op.setdefault("nobj", NOBJ)
        op.setdefault("long", 0.75)
        op.setdefault("vtol", 0.55)
    # ---------------------------------- dimensions of rectangles
    nd = rp.shape[1] // 2
    ni = rp.shape[0]
    # ------------------------------------------ alloc. workspace
    xl = np.zeros((ni, nd))
    xr = np.zeros((ni, nd))
    ii = np.zeros((ni, 2), dtype=int)
    ll = [None] * ni
    ss = np.zeros(ni, dtype=int)
    # ------------------------------------ min & max coord. masks
    lv = np.zeros(rp.shape[1], dtype=bool)
    rv = np.zeros(rp.shape[1], dtype=bool)
    lv[:nd] = True
    rv[nd:] = True
    # ---------------------------------------- inflate rectangle
    r0 = np.min(rp[:, lv], axis=0)
    r1 = np.max(rp[:, rv], axis=0)
    rd = np.tile(r1 - r0, (ni, 1))
    rp[:, lv] -= rd * np.power(np.finfo(float).eps, 0.8)
    rp[:, rv] += rd * np.power(np.finfo(float).eps, 0.8)
    # ----------------------------------------- rectangle centres
    rc = 0.5 * (rp[:, lv] + rp[:, rv])
    # ----------------------------------------- rectangle lengths
    rd = rp[:, rv] - rp[:, lv]
    # ------------------------------ root contains all rectangles
    ll[0] = np.arange(ni)
    # ------------------------------------ indexing for root node
    ii[0, :] = 0
    # ------------------------------ root contains all rectangles
    xl[0, :] = np.min(rp[:, lv], axis=0)
    xr[0, :] = np.max(rp[:, rv], axis=0)
    # -- main loop : divide nodes until all constraints satisfied
    ss[0] = 0
    ns = 1
    nn = 1

    while ns != 0:
        # ----------------------------------- pop node from stack
        ni_node = ss[ns - 1]
        ns -= 1
        # ----------------------------------- push child indexing
        n1 = nn
        n2 = nn + 1
        # --------------------------- set of rectangles in parent
        li = ll[ni_node]
        # --------------------------- split plane on longest axis
        dd = xr[ni_node, :] - xl[ni_node, :]
        ia = np.argsort(dd)

        for id in range(nd - 1, -1, -1):
            # --------------------------- push rectangles to children
            ax = ia[id]
            mx = dd[ax]

            il = rd[li, ax] > op["long"] * mx
            lp = li[il]  #  "long" rectangles
            ls = li[~il]  #  "short" rectangles

            if len(lp) < 0.5 * len(ls) and len(lp) < 0.5 * op["nobj"]:
                break

        # select the split position: take the mean of the set of
        # (non-"long") rectangle centres along axis AX
        if len(ls) == 0:
            # -------------------------------- partition empty, done!
            continue

        sp = np.mean(rc[ls, ax])
        # ---------------------------- partition based on centres
        i2 = rc[ls, ax] > sp
        l1 = ls[~i2]  #  "left" rectangles
        l2 = ls[i2]  #  "right" rectangles

        if len(l1) <= 1 or len(l2) <= 1:
            # -------------------------------- partition empty, done!
            continue

        # ------------------------------- finalise node position
        xl[n1, :] = np.min(rp[l1[:, None], lv], axis=0)
        xr[n1, :] = np.max(rp[l1[:, None], rv], axis=0)
        xl[n2, :] = np.min(rp[l2[:, None], lv], axis=0)
        xr[n2, :] = np.max(rp[l2[:, None], rv], axis=0)
        # --------------------------- push child nodes onto stack
        if len(li) <= op["nobj"]:
            vi = np.prod(xr[ni_node, :] - xl[ni_node, :])  #  upper d-dim "vol."
            v1 = np.prod(xr[n1, :] - xl[n1, :])  # lower d-dim "vol."
            v2 = np.prod(xr[n2, :] - xl[n2, :])

            if v1 + v2 < op["vtol"] * vi:
                # -------------------------------- parent--child indexing
                ii[n1, 0] = ni_node
                ii[n2, 0] = ni_node
                ii[ni_node, 1] = n1
                # -------------------------------- finalise list indexing
                ll[ni_node] = lp
                ll[n1] = l1
                ll[n2] = l2

                ss[ns] = n1
                ss[ns + 1] = n2
                ns += 2
                nn += 2
        else:
            # -------------------------------- parent--child indexing
            ii[n1, 0] = ni_node
            ii[n2, 0] = ni_node
            ii[ni_node, 1] = n1
            # -------------------------------- finalise list indexing
            ll[ni_node] = lp
            ll[n1] = l1
            ll[n2] = l2

            ss[ns] = n1
            ss[ns + 1] = n2
            ns += 2
            nn += 2
    # ---------------------------------------------- trim alloc.
    xl = xl[:nn, :]
    xr = xr[:nn, :]
    ii = ii[:nn, :]
    ll = ll[:nn]
    # ----------------------------------------------- pack struct
    tr["xx"] = np.hstack((xl, xr))
    tr["ii"] = ii
    tr["ll"] = ll

    return tr
