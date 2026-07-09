import numpy as np


def maketree(rp, op=None):
    """Assemble an AABB search tree for a collection of (hyper-)rectangles.

    Parameters
    ----------
    rp : ndarray of shape (NR, 2*NDIM)
        Rectangles defined as ``[pmin, pmax]``.
    op : dict, optional
        Options: ``"nobj"`` (max rectangles per node, default 32),
        ``"long"`` (relative length tolerance, default 0.75), and ``"vtol"``
        (volume tolerance, default 0.55).

    Returns
    -------
    tr : dict
        Tree with keys ``"xx"`` (node bounding boxes), ``"ii"`` (parent/child
        indexing), and ``"ll"`` (rectangle indices per node).

    References
    ----------
    Translation of the MESH2D function ``MAKETREE``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    tr = {"xx": [], "ii": [], "ll": []}

    if rp is None or len(rp) == 0:
        return tr

    if not isinstance(rp, np.ndarray):
        raise TypeError("maketree: incorrect input class")

    if rp.ndim != 2 or rp.shape[1] % 2 != 0:
        raise ValueError("maketree: incorrect input dimensions")

    NOBJ = 32
    if op is None:
        op = {"nobj": NOBJ, "long": 0.75, "vtol": 0.55}
    else:
        op.setdefault("nobj", NOBJ)
        op.setdefault("long", 0.75)
        op.setdefault("vtol", 0.55)
    nd = rp.shape[1] // 2
    ni = rp.shape[0]
    xl = np.zeros((ni, nd))
    xr = np.zeros((ni, nd))
    ii = np.zeros((ni, 2), dtype=int)
    ll = [None] * ni
    ss = np.zeros(ni, dtype=int)
    lv = np.zeros(rp.shape[1], dtype=bool)
    rv = np.zeros(rp.shape[1], dtype=bool)
    lv[:nd] = True
    rv[nd:] = True
    r0 = np.min(rp[:, lv], axis=0)
    r1 = np.max(rp[:, rv], axis=0)
    rd = np.tile(r1 - r0, (ni, 1))
    rp[:, lv] -= rd * np.power(np.finfo(float).eps, 0.8)
    rp[:, rv] += rd * np.power(np.finfo(float).eps, 0.8)
    rc = 0.5 * (rp[:, lv] + rp[:, rv])
    rd = rp[:, rv] - rp[:, lv]
    ll[0] = np.arange(ni)
    ii[0, :] = 0
    xl[0, :] = np.min(rp[:, lv], axis=0)
    xr[0, :] = np.max(rp[:, rv], axis=0)
    # -- main loop : divide nodes until all constraints satisfied
    ss[0] = 0
    ns = 1
    nn = 1

    while ns != 0:
        ni_node = ss[ns - 1]
        ns -= 1
        n1 = nn
        n2 = nn + 1
        li = ll[ni_node]
        dd = xr[ni_node, :] - xl[ni_node, :]
        ia = np.argsort(dd)

        for id in range(nd - 1, -1, -1):
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
            continue

        sp = np.mean(rc[ls, ax])
        i2 = rc[ls, ax] > sp
        l1 = ls[~i2]  #  "left" rectangles
        l2 = ls[i2]  #  "right" rectangles

        if len(l1) <= 1 or len(l2) <= 1:
            continue

        xl[n1, :] = np.min(rp[l1[:, None], lv], axis=0)
        xr[n1, :] = np.max(rp[l1[:, None], rv], axis=0)
        xl[n2, :] = np.min(rp[l2[:, None], lv], axis=0)
        xr[n2, :] = np.max(rp[l2[:, None], rv], axis=0)
        if len(li) <= op["nobj"]:
            vi = np.prod(xr[ni_node, :] - xl[ni_node, :])  #  upper d-dim "vol."
            v1 = np.prod(xr[n1, :] - xl[n1, :])  # lower d-dim "vol."
            v2 = np.prod(xr[n2, :] - xl[n2, :])

            if v1 + v2 < op["vtol"] * vi:
                ii[n1, 0] = ni_node
                ii[n2, 0] = ni_node
                ii[ni_node, 1] = n1
                ll[ni_node] = lp
                ll[n1] = l1
                ll[n2] = l2

                ss[ns] = n1
                ss[ns + 1] = n2
                ns += 2
                nn += 2
        else:
            ii[n1, 0] = ni_node
            ii[n2, 0] = ni_node
            ii[ni_node, 1] = n1
            ll[ni_node] = lp
            ll[n1] = l1
            ll[n2] = l2

            ss[ns] = n1
            ss[ns + 1] = n2
            ns += 2
            nn += 2
    xl = xl[:nn, :]
    xr = xr[:nn, :]
    ii = ii[:nn, :]
    ll = ll[:nn]
    tr["xx"] = np.hstack((xl, xr))
    tr["ii"] = ii
    tr["ll"] = ll

    return tr
