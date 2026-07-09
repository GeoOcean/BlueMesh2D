import numpy as np


def inpoly_mat(vert, node, edge, fTOL, lbar):
    """Run the crossing-number point-in-polygon test on sorted points.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Query points, sorted by y-coordinate.
    node : ndarray of shape (M, 2)
        Polygon vertex coordinates.
    edge : ndarray of shape (P, 2)
        Edge connectivity as vertex-index pairs.
    fTOL : float
        Relative floating-point tolerance scale.
    lbar : float
        Characteristic polygon length scale.

    Returns
    -------
    stat : ndarray of bool, shape (N,)
        ``True`` for points classified as inside the polygon.
    bnds : ndarray of bool, shape (N,)
        ``True`` for points lying on a polygon edge.

    References
    ----------
    Translation of the MESH2D function ``INPOLY2_MAT``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    feps = fTOL * lbar**1
    veps = fTOL * lbar**1

    nvrt = vert.shape[0]
    _nnod = node.shape[0]
    nedg = edge.shape[0]

    stat = np.zeros(nvrt, dtype=bool)
    bnds = np.zeros(nvrt, dtype=bool)
    for epos in range(nedg):
        inod = edge[epos, 0]
        jnod = edge[epos, 1]
        yone = node[inod, 1]
        ytwo = node[jnod, 1]
        xone = node[inod, 0]
        xtwo = node[jnod, 0]

        xmin = min(xone, xtwo) - veps
        xmax = max(xone, xtwo) + veps

        ymin = yone - veps
        ymax = ytwo + veps

        ydel = ytwo - yone
        xdel = xtwo - xone

        edel = abs(xdel) + ydel
        ilow = 0
        iupp = nvrt - 1

        while ilow < iupp - 1:  # binary search
            imid = ilow + (iupp - ilow) // 2
            if vert[imid, 1] < ymin:
                ilow = imid
            else:
                iupp = imid

        if vert[ilow, 1] >= ymin:
            ilow -= 1
        for jpos in range(ilow + 1, nvrt):
            if bnds[jpos]:
                continue

            xpos = vert[jpos, 0]
            ypos = vert[jpos, 1]

            if ypos <= ymax:
                if xpos >= xmin:
                    if xpos <= xmax:
                        mul1 = ydel * (xpos - xone)
                        mul2 = xdel * (ypos - yone)

                        if (feps * edel) >= abs(mul2 - mul1):
                            bnds[jpos] = True
                            stat[jpos] = True
                        elif ypos == yone and xpos == xone:
                            bnds[jpos] = True
                            stat[jpos] = True
                        elif ypos == ytwo and xpos == xtwo:
                            bnds[jpos] = True
                            stat[jpos] = True
                        elif mul1 < mul2:
                            if ypos >= yone and ypos < ytwo:
                                stat[jpos] = ~stat[jpos]

                else:
                    if ypos >= yone and ypos < ytwo:
                        stat[jpos] = ~stat[jpos]
            else:
                break  # -- done -- due to the sort

    return stat, bnds
