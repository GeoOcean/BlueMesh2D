import numpy as np


def inpoly_mat(vert, node, edge, fTOL, lbar):
    """
    INPOLY_MAT : local MATLAB implementation of the crossing-number test.

    Performs a "point-in-polygon" classification by looping over polygon
    edges and evaluating crossings efficiently using a binary search
    approach on the y-range of edges.

    Description
    -----------
    This function implements the local m-code version of the crossing-number
    algorithm used in `inpoly2`. For each polygon edge:
      - Performs a binary search to locate the first vertex whose y-coordinate
        intersects the edge’s y-range.
      - Executes crossing-number comparisons to determine ray–edge intersections.
      - Terminates once the y-range of the current edge is exceeded.

    Notes
    -----
    - The algorithm minimizes unnecessary edge–point intersection checks by
      restricting operations to relevant y-intervals.
    - Designed for improved efficiency compared to the naive O(N*M) version,
      where N is the number of test points and M the number of polygon edges.

    References
    ----------
    Translation of the MESH2D function `INPOLY2_MAT`.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    feps = fTOL * lbar**1
    veps = fTOL * lbar**1

    nvrt = vert.shape[0]
    _nnod = node.shape[0]
    nedg = edge.shape[0]

    stat = np.zeros(nvrt, dtype=bool)
    bnds = np.zeros(nvrt, dtype=bool)
    # ----------------------------------- loop over polygon edges
    for epos in range(nedg):
        inod = edge[epos, 0]
        jnod = edge[epos, 1]
        # ------------------------------ calc. edge bounding-box
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
        # ------------------------------- find top VERT(:,2)<YONE
        ilow = 0
        iupp = nvrt - 1

        while ilow < iupp - 1:  # --------binary search
            imid = ilow + (iupp - ilow) // 2
            if vert[imid, 1] < ymin:
                ilow = imid
            else:
                iupp = imid

        if vert[ilow, 1] >= ymin:
            ilow -= 1
        # ------------------------------- calc. edge-intersection
        for jpos in range(ilow + 1, nvrt):
            if bnds[jpos]:
                continue

            xpos = vert[jpos, 0]
            ypos = vert[jpos, 1]

            if ypos <= ymax:
                if xpos >= xmin:
                    if xpos <= xmax:
                        # ------------------- compute crossing number
                        mul1 = ydel * (xpos - xone)
                        mul2 = xdel * (ypos - yone)

                        if (feps * edel) >= abs(mul2 - mul1):
                            # ------------------- BNDS -- approx. on edge
                            bnds[jpos] = True
                            stat[jpos] = True
                        elif ypos == yone and xpos == xone:
                            # ------------------ BNDS -- match about ONE
                            bnds[jpos] = True
                            stat[jpos] = True
                        elif ypos == ytwo and xpos == xtwo:
                            # ------------------- BNDS -- match about TWO
                            bnds[jpos] = True
                            stat[jpos] = True
                        elif mul1 < mul2:
                            # ------------------- advance crossing number
                            if ypos >= yone and ypos < ytwo:
                                stat[jpos] = ~stat[jpos]

                else:
                    if ypos >= yone and ypos < ytwo:
                        # ------------------- advance crossing number
                        stat[jpos] = ~stat[jpos]
            else:
                break  # -- done -- due to the sort

    return stat, bnds
