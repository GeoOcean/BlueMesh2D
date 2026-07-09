import time

import numpy as np
from scipy.sparse import csr_matrix

from .mesh_cost.triscr import triscr
from .mesh_util.deltri import deltri
from .mesh_util.setset import setset
from .mesh_util.tricon import tricon


def smooth(vert=None, conn=None, tria=None, tnum=None, opts=None, hfun=None, harg=[]):
    """Perform hill-climbing mesh smoothing for 2D triangle meshes.

    Optimize vertex positions and local topology via a spring-based update
    with monotonic quality guarantees.

    Parameters
    ----------
    vert : ndarray of shape (V, 2), optional
        Vertex coordinates.
    conn : ndarray of shape (E, 2), optional
        Constrained edges.
    tria : ndarray of shape (T, 3), optional
        Triangle connectivity (vertex indices).
    tnum : ndarray of shape (T, 1), optional
        Part index per triangle.
    opts : dict, optional
        Smoothing options (defaults via :func:`makeopt`):

        - ``vtol`` : float, default ``1.0e-2`` — relative vertex movement tolerance
        - ``iter`` : int, default ``32`` — maximum iterations
        - ``disp`` : int or float, default ``4`` — progress interval; ``np.inf`` for quiet
    hfun : callable, optional
        Mesh-size function for local edge-length control.
    harg : tuple, optional
        Extra arguments passed to ``hfun``.

    Returns
    -------
    vert : ndarray of shape (V, 2)
        Smoothed vertex coordinates.
    conn : ndarray of shape (E, 2)
        Updated constrained edges.
    tria : ndarray of shape (T, 3)
        Updated triangle connectivity.
    tnum : ndarray of shape (T, 1)
        Updated part indices.

    Notes
    -----
    Loosely based on the DISTMESH spring analogy with additional hill-climbing
    quality and density control. See P.-O. Persson and G. Strang (2004),
    *A Simple Mesh Generator in MATLAB*, SIAM Review 46(2): 329–345.

    References
    ----------
    Translation of the MESH2D function ``SMOOTH2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if vert is None:
        vert = np.empty((0, 2))
    if conn is None:
        conn = np.empty((0, 2), dtype=int)
    if tria is None:
        tria = np.empty((0, 3), dtype=int)
    if tnum is None:
        tnum = np.empty((0, 1), dtype=int)
    if opts is None:
        opts = {}

    opts = makeopt(opts)

    if conn.size == 0:
        edge, _ = tricon(tria)
        ebnd = edge[:, 3] < 1  # use boundary edge
        conn = edge[ebnd, 0:2]

    if tnum.size == 0:
        tnum = np.ones((tria.shape[0], 1), dtype=int)

    if not (
        isinstance(vert, np.ndarray)
        and isinstance(conn, np.ndarray)
        and isinstance(tria, np.ndarray)
        and isinstance(tnum, np.ndarray)
        and isinstance(opts, dict)
    ):
        raise TypeError("smooth:incorrectInputClass - Incorrect input class.")

    nvrt = vert.shape[0]

    if np.min(conn[:, :2]) < 0 or np.max(conn[:, :2]) > nvrt:
        raise ValueError("smooth:invalidInputs - Invalid EDGE input array.")

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) > nvrt:
        raise ValueError("smooth:invalidInputs - Invalid TRIA input array.")

    if not np.isinf(opts["disp"]):
        print("\n Smooth triangulation...\n")
        print(" -------------------------------------------------------")
        print("      |ITER.|          |MOVE(X)|          |DTRI(X)|     ")
        print(" -------------------------------------------------------")

    node = vert.copy()
    PSLG = conn.copy()
    pmax = int(np.max(tnum))
    part = [None for _ in range(pmax)]

    for ppos in range(pmax):
        tsel = tnum.flatten() == (ppos + 1)
        tcur = tria[tsel, :]
        ecur, tcur = tricon(tcur)
        ebnd = ecur[:, 3] == -1
        same, _ = setset(PSLG, ecur[ebnd, 0:2])
        part[ppos] = np.where(same)[0]

    vmin = np.min(vert, axis=0)
    vmax = np.max(vert, axis=0)

    vdel = vmax - 1.0 * vmin
    vmin = vmin - 0.5 * vdel
    vmax = vmax + 0.5 * vdel

    vbox = np.array(
        [
            [vmin[0], vmin[1]],
            [vmax[0], vmin[1]],
            [vmax[0], vmax[1]],
            [vmin[0], vmax[1]],
        ]
    )

    vert = np.vstack((vert, vbox))

    tnow = time.time()
    tcpu = {
        "full": 0.0,
        "dtri": 0.0,
        "tcon": 0.0,
        "iter": 0.0,
        "undo": 0.0,
        "keep": 0.0,
    }

    for iter in range(int(opts["iter"])):
        ttic = time.time()
        edge, tria = tricon(tria, conn)
        tcpu["tcon"] += time.time() - ttic

        oscr = triscr(vert, tria)

        ttic = time.time()
        nvrt = vert.shape[0]
        nedg = edge.shape[0]

        IMAT = csr_matrix(
            (np.ones(nedg), (edge[:, 0], np.arange(nedg))), shape=(nvrt, nedg)
        )
        JMAT = csr_matrix(
            (np.ones(nedg), (edge[:, 1], np.arange(nedg))), shape=(nvrt, nedg)
        )

        EMAT = IMAT + JMAT
        vdeg = np.array(EMAT.sum(axis=1)).flatten()  # vertex |deg|
        free = vdeg == 0

        vold = vert.copy()

        # Local iterations
        for isub in range(max(1, min(8, iter))):
            # compute HFUN at vert/midpoints
            hvrt = evalhfn(vert, edge, EMAT, hfun, harg)
            hmid = 0.5 * (hvrt[edge[:, 0]] + hvrt[edge[:, 1]])
            # calc. relative edge extensions
            evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
            elen = np.sqrt(np.sum(evec**2, axis=1))

            scal = 1.0 - elen / hmid
            scal = np.clip(scal, -1.0, 1.0)
            # projected points from each end
            ipos = vert[edge[:, 0], :] - 0.67 * (scal[:, None] * evec)
            jpos = vert[edge[:, 1], :] + 0.67 * (scal[:, None] * evec)
            scal = np.maximum(np.abs(scal) ** 1, np.finfo(float).eps ** 0.75)
            # sum contributions edge-to-vert
            vnew = IMAT.dot(scal[:, None] * ipos) + JMAT.dot(scal[:, None] * jpos)
            vsum = np.maximum(EMAT.dot(scal), np.finfo(float).eps ** 0.75)

            vnew = vnew / vsum[:, None]
            vnew[conn.flatten(), :] = vert[conn.flatten(), :]
            vnew[vdeg == 0, :] = vert[vdeg == 0, :]
            vert = vnew

        tcpu["iter"] += time.time() - ttic
        ttic = time.time()

        # unwind vert. upadte if score lower
        nscr = np.ones(tria.shape[0])
        btri = np.ones(tria.shape[0], dtype=bool)

        umax = 8
        for undo in range(umax):
            nscr[btri] = triscr(vert, tria[btri, :])

            # TRUE if tria needs "unwinding"
            smin = 0.70
            smax = 0.90
            sdel = 0.025

            stol = smin + iter * sdel
            stol = min(smax, stol)

            btri = (nscr <= stol) & (nscr < oscr)

            if not np.any(btri):
                break

            # relax toward old vert. coord's
            ivrt = np.unique(tria[btri, :3])
            bvrt = np.zeros(vert.shape[0], dtype=bool)
            bvrt[ivrt] = True

            if undo != umax:
                bnew = 0.75**undo
                bold = 1.0 - bnew
            else:
                bnew = 0.0
                bold = 1.0 - bnew

            vert[bvrt, :] = bold * vold[bvrt, :] + bnew * vert[bvrt, :]

            btri = np.any(bvrt[tria[:, :3]], axis=1)

        oscr = nscr
        tcpu["undo"] += time.time() - ttic

        ttic = time.time()

        vdel = np.sum((vert - vold) ** 2, axis=1)

        evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
        elen = np.sqrt(np.sum(evec**2, axis=1))

        hvrt = evalhfn(vert, edge, EMAT, hfun, harg)
        hmid = 0.5 * (hvrt[edge[:, 0]] + hvrt[edge[:, 1]])
        scal = elen / hmid

        emid = 0.5 * (vert[edge[:, 0], :] + vert[edge[:, 1], :])

        keep = np.zeros(vert.shape[0], dtype=bool)
        keep[vdeg > 4] = True
        keep[conn.flatten()] = True
        keep[free.flatten()] = True

        lmax = 5.0 / 4.0
        lmin = 1.0 / lmax

        less = scal <= lmin
        more = scal >= lmax

        vbnd = np.zeros(vert.shape[0], dtype=bool)
        vbnd[conn[:, 0]] = True
        vbnd[conn[:, 1]] = True

        ebad = vbnd[edge[:, 0]] | vbnd[edge[:, 1]]  # not at boundaries

        less[ebad] = False
        more[ebad] = False

        lidx = np.where(less)[0]
        for epos in lidx:
            inod = edge[epos, 0]
            jnod = edge[epos, 1]
            if keep[inod] and keep[jnod]:
                keep[inod] = False
                keep[jnod] = False
            else:
                less[epos] = False

        ebad = keep[edge[less, 0]] & keep[edge[less, 1]]
        indices = np.flatnonzero(ebad)
        indices = indices[indices < more.size]
        more[indices] = False
        # more[ebad] = False

        redo = np.zeros(vert.shape[0], dtype=int)
        itop = np.count_nonzero(keep)
        iend = np.count_nonzero(less)

        redo[keep] = np.arange(itop)
        redo[edge[less, 0]] = np.arange(itop, itop + iend)  # to new midpoints
        redo[edge[less, 1]] = np.arange(itop, itop + iend)

        vnew = np.vstack([vert[keep, :], emid[less, :]])
        tnew = redo[tria[:, 0:3]]

        ttmp = np.sort(tnew, axis=1)  # filter collapsed
        okay = np.all(np.diff(ttmp, axis=1) != 0, axis=1)
        okay = okay & (ttmp[:, 0] > 0)
        tnew = tnew[okay, :]

        nscr = triscr(vnew, tnew)

        stol = 0.80
        tbad = (nscr < stol) & (nscr < oscr[okay])

        vbad = np.zeros(vnew.shape[0], dtype=bool)
        vbad[tnew[tbad, :]] = True

        lidx = np.where(less)[0]
        ebad = vbad[redo[edge[lidx, 0]]] | vbad[redo[edge[lidx, 1]]]

        less[lidx[ebad]] = False
        keep[edge[lidx[ebad], 0:2].flatten()] = True

        redo = np.zeros(vert.shape[0], dtype=int)
        itop = np.count_nonzero(keep)
        iend = np.count_nonzero(less)

        redo[keep] = np.arange(itop)
        redo[edge[less, 0]] = np.arange(itop, itop + iend)
        redo[edge[less, 1]] = np.arange(itop, itop + iend)

        vert = np.vstack([vert[keep, :], emid[less, :], emid[more, :]])
        conn = redo[conn]

        tcpu["keep"] += time.time() - ttic

        ttic = time.time()
        vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part)
        tcpu["dtri"] += time.time() - ttic

        vdel = vdel / (hvrt.flatten() ** 2)
        move = vdel > opts["vtol"] ** 2

        nmov = np.count_nonzero(move)

        ntri = tria.shape[0]
        if iter % opts["disp"] == 0:
            print(f"{iter:11d} {nmov:18d} {ntri:18d}")

        if nmov == 0:
            break

    tria = tria[:, 0:3]

    keep = np.zeros(vert.shape[0], dtype=bool)
    keep[tria.flatten()] = True
    keep[conn.flatten()] = True

    redo = np.zeros(vert.shape[0], dtype=int)
    redo[keep] = np.arange(np.count_nonzero(keep))

    conn = redo[conn]
    tria = redo[tria]

    vert = vert[keep, :]

    tcpu["full"] += time.time() - tnow

    if opts["dbug"]:
        print("\n Mesh smoothing timer...\n")
        print(f" FULL: {tcpu['full']:.6f}")
        print(f" DTRI: {tcpu['dtri']:.6f}")
        print(f" TCON: {tcpu['tcon']:.6f}")
        print(f" ITER: {tcpu['iter']:.6f}")
        print(f" UNDO: {tcpu['undo']:.6f}")
        print(f" KEEP: {tcpu['keep']:.6f}\n")

    if not np.isinf(opts["disp"]):
        print("")

    return vert, conn, tria, tnum


def evalhfn(vert, edge, EMAT, hfun=None, harg=[]):
    """Evaluate the mesh-size function at mesh vertices.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Vertex coordinates.
    edge : ndarray of shape (E, 2)
        Edge connectivity.
    EMAT : scipy.sparse matrix
        Vertex–edge incidence matrix.
    hfun : float, callable, or None, optional
        Mesh-size function or constant spacing.
    harg : tuple, optional
        Extra arguments passed to ``hfun``.

    Returns
    -------
    hvrt : ndarray of shape (N,)
        Mesh-size value at each vertex.

    References
    ----------
    Translation of the MESH2D function ``EVALHFN``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # Default: mean edge length at each vertex (used when no hfun or as fallback)
    evec = vert[edge[:, 1], :] - vert[edge[:, 0], :]
    elen = np.sqrt(np.sum(evec**2, axis=1))
    default_hvrt = np.ravel(EMAT @ elen) / np.maximum(
        np.ravel(np.sum(EMAT, axis=1)), np.finfo(float).eps
    )
    free = np.ones(vert.shape[0], dtype=bool)
    free[edge[:, 0]] = False
    free[edge[:, 1]] = False
    default_hvrt[free] = np.inf

    if hfun is not None and (np.isscalar(hfun) or callable(hfun)):
        if np.isscalar(hfun):
            hvrt = hfun * np.ones(vert.shape[0])
        else:
            try:
                hvrt = np.asarray(hfun(vert, *harg)).flatten()
            except Exception:
                # hfun may fail on out-of-domain vertices (e.g. bbox) -> fall back to default
                hvrt = default_hvrt.copy()
            else:
                if hvrt.size != vert.shape[0]:
                    raise ValueError(
                        "smooth:evalhfn - hfun must return one value per vertex, "
                        f"got size {hvrt.size} for {vert.shape[0]} vertices."
                    )
                # out-of-domain vertices (e.g. bbox) may yield NaN/inf: fall back to default
                bad = ~np.isfinite(hvrt) | (hvrt <= 0)
                hvrt[bad] = default_hvrt[bad]
    else:
        hvrt = default_hvrt.copy()

    return hvrt


def makeopt(opts=None):
    """Set up and validate the options dictionary for :func:`smooth`.

    Parameters
    ----------
    opts : dict or None, optional
        User options; if ``None``, start from an empty dict.

    Returns
    -------
    opts : dict
        Validated options dictionary.

    References
    ----------
    Translation of the MESH2D function ``MAKEOPT``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if opts is None:
        opts = {}

    if "iter" not in opts:
        opts["iter"] = 32
    else:
        if not isinstance(opts["iter"], (int, float)):
            raise TypeError("smooth:incorrectInputClass - Incorrect input class.")
        if not (
            isinstance(opts["iter"], (int, float))
            and not isinstance(opts["iter"], bool)
        ):
            raise TypeError("smooth:incorrectInputClass - Incorrect input class.")
        if isinstance(opts["iter"], (list, tuple)) or hasattr(opts["iter"], "__len__"):
            raise ValueError("smooth:incorrectDimensions - Incorrect input dimensions.")
        if opts["iter"] <= 0:
            raise ValueError("smooth:invalidOptionValues - Invalid OPT.ITER selection.")

    if "disp" not in opts:
        opts["disp"] = 4
    else:
        if not isinstance(opts["disp"], (int, float)):
            raise TypeError("smooth:incorrectInputClass - Incorrect input class.")
        if isinstance(opts["disp"], (list, tuple)) or hasattr(opts["disp"], "__len__"):
            raise ValueError("smooth:incorrectDimensions - Incorrect input dimensions.")
        if opts["disp"] <= 0:
            raise ValueError("smooth:invalidOptionValues - Invalid OPT.DISP selection.")

    if "vtol" not in opts:
        opts["vtol"] = 1.0e-2
    else:
        if not isinstance(opts["vtol"], (int, float)):
            raise TypeError("smooth:incorrectInputClass - Incorrect input class.")
        if isinstance(opts["vtol"], (list, tuple)) or hasattr(opts["vtol"], "__len__"):
            raise ValueError("smooth:incorrectDimensions - Incorrect input dimensions.")
        if opts["vtol"] <= 0:
            raise ValueError("smooth:invalidOptionValues - Invalid OPT.VTOL selection.")

    if "dbug" not in opts:
        opts["dbug"] = False
    else:
        if not isinstance(opts["dbug"], bool):
            raise TypeError("refine:incorrectInputClass - Incorrect input class.")
        if isinstance(opts["dbug"], (list, tuple)) or hasattr(opts["dbug"], "__len__"):
            raise ValueError("refine:incorrectDimensions - Incorrect input dimensions.")

    return opts
