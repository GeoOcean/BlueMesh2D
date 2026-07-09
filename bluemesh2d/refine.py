import math
import time

import numpy as np

from .aabb_tree.findball import findball
from .mesh_ball.cdtbal1 import cdtbal1
from .mesh_ball.cdtbal2 import cdtbal2
from .mesh_util.deltri import deltri
from .mesh_util.isfeat import isfeat
from .mesh_util.minlen import minlen
from .mesh_util.setset import setset
from .mesh_util.tricon import tricon


def refine(node=None, edge=None, part=None, opts=None, hfun=None, *harg):
    """Perform (Frontal-)Delaunay refinement for 2D polygonal geometries.

    Build a constrained Delaunay triangulation of the region defined by
    ``node`` and ``edge``.

    Parameters
    ----------
    node : ndarray of shape (N, 2), optional
        XY coordinates of polygon vertices.
    edge : ndarray of shape (E, 2), optional
        Edge connectivity as vertex-index pairs. If omitted, ``node`` vertices
        are connected sequentially.
    part : list of ndarray, optional
        For multi-region geometries, each entry lists edge indices defining
        one polygonal subdomain.
    opts : dict, optional
        Refinement options (defaults applied via :func:`makeopt`):

        - ``kind`` : ``{'delfront', 'delaunay'}``, default ``'delfront'``
        - ``rho2`` : float, default ``1.025`` — maximum radius–edge ratio
        - ``ref1``, ``ref2`` : ``{'refine', 'preserve'}``, default ``'refine'``
        - ``siz1`` : float, default ``1.333`` — edge relative-length threshold
        - ``siz2`` : float, default ``1.300`` — triangle relative-length threshold
        - ``disp`` : int or float, default ``10`` — progress interval; ``np.inf`` for quiet
    hfun : float or callable, optional
        Mesh-size function or constant spacing. If callable, must accept
        ``vert`` ``(N, 2)`` and return ``hvrt`` ``(N,)`` (fully vectorized).
    *harg : tuple, optional
        Extra arguments passed to ``hfun``.

    Returns
    -------
    vert : ndarray of shape (V, 2)
        Triangulation vertex coordinates.
    edge : ndarray of shape (E, 2)
        Constrained edges.
    tria : ndarray of shape (T, 3)
        Triangle connectivity (vertex indices).
    tnum : ndarray of shape (T,)
        Part index for each triangle.

    Notes
    -----
    Supports classical Delaunay-refinement and Frontal-Delaunay variants.
    The Frontal-Delaunay method follows the approach used in the JIGSAW library.

    References
    ----------
    D. Engwirda (2014), *Locally-optimal Delaunay-refinement and
    optimisation-based mesh generation*, PhD Thesis, University of Sydney.
    http://hdl.handle.net/2123/13148

    Translation of the MESH2D function ``REFINE2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    node = np.array([]) if node is None else node
    PSLG = np.array([]) if edge is None else edge
    part = [] if part is None else part
    hfun = [] if hfun is None else hfun

    opts = makeopt({} if opts is None else opts)

    nnod = node.shape[0]

    if PSLG.size == 0:
        PSLG = np.vstack(
            [
                np.column_stack([np.arange(0, nnod - 1), np.arange(1, nnod)]),
                np.array([[nnod - 1, 0]]),
            ]
        )

    ncon = PSLG.shape[0]
    if len(part) == 0:
        part = [np.arange(ncon)]

    if (
        not isinstance(node, np.ndarray)
        or not isinstance(PSLG, np.ndarray)
        or not isinstance(part, list)
        or not isinstance(opts, dict)
    ):
        raise TypeError("refine:incorrectInputClass")

    if node.ndim != 2 or PSLG.ndim != 2:
        raise ValueError("refine:incorrectDimensions")
    if node.shape[1] < 2 or PSLG.shape[1] < 2:
        raise ValueError("refine:incorrectDimensions")

    if PSLG.min() < 0 or PSLG.max() > nnod:
        raise ValueError("refine:invalid EDGE input array")

    for p in part:
        if np.min(p) < 0 or np.max(p) >= ncon:
            raise ValueError("refine:invalid PART input array")

    PSLG_sorted = np.sort(PSLG, axis=1)
    _, ivec, jvec = np.unique(
        PSLG_sorted, axis=0, return_index=True, return_inverse=True
    )
    PSLG = PSLG[ivec, :]

    newpart = []
    for p in part:
        newpart.append(np.unique(jvec[np.array(p)]))
    part = newpart

    for p in part:
        eloc = PSLG[p, :]
        # cast only for bincount computation
        eloc_int = eloc.astype(np.int64, copy=False)
        nadj = np.bincount(eloc_int.ravel(), minlength=nnod)
        if np.any(nadj % 2 != 0):
            raise ValueError("refine:nonmanifoldInputs")

    if not np.isinf(opts["disp"]):
        print("\n Refine triangulation...\n")
        print(" -------------------------------------------------------")
        print("      |ITER.|          |CDT1(X)|          |CDT2(X)|     ")
        print(" -------------------------------------------------------")

    vert = node.copy()
    tria = np.zeros((0, 3), dtype=int)
    tnum = np.zeros((0,), dtype=int)
    iter = 0
    conn = PSLG.copy()

    vmin = vert.min(axis=0)  # inflate bbox for stability
    vmax = vert.max(axis=0)
    vdel = vmax - 1.0 * vmin
    vmin = vmin - 0.5 * vdel
    vmax = vmax + 0.5 * vdel

    vbox = np.array(
        [[vmin[0], vmin[1]], [vmax[0], vmin[1]], [vmax[0], vmax[1]], [vmin[0], vmax[1]]]
    )
    vert = np.vstack([vert, vbox])


    vert, conn, tria, tnum, iter = cdtbal0(
        vert, conn, tria, tnum, node, PSLG, part, opts, hfun, harg, iter
    )

    vert, conn, tria, tnum, iter = cdtref1(
        vert, conn, tria, tnum, node, PSLG, part, opts, hfun, harg, iter
    )


    vert, conn, tria, tnum, iter = cdtref2(
        vert, conn, tria, tnum, node, PSLG, part, opts, hfun, harg, iter
    )

    if not np.isinf(opts["disp"]):
        print("")

    tria = tria[:, :3]

    keep = np.zeros(vert.shape[0], dtype=bool)
    keep[tria.ravel()] = True
    keep[conn.ravel()] = True

    redo = np.zeros(vert.shape[0], dtype=int)
    redo[keep] = np.arange(0, np.sum(keep))

    conn = (redo[conn]).reshape(-1, 2)
    tria = (redo[tria]).reshape(-1, 3)
    vert = vert[keep, :]

    return vert, conn, tria, tnum


def cdtbal0(vert, conn, tria, tnum, node, PSLG, part, opts, hfun, harg, iter):
    """Refine sharp PSLG vertices with collar points.

    Subdivide edges incident to acute features by inserting collar vertices
    sized from local edge lengths and mesh-spacing constraints.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Current vertex coordinates.
    conn : ndarray of shape (E, 2)
        Constrained edge connectivity.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.
    tnum : ndarray of shape (T,)
        Part indices per triangle.
    node : ndarray of shape (N0, 2)
        Original PSLG vertex coordinates.
    PSLG : ndarray of shape (E0, 2)
        Original PSLG edge indices.
    part : list of ndarray
        Polygon part edge-index lists.
    opts : dict
        Refinement options.
    hfun : float or callable, optional
        Mesh-size function or constant spacing.
    harg : tuple
        Extra arguments for ``hfun``.
    iter : int
        Current iteration counter.

    Returns
    -------
    vert : ndarray of shape (N', 2)
        Updated vertex coordinates.
    conn : ndarray of shape (E', 2)
        Updated constrained edges.
    tria : ndarray of shape (T', 3)
        Updated triangle connectivity.
    tnum : ndarray of shape (T',)
        Updated part indices.
    iter : int
        Unchanged iteration counter.

    References
    ----------
    Translation of the MESH2D function ``CDTBAL0``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    if iter <= opts["iter"]:
        vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part, opts["dtri"])
        edge, tria = tricon(tria, conn)
        feat, ftri = isfeat(vert, edge, tria)
        apex = np.zeros(vert.shape[0], dtype=bool)
        apex[tria[:, :3][ftri].ravel()] = True
        if hfun is not None and len(np.atleast_1d(hfun)) > 0:
            if isinstance(hfun, (int, float, np.number)):
                vlen = hfun * np.ones((vert.shape[0],), dtype=float)
            else:
                vlen = hfun(vert, *harg)
                vlen = np.ravel(vlen)
        else:
            vlen = np.full((vert.shape[0],), np.inf)

        conn = np.sort(conn, axis=1)
        evec = vert[conn[:, 1], :] - vert[conn[:, 0], :]
        elen = np.sqrt(np.sum(evec**2, axis=1))

        # avoid division by zero (mirrors MATLAB's behaviour)
        eps = np.finfo(float).eps
        mask_zero = elen < eps
        if np.any(mask_zero):
            elen[mask_zero] = eps

        evec = evec / np.column_stack([elen, elen])
        for epos in range(conn.shape[0]):
            ivrt = conn[epos, 0]
            jvrt = conn[epos, 1]
            vlen[ivrt] = min(vlen[ivrt], 0.67 * elen[epos])
            vlen[jvrt] = min(vlen[jvrt], 0.67 * elen[epos])

        iref = apex[conn[:, 0]] & ~apex[conn[:, 1]]  # refine at vert. 1
        jref = apex[conn[:, 1]] & ~apex[conn[:, 0]]  # refine at vert. 2
        dref = apex[conn[:, 0]] & apex[conn[:, 1]]  # refine at both!
        keep = ~apex[conn[:, 0]] & ~apex[conn[:, 1]]  # refine at neither
        ilen = vlen[conn[iref, 0]]
        inew = vert[conn[iref, 0], :] + np.column_stack([ilen, ilen]) * evec[iref, :]

        jlen = vlen[conn[jref, 1]]
        jnew = vert[conn[jref, 1], :] - np.column_stack([jlen, jlen]) * evec[jref, :]

        Ilen = vlen[conn[dref, 0]]
        Inew = vert[conn[dref, 0], :] + np.column_stack([Ilen, Ilen]) * evec[dref, :]

        Jlen = vlen[conn[dref, 1]]
        Jnew = vert[conn[dref, 1], :] - np.column_stack([Jlen, Jlen]) * evec[dref, :]

        vnew = np.vstack([inew, jnew, Inew, Jnew])

        iset = np.arange(0, inew.shape[0]) + vert.shape[0]

        jset = np.arange(0, jnew.shape[0]) + inew.shape[0] + vert.shape[0]

        Iset = (
            np.arange(0, Inew.shape[0]) + inew.shape[0] + jnew.shape[0] + vert.shape[0]
        )

        Jset = (
            np.arange(0, Jnew.shape[0])
            + inew.shape[0]
            + jnew.shape[0]
            + Inew.shape[0]
            + vert.shape[0]
        )

        vert = np.vstack([vert, vnew])

        cnew = np.vstack(
            [
                np.column_stack([conn[iref, 0], iset]),
                np.column_stack([conn[iref, 1], iset]),
                np.column_stack([conn[jref, 1], jset]),
                np.column_stack([conn[jref, 0], jset]),
                np.column_stack([conn[dref, 0], Iset]),
                np.column_stack([conn[dref, 1], Jset]),
                np.column_stack([Iset, Jset]),
            ]
        )

        conn = np.vstack([conn[keep, :], cnew])

    return vert, conn, tria, tnum, iter


def cdtref1(vert, conn, tria, tnum, node, PSLG, part, opts, hfun, harg, iter):
    """Refine 1-simplex (edge) elements until constraints are satisfied.

    Split encroached or oversized edges using Delaunay-refinement (midpoint
    splits) or Frontal-Delaunay (locally optimal off-centre splits), per
    ``opts['kind']``.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Vertex coordinates.
    conn : ndarray of shape (E, 2)
        Constrained edge connectivity.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.
    tnum : ndarray of shape (T,)
        Part index per triangle.
    node : ndarray of shape (N0, 2)
        Original PSLG vertex coordinates.
    PSLG : ndarray of shape (E0, 2)
        Original PSLG edge indices.
    part : list of ndarray
        Polygon part edge-index lists.
    opts : dict
        Refinement options (``kind``, ``siz1``, ``ref1``, ``disp``, …).
    hfun : float or callable, optional
        Mesh-size function or constant spacing.
    harg : tuple
        Extra arguments for ``hfun``.
    iter : int
        Current iteration counter.

    Returns
    -------
    vert : ndarray of shape (N', 2)
        Updated vertex coordinates.
    conn : ndarray of shape (E', 2)
        Updated constrained edges.
    tria : ndarray of shape (T', 3)
        Updated triangle connectivity.
    tnum : ndarray of shape (T',)
        Updated part indices.
    iter : int
        Updated iteration counter.

    References
    ----------
    Translation of the MESH2D function ``CDTREF1``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    # timers
    tcpu = {"full": 0.0, "ball": 0.0, "hfun": 0.0, "encr": 0.0, "offc": 0.0}

    vidx = np.arange(vert.shape[0])  # "new" vert list to test
    tnow = time.time()

    ntol = 1.55

    while opts["ref1"].lower() == "refine":
        iter += 1
        if iter >= opts["iter"]:
            break

        ttic = time.time()
        bal1 = cdtbal1(vert, conn)
        tcpu["ball"] += time.time() - ttic

        ttic = time.time()
        if hfun is not None and len(np.atleast_1d(hfun)) > 0:
            if isinstance(hfun, (int, float, np.number)):
                fun0 = hfun * np.ones((vert.shape[0],), dtype=float)
                fun1 = hfun
            else:
                if "fun0" not in locals():
                    fun0 = np.zeros((vert.shape[0],))
                if np.max(vidx) >= len(fun0):
                    new_size = np.max(vidx) + 1
                    extended = np.full((new_size,), np.nan, dtype=float)
                    extended[: len(fun0)] = fun0
                    fun0 = extended
                fun0[vidx] = np.ravel(hfun(vert[vidx, :], *harg))
                fun1 = (fun0[conn[:, 0]] + fun0[conn[:, 1]]) / 2.0
        else:
            fun0 = np.full((vert.shape[0],), np.inf)
            fun1 = np.inf

        siz1 = 4.0 * bal1[:, 2] / (fun1 * fun1)
        tcpu["hfun"] += time.time() - ttic

        ttic = time.time()
        bal1[:, 2] = (1.0 - np.finfo(float).eps ** 0.75) * bal1[:, 2]

        vp, vi, _ = findball(bal1, vert[:, 0:2])
        nexti = 0
        ebad = np.zeros((conn.shape[0],), dtype=bool)
        near = np.zeros((conn.shape[0], 2), dtype=int)

        for ii in range(vp.shape[0]):
            for ip in range(vp[ii, 0], vp[ii, 1] + 1):
                jj = vi[ip]
                if ii != conn[jj, 0] and ii != conn[jj, 1]:
                    if nexti >= len(near):
                        near = np.vstack((near, np.array([[ii, jj]])))
                    else:
                        near[nexti, 0] = ii
                        near[nexti, 1] = jj
                    nexti += 1

        near = near[:nexti, :]

        if near.shape[0] > 0:
            """
            mark edge "encroached" if there is a vert within its
            dia.-ball that is not joined to either of its vert's
            via an existing edge...
            """
            ivrt = conn[near[:, 1], 0]
            jvrt = conn[near[:, 1], 1]

            pair = np.column_stack([near[:, 0], ivrt])
            ivec, _ = setset(pair, conn)

            pair = np.column_stack([near[:, 0], jvrt])
            jvec, _ = setset(pair, conn)

            okay = ~ivec & ~jvec
            ebad[near[okay, 1]] = True

        tcpu["encr"] += time.time() - ttic

        ref1 = np.zeros((conn.shape[0],), dtype=bool)  # edge encroachment
        ref1[ebad] = True
        ref1[siz1 > opts["siz1"] * opts["siz1"]] = True  # bad equiv. length

        num1 = np.where(ref1)[0]

        if iter % opts["disp"] == 0:
            numc = conn.shape[0]
            numt = tria.shape[0]
            print(f"{iter:11d} {numc:18d} {numt:18d}")

        if num1.size == 0:
            break

        if opts["kind"].lower() == "delaunay":
            new1 = bal1[ref1, 0:2]

            vidx = np.arange(new1.shape[0]) + vert.shape[0]

            cnew = np.vstack(
                [
                    np.column_stack([conn[ref1, 0], vidx]),
                    np.column_stack([conn[ref1, 1], vidx]),
                ]
            )

            conn = np.vstack([conn[~ref1, :], cnew])
            vert = np.vstack([vert, new1[:, 0:2]])

        elif opts["kind"].lower() == "delfront":
            """
            symmetric off-centre scheme:- refine edges from both
            ends simultaneously, placing new vertices to satisfy
            the worst of mesh-spacing and local voronoi constra-
            ints.
            """

            ttic = time.time()

            evec = vert[conn[ref1, 1], :] - vert[conn[ref1, 0], :]
            elen = np.sqrt(np.sum(evec**2, axis=1))
            evec = evec / np.column_stack([elen, elen])
            vlen = np.sqrt(bal1[ref1, 2])
            ihfn = fun0[conn[ref1, 0]]
            jhfn = fun0[conn[ref1, 1]]
            ilen = np.minimum(vlen, ihfn)
            jlen = np.minimum(vlen, jhfn)

            inew = vert[conn[ref1, 0], :] + np.column_stack([ilen, ilen]) * evec
            jnew = vert[conn[ref1, 1], :] - np.column_stack([jlen, jlen]) * evec
            for _ in range(3):
                if hfun is not None and len(np.atleast_1d(hfun)) > 0:
                    if isinstance(hfun, (int, float, np.number)):
                        iprj = hfun * np.ones((inew.shape[0],))
                        jprj = hfun * np.ones((jnew.shape[0],))
                    else:
                        iprj = np.ravel(hfun(inew, *harg))
                        jprj = np.ravel(hfun(jnew, *harg))
                else:
                    iprj = np.full((inew.shape[0],), np.inf)
                    jprj = np.full((jnew.shape[0],), np.inf)

                iprj = 0.5 * ihfn + 0.5 * iprj
                jprj = 0.5 * jhfn + 0.5 * jprj
                ilen = np.minimum(vlen, iprj)
                jlen = np.minimum(vlen, jprj)
                inew = vert[conn[ref1, 0], :] + np.column_stack([ilen, ilen]) * evec
                jnew = vert[conn[ref1, 1], :] - np.column_stack([jlen, jlen]) * evec
            near = ilen + jlen >= vlen * ntol

            znew = 0.5 * (inew[near, :] + jnew[near, :])
            inew = inew[~near, :]
            jnew = jnew[~near, :]
            zset = np.arange(znew.shape[0]) + vert.shape[0]
            iset = np.arange(inew.shape[0]) + znew.shape[0] + vert.shape[0]
            jset = (
                np.arange(jnew.shape[0]) + znew.shape[0] + inew.shape[0] + vert.shape[0]
            )

            set1 = num1[near]
            set2 = num1[~near]

            cnew = np.vstack(
                [
                    np.column_stack([conn[set1, 0], zset]),
                    np.column_stack([conn[set1, 1], zset]),
                    np.column_stack([conn[set2, 0], iset]),
                    np.column_stack([conn[set2, 1], jset]),
                    np.column_stack([iset, jset]),
                ]
            )

            conn = np.vstack([conn[~ref1, :], cnew])
            vert = np.vstack([vert, znew, inew, jnew])
            vidx = np.concatenate([zset, iset, jset])

            tcpu["offc"] += time.time() - ttic

    tcpu["full"] += time.time() - tnow

    if not np.isinf(opts["disp"]):
        numc = conn.shape[0]
        numt = tria.shape[0]
        print(f"{iter:11d} {numc:18d} {numt:18d}")

    if opts["dbug"]:
        print("\n 1-simplex REF. timer...\n")
        print(f" FULL: {tcpu['full']:.6f}")
        print(f" BALL: {tcpu['ball']:.6f}")
        print(f" HFUN: {tcpu['hfun']:.6f}")
        print(f" ENCR: {tcpu['encr']:.6f}")
        print(f" OFFC: {tcpu['offc']:.6f}\n")

    return vert, conn, tria, tnum, iter


def cdtref2(vert, conn, tria, tnum, node, PSLG, part, opts, hfun, harg, iter):
    """Refine 2-simplex (triangle) elements until constraints are satisfied.

    Split bad triangles (poor radius–edge ratio or oversized) using
    Delaunay-refinement (circumball splits) or Frontal-Delaunay (off-centre
    splits), per ``opts['kind']``.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Vertex coordinates.
    conn : ndarray of shape (E, 2)
        Constrained edge connectivity.
    tria : ndarray of shape (T, 3)
        Triangle connectivity.
    tnum : ndarray of shape (T,)
        Part index per triangle.
    node : ndarray of shape (N0, 2)
        Original PSLG vertex coordinates.
    PSLG : ndarray of shape (E0, 2)
        Original PSLG edge indices.
    part : list of ndarray
        Polygon part edge-index lists.
    opts : dict
        Refinement options (``kind``, ``siz2``, ``rho2``, ``ref2``, …).
    hfun : float or callable, optional
        Mesh-size function or constant spacing.
    harg : tuple
        Extra arguments for ``hfun``.
    iter : int
        Current iteration counter.

    Returns
    -------
    vert : ndarray of shape (N', 2)
        Updated vertex coordinates.
    conn : ndarray of shape (E', 2)
        Updated constrained edges.
    tria : ndarray of shape (T', 3)
        Updated triangle connectivity.
    tnum : ndarray of shape (T',)
        Updated part indices.
    iter : int
        Updated iteration counter.

    References
    ----------
    Translation of the MESH2D function ``CDTREF2``.
    Original MATLAB source: https://github.com/dengwirda/mesh2d
    """

    tcpu = {
        "full": 0.0,
        "dtri": 0.0,
        "tcon": 0.0,
        "ball": 0.0,
        "hfun": 0.0,
        "offc": 0.0,
        "filt": 0.0,
    }

    vidx = np.arange(vert.shape[0])  # "new" vert list

    tnow = time.time()
    near = 0.775

    while opts["ref2"].lower() == "refine":
        iter += 1

        ttic = time.time()
        nold = vert.shape[0]

        vert, conn, tria, tnum = deltri(vert, conn, node, PSLG, part, opts["dtri"])

        nnew = vert.shape[0]
        vidx = np.concatenate([vidx, np.arange(nold, nnew)])

        tcpu["dtri"] += time.time() - ttic
        ttic = time.time()
        edge, tria = tricon(tria, conn)
        tcpu["tcon"] += time.time() - ttic

        if iter >= opts["iter"]:
            break

        ttic = time.time()
        bal1 = cdtbal1(vert, conn)
        bal2 = cdtbal2(vert, edge, tria)
        len2, _ = minlen(vert, tria)
        rho2 = bal2[:, 2] / len2  # rad-edge ratio
        scr2 = rho2 * bal2[:, 2]
        tcpu["ball"] += time.time() - ttic

        ttic = time.time()
        if hfun is not None and len(np.atleast_1d(hfun)) > 0:
            if isinstance(hfun, (int, float, np.number)):
                fun0 = hfun * np.ones(vert.shape[0])
                fun2 = hfun
            else:
                if "fun0" not in locals():
                    fun0 = np.zeros((vert.shape[0],))
                if np.max(vidx) >= len(fun0):
                    new_size = np.max(vidx) + 1
                    extended = np.full((new_size,), np.nan, dtype=float)
                    extended[: len(fun0)] = fun0
                    fun0 = extended
                fun0[vidx] = hfun(vert[vidx, :], *harg).ravel()
                fun2 = (fun0[tria[:, 0]] + fun0[tria[:, 1]] + fun0[tria[:, 2]]) / 3.0
        else:
            fun0 = np.full(vert.shape[0], np.inf)
            fun2 = np.inf

        siz2 = 3.0 * bal2[:, 2] / (fun2 * fun2)
        tcpu["hfun"] += time.time() - ttic

        ref1 = np.zeros(conn.shape[0], dtype=bool)
        ref2 = np.zeros(tria.shape[0], dtype=bool)

        stri, _ = isfeat(vert, edge, tria)

        ref2[rho2 > opts["rho2"] * opts["rho2"]] = True  # bad rad-edge len.
        ref2[stri] = False
        ref2[siz2 > opts["siz2"] * opts["siz2"]] = True  # bad equiv. length

        num2 = np.where(ref2)[0]

        if iter % opts["disp"] == 0:
            numc = conn.shape[0]
            numt = tria.shape[0]
            print(f"{iter:11d} {numc:18d} {numt:18d}")

        if num2.size == 0:
            break

        scr2_sorted = scr2[num2]
        idx2 = np.argsort(-scr2_sorted)
        num2 = num2[idx2]

        if opts["kind"].lower() == "delaunay":
            new2 = np.zeros((len(num2), 3))
            new2[:, 0:2] = bal2[num2, 0:2]
            rmin = (
                len2[num2] * (1.0 - np.finfo(float).eps ** 0.75) ** 2
            )  # min. insert radii
            new2[:, 2] = np.maximum(bal2[num2, 2] * near**2, rmin)

        elif opts["kind"].lower() == "delfront":
            """
            off-centre scheme -- refine triangles by positioning
            new vertices along a local segment of the voronoi
            diagram, bounded by assoc. circmballs. New points
            are placed to satisfy the worst of local mesh-length
            and element-shape constraints.
            """
            lmin, emin = minlen(vert, tria[num2, :])
            ftri = np.zeros(len(num2), dtype=bool)
            epos = np.zeros(len(num2), dtype=int)
            tadj = np.zeros(len(num2), dtype=int)

            for ii in range(len(epos)):
                epos[ii] = tria[num2[ii], emin[ii] + 3]

            for enum in range(3):
                eidx = tria[num2, enum + 3]
                ftri = ftri | (edge[eidx, 4] > 0)

                ione = num2 != edge[eidx, 2]
                itwo = ~ione

                tadj[ione] = edge[eidx[ione], 2]
                tadj[itwo] = edge[eidx[itwo], 3]

                okay = tadj > 0
                tidx = tadj[okay]
                ftri[okay] = ftri[okay] | ~ref2[tidx]

            if not np.any(ftri):
                ftri[:] = True  # can this happen!?

            emid = (vert[edge[epos, 0], :] + vert[edge[epos, 1], :]) * 0.5
            elen = np.sqrt(lmin[:])
            vvec = bal2[num2, 0:2] - emid
            vlen = np.sqrt(np.sum(vvec**2, axis=1))
            vvec = vvec / np.column_stack([vlen, vlen])

            hmid = (fun0[edge[epos, 0]] + fun0[edge[epos, 1]]) * 0.5
            rtri = elen * opts["off2"]
            rfac = elen * 0.5
            dsqr = rtri**2 - rfac**2
            doff = rtri + np.sqrt(np.maximum(0.0, dsqr))
            dsiz = np.sqrt(3.0) / 2.0 * hmid
            dist = np.minimum.reduce([dsiz, doff, vlen])
            off2 = emid + np.column_stack([dist, dist]) * vvec

            for _ in range(3):
                if hfun is not None and len(np.atleast_1d(hfun)) > 0:
                    if isinstance(hfun, (int, float, np.number)):
                        hprj = hfun * np.ones(off2.shape[0])
                    else:
                        hprj = hfun(off2, *harg).ravel()
                else:
                    hprj = np.full(off2.shape[0], np.inf)
                hprj = 0.33 * hmid + 0.67 * hprj
                dsiz = np.sqrt(3.0) / 2.0 * hprj
                dsiz[dsiz < elen * 0.50] = np.inf  # edge-ball limiter
                dsiz[dsiz > vlen * 0.95] = np.inf  # circ-ball limiter
                dist = np.minimum.reduce([dsiz, doff, vlen])
                off2 = emid + np.column_stack([dist, dist]) * vvec

            orad = np.sqrt((elen * 0.5) ** 2 + dist**2)
            new2 = np.zeros((np.count_nonzero(ftri), 3))
            new2[:, 0:2] = off2[ftri, 0:2]
            rmin = (
                lmin[ftri] * (1.0 - np.finfo(float).eps ** 0.75) ** 2
            )  # min. insert radii
            new2[:, 2] = np.maximum((orad[ftri] * near) ** 2, rmin)

            tcpu["offc"] += time.time() - ttic

        ttic = time.time()
        vp, vi, _ = findball(new2, new2[:, 0:2])

        keep = np.ones(new2.shape[0], dtype=bool)
        for ii in range(vp.shape[0] - 1, -1, -1):
            for ip in range(vp[ii, 0], vp[ii, 1] + 1):
                jj = vi[ip]
                if keep[jj] and keep[ii] and jj < ii:
                    keep[ii] = False
                    break
        new2 = new2[keep, :]

        bal1[:, 2] = (1.0 - np.finfo(float).eps ** 0.75) * bal1[:, 2]
        vp, vi, _ = findball(bal1, new2[:, 0:2])
        keep = np.ones(new2.shape[0], dtype=bool)
        for ii in range(vp.shape[0]):
            for ip in range(vp[ii, 0], vp[ii, 1] + 1):
                jj = vi[ip]
                ref1[jj] = True
                keep[ii] = False
        ebnd = np.zeros(edge.shape[0], dtype=bool)
        ebnd[tria[stri, 3:6].ravel()] = True

        enot, _ = setset(conn, edge[ebnd, 0:2])

        ref1[enot] = False

        if opts["ref1"].lower() == "preserve":
            ref1[:] = False

        new2 = new2[keep, :]
        new1 = bal1[ref1, :]
        idx1 = np.arange(new1.shape[0]) + vert.shape[0]
        idx2 = np.arange(new2.shape[0]) + new1.shape[0] + vert.shape[0]

        cnew = np.vstack(
            [
                np.column_stack([conn[ref1, 0], idx1]),
                np.column_stack([conn[ref1, 1], idx1]),
            ]
        )
        conn = np.vstack([conn[~ref1, :], cnew])
        vidx = np.concatenate([idx1, idx2])
        nold = vert.shape[0]
        vert = np.vstack([vert, new1[:, 0:2], new2[:, 0:2]])
        nnew = vert.shape[0]
        if nnew == nold:
            break  # we *must* be done

        tcpu["filt"] += time.time() - ttic

    tcpu["full"] += time.time() - tnow

    if not np.isinf(opts["disp"]):
        numc = conn.shape[0]
        numt = tria.shape[0]
        print(f"{iter:11d} {numc:18d} {numt:18d}")

    if opts["dbug"]:
        print("\n 2-simplex REF. timer...\n")
        for k, v in tcpu.items():
            print(f" {k.upper():5s}: {v:.6f}")
    return vert, conn, tria, tnum, iter


def makeopt(opts):
    """Set up and validate the options dictionary for :func:`refine`.

    Parameters
    ----------
    opts : dict
        User options; missing keys receive defaults.

    Returns
    -------
    opts : dict
        Validated options dictionary.
    """

    # dtri option
    if "dtri" not in opts:
        opts["dtri"] = "constrained"
    else:
        if opts["dtri"].lower() not in ["conforming", "constrained"]:
            raise ValueError("refine:invalidOption - Invalid constraint DTRI.")

    # kind option
    if "kind" not in opts:
        opts["kind"] = "delfront"
    else:
        if opts["kind"].lower() not in ["delfront", "delaunay"]:
            raise ValueError("refine:invalidOption - Invalid refinement KIND.")

    # ref1 option
    if "ref1" not in opts:
        opts["ref1"] = "refine"
    else:
        if opts["ref1"].lower() not in ["refine", "preserve"]:
            raise ValueError("refine:invalidOption - Invalid refinement REF1.")

    # ref2 option
    if "ref2" not in opts:
        opts["ref2"] = "refine"
    else:
        if opts["ref2"].lower() not in ["refine", "preserve"]:
            raise ValueError("refine:invalidOption - Invalid refinement REF2.")

    # iter option
    if "iter" not in opts:
        opts["iter"] = math.inf
    else:
        if not isinstance(opts["iter"], (int, float)):
            raise TypeError("refine:incorrectInputClass - Incorrect input class.")
        if not isinstance(opts["iter"], (int, float)):
            raise ValueError("refine:incorrectDimensions - Incorrect input dimensions.")
        if opts["iter"] <= 0:
            raise ValueError("refine:invalidOptionValues - Invalid OPT.ITER selection.")

    # disp option
    if "disp" not in opts:
        opts["disp"] = 10
    else:
        if not isinstance(opts["disp"], (int, float)):
            raise TypeError("refine:incorrectInputClass - Incorrect input class.")
        if not isinstance(opts["disp"], (int, float)):
            raise ValueError("refine:incorrectDimensions - Incorrect input dimensions.")
        if opts["disp"] <= 0:
            raise ValueError("refine:invalidOptionValues - Invalid OPT.DISP selection.")

    # rho2 option
    if "rho2" not in opts:
        opts["rho2"] = 1.025
    else:
        if not isinstance(opts["rho2"], (int, float)):
            raise TypeError("refine:incorrectInputClass - Incorrect input class.")
        if not isinstance(opts["rho2"], (int, float)):
            raise ValueError("refine:incorrectDimensions - Incorrect input dimensions.")
        if opts["rho2"] < 1.0:
            raise ValueError("refine:invalidOptionValues - Invalid OPT.RHO2 selection.")

    # off2 option
    if "off2" not in opts:
        opts["off2"] = 0.933
    else:
        if not isinstance(opts["off2"], (int, float)):
            raise TypeError("refine:incorrectInputClass - Incorrect input class.")
        if not isinstance(opts["off2"], (int, float)):
            raise ValueError("refine:incorrectDimensions - Incorrect input dimensions.")
        if opts["off2"] < 0.7:
            raise ValueError("refine:invalidOptionValues - Invalid OPT.OFF2 selection.")

    # siz1 option
    if "siz1" not in opts:
        opts["siz1"] = 1.333
    else:
        if not isinstance(opts["siz1"], (int, float)):
            raise TypeError("refine:incorrectInputClass - Incorrect input class.")
        if not isinstance(opts["siz1"], (int, float)):
            raise ValueError("refine:incorrectDimensions - Incorrect input dimensions.")
        if opts["siz1"] <= 0.0:
            raise ValueError("refine:invalidOptionValues - Invalid OPT.SIZ1 selection.")

    # siz2 option
    if "siz2" not in opts:
        opts["siz2"] = 1.300
    else:
        if not isinstance(opts["siz2"], (int, float)):
            raise TypeError("refine:incorrectInputClass - Incorrect input class.")
        if not isinstance(opts["siz2"], (int, float)):
            raise ValueError("refine:incorrectDimensions - Incorrect input dimensions.")
        if opts["siz2"] <= 0.0:
            raise ValueError("refine:invalidOptionValues - Invalid OPT.SIZ2 selection.")

    # dbug option
    if "dbug" not in opts:
        opts["dbug"] = False
    else:
        if not isinstance(opts["dbug"], (bool, np.bool_)):
            raise TypeError("refine:incorrectInputClass - Incorrect input class.")
        if not isinstance(opts["dbug"], (bool, np.bool_)):
            raise ValueError("refine:incorrectDimensions - Incorrect input dimensions.")

    return opts
