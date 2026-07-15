"""Stage 4: PSLG + hfun -> triangular mesh (refine / smooth / smood)."""
from __future__ import annotations

import contextlib

from .dependencies import smood_dependencies
from .feedback import _LogWriter, _NullFeedback, _available_ram_bytes, _check


def _warn_if_mesh_too_big(node, edge, hfuns, feedback):
    """Estimate the refined mesh size and warn when it looks RAM-risky.

    The expected triangle count is ``~(2/sqrt(3)) * integral(dA / h^2)``,
    evaluated by sampling `hfuns` on a coarse grid over the PSLG polygons.
    Estimation errors are fine here -- this only decides whether to warn.
    """
    try:
        import numpy as np
        import shapely
        from shapely.ops import polygonize

        rings = polygonize(
            [((node[a][0], node[a][1]), (node[b][0], node[b][1]))
             for a, b in edge])
        area_geom = shapely.unary_union(list(rings))
        if area_geom.is_empty:
            return
        xmin, ymin, xmax, ymax = area_geom.bounds
        n = 128
        xs = np.linspace(xmin, xmax, n)
        ys = np.linspace(ymin, ymax, n)
        X, Y = np.meshgrid(xs, ys)
        xy = np.column_stack([X.ravel(), Y.ravel()])
        inside = shapely.contains_xy(area_geom, xy[:, 0], xy[:, 1])
        if not inside.any():
            return
        h = np.asarray(hfuns(xy[inside]), dtype=float)
        cell_area = area_geom.area / inside.sum()
        n_tria = 2.0 / np.sqrt(3.0) * cell_area * np.sum(1.0 / h ** 2)
        if n_tria > 500_000:
            msg = (f"The size function implies roughly {n_tria / 1e6:.1f} "
                   "million triangles; refinement may take a long time.")
            est_bytes = 1000.0 * n_tria  # ~1 kB per triangle during refine/smooth
            avail = _available_ram_bytes()
            if avail is not None and est_bytes > 0.5 * avail:
                msg += (f" Estimated memory ~{est_bytes / 1e9:.1f} GB of "
                        f"~{avail / 1e9:.1f} GB available -- QGIS may become "
                        "unresponsive or crash.")
            msg += (" Consider increasing Min element size or reducing the "
                    "domain.")
            feedback.pushWarning(msg)
    except Exception:
        pass  # a failed estimate must never block the run


def _locate_fixed(vert, fixed_points, feedback, tol=1e-6):
    """Return indices in ``vert`` of each fixed point (nearest within tol)."""
    import numpy as np

    idx = []
    for p in np.asarray(fixed_points, dtype=float):
        d2 = np.sum((vert - p) ** 2, axis=1)
        j = int(np.argmin(d2))
        if d2[j] <= tol * tol:
            idx.append(j)
        else:
            feedback.pushWarning(
                f"Fixed point ({p[0]:.3f}, {p[1]:.3f}) not found in mesh "
                f"(nearest node {np.sqrt(d2[j]):.3g} m away); skipped.")
    return np.asarray(idx, dtype=int)


def mesh_pslg(node, edge, hfuns, kind="delaunay", do_smooth=True,
              do_smood=False, smood_merge_small_links=False,
              fixed_points=None, feedback=None):
    """Refine a PSLG, then optionally smooth and/or smood it.

    Parameters
    ----------
    node : ndarray of shape (N, 2)
        PSLG node coordinates (working CRS).
    edge : ndarray of shape (E, 2)
        PSLG edges (0-based node indices).
    hfuns : callable
        Element-size function, ``hfuns(xy) -> h``.
    kind : {'delaunay', 'delfront'}, optional
        Refinement scheme passed to ``refine``. Default is ``'delaunay'``.
    do_smooth : bool, optional
        If ``True`` (default), run non-linear mesh optimisation (``smooth``)
        after refinement.
    do_smood : bool, optional
        If ``True``, additionally run orthogonalization (``smood``) after
        smoothing. Default is ``False``.
    smood_merge_small_links : bool, optional
        Enable the merge step inside smood's ortho-merge cycles (pairs of
        triangles whose circumcenters are too close are merged, then
        re-split). Use only when the default triangle-only smood cannot
        remove the remaining small flow links. Default is ``False``.
    fixed_points : ndarray of shape (K, 2), optional
        XY coordinates (working CRS) of points that must appear as mesh
        nodes at exactly these positions: they are inserted before
        refinement and pinned during smoothing and orthogonalization.
        Points outside the meshed domain or coincident with boundary
        nodes are ignored (with a warning).
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    vert : ndarray of shape (M, 2)
        Mesh vertex coordinates (working CRS).
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based vertex indices).
    """
    feedback = feedback or _NullFeedback()
    from bluemesh2d.refine import refine
    from bluemesh2d.smooth import smooth

    kind = str(kind).lower()
    if kind not in ("delaunay", "delfront"):
        raise ValueError("kind must be 'delaunay' or 'delfront'")

    import numpy as np

    if fixed_points is not None:
        fixed_points = np.asarray(fixed_points, dtype=float).reshape(-1, 2)
        if fixed_points.size:
            # drop fixed points (nearly) coincident with existing PSLG nodes:
            # a duplicate vertex would break the triangulation
            keep_fp = np.ones(fixed_points.shape[0], dtype=bool)
            for i, p in enumerate(fixed_points):
                if np.min(np.sum((node - p) ** 2, axis=1)) < 1e-6:
                    keep_fp[i] = False
                    feedback.pushWarning(
                        f"Fixed point ({p[0]:.3f}, {p[1]:.3f}) coincides with "
                        "a boundary node; skipped.")
            fixed_points = fixed_points[keep_fp]
        if fixed_points.size:
            feedback.pushInfo(f"Inserting {len(fixed_points)} fixed point(s)")
            node = np.vstack([node, fixed_points])
        else:
            fixed_points = None

    _warn_if_mesh_too_big(node, edge, hfuns, feedback)
    feedback.setProgress(5)

    feedback.pushInfo(f"Refining mesh ({len(node)} boundary nodes, kind={kind}) ...")
    with contextlib.redirect_stdout(_LogWriter(feedback)):
        vert, etri, tria, tnum = refine(node, edge, [], {"kind": kind}, hfuns)
    _check(feedback)
    feedback.pushInfo(f"Refined: {len(vert)} nodes, {len(tria)} triangles")
    feedback.setProgress(55)

    # refine never moves input nodes, so fixed points can be re-located by
    # coordinate after each stage (indices change with mesh compaction)
    fixed_idx = None
    if fixed_points is not None:
        fixed_idx = _locate_fixed(vert, fixed_points, feedback)
        # keep only the points actually present (e.g. outside the domain,
        # dropped by refine) so later stages don't re-warn about them
        fixed_points = vert[fixed_idx, :].copy()
        if fixed_points.size == 0:
            fixed_points = None
            fixed_idx = None

    if do_smooth:
        feedback.pushInfo("Smoothing mesh ...")
        with contextlib.redirect_stdout(_LogWriter(feedback)):
            vert, etri, tria, tnum = smooth(vert, etri, tria, tnum, {}, hfuns,
                                            fixed=fixed_idx)
        _check(feedback)
        if fixed_points is not None:
            fixed_idx = _locate_fixed(vert, fixed_points, feedback)
        feedback.setProgress(85)

    if do_smood:
        missing = smood_dependencies()
        if missing:
            raise RuntimeError(
                "smood (orthogonalization) requires: " + ", ".join(missing)
                + ". Install it, or disable the smood option.")
        feedback.pushInfo("Applying smood (orthogonalization) ...")
        from bluemesh2d.smood import smood
        smood_opts = {}
        if smood_merge_small_links:
            feedback.pushInfo("smood: small-link merging enabled")
            smood_opts["merge_small_links"] = True
        try:
            with contextlib.redirect_stdout(_LogWriter(feedback)):
                vert, etri, tria, tnum = smood(vert, etri, tria, tnum, smood_opts,
                                               fixed=fixed_idx)
        except ImportError as exc:
            raise RuntimeError(
                f"smood needs an optional package that is not installed: {exc}. "
                "Install it or disable the smood option.")
        _check(feedback)
        feedback.pushInfo(f"After smood: {len(vert)} nodes, {len(tria)} faces")

    return vert, tria


# UGRID fill values matching bluemesh2d.geomesh_util.grd_util
