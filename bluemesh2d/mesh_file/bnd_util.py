"""Boundary classification and boundary-condition / .grd / .pli exports."""
from __future__ import annotations

import os

from ..feedback import _NullFeedback
from .ugrid import read_ugrid_mesh


def export_boundary_conditions(nc_path, out_dir, zlim=20.0,
                               pli_name="Boundary01", bc_name="Riemann",
                               ext_name="FlowFM_bnd", feedback=None):
    """Write Delft3D-FM open-boundary files (.pli, .bc, .ext) for a mesh.

    Reproduces the notebook: boundary edges deeper than `zlim` are open; the
    **longest** open contour becomes the boundary polyline, with a Riemann
    time-series stanza per point.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    out_dir : str
        Output directory for the three files.
    zlim : float, optional
        Depth threshold (m); boundary edges deeper than `zlim` are
        classified open. Default is 20.0.
    pli_name, bc_name, ext_name : str, optional
        Base names (without extension) for the ``.pli``, ``.bc`` and ``.ext``
        files. Defaults are ``'Boundary01'``, ``'Riemann'`` and
        ``'FlowFM_bnd'``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    pli_path, bc_path, ext_path : str
        Paths to the three written files.

    Raises
    ------
    RuntimeError
        If no open boundary is found at the given threshold.
    """
    feedback = feedback or _NullFeedback()
    import os

    from bluemesh2d.geomesh_util.border_util import identify_boundary

    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    feedback.pushInfo(f"Identifying boundaries (open where depth > {zlim} m) ...")
    boundary = identify_boundary(vert, tria, z_depth, zlim=zlim)
    open_contours = boundary["open_contours"]
    if not open_contours:
        raise RuntimeError(
            f"No open boundary found with threshold {zlim} m; "
            "lower the threshold.")
    contour = max(open_contours, key=len)
    feedback.pushInfo(
        f"Open contours: {len(open_contours)}; using the longest "
        f"({len(contour)} points).")

    xb = vert[contour, 0]
    yb = vert[contour, 1]

    pli_path = os.path.join(out_dir, f"{pli_name}.pli")
    bc_path = os.path.join(out_dir, f"{bc_name}.bc")
    ext_path = os.path.join(out_dir, f"{ext_name}.ext")

    with open(pli_path, "w") as f_pli, open(bc_path, "w") as f_bc:
        f_pli.write(f"{pli_name}\n")
        f_pli.write(f"    {len(xb)}    2\n")
        for i, (xi, yi) in enumerate(zip(xb, yb)):
            boundary_id = f"{pli_name}_{i:04d}"
            f_pli.write(f"{xi:.15E}  {yi:.15E} {boundary_id}\n")
            f_bc.write("[forcing]\n")
            f_bc.write(f"Name = {boundary_id}\n")
            f_bc.write("Function = timeseries\n")
            f_bc.write("Time-interpolation = linear\n")
            f_bc.write("Quantity = time\n")
            f_bc.write("Unit = seconds since 2000-01-01 00:00:00\n")
            f_bc.write("Quantity = riemannbnd\n")
            f_bc.write("Unit = m\n")
            f_bc.write("0    0\n")
            f_bc.write("9999999999   0\n\n")

    with open(ext_path, "w") as f:
        f.write("[general]\n")
        f.write("fileVersion=2.01\n")
        f.write("fileType=extForce\n\n")
        f.write("[boundary]\n")
        f.write("quantity=riemannbnd\n")
        f.write(f"locationFile={pli_name}.pli\n")
        f.write(f"forcingFile={bc_name}.bc\n")

    return pli_path, bc_path, ext_path


def export_grd(nc_path, out_grd, zlim=20.0, crs="EPSG:4326", feedback=None):
    """Write an ADCIRC-style .grd with open/land boundary loops.

    Boundary edges deeper than `zlim` are tagged open, the rest land
    (``identify_boundary`` + ``export_to_grd``, as in the notebook).

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    out_grd : str
        Output ``.grd`` path.
    zlim : float, optional
        Depth threshold (m); boundary edges deeper than `zlim` are
        classified open. Default is 20.0.
    crs : str, optional
        CRS string written to the ``.grd`` header. Default is
        ``'EPSG:4326'``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    out_grd : str
        Path to the written file (same as the input `out_grd`).
    """
    feedback = feedback or _NullFeedback()
    from bluemesh2d.geomesh_util.border_util import identify_boundary
    from bluemesh2d.geomesh_util.grd_util import export_to_grd

    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    feedback.pushInfo(f"Identifying boundaries (open where depth > {zlim} m) ...")
    boundary = identify_boundary(vert, tria, z_depth, zlim=zlim)
    feedback.pushInfo(f"Writing .grd -> {out_grd}")
    export_to_grd(
        out_grd, vert=vert, tria=tria, z=z_depth, crs=crs,
        edge_tag=boundary["edge_tag"],
        edge_open=boundary["edge_open"],
        edge_land=boundary["edge_land"],
    )
    return out_grd


# ===========================================================================
# Boundary condition generation (editable open / closed / island lines)
# ===========================================================================

def _boundary_loops(vert, tria):
    """Assemble the mesh boundary (free) edges into ordered node loops.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Node coordinates.
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based).

    Returns
    -------
    loops : list of list of int
        One list of node indices per closed boundary loop (the first node is
        not repeated at the end).
    """
    import numpy as np
    from collections import defaultdict

    edges = np.vstack([tria[:, [0, 1]], tria[:, [1, 2]], tria[:, [2, 0]]])
    es = np.sort(edges, axis=1)
    uniq, counts = np.unique(es, axis=0, return_counts=True)
    free = uniq[counts == 1]

    adj = defaultdict(list)
    for a, b in free:
        adj[int(a)].append(int(b))
        adj[int(b)].append(int(a))

    def key(a, b):
        return (a, b) if a < b else (b, a)

    used = set()
    loops = []
    for a0, b0 in free:
        a0, b0 = int(a0), int(b0)
        if key(a0, b0) in used:
            continue
        used.add(key(a0, b0))
        loop = [a0]
        prev, cur = a0, b0
        while cur != a0:
            loop.append(cur)
            nbrs = [n for n in adj[cur] if key(cur, n) not in used]
            pref = [n for n in nbrs if n != prev]
            step = pref or nbrs
            if not step:
                break
            nxt = step[0]
            used.add(key(cur, nxt))
            prev, cur = cur, nxt
        loops.append(loop)
    return loops


def classify_boundary_lines(vert, tria, z_depth, zlim=20.0):
    """Split the mesh boundary into open / closed / island polylines.

    The boundary free edges are assembled into loops. Loops contained inside
    another loop are *islands* (all their edges are coastline). On each outer
    loop, an edge is *open* where the mean node depth exceeds `zlim`, otherwise
    *closed* (land); consecutive edges of the same class form one continuous
    polyline.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Node coordinates (mesh CRS).
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based).
    z_depth : ndarray of shape (N,)
        Node depth (positive down), as returned by :func:`read_ugrid_mesh`.
    zlim : float, optional
        Depth threshold (m); outer-boundary edges deeper than `zlim` are open.
        Default is 20.0.

    Returns
    -------
    lines : dict of {str: list of ndarray}
        Keys ``'open'``, ``'closed'`` and ``'island'``; each value is a list
        of ``(M, 2)`` coordinate arrays (polylines in the mesh CRS).
    """
    import numpy as np
    from shapely.geometry import Polygon

    loops = _boundary_loops(vert, tria)
    polys = [Polygon(vert[lp]) if len(lp) >= 3 else None for lp in loops]

    # a loop is an island if it lies inside another (larger) loop
    is_island = [False] * len(loops)
    for i, pi in enumerate(polys):
        if pi is None or not pi.is_valid:
            continue
        for j, pj in enumerate(polys):
            if i == j or pj is None or not pj.is_valid:
                continue
            if pj.area > pi.area and pj.contains(pi.representative_point()):
                is_island[i] = True
                break

    out = {"open": [], "closed": [], "island": []}
    for lp, island in zip(loops, is_island):
        coords = vert[lp]
        ring = np.vstack([coords, coords[0]])  # close the ring for display
        if island:
            out["island"].append(ring)
            continue

        n = len(lp)
        tags = [0.5 * (z_depth[lp[k]] + z_depth[lp[(k + 1) % n]]) > zlim
                for k in range(n)]
        if all(tags):
            out["open"].append(ring)
            continue
        if not any(tags):
            out["closed"].append(ring)
            continue

        # rotate so the walk starts at a class transition (avoids wrap-around)
        start = next(k for k in range(n) if tags[k] != tags[k - 1])
        eord = [(start + k) % n for k in range(n)]
        runs = [[eord[0]]]
        for e in eord[1:]:
            if tags[e] == tags[runs[-1][-1]]:
                runs[-1].append(e)
            else:
                runs.append([e])
        for run in runs:
            node_seq = [lp[run[0]]] + [lp[(e + 1) % n] for e in run]
            line = vert[node_seq]
            out["open" if tags[run[0]] else "closed"].append(line)
    return out


def classify_boundary_points(vert, tria, z_depth, zlim=20.0):
    """Classify each mesh boundary node as open / closed / island.

    The boundary free edges are assembled into loops (see
    :func:`_boundary_loops`). Loops contained inside another loop are
    *islands* (every node tagged ``'island'``). On outer loops, a node is
    ``'open'`` where its depth exceeds `zlim`, otherwise ``'closed'``.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Node coordinates (mesh CRS).
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based).
    z_depth : ndarray of shape (N,)
        Node depth (positive down), as returned by :func:`read_ugrid_mesh`.
    zlim : float, optional
        Depth threshold (m); outer-boundary nodes deeper than `zlim` are
        open. Default is 20.0.

    Returns
    -------
    loops : list of dict
        One dict per boundary loop, with keys ``'coords'`` (``(n, 2)``
        node coordinates, in walk order, first node not repeated),
        ``'btype'`` (list of ``n`` strings), ``'depth'`` (list of ``n``
        floats) and ``'island'`` (bool).
    """
    import numpy as np
    from shapely.geometry import Polygon

    loops = _boundary_loops(vert, tria)
    polys = [Polygon(vert[lp]) if len(lp) >= 3 else None for lp in loops]

    # a loop is an island if it lies inside another (larger) loop
    is_island = [False] * len(loops)
    for i, pi in enumerate(polys):
        if pi is None or not pi.is_valid:
            continue
        for j, pj in enumerate(polys):
            if i == j or pj is None or not pj.is_valid:
                continue
            if pj.area > pi.area and pj.contains(pi.representative_point()):
                is_island[i] = True
                break

    out = []
    for lp, island in zip(loops, is_island):
        depths = [float(z_depth[k]) for k in lp]
        if island:
            btype = ["island"] * len(lp)
        else:
            btype = ["open" if d > zlim else "closed" for d in depths]
        out.append({"coords": np.asarray(vert[lp], dtype=float),
                    "btype": btype, "depth": depths, "island": island})
    return out


def boundary_lines_from_points(loops):
    """Rebuild open / closed / island polylines from per-node classifications.

    Inverse companion of :func:`classify_boundary_points`, applied after the
    user has edited node types: consecutive edges of the same class form one
    polyline. An edge takes the type of its two nodes when they agree; at an
    open/other transition the edge is not open (the ``.pli`` open boundary
    only spans fully-open stretches), otherwise it takes its first node's
    type.

    Parameters
    ----------
    loops : list of (ndarray of shape (n, 2), list of str)
        Per loop: node coordinates in walk order (first node not repeated)
        and one type string per node.

    Returns
    -------
    lines : dict of {str: list of ndarray}
        Type -> list of ``(M, 2)`` coordinate polylines, as in
        :func:`classify_boundary_lines`.
    """
    import numpy as np

    out = {}
    for coords, btype in loops:
        coords = np.asarray(coords, dtype=float)
        n = len(coords)
        if n < 2:
            continue

        def edge_type(k):
            a, b = btype[k], btype[(k + 1) % n]
            if a == b:
                return a
            if a == "open":
                return b
            if b == "open":
                return a
            return a

        tags = [edge_type(k) for k in range(n)]
        if all(t == tags[0] for t in tags):
            ring = np.vstack([coords, coords[:1]])
            out.setdefault(tags[0], []).append(ring)
            continue

        # rotate so the walk starts at a class transition (avoids wrap-around)
        start = next(k for k in range(n) if tags[k] != tags[k - 1])
        eord = [(start + k) % n for k in range(n)]
        runs = [[eord[0]]]
        for e in eord[1:]:
            if tags[e] == tags[runs[-1][-1]]:
                runs[-1].append(e)
            else:
                runs.append([e])
        for run in runs:
            node_seq = [run[0]] + [(e + 1) % n for e in run]
            out.setdefault(tags[run[0]], []).append(coords[node_seq])
    return out


def generate_boundary_condition_points(nc_path, zlim=20.0, feedback=None):
    """Classify each mesh boundary node as open / closed / island.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    zlim : float, optional
        Depth threshold (m) for the initial open/closed split. Default 20.0.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    loops : list of dict
        See :func:`classify_boundary_points`.
    """
    feedback = feedback or _NullFeedback()
    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    feedback.pushInfo(f"Classifying boundary (open where depth > {zlim} m) ...")
    loops = classify_boundary_points(vert, tria, z_depth, zlim=zlim)
    n_island = sum(1 for lp in loops if lp["island"])
    n_pts = sum(len(lp["btype"]) for lp in loops)
    feedback.pushInfo(
        f"Boundary points: {n_pts} on {len(loops)} loop(s) "
        f"({n_island} island).")
    return loops


def generate_boundary_conditions(nc_path, zlim=20.0, feedback=None):
    """Classify a mesh's boundary into open / closed / island polylines.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    zlim : float, optional
        Depth threshold (m) for the initial open/closed split. Default 20.0.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    lines : dict of {str: list of ndarray}
        See :func:`classify_boundary_lines`.
    """
    feedback = feedback or _NullFeedback()
    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    feedback.pushInfo(f"Classifying boundary (open where depth > {zlim} m) ...")
    lines = classify_boundary_lines(vert, tria, z_depth, zlim=zlim)
    feedback.pushInfo(
        f"Boundary lines: {len(lines['open'])} open, "
        f"{len(lines['closed'])} closed, {len(lines['island'])} island.")
    return lines


def write_open_boundary_pli(out_dir, open_lines, pli_name="Boundary01",
                            feedback=None):
    """Write a Delft3D-FM ``.pli`` polyline file from open boundary polylines.

    Parameters
    ----------
    out_dir : str
        Output directory.
    open_lines : list of array_like of shape (M, 2)
        Open-boundary polylines (coordinates), e.g. the ``'open'`` features of
        the stage-5 boundary-condition layer.
    pli_name : str, optional
        Base name for the ``.pli`` file. Default ``'Boundary01'``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    pli_path : str
        Path to the written file.
    boundary_ids : list of list of str
        Per-line lists of the boundary point ids written to the ``.pli``
        file, in the same order/nesting as `open_lines` -- for use e.g. when
        writing matching ``.bc`` forcing blocks.

    Raises
    ------
    RuntimeError
        If `open_lines` is empty.
    """
    import os
    import numpy as np

    feedback = feedback or _NullFeedback()
    lines = [np.atleast_2d(np.asarray(ln, dtype=float)) for ln in open_lines
             if len(ln) >= 2]
    if not lines:
        raise RuntimeError(
            "No open boundary polyline provided; classify one in "
            "'5 - Generate boundary conditions' (or lower the depth threshold).")

    pli_path = os.path.join(out_dir, f"{pli_name}.pli")
    idx = 0
    boundary_ids = []
    with open(pli_path, "w") as f_pli:
        for li, line in enumerate(lines):
            block = pli_name if len(lines) == 1 else f"{pli_name}_{li:03d}"
            f_pli.write(f"{block}\n")
            f_pli.write(f"    {len(line)}    2\n")
            ids = []
            for xi, yi in line:
                boundary_id = f"{pli_name}_{idx:04d}"
                idx += 1
                ids.append(boundary_id)
                f_pli.write(f"{xi:.15E}  {yi:.15E} {boundary_id}\n")
            boundary_ids.append(ids)

    feedback.pushInfo(f"Open boundary file: {pli_path}")
    return pli_path, boundary_ids


def write_open_boundary_files(out_dir, open_lines, pli_name="Boundary01",
                              bc_name="Riemann", ext_name="FlowFM_bnd",
                              feedback=None):
    """Write Delft3D-FM open-boundary files from open boundary polylines.

    Parameters
    ----------
    out_dir : str
        Output directory.
    open_lines : list of array_like of shape (M, 2)
        Open-boundary polylines (coordinates), e.g. the ``'open'`` features of
        the stage-5 boundary-condition layer.
    pli_name, bc_name, ext_name : str, optional
        Base names for the ``.pli``, ``.bc`` and ``.ext`` files. Defaults
        ``'Boundary01'``, ``'Riemann'``, ``'FlowFM_bnd'``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    pli_path, bc_path, ext_path : str
        Paths to the three written files.

    Raises
    ------
    RuntimeError
        If `open_lines` is empty.
    """
    import os

    feedback = feedback or _NullFeedback()
    pli_path, boundary_ids = write_open_boundary_pli(
        out_dir, open_lines, pli_name=pli_name, feedback=feedback)

    bc_path = os.path.join(out_dir, f"{bc_name}.bc")
    with open(bc_path, "w") as f_bc:
        for ids in boundary_ids:
            for boundary_id in ids:
                f_bc.write("[forcing]\n")
                f_bc.write(f"Name = {boundary_id}\n")
                f_bc.write("Function = timeseries\n")
                f_bc.write("Time-interpolation = linear\n")
                f_bc.write("Quantity = time\n")
                f_bc.write("Unit = seconds since 2000-01-01 00:00:00\n")
                f_bc.write("Quantity = riemannbnd\n")
                f_bc.write("Unit = m\n")
                f_bc.write("0    0\n")
                f_bc.write("9999999999   0\n\n")

    ext_path = os.path.join(out_dir, f"{ext_name}.ext")
    with open(ext_path, "w") as f:
        f.write("[general]\n")
        f.write("fileVersion=2.01\n")
        f.write("fileType=extForce\n\n")
        f.write("[boundary]\n")
        f.write("quantity=riemannbnd\n")
        f.write(f"locationFile={pli_name}.pli\n")
        f.write(f"forcingFile={bc_name}.bc\n")

    feedback.pushInfo(f"Open boundary files: {pli_path}, {bc_path}, {ext_path}")
    return pli_path, bc_path, ext_path


def export_grd_from_lines(nc_path, out_grd, open_lines, land_lines,
                          crs="EPSG:4326", snap_tol=None, feedback=None):
    """Write an ADCIRC ``.grd`` using an edited open/land boundary classification.

    Each polyline vertex is snapped to the nearest mesh node, so the (possibly
    edited) stage-5 lines are mapped back to mesh boundary edges and contours.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.
    out_grd : str
        Output ``.grd`` path.
    open_lines : list of array_like of shape (M, 2)
        Open-boundary polylines (the ``'open'`` features from stage 5).
    land_lines : list of array_like of shape (M, 2)
        Land-boundary polylines (the ``'closed'`` and ``'island'`` features).
    crs : str, optional
        CRS string written to the ``.grd`` header. Default ``'EPSG:4326'``.
    snap_tol : float or None, optional
        Maximum distance for snapping a vertex to a mesh node; auto-derived
        from the median boundary edge length when ``None``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    out_grd : str
        Path to the written file.
    """
    import numpy as np
    from scipy.spatial import cKDTree

    from bluemesh2d.geomesh_util.grd_util import export_to_grd

    feedback = feedback or _NullFeedback()
    vert, tria, z_depth = read_ugrid_mesh(nc_path)
    tree = cKDTree(vert)
    if snap_tol is None:
        # median boundary edge length as a lenient default tolerance
        loops = _boundary_loops(vert, tria)
        d = [np.linalg.norm(vert[lp[k]] - vert[lp[(k + 1) % len(lp)]])
             for lp in loops for k in range(len(lp))]
        snap_tol = (np.median(d) if d else 1.0) * 0.75

    def lines_to_edges_contours(lines):
        edges, contours = [], []
        for ln in lines:
            ln = np.atleast_2d(np.asarray(ln, dtype=float))
            dist, idx = tree.query(ln)
            if np.any(dist > snap_tol):
                feedback.pushWarning(
                    "Some boundary vertices are far from any mesh node; "
                    "the classification may be imprecise (avoid moving "
                    "vertices when editing).")
            seq = [int(i) for i, _ in zip(idx, range(len(idx)))]
            # drop consecutive duplicates from snapping
            seq = [seq[0]] + [b for a, b in zip(seq[:-1], seq[1:]) if a != b]
            if len(seq) >= 2:
                contours.append(np.asarray(seq, dtype=int))
                edges.extend([seq[k], seq[k + 1]] for k in range(len(seq) - 1))
        return (np.asarray(edges, dtype=int) if edges
                else np.empty((0, 2), dtype=int)), contours

    edge_open, open_contours = lines_to_edges_contours(open_lines)
    edge_land, land_contours = lines_to_edges_contours(land_lines)

    tag_o = np.ones((edge_open.shape[0], 1), dtype=int)
    tag_l = np.full((edge_land.shape[0], 1), 2, dtype=int)
    parts = []
    if edge_open.shape[0]:
        parts.append(np.hstack([edge_open, tag_o]))
    if edge_land.shape[0]:
        parts.append(np.hstack([edge_land, tag_l]))
    edge_tag = np.vstack(parts) if parts else np.empty((0, 3), dtype=int)

    feedback.pushInfo(
        f"Writing .grd -> {out_grd} ({edge_open.shape[0]} open, "
        f"{edge_land.shape[0]} land edges)")
    export_to_grd(
        out_grd, vert=vert, tria=tria, z=z_depth, crs=crs,
        edge_tag=edge_tag, edge_open=edge_open, edge_land=edge_land,
        open_contours=open_contours, land_contours=land_contours,
    )
    return out_grd
