"""
Iterative pipeline: orthogonalize <-> merge_circumcenters (with optional recovery).

Used by :mod:`bluemesh2d.smood` as its orthogonalization/merge pipeline. The
underlying numeric orthogonality/small-link kernels live in
:mod:`bluemesh2d.ortho_merge.orthogonalize`.

Intended process (Delft3D-FM / dual mesh)
-----------------------------------------
1. **Orthogonalize** on a triangulation of the current (mixed) faces **consistent with**
   ``merge_circumcenters`` / UGRID export: quads ``[a,v1,b,v2]`` are split on diagonal
   ``(v1,v2)`` (see ``geomesh_util.grd_util.triangulate_mixed_face_row_to_tris``), **not**
   fan-from-``a`` (diagonal ``(a,b)``), which skews the dual w.r.t. Delft3D-FM.
2. **Remove short flow links**: ``merge_circumcenters`` merges triangle pairs into quads
   (skipped when ``merge_small_links=False``: the mesh then stays pure triangles).
3. Repeat; optional **recovery** cycles if ``max|cosφ|`` or small-link count still fails
   the dual criteria (see ``require_both_criteria``).
4. With ``require_both_criteria=True``, **raise** if criteria are still not met after the
   main and recovery cycles. Default **False** skips the global dual check and recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class OrthoMergeStats:
    outer_iter: int
    max_cosphi: float
    n_small_flow_links: int
    merged_this_iter: int
    n_zones_orthogonalized: int
    recovery: bool = False
    recovery_iter: int = 0


def _faces_from_face_nodes(face_nodes: np.ndarray) -> List[np.ndarray]:
    """
    Convert internal UGRID `face_nodes` (0-based, invalid = -1) to a list of faces.

    The returned faces contain only valid node indices (>=0) and keep original polygon
    size (3 or 4+ depending on input).
    """
    face_nodes = np.asarray(face_nodes, dtype=np.int64)
    faces: List[np.ndarray] = []
    for f in range(face_nodes.shape[0]):
        nodes = face_nodes[f, :]
        nodes = nodes[nodes >= 0]
        if nodes.size >= 3:
            faces.append(nodes.astype(np.int64, copy=True))
    return faces


def _triangulate_faces_for_ortho(
    vert_xy: np.ndarray, faces: Sequence[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Triangle rows for ``orthogonalize_tria_mesh``, aligned with ``merge_circumcenters`` quads
    and NetCDF export: use :func:`~bluemesh2d.geomesh_util.grd_util.triangulate_mixed_face_row_to_tris`
    so 4-node faces use diagonal ``(v1,v2)``, not fan-from-``a``.

    Returns
    -------
    (tria, tri_face_id)
        - tria: (T,3) triangle rows.
        - tri_face_id: (T,) index into ``faces`` of the origin face of each row.
    """
    from ..geomesh_util.grd_util import triangulate_mixed_face_row_to_tris

    vert_xy = np.asarray(vert_xy, dtype=np.float64)
    if vert_xy.ndim != 2 or vert_xy.shape[1] < 2:
        raise ValueError("vert_xy must have shape (N, 2) or (N, >=2) for x,y")
    xy = vert_xy[:, :2]
    out: List[Tuple[int, int, int]] = []
    face_ids: List[int] = []
    for fid, nodes in enumerate(faces):
        n = np.asarray(nodes, dtype=np.int64).reshape(-1)
        if n.size < 3:
            continue
        tris = triangulate_mixed_face_row_to_tris(xy, n)
        out.extend(tris)
        face_ids.extend([fid] * len(tris))
    if out:
        return np.asarray(out, dtype=np.int64), np.asarray(face_ids, dtype=np.int64)
    return np.empty((0, 3), dtype=np.int64), np.empty(0, dtype=np.int64)


def _faces_with_ortho_topology(
    faces: Sequence[np.ndarray],
    tria_in: np.ndarray,
    tria_out: np.ndarray,
    tri_face_id: np.ndarray,
) -> Sequence[np.ndarray]:
    """
    Propagate topology changes (edge flips) made by ``orthogonalize_tria_mesh``
    on the triangle proxy back onto the mixed face list.

    ``orthogonalize_tria_mesh`` flips edges by rewriting the two triangle rows
    in place, so rows of ``tria_out`` correspond 1:1 to rows of ``tria_in``.
    Faces whose proxy rows are unchanged are kept as-is (including quads). A
    quad whose proxy rows changed only by re-diagonalization (no external node
    entered) is also kept — the 4-node face is the same, only the proxy
    diagonal moved. Any other changed face is replaced by its flipped triangle
    rows: without this, flips were silently discarded by the dataset rebuild
    and the outer/recovery cycles could stall forever on an edge that only a
    flip can fix, while the ortho stage kept reporting it as solved.
    """
    if tria_out.shape != tria_in.shape:
        # Unexpected: fall back to the original faces (previous behaviour).
        return faces
    row_changed = np.any(np.sort(tria_in, axis=1) != np.sort(tria_out, axis=1), axis=1)
    if not np.any(row_changed):
        return faces

    changed_faces = set(int(f) for f in np.unique(tri_face_id[row_changed]))
    faces_out: List[np.ndarray] = []
    for fid, f in enumerate(faces):
        if fid not in changed_faces:
            faces_out.append(f)
            continue
        rows = np.where(tri_face_id == fid)[0]
        face_nodes = set(int(v) for v in np.asarray(f).reshape(-1))
        out_nodes = set(int(v) for v in tria_out[rows].ravel())
        if len(f) >= 4 and out_nodes <= face_nodes:
            # Internal re-diagonalization of a quad: same 4-node face.
            faces_out.append(f)
        else:
            for r in rows:
                faces_out.append(np.asarray(tria_out[r], dtype=np.int64).copy())
    return faces_out


def _face_nodes_raw_to_0b(face_nodes_raw: np.ndarray, start_index: int) -> np.ndarray:
    """UGRID face_nodes array to 0-based with -1 fill."""
    face_nodes_raw = np.asarray(face_nodes_raw, dtype=np.int64)
    if int(start_index) == 1:
        face_nodes = np.full_like(face_nodes_raw, -1)
        valid = face_nodes_raw > 0
        face_nodes[valid] = face_nodes_raw[valid] - 1
        return face_nodes
    return face_nodes_raw.copy()


def _fan_vert_tria_from_ds(
    ds_cur,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Triangle proxy of the mixed UGRID dataset, aligned with quad splitting
    used by the rest of this repository.

    Returns
    -------
    vert_xy : (N,2) float64
        Triangle-proxy vertex coordinates.
    tria : (T,3) int64
        Triangle connectivity of the proxy.
    tri_origin_face_id : (T,) int64
        Index of the original mixed-face row from which each triangle originates.
        (Triangles coming from the same quad share the same origin id.)
    quad_face_mask : (F,) bool
        Whether each original mixed-face row is a quad (len==4).
    """
    face_nodes_raw = np.asarray(ds_cur["mesh2d_face_nodes"].values, dtype=np.int64)
    start_index = int(ds_cur["mesh2d_face_nodes"].attrs.get("start_index", 1))
    face_nodes = _face_nodes_raw_to_0b(face_nodes_raw, start_index)
    faces = _faces_from_face_nodes(face_nodes)

    node_x = np.asarray(ds_cur["mesh2d_node_x"].values, dtype=np.float64)
    node_y = np.asarray(ds_cur["mesh2d_node_y"].values, dtype=np.float64)
    vert_xy = np.column_stack([node_x, node_y])
    xy = vert_xy[:, :2]

    from ..geomesh_util.grd_util import triangulate_mixed_face_row_to_tris

    quad_face_mask = np.asarray([(f.size == 4) for f in faces], dtype=bool)
    tris: List[Tuple[int, int, int]] = []
    tri_origin_face_id: List[int] = []

    for fid, nodes in enumerate(faces):
        tri_list = triangulate_mixed_face_row_to_tris(
            xy, np.asarray(nodes, dtype=np.int64)
        )
        for tri in tri_list:
            tris.append((int(tri[0]), int(tri[1]), int(tri[2])))
            tri_origin_face_id.append(int(fid))

    tria = np.asarray(tris, dtype=np.int64)
    tri_origin_face_id = np.asarray(tri_origin_face_id, dtype=np.int64)

    return vert_xy, tria, tri_origin_face_id, quad_face_mask


def dual_criteria_on_fan_mesh(
    vert: np.ndarray,
    tria: np.ndarray,
    tri_origin_face_id: np.ndarray,
    quad_face_mask: np.ndarray,
    *,
    cosphi_threshold: float,
    removesmalllinkstrsh: float,
    jsferic: int = 1,
) -> Tuple[bool, float, int]:
    """Check dual-mesh orthogonality and small-link criteria on a triangle proxy.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Node coordinates (lon/lat degrees or planar x/y per ``jsferic``).
    tria : ndarray of shape (T, 3)
        0-based triangle connectivity.
    tri_origin_face_id : ndarray of shape (T,)
        Origin mixed-face index for each proxy triangle.
    quad_face_mask : ndarray of shape (F,)
        Whether each original mixed face is a quad.
    cosphi_threshold : float
        Maximum allowed ``|cos φ|`` on internal flow links.
    removesmalllinkstrsh : float
        Small flow-link threshold (Delft3D convention).
    jsferic : int, optional
        ``1`` for spherical lon/lat; ``0`` for planar coordinates.

    Returns
    -------
    ok : bool
        ``True`` when both criteria are satisfied.
    max_c : float
        Maximum ``|cos φ|`` over counted edges.
    n_small : int
        Number of small flow links detected.
    """
    from . import orthogonalize as ortho
    from .geometry import build_edges_from_tria

    vert = np.asarray(vert, dtype=np.float64)
    tria = np.asarray(tria, dtype=np.int64)
    if tria.size == 0:
        return True, 0.0, 0

    edge_nodes, edge_faces = build_edges_from_tria(tria)
    _, _, cosphi_abs = ortho.compute_cosphi_abs_from_arrays(
        vert[:, 0],
        vert[:, 1],
        tria,
        edge_nodes,
        edge_faces,
        use_file_centers=False,
        use_circumcenter_3d=True,
        jsferic=jsferic,
    )
    # MeshKernel small-flow-links should ignore edges internal to a quad in the
    # original mixed mesh. In the triangle-proxy, those correspond to the shared
    # diagonal between the two triangles coming from the same quad-face row.
    # Apply the same exclusion to max|cos φ|: internal diagonals are not flow
    # links, so they must not force recovery when merge introduces quads.
    n_edges = edge_faces.shape[0]
    keep_edge_indices = np.arange(n_edges, dtype=np.int64)
    exclude_mask = np.zeros(n_edges, dtype=bool)
    for e in range(n_edges):
        f1 = int(edge_faces[e, 0])
        f2 = int(edge_faces[e, 1])
        if f1 < 0 or f2 < 0 or f1 == f2:
            continue
        o1 = int(tri_origin_face_id[f1])
        o2 = int(tri_origin_face_id[f2])
        if o1 == o2 and bool(quad_face_mask[o1]):
            exclude_mask[e] = True

    mask = ~np.isnan(cosphi_abs) & ~exclude_mask
    max_c = float(np.nanmax(cosphi_abs[mask])) if np.any(mask) else 0.0

    keep_edge_indices = keep_edge_indices[~exclude_mask]
    n_small, _ = ortho.compute_small_links_from_arrays(
        vert[:, 0],
        vert[:, 1],
        tria,
        edge_nodes,
        edge_faces,
        removesmalllinkstrsh=float(removesmalllinkstrsh),
        edge_indices=keep_edge_indices,
        jsferic=jsferic,
    )

    ok = (max_c <= float(cosphi_threshold) + 1.0e-9) and (int(n_small) == 0)
    return ok, max_c, int(n_small)


def ortho_merge_iterate_dataset(
    ds,
    *,
    outer_iter_max: int = 5,
    cosphi_threshold: float = 0.49,
    removesmalllinkstrsh: float = 0.11,
    buffer_layers: int = 2,
    max_global_iter: int = 6,
    smooth_iter: int = 16,
    enable_edge_flips: bool = True,
    stop_if_no_merge: bool = True,
    ortho_disable_smalllink_logic: bool = True,
    require_both_criteria: bool = False,
    max_recovery_iterations: int = 25,
    recovery_stagnation_break: int = 3,
    outer_stagnation_break: int = 2,
    adaptive_recovery: bool = True,
    recovery_buffer_growth: int = 1,
    recovery_smooth_iter_growth: int = 6,
    recovery_global_iter_growth: int = 1,
    on_state: Optional[Callable[[OrthoMergeStats], None]] = None,
    verbose: bool = True,
    jsferic: int = 1,
    merge_small_links: bool = True,
) -> tuple:
    """Iteratively orthogonalize and merge circumcenters on a UGRID dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Delft3D-FM UGRID mesh dataset.
    outer_iter_max : int, optional
        Number of outer ortho+merge cycles.
    cosphi_threshold : float, optional
        Maximum allowed ``|cos φ|`` passed to :func:`orthogonalize_tria_mesh`.
    removesmalllinkstrsh : float, optional
        Small flow-link threshold.
    buffer_layers : int, optional
        Zone buffer depth for orthogonalization.
    max_global_iter : int, optional
        Global orthogonalization passes per cycle.
    smooth_iter : int, optional
        Smoothing iterations per zone pass.
    enable_edge_flips : bool, optional
        Allow quality-guarded edge flips during orthogonalization.
    stop_if_no_merge : bool, optional
        Stop when a merge cycle reduces no faces.
    ortho_disable_smalllink_logic : bool, optional
        If ``True``, delegate small-link removal to merge (not ortho).
    require_both_criteria : bool, optional
        If ``True``, require dual criteria after main and recovery cycles.
    max_recovery_iterations : int, optional
        Extra recovery cycles when criteria fail.
    recovery_stagnation_break : int, optional
        Stop recovery after this many stagnant cycles (0 = disabled).
    outer_stagnation_break : int, optional
        Stop main loop after this many stagnant outer cycles.
    adaptive_recovery : bool, optional
        Increase recovery effort when criteria remain unmet.
    recovery_buffer_growth, recovery_smooth_iter_growth, recovery_global_iter_growth : int, optional
        Per-recovery growth of buffer/smooth/global iteration counts.
    on_state : callable, optional
        Called with :class:`OrthoMergeStats` after each cycle.
    verbose : bool, optional
        Enable per-zone orthogonalization logs.
    jsferic : int, optional
        ``1`` for spherical lon/lat; ``0`` for planar coordinates.
    merge_small_links : bool, optional
        If ``False``, keep pure triangles (no quad merge step).

    Returns
    -------
    ds_final : xarray.Dataset
        Updated mesh dataset.
    stats : list of OrthoMergeStats
        Per-cycle statistics.
    """
    import xarray as xr

    from ..geomesh_util.merge_circumcenters import (
        _rebuild_ds_from_form,
        build_ugrid_arrays_mixed,
        merge_circumcenters,
    )

    from .orthogonalize import orthogonalize_tria_mesh

    if not isinstance(ds, xr.Dataset):
        raise TypeError("ds must be an xarray.Dataset")

    stats: List[OrthoMergeStats] = []

    # Work on a copy to avoid mutating caller data
    ds_cur = ds.copy(deep=True)

    def _run_ortho_merge_cycle(
        ds_in,
        *,
        buffer_layers_override: Optional[int] = None,
        max_global_iter_override: Optional[int] = None,
        smooth_iter_override: Optional[int] = None,
    ):
        """One ortho (dual-consistent tris of mixed faces) + merge_circumcenters. Returns updated ds."""
        ds_before_outer = ds_in.copy(deep=True)
        node_x = np.asarray(ds_in["mesh2d_node_x"].values, dtype=np.float64)
        node_y = np.asarray(ds_in["mesh2d_node_y"].values, dtype=np.float64)
        node_z = (
            np.asarray(ds_in["mesh2d_node_z"].values, dtype=np.float64)
            if "mesh2d_node_z" in ds_in
            else np.zeros((node_x.shape[0],), dtype=np.float64)
        )
        vert = np.column_stack([node_x, node_y])
        face_nodes_raw = np.asarray(ds_in["mesh2d_face_nodes"].values, dtype=np.int64)
        start_index = int(ds_in["mesh2d_face_nodes"].attrs.get("start_index", 1))
        face_nodes = _face_nodes_raw_to_0b(face_nodes_raw, start_index)
        faces = _faces_from_face_nodes(face_nodes)
        tria_for_ortho, tri_face_id = _triangulate_faces_for_ortho(vert, faces)

        bl = int(
            buffer_layers if buffer_layers_override is None else buffer_layers_override
        )
        mgi = int(
            max_global_iter
            if max_global_iter_override is None
            else max_global_iter_override
        )
        si = int(smooth_iter if smooth_iter_override is None else smooth_iter_override)

        if merge_small_links:
            # Small links are removed by merge_circumcenters below; keep the
            # orthogonalizer focused on |cosphi| only.
            ortho_smalllink_trsh = (
                1.0e-12 if ortho_disable_smalllink_logic else removesmalllinkstrsh
            )
        else:
            # Triangles-only mode: the orthogonalizer itself must clear small
            # links (guarded flips + circumcenter-separation displacement).
            ortho_smalllink_trsh = removesmalllinkstrsh
        ortho_res = orthogonalize_tria_mesh(
            vert,
            tria_for_ortho,
            cosphi_threshold=cosphi_threshold,
            removesmalllinkstrsh=ortho_smalllink_trsh,
            buffer_layers=bl,
            max_global_iter=mgi,
            smooth_iter=si,
            enable_edge_flips=enable_edge_flips,
            verbose=verbose,
            jsferic=jsferic,
            smalllink_priority=not merge_small_links,
        )

        NODE = np.column_stack([ortho_res.vert[:, 0], ortho_res.vert[:, 1], node_z])
        faces_after_ortho = _faces_with_ortho_topology(
            faces,
            tria_for_ortho,
            np.asarray(ortho_res.tria, dtype=np.int64),
            tri_face_id,
        )
        ugrid_arrays = build_ugrid_arrays_mixed(NODE, faces_after_ortho)
        ds_after_ortho = _rebuild_ds_from_form(ds_in, ugrid_arrays)

        if not merge_small_links:
            return ds_after_ortho, ortho_res, 0, ds_before_outer

        nfaces_before = int(
            ds_after_ortho.sizes.get(
                "mesh2d_nFaces", ds_after_ortho["mesh2d_face_nodes"].shape[0]
            )
        )
        ds_merged = merge_circumcenters(
            ds_after_ortho, removesmalllinkstrsh=removesmalllinkstrsh, jsferic=jsferic
        )
        nfaces_after = int(
            ds_merged.sizes.get(
                "mesh2d_nFaces", ds_merged["mesh2d_face_nodes"].shape[0]
            )
        )
        merged_this_iter = max(0, nfaces_before - nfaces_after)

        return ds_merged, ortho_res, merged_this_iter, ds_before_outer

    outer_stall_limit = max(0, int(outer_stagnation_break))
    outer_stall_count = 0
    for outer in range(int(outer_iter_max)):
        ds_cur, ortho_res, merged_this_iter, ds_before_outer = _run_ortho_merge_cycle(
            ds_cur
        )

        # Early stop on stagnation: if the same outer-cycle metrics repeat,
        # further global passes are unlikely to help and are expensive.
        if len(stats) > 0:
            prev = stats[-1]
            same_metric = (
                abs(float(ortho_res.max_cosphi) - float(prev.max_cosphi)) <= 1.0e-9
                and int(ortho_res.n_small_flow_links) == int(prev.n_small_flow_links)
                and int(merged_this_iter) == int(prev.merged_this_iter)
            )
            if same_metric:
                outer_stall_count += 1
                if outer_stall_limit > 0 and outer_stall_count >= outer_stall_limit:
                    stats.append(
                        OrthoMergeStats(
                            outer_iter=outer,
                            max_cosphi=float(ortho_res.max_cosphi),
                            n_small_flow_links=int(ortho_res.n_small_flow_links),
                            merged_this_iter=int(merged_this_iter),
                            n_zones_orthogonalized=int(
                                getattr(ortho_res, "n_zones_orthogonalized", 0)
                            ),
                        )
                    )
                    if on_state is not None:
                        on_state(stats[-1])
                    break
            else:
                outer_stall_count = 0

        # Guardrail: if no merge happened and orthogonality got worse than previous outer-iter,
        # revert this outer step and stop.
        if (
            merged_this_iter == 0
            and len(stats) > 0
            and float(ortho_res.max_cosphi) > (float(stats[-1].max_cosphi) + 1.0e-9)
        ):
            ds_cur = ds_before_outer
            stats.append(
                OrthoMergeStats(
                    outer_iter=outer,
                    max_cosphi=float(stats[-1].max_cosphi),
                    n_small_flow_links=int(stats[-1].n_small_flow_links),
                    merged_this_iter=0,
                    n_zones_orthogonalized=int(
                        getattr(ortho_res, "n_zones_orthogonalized", 0)
                    ),
                )
            )
            if on_state is not None:
                on_state(stats[-1])
            break
        stats.append(
            OrthoMergeStats(
                outer_iter=outer,
                max_cosphi=float(ortho_res.max_cosphi),
                n_small_flow_links=int(ortho_res.n_small_flow_links),
                merged_this_iter=int(merged_this_iter),
                n_zones_orthogonalized=int(
                    getattr(ortho_res, "n_zones_orthogonalized", 0)
                ),
            )
        )
        if on_state is not None:
            on_state(stats[-1])

        # Not meaningful in triangles-only mode (nothing is ever merged).
        if merge_small_links and stop_if_no_merge and merged_this_iter == 0:
            break

    if bool(require_both_criteria):
        v_chk, t_chk, tri_origin_face_id, quad_face_mask = _fan_vert_tria_from_ds(
            ds_cur
        )
        ok, max_c, n_s = dual_criteria_on_fan_mesh(
            v_chk,
            t_chk,
            tri_origin_face_id,
            quad_face_mask,
            cosphi_threshold=cosphi_threshold,
            removesmalllinkstrsh=removesmalllinkstrsh,
            jsferic=jsferic,
        )
        max_rec = max(0, int(max_recovery_iterations))
        stall_need = int(recovery_stagnation_break)
        prev_metric_key = (round(max_c, 9), int(n_s))
        stall = 0
        r = 0
        while (not ok) and r < max_rec:
            if bool(adaptive_recovery):
                growth_steps = 1 + r + stall
                bl_rec = int(
                    max(1, buffer_layers + int(recovery_buffer_growth) * growth_steps)
                )
                mgi_rec = int(
                    max(
                        1,
                        max_global_iter
                        + int(recovery_global_iter_growth) * growth_steps,
                    )
                )
                si_rec = int(
                    max(
                        1, smooth_iter + int(recovery_smooth_iter_growth) * growth_steps
                    )
                )
            else:
                bl_rec = int(buffer_layers)
                mgi_rec = int(max_global_iter)
                si_rec = int(smooth_iter)

            ds_cur, ortho_res, merged_this_iter, ds_before_outer = (
                _run_ortho_merge_cycle(
                    ds_cur,
                    buffer_layers_override=bl_rec,
                    max_global_iter_override=mgi_rec,
                    smooth_iter_override=si_rec,
                )
            )
            # Recovery: no guardrail revert (keep trying); always log cycle.
            stats.append(
                OrthoMergeStats(
                    outer_iter=r,
                    max_cosphi=float(ortho_res.max_cosphi),
                    n_small_flow_links=int(ortho_res.n_small_flow_links),
                    merged_this_iter=int(merged_this_iter),
                    n_zones_orthogonalized=int(
                        getattr(ortho_res, "n_zones_orthogonalized", 0)
                    ),
                    recovery=True,
                    recovery_iter=r,
                )
            )
            if on_state is not None:
                on_state(stats[-1])
            v_chk, t_chk, tri_origin_face_id, quad_face_mask = _fan_vert_tria_from_ds(
                ds_cur
            )
            ok, max_c, n_s = dual_criteria_on_fan_mesh(
                v_chk,
                t_chk,
                tri_origin_face_id,
                quad_face_mask,
                cosphi_threshold=cosphi_threshold,
                removesmalllinkstrsh=removesmalllinkstrsh,
                jsferic=jsferic,
            )
            r += 1
            if stall_need > 0:
                key = (round(max_c, 9), int(n_s))
                if key == prev_metric_key:
                    stall += 1
                    if stall >= stall_need:
                        break
                else:
                    stall = 0
                    prev_metric_key = key

        if not ok:
            raise RuntimeError(
                "ortho_merge_iterate_dataset: mesh still violates dual criteria after "
                f"{int(outer_iter_max)} main cycle(s) and {r} recovery cycle(s). "
                f"Required: max|cosφ| <= {cosphi_threshold} and n_small_flow_links == 0 "
                f"(triangle proxy). Got max|cosφ|={max_c:.6f}, n_small={n_s}. "
                "Increase max_recovery_iterations / outer_iter_max, relax thresholds, "
                "or improve the initial mesh."
            )

    return ds_cur, stats


def print_stats(
    stats: Sequence[OrthoMergeStats], *, print_header: bool = False
) -> None:
    """Print a compact ortho+merge progress table.

    Parameters
    ----------
    stats : sequence of OrthoMergeStats
        Statistics rows to display.
    print_header : bool, optional
        If ``True``, print the column header first.
    """
    if print_header:
        print(" -------------------------------------------------------")
        print("      |STATE.|      |MAX|COS(PHI)| |N_SMALL| |N_ZONES|")
        print(" -------------------------------------------------------")

    for s in stats:
        if getattr(s, "recovery", False):
            head = f"recovery={s.recovery_iter}"
        else:
            # Special markers used by smood for start/end snapshots.
            if int(s.outer_iter) == -1:
                head = "initial"
            elif int(s.outer_iter) == -2:
                head = "final"
            else:
                head = f"outer={s.outer_iter}"

        print(
            f"    {head:<11}{s.max_cosphi:>13.6f}{int(s.n_small_flow_links):>14d}{int(s.n_zones_orthogonalized):>11d}",
            flush=True,
        )


def ortho_merge_iterate_tria(
    vert: np.ndarray,
    tria: np.ndarray,
    *,
    node_z: Optional[np.ndarray] = None,
    outer_iter_max: int = 5,
    cosphi_threshold: float = 0.49,
    removesmalllinkstrsh: float = 0.11,
    buffer_layers: int = 2,
    max_global_iter: int = 6,
    smooth_iter: int = 16,
    enable_edge_flips: bool = True,
    stop_if_no_merge: bool = True,
    ortho_disable_smalllink_logic: bool = True,
    require_both_criteria: bool = False,
    max_recovery_iterations: int = 25,
    recovery_stagnation_break: int = 3,
    outer_stagnation_break: int = 2,
    adaptive_recovery: bool = True,
    recovery_buffer_growth: int = 1,
    recovery_smooth_iter_growth: int = 6,
    recovery_global_iter_growth: int = 1,
    on_state: Optional[Callable[[OrthoMergeStats], None]] = None,
    verbose: bool = True,
    jsferic: int = 1,
    merge_small_links: bool = True,
) -> tuple:
    """Run the ortho+merge pipeline starting from a pure triangle mesh.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Node coordinates (lon/lat degrees or planar x/y per ``jsferic``).
    tria : ndarray of shape (T, 3)
        0-based triangle connectivity.
    node_z : ndarray of shape (N,), optional
        Node elevations; zeros used if ``None``.
    outer_iter_max, cosphi_threshold, removesmalllinkstrsh, buffer_layers,
    max_global_iter, smooth_iter, enable_edge_flips, stop_if_no_merge,
    ortho_disable_smalllink_logic, require_both_criteria,
    max_recovery_iterations, recovery_stagnation_break, outer_stagnation_break,
    adaptive_recovery, recovery_buffer_growth, recovery_smooth_iter_growth,
    recovery_global_iter_growth, on_state, verbose, jsferic, merge_small_links
        Same meaning as in :func:`ortho_merge_iterate_dataset`.

    Returns
    -------
    vert_out : ndarray of shape (N, 2)
        Updated node coordinates.
    face_nodes_0b : ndarray of shape (F, 4)
        Mixed face-node rows (0-based, ``-1`` fill for triangles).
    stats : list of OrthoMergeStats
        Per-cycle statistics.
    """
    import xarray as xr

    from ..geomesh_util.grd_util import adcirc2DFlowFM

    vert = np.asarray(vert, dtype=np.float64)
    tria = np.asarray(tria, dtype=np.int64)
    if vert.ndim != 2 or vert.shape[1] != 2:
        raise ValueError("vert must have shape (N,2)")
    if tria.ndim != 2 or tria.shape[1] != 3:
        raise ValueError("tria must have shape (T,3)")

    if node_z is None:
        node_z = np.zeros((vert.shape[0],), dtype=np.float64)
    else:
        node_z = np.asarray(node_z, dtype=np.float64).reshape(-1)
        if node_z.shape[0] != vert.shape[0]:
            raise ValueError("node_z must have length N (same as vert)")

    NODE = np.column_stack([vert[:, 0], vert[:, 1], node_z])
    ds0 = adcirc2DFlowFM(NODE=NODE, EDGE=tria)
    if not isinstance(ds0, xr.Dataset):
        raise TypeError(
            "adcirc2DFlowFM must return an xarray.Dataset in this repository"
        )

    ds_final, stats = ortho_merge_iterate_dataset(
        ds0,
        outer_iter_max=outer_iter_max,
        cosphi_threshold=cosphi_threshold,
        removesmalllinkstrsh=removesmalllinkstrsh,
        buffer_layers=buffer_layers,
        max_global_iter=max_global_iter,
        smooth_iter=smooth_iter,
        enable_edge_flips=enable_edge_flips,
        stop_if_no_merge=stop_if_no_merge,
        ortho_disable_smalllink_logic=ortho_disable_smalllink_logic,
        require_both_criteria=require_both_criteria,
        max_recovery_iterations=max_recovery_iterations,
        recovery_stagnation_break=recovery_stagnation_break,
        outer_stagnation_break=outer_stagnation_break,
        adaptive_recovery=adaptive_recovery,
        recovery_buffer_growth=recovery_buffer_growth,
        recovery_smooth_iter_growth=recovery_smooth_iter_growth,
        recovery_global_iter_growth=recovery_global_iter_growth,
        on_state=on_state,
        verbose=verbose,
        jsferic=jsferic,
        merge_small_links=merge_small_links,
    )

    vert_out = np.column_stack(
        [
            np.asarray(ds_final["mesh2d_node_x"].values, dtype=np.float64),
            np.asarray(ds_final["mesh2d_node_y"].values, dtype=np.float64),
        ]
    )
    face_nodes_1b = np.asarray(ds_final["mesh2d_face_nodes"].values, dtype=np.int64)
    start_index = int(ds_final["mesh2d_face_nodes"].attrs.get("start_index", 1))
    fill = -1
    if start_index == 1:
        # Common convention: invalid nodes are stored as 0 for 1-based meshes.
        face_nodes_0b = np.where(face_nodes_1b > 0, face_nodes_1b - 1, fill)
    else:
        # Assume already 0-based with invalid encoded as -1.
        face_nodes_0b = face_nodes_1b.copy()

    return vert_out, face_nodes_0b, stats
