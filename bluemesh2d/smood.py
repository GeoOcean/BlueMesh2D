import numpy as np
from .geomesh_util.grd_util import triangulate_mixed_face_row_to_tris
from .mesh_util.tricon import tricon
from .ortho_merge.constants import DEFAULT_SMALLLINK_THRESHOLD

DEFAULT_REQUIRE_STRICT_DUAL: bool = True


def _ortho_merge_pipeline(vert, conn, tria, tnum, opts, fixed=None):
    """Run repeated orthogonalize and merge_circumcenters cycles on a triangle mesh."""
    from .ortho_merge.ortho_merge_iter import ortho_merge_iterate_tria, print_stats

    vert_in = np.asarray(vert, dtype=np.float64)
    tria_in = np.asarray(tria, dtype=np.int64)
    tnum_in = np.asarray(tnum, dtype=np.int64).reshape(-1)

    outer_iter_max = max(1, int(opts.get("iter", 4)))

    smalllink_trsh = float(opts.get("smalllink_threshold", DEFAULT_SMALLLINK_THRESHOLD))
    require_strict = bool(
        opts.get("require_both_criteria", DEFAULT_REQUIRE_STRICT_DUAL)
    )

    # jsferic=1 -> spherical (lon/lat degrees); jsferic=0 -> planar x/y.
    jsferic = 1 if bool(opts.get("spherical", False)) else 0

    # Triangles-only mode: clear small flow links by guarded flips and node
    # movement instead of merging triangle pairs into quads.
    merge_small_links = bool(opts.get("merge_small_links", False))

    # Last-resort merge for the elements triangles-only recovery cannot fix.
    recovery_merge = bool(opts.get("recovery_merge_small_links", True))
    recovery_merge_from = int(opts.get("recovery_merge_from_iter", 2))

    do_log = not np.isinf(opts.get("disp", 4))

    # Initial snapshot (triangle proxy: 1 mixed-face row per input triangle).
    init_max_c = None
    init_n_small = None
    if do_log:
        cosphi_threshold = float(opts.get("orthogonality_threshold", 0.49))
        removesmalllinkstrsh = smalllink_trsh
        tri_origin_face_id = np.arange(tria_in.shape[0], dtype=np.int64)
        quad_face_mask = np.zeros(tria_in.shape[0], dtype=bool)

        from .ortho_merge.ortho_merge_iter import (
            dual_criteria_on_fan_mesh,
            OrthoMergeStats,
        )

        _, init_max_c, init_n_small = dual_criteria_on_fan_mesh(
            np.asarray(vert_in, dtype=np.float64),
            tria_in,
            tri_origin_face_id,
            quad_face_mask,
            cosphi_threshold=cosphi_threshold,
            removesmalllinkstrsh=removesmalllinkstrsh,
            jsferic=jsferic,
        )
        print_stats(
            [
                OrthoMergeStats(
                    outer_iter=-1,
                    max_cosphi=float(init_max_c),
                    n_small_flow_links=int(init_n_small),
                    merged_this_iter=0,
                    n_zones_orthogonalized=0,
                )
            ],
            print_header=True,
        )

    def _on_state(s):
        if do_log:
            print_stats([s], print_header=False)

    # Keep output compact: disable verbose per-zone logs inside meshkernel orthogonalization.
    vert_out, face_nodes_0b, stats = ortho_merge_iterate_tria(
        vert_in,
        tria_in,
        node_z=None,
        outer_iter_max=outer_iter_max,
        cosphi_threshold=float(opts.get("orthogonality_threshold", 0.49)),
        removesmalllinkstrsh=smalllink_trsh,
        buffer_layers=int(opts.get("buffer_layers", 2)),
        max_global_iter=int(
            opts.get("max_global_iter", int(opts.get("inner_iter", 4)) + 2)
        ),
        smooth_iter=int(opts.get("smooth_iter", int(opts.get("inner_iter", 4)) * 4)),
        enable_edge_flips=bool(opts.get("enable_edge_flips", True)),
        stop_if_no_merge=bool(opts.get("stop_if_no_merge", False)),
        ortho_disable_smalllink_logic=True,
        require_both_criteria=require_strict,
        max_recovery_iterations=int(opts.get("max_recovery_iterations", 100)),
        recovery_stagnation_break=int(opts.get("recovery_stagnation_break", 10)),
        outer_stagnation_break=int(opts.get("outer_stagnation_break", 2)),
        adaptive_recovery=bool(opts.get("adaptive_recovery", True)),
        recovery_buffer_growth=int(opts.get("recovery_buffer_growth", 1)),
        recovery_smooth_iter_growth=int(opts.get("recovery_smooth_iter_growth", 6)),
        recovery_global_iter_growth=int(opts.get("recovery_global_iter_growth", 1)),
        on_state=_on_state if do_log else None,
        verbose=False,
        jsferic=jsferic,
        merge_small_links=merge_small_links,
        recovery_merge_small_links=recovery_merge,
        recovery_merge_from_iter=recovery_merge_from,
        fixed=fixed,
    )

    # Final snapshot (triangle proxy built from mixed faces).
    if do_log:
        cosphi_threshold = float(opts.get("orthogonality_threshold", 0.49))
        removesmalllinkstrsh = smalllink_trsh

        from .ortho_merge.ortho_merge_iter import (
            dual_criteria_on_fan_mesh,
            OrthoMergeStats,
        )

        face_nodes_0b_arr = np.asarray(face_nodes_0b, dtype=np.int64)
        vert_xy = np.asarray(vert_out, dtype=np.float64)

        tria_proxy = []
        tri_origin_face_id = []
        quad_face_mask = np.zeros(face_nodes_0b_arr.shape[0], dtype=bool)

        for fid, row in enumerate(face_nodes_0b_arr):
            nodes = row[row >= 0]
            if nodes.size < 3:
                continue
            quad_face_mask[fid] = nodes.size == 4
            for t in triangulate_mixed_face_row_to_tris(vert_xy, nodes):
                tria_proxy.append(t)
                tri_origin_face_id.append(fid)

        tria_proxy = np.asarray(tria_proxy, dtype=np.int64)
        tri_origin_face_id = np.asarray(tri_origin_face_id, dtype=np.int64)

        _, fin_max_c, fin_n_small = dual_criteria_on_fan_mesh(
            vert_xy,
            tria_proxy,
            tri_origin_face_id,
            quad_face_mask,
            cosphi_threshold=cosphi_threshold,
            removesmalllinkstrsh=removesmalllinkstrsh,
            jsferic=jsferic,
        )

        last_zones = (
            int(getattr(stats[-1], "n_zones_orthogonalized", 0)) if stats else 0
        )
        print_stats(
            [
                OrthoMergeStats(
                    outer_iter=-2,
                    max_cosphi=float(fin_max_c),
                    n_small_flow_links=int(fin_n_small),
                    merged_this_iter=0,
                    n_zones_orthogonalized=last_zones,
                )
            ],
            print_header=False,
        )

    # Optionally keep the merged mixed faces (quads) for UGRID export.
    preserve_merged_quads = bool(opts.get("preserve_merged_quads", False))
    face_nodes_arr = np.asarray(face_nodes_0b, dtype=np.int64)

    if preserve_merged_quads:
        opts["_mixed_face_nodes_0b"] = face_nodes_arr.copy()
    else:
        opts.pop("_mixed_face_nodes_0b", None)

    # If we actually have quads and the caller asked to preserve them, return
    # ``face_nodes_0b`` directly as the 3rd output (`tria`), so callers can do:
    #   ds_out = adcirc2DFlowFM(NODE, tria)
    # without needing the separate `mixed_fn` conditional.
    #
    # For purely triangulated meshes (no quads), keep the historical return
    # type: triangle connectivity (T,3).
    valid_counts = np.sum(face_nodes_arr >= 0, axis=1)
    has_quads = bool(np.any(valid_counts == 4))
    if preserve_merged_quads and has_quads:
        tria_out = face_nodes_arr.copy()
        # `tnum` is not used by `adcirc2DFlowFM`, but keep the shape consistent
        # with the returned face rows.
        tnum_out = np.ones((tria_out.shape[0], 1), dtype=np.int64)
        return np.asarray(vert_out, dtype=np.float64), conn, tria_out, tnum_out

    # Triangle-only connectivity.
    # Quads from merge_circumcenters: split on diagonal (v1,v2), not fan-from-a.
    new_tris = []
    new_parts = []
    tri_origin_face_id_export: list = []
    quad_face_mask_export = np.zeros(face_nodes_arr.shape[0], dtype=bool)
    vert_xy = np.asarray(vert_out, dtype=np.float64)
    for fid, row in enumerate(face_nodes_arr):
        nodes = row[row >= 0]
        if nodes.size < 3:
            continue
        quad_face_mask_export[fid] = nodes.size == 4
        for t in triangulate_mixed_face_row_to_tris(vert_xy, nodes):
            new_tris.append(t)
            new_parts.append(1)
            tri_origin_face_id_export.append(int(fid))

    tria_out = np.asarray(new_tris, dtype=np.int64)
    if tria_out.size == 0:
        tria_out = tria_in.copy()
        tnum_out = np.asarray(tnum, dtype=np.int64)
        tri_origin_face_id_for_dual = np.arange(tria_out.shape[0], dtype=np.int64)
        quad_face_mask_for_dual = np.zeros(tria_out.shape[0], dtype=bool)
    else:
        tnum_out = np.asarray(new_parts, dtype=np.int64).reshape(-1, 1)
        tri_origin_face_id_for_dual = np.asarray(
            tri_origin_face_id_export, dtype=np.int64
        )
        quad_face_mask_for_dual = quad_face_mask_export

    # Optional post-pass: enforce dual criteria on the final triangle output.
    # This addresses cases where outer-cycle stats are good but exported triangle
    # connectivity still shows degraded max|cos(phi)|.
    enforce_output_dual = bool(
        opts.get("enforce_output_dual_criteria", bool(require_strict))
    )
    if enforce_output_dual and tria_out.size > 0:
        from .ortho_merge.orthogonalize import orthogonalize_tria_mesh
        from .ortho_merge.ortho_merge_iter import dual_criteria_on_fan_mesh

        post_iter = int(opts.get("post_output_ortho_iter", 3))
        post_iter = max(1, post_iter)
        cosphi_threshold = float(opts.get("orthogonality_threshold", 0.49))
        removesmalllinkstrsh = float(
            opts.get("smalllink_threshold", DEFAULT_SMALLLINK_THRESHOLD)
        )

        # A few short recovery cycles are enough in practice and avoid over-smoothing.
        for _ in range(post_iter):
            _, max_c_now, n_small_now = dual_criteria_on_fan_mesh(
                np.asarray(vert_out, dtype=np.float64),
                tria_out,
                tri_origin_face_id_for_dual,
                quad_face_mask_for_dual,
                cosphi_threshold=cosphi_threshold,
                removesmalllinkstrsh=removesmalllinkstrsh,
                jsferic=jsferic,
            )
            if (float(max_c_now) <= cosphi_threshold + 1.0e-9) and (
                int(n_small_now) == 0
            ):
                break

            # Same compact logs as the main ortho+merge loop: each orthogonalize pass
            # visits every zone once — without this, [ZONE] lines look like a hang.
            ortho_res = orthogonalize_tria_mesh(
                np.asarray(vert_out, dtype=np.float64),
                np.asarray(tria_out, dtype=np.int64),
                cosphi_threshold=cosphi_threshold,
                removesmalllinkstrsh=removesmalllinkstrsh,
                buffer_layers=int(opts.get("buffer_layers", 2)),
                max_global_iter=int(
                    opts.get(
                        "max_global_iter",
                        int(opts.get("inner_iter", 4)) + 2,
                    )
                ),
                smooth_iter=int(
                    opts.get("smooth_iter", int(opts.get("inner_iter", 4)) * 4)
                ),
                enable_edge_flips=bool(opts.get("enable_edge_flips", True)),
                verbose=False,
                jsferic=jsferic,
                fixed=fixed,
            )
            vert_out = ortho_res.vert
            tria_out = ortho_res.tria

    return np.asarray(vert_out, dtype=np.float64), conn, tria_out, tnum_out


def smood(
    vert=None, conn=None, tria=None, tnum=None, opts=None, hfun=None, harg=[],
    fixed=None,
):
    """Smooth a mesh with orthogonalization for flow simulations.

    Combine orthogonalization (aspect-ratio control) with hill-climbing
    smoothing (angle optimization) via the ortho-merge pipeline.

    Parameters
    ----------
    vert : ndarray of shape (V, 2), optional
        Vertex coordinates.
    conn : ndarray of shape (E, 2), optional
        Constrained edges.
    tria : ndarray of shape (T, 3), optional
        Triangle connectivity.
    tnum : ndarray of shape (T, 1), optional
        Part index per triangle.
    opts : dict, optional
        Pipeline options (defaults via :func:`makeopt_smood`):

        - ``vtol`` : float, default ``1.0e-3`` — vertex movement tolerance
        - ``iter`` : int, default ``4`` — outer ortho/merge cycles
        - ``inner_iter`` : int, default ``4`` — inner iterations per cycle
        - ``ortho_factor`` : float, default ``0.5`` — ortho vs. smooth blend
        - ``relaxation`` : float, default ``0.75`` — coordinate update relaxation
        - ``orthogonality_threshold`` : float, default ``0.49`` — max ``|cos φ|``
        - ``smalllink_threshold`` : float, default ``0.11`` — small flow-link threshold
        - ``require_both_criteria`` : bool, default ``False`` — enforce dual criteria
        - ``enforce_output_dual_criteria`` : bool — post-pass on final triangles
        - ``post_output_ortho_iter`` : int, default ``3`` — post-recovery cycles
        - ``max_recovery_iterations`` : int, default ``100`` — recovery cycle cap
        - ``recovery_stagnation_break`` : int, default ``10`` — stagnation stop
        - ``preserve_merged_quads`` : bool, default ``False`` — keep quad faces
        - ``spherical`` : bool, default ``False`` — lon/lat vs. planar geometry
        - ``merge_small_links`` : bool, default ``False`` — merge vs. flip-only mode
        - ``recovery_merge_small_links`` : bool, default ``True`` — in flip-only
          mode, merge the elements recovery cannot fix instead of failing
        - ``recovery_merge_from_iter`` : int, default ``2`` — recovery cycle
          (``recovery=N`` in the log) the last-resort merge starts at
        - ``disp`` : int or float, default ``4`` — progress interval; ``np.inf`` for quiet
    hfun : callable, optional
        Mesh-size function (reserved for future use).
    harg : tuple, optional
        Extra arguments for ``hfun``.
    fixed : array_like of int, optional
        Indices (into ``vert``) of vertices to hold fixed: they are never
        displaced by orthogonalization or smoothing. Node numbering is
        preserved by the pipeline, so the indices stay valid throughout.

    Returns
    -------
    vert : ndarray of shape (V, 2)
        Updated vertex coordinates.
    conn : ndarray of shape (E, 2)
        Updated constrained edges.
    tria : ndarray of shape (T, 3)
        Updated triangle connectivity (or mixed face rows if quads preserved).
    tnum : ndarray of shape (T, 1)
        Updated part indices.

    Notes
    -----
    Delegates to :mod:`bluemesh2d.ortho_merge.ortho_merge_iter`. See
    :mod:`bluemesh2d.ortho_merge.orthogonalize` for MeshKernel / Delft3D-FM
    references.
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

    opts = makeopt_smood(opts)

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
        raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    nvrt = vert.shape[0]

    if fixed is not None:
        fixed = np.unique(np.asarray(fixed, dtype=int).ravel())
        if fixed.size and (fixed.min() < 0 or fixed.max() >= nvrt):
            raise ValueError("smood:invalidInputs - Invalid FIXED input array.")

    if np.min(conn[:, :2]) < 0 or np.max(conn[:, :2]) > nvrt:
        raise ValueError("smood:invalidInputs - Invalid CONN input array.")

    if np.min(tria[:, :3]) < 0 or np.max(tria[:, :3]) > nvrt:
        raise ValueError("smood:invalidInputs - Invalid TRIA input array.")

    if not np.isinf(opts["disp"]):
        print("\n Smooth triangulation for Delft3D-FM computation...\n")

    return _ortho_merge_pipeline(vert, conn, tria, tnum, opts, fixed=fixed)


def makeopt_smood(opts=None):
    """Set up and validate the options dictionary for :func:`smood`.

    Parameters
    ----------
    opts : dict or None, optional
        User options; if ``None``, start from an empty dict.

    Returns
    -------
    opts : dict
        Validated options dictionary.
    """
    if opts is None:
        opts = {}

    if "iter" not in opts:
        # Default pipeline: 4 ortho <-> merge outer cycles.
        opts["iter"] = 4
    else:
        if not isinstance(opts["iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["iter"] <= 0:
            raise ValueError("smood:invalidOptionValues - Invalid OPT.ITER selection.")

    if "inner_iter" not in opts:
        opts["inner_iter"] = 4
    else:
        if not isinstance(opts["inner_iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["inner_iter"] <= 0:
            raise ValueError(
                "smood:invalidOptionValues - Invalid OPT.INNER_ITER selection."
            )

    if "ortho_factor" not in opts:
        opts["ortho_factor"] = 0.5
    else:
        if not isinstance(opts["ortho_factor"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if not (0.0 <= opts["ortho_factor"] <= 1.0):
            raise ValueError(
                "smood:invalidOptionValues - ORTHO_FACTOR must be in [0, 1]."
            )

    if "relaxation" not in opts:
        opts["relaxation"] = 0.75
    else:
        if not isinstance(opts["relaxation"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if not (0.0 < opts["relaxation"] <= 1.0):
            raise ValueError(
                "smood:invalidOptionValues - RELAXATION must be in (0, 1]."
            )

    if "disp" not in opts:
        opts["disp"] = 8
    else:
        if not isinstance(opts["disp"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["disp"] <= 0:
            raise ValueError("smood:invalidOptionValues - Invalid OPT.DISP selection.")

    if "vtol" not in opts:
        opts["vtol"] = 1.0e-3
    else:
        if not isinstance(opts["vtol"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if opts["vtol"] <= 0:
            raise ValueError("smood:invalidOptionValues - Invalid OPT.VTOL selection.")

    if "dbug" not in opts:
        opts["dbug"] = False
    else:
        if not isinstance(opts["dbug"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    if "orthogonality_threshold" not in opts:
        opts["orthogonality_threshold"] = 0.49  # max|cosphi|
    else:
        if not isinstance(opts["orthogonality_threshold"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        if not (0.0 <= opts["orthogonality_threshold"] <= 1.0):
            raise ValueError(
                "smood:invalidOptionValues - ORTHOGONALITY_THRESHOLD must be in [0, 1]."
            )

    if "smalllink_threshold" not in opts:
        opts["smalllink_threshold"] = DEFAULT_SMALLLINK_THRESHOLD
    else:
        if not isinstance(opts["smalllink_threshold"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["smalllink_threshold"] = float(opts["smalllink_threshold"])

    if "buffer_layers" not in opts:
        opts["buffer_layers"] = 2
    else:
        if not isinstance(opts["buffer_layers"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["buffer_layers"] = int(opts["buffer_layers"])
        if opts["buffer_layers"] <= 0:
            raise ValueError("smood:invalidOptionValues - buffer_layers must be > 0.")

    if "enable_edge_flips" not in opts:
        opts["enable_edge_flips"] = True
    else:
        if not isinstance(opts["enable_edge_flips"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    if "max_global_iter" not in opts:
        opts["max_global_iter"] = int(opts["inner_iter"]) + 2
    else:
        if not isinstance(opts["max_global_iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["max_global_iter"] = int(opts["max_global_iter"])
        if opts["max_global_iter"] <= 0:
            raise ValueError("smood:invalidOptionValues - max_global_iter must be > 0.")

    if "smooth_iter" not in opts:
        opts["smooth_iter"] = int(opts["inner_iter"]) * 4
    else:
        if not isinstance(opts["smooth_iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["smooth_iter"] = int(opts["smooth_iter"])
        if opts["smooth_iter"] <= 0:
            raise ValueError("smood:invalidOptionValues - smooth_iter must be > 0.")

    if "stop_if_no_merge" not in opts:
        opts["stop_if_no_merge"] = False
    else:
        if not isinstance(opts["stop_if_no_merge"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    # REQUIRE_BOTH_CRITERIA (fan-proxy dual check + recovery)
    if "require_both_criteria" not in opts:
        opts["require_both_criteria"] = DEFAULT_REQUIRE_STRICT_DUAL
    else:
        if not isinstance(opts["require_both_criteria"], bool):
            raise TypeError(
                "smood:incorrectInputClass - require_both_criteria must be bool."
            )

    if "max_recovery_iterations" not in opts:
        opts["max_recovery_iterations"] = 100
    else:
        if not isinstance(opts["max_recovery_iterations"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["max_recovery_iterations"] = int(opts["max_recovery_iterations"])
        if opts["max_recovery_iterations"] < 0:
            raise ValueError(
                "smood:invalidOptionValues - max_recovery_iterations must be >= 0."
            )

    if "recovery_stagnation_break" not in opts:
        opts["recovery_stagnation_break"] = 10
    else:
        if not isinstance(opts["recovery_stagnation_break"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["recovery_stagnation_break"] = int(opts["recovery_stagnation_break"])
        if opts["recovery_stagnation_break"] < 0:
            raise ValueError(
                "smood:invalidOptionValues - recovery_stagnation_break must be >= 0."
            )

    if "outer_stagnation_break" not in opts:
        opts["outer_stagnation_break"] = 2
    else:
        if not isinstance(opts["outer_stagnation_break"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["outer_stagnation_break"] = int(opts["outer_stagnation_break"])
        if opts["outer_stagnation_break"] < 0:
            raise ValueError(
                "smood:invalidOptionValues - outer_stagnation_break must be >= 0."
            )

    # Last-resort merge during recovery (flip-only mode) + the recovery cycle
    # it starts at, matching the `recovery=N` rows of the progress table.
    if "recovery_merge_small_links" not in opts:
        opts["recovery_merge_small_links"] = True
    else:
        if not isinstance(opts["recovery_merge_small_links"], bool):
            raise TypeError(
                "smood:incorrectInputClass - recovery_merge_small_links must be bool."
            )

    if "recovery_merge_from_iter" not in opts:
        opts["recovery_merge_from_iter"] = 2
    else:
        if not isinstance(opts["recovery_merge_from_iter"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["recovery_merge_from_iter"] = int(opts["recovery_merge_from_iter"])
        if opts["recovery_merge_from_iter"] < 0:
            raise ValueError(
                "smood:invalidOptionValues - recovery_merge_from_iter must be >= 0."
            )

    if "adaptive_recovery" not in opts:
        opts["adaptive_recovery"] = True
    else:
        if not isinstance(opts["adaptive_recovery"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    if "recovery_buffer_growth" not in opts:
        opts["recovery_buffer_growth"] = 1
    else:
        if not isinstance(opts["recovery_buffer_growth"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["recovery_buffer_growth"] = int(opts["recovery_buffer_growth"])
        if opts["recovery_buffer_growth"] < 0:
            raise ValueError(
                "smood:invalidOptionValues - recovery_buffer_growth must be >= 0."
            )

    if "recovery_smooth_iter_growth" not in opts:
        opts["recovery_smooth_iter_growth"] = 6
    else:
        if not isinstance(opts["recovery_smooth_iter_growth"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["recovery_smooth_iter_growth"] = int(opts["recovery_smooth_iter_growth"])
        if opts["recovery_smooth_iter_growth"] < 0:
            raise ValueError(
                "smood:invalidOptionValues - recovery_smooth_iter_growth must be >= 0."
            )

    if "recovery_global_iter_growth" not in opts:
        opts["recovery_global_iter_growth"] = 1
    else:
        if not isinstance(opts["recovery_global_iter_growth"], (int, float)):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")
        opts["recovery_global_iter_growth"] = int(opts["recovery_global_iter_growth"])
        if opts["recovery_global_iter_growth"] < 0:
            raise ValueError(
                "smood:invalidOptionValues - recovery_global_iter_growth must be >= 0."
            )

    if "preserve_merged_quads" not in opts:
        opts["preserve_merged_quads"] = False
    else:
        if not isinstance(opts["preserve_merged_quads"], bool):
            raise TypeError("smood:incorrectInputClass - Incorrect input class.")

    if "spherical" not in opts:
        opts["spherical"] = False
    else:
        if not isinstance(opts["spherical"], bool):
            raise TypeError("smood:incorrectInputClass - spherical must be bool.")

    if "merge_small_links" not in opts:
        opts["merge_small_links"] = False
    else:
        if not isinstance(opts["merge_small_links"], bool):
            raise TypeError(
                "smood:incorrectInputClass - merge_small_links must be bool."
            )

    return opts
