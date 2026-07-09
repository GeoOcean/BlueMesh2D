"""Merge circumcenters along small dual links into quadrilateral faces.

Identifies short flow links, merges adjacent triangle pairs into quads, and
rebuilds a mixed tri/quad UGRID dataset compatible with Delft3D-FM export.
"""

import numpy as np
import xarray as xr

from .grd_util import build_ugrid_arrays_mixed, triangulate_mixed_face_row_to_tris


def _faces_list_from_ds(ds: xr.Dataset) -> list:
    """0-based polygon faces (len 3 or 4) from ``mesh2d_face_nodes``."""
    face_nodes_raw = np.asarray(ds["mesh2d_face_nodes"].values, dtype=np.int64)
    start = int(ds["mesh2d_face_nodes"].attrs.get("start_index", 1))
    if start == 1:
        face0b = np.full_like(face_nodes_raw, -1)
        valid = face_nodes_raw > 0
        face0b[valid] = face_nodes_raw[valid] - 1
    else:
        face0b = face_nodes_raw.copy()
    faces: list = []
    for row in face0b:
        nodes = row[row >= 0]
        if nodes.size >= 3:
            faces.append(nodes.astype(np.int64, copy=True))
    return faces


def _mixed_faces_to_triangles(vert_xy: np.ndarray, faces: list) -> np.ndarray:
    """Expand mixed faces to triangle rows using the merge diagonal ``(v1, v2)``."""
    xy = np.asarray(vert_xy, dtype=np.float64)[:, :2]
    tris: list = []
    for f in faces:
        fn = np.asarray(f, dtype=np.int64).reshape(-1)
        if fn.size < 3:
            continue
        tris.extend(triangulate_mixed_face_row_to_tris(xy, fn))
    if not tris:
        return np.empty((0, 3), dtype=np.int64)
    return np.asarray(tris, dtype=np.int64)


def _signed_area_quad(vert, quad):
    """Signed area (doubled) of quadrilateral for orientation check."""
    v = vert[quad]
    return (
        (v[1, 0] - v[0, 0]) * (v[2, 1] - v[0, 1])
        - (v[2, 0] - v[0, 0]) * (v[1, 1] - v[0, 1])
        + (v[2, 0] - v[1, 0]) * (v[3, 1] - v[1, 1])
        - (v[3, 0] - v[1, 0]) * (v[2, 1] - v[1, 1])
    )


def _merge_small_links_into_faces(tria, edge_cc, small_link_indices, vert):
    """Merge triangle pairs along small links into quadrilateral faces."""
    merged = set()
    quads = []
    for idx in small_link_indices:
        e = edge_cc[idx]
        v1, v2, t1, t2 = int(e[0]), int(e[1]), int(e[2]), int(e[3])
        if t2 < 0 or t1 in merged or t2 in merged:
            continue
        tri1 = tria[t1]
        tri2 = tria[t2]
        mask1 = (tri1 != v1) & (tri1 != v2)
        mask2 = (tri2 != v1) & (tri2 != v2)
        if not np.any(mask1) or not np.any(mask2):
            continue
        a = int(tri1[mask1][0])
        b = int(tri2[mask2][0])
        quad = np.array([a, v1, b, v2], dtype=np.int32)
        if _signed_area_quad(vert, quad) < 0:
            quad = np.array([a, v2, b, v1], dtype=np.int32)
        quads.append((t1, t2, quad))
        merged.add(t1)
        merged.add(t2)

    faces_list = []
    for t1, t2, q in quads:
        faces_list.append(q)
    for i in range(len(tria)):
        if i not in merged:
            faces_list.append(tria[i])
    return faces_list


def _rebuild_ds_from_form(ds_ori, ugrid_arrays):
    """Rebuild an xarray dataset preserving the structure of ``ds_ori``."""

    def _da(name, data, dims, attrs=None):
        if name in ds_ori.variables and hasattr(ds_ori.variables[name], "attrs"):
            base_attrs = dict(ds_ori.variables[name].attrs)
        else:
            base_attrs = {}
        if attrs:
            base_attrs.update(attrs)
        return xr.DataArray(data=data, dims=dims, attrs=base_attrs)

    coords = {
        "mesh2d_node_x": _da(
            "mesh2d_node_x",
            ugrid_arrays["node_x"],
            ("mesh2d_nNodes",),
        ),
        "mesh2d_node_y": _da(
            "mesh2d_node_y",
            ugrid_arrays["node_y"],
            ("mesh2d_nNodes",),
        ),
    }

    data_vars = {
        "mesh2d_node_z": _da(
            "mesh2d_node_z",
            ugrid_arrays["node_z"],
            ("mesh2d_nNodes",),
        ),
        "mesh2d_edge_x": _da(
            "mesh2d_edge_x",
            ugrid_arrays["edge_x"],
            ("mesh2d_nEdges",),
        ),
        "mesh2d_edge_y": _da(
            "mesh2d_edge_y",
            ugrid_arrays["edge_y"],
            ("mesh2d_nEdges",),
        ),
        "mesh2d_edge_nodes": _da(
            "mesh2d_edge_nodes",
            ugrid_arrays["edge_nodes"],
            ("mesh2d_nEdges", "Two"),
        ),
        "mesh2d_edge_faces": _da(
            "mesh2d_edge_faces",
            ugrid_arrays["edge_faces"],
            ("mesh2d_nEdges", "Two"),
        ),
        "mesh2d_face_nodes": _da(
            "mesh2d_face_nodes",
            ugrid_arrays["face_nodes"],
            ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
        ),
        "mesh2d_face_x": _da(
            "mesh2d_face_x",
            ugrid_arrays["face_x"],
            ("mesh2d_nFaces",),
        ),
        "mesh2d_face_y": _da(
            "mesh2d_face_y",
            ugrid_arrays["face_y"],
            ("mesh2d_nFaces",),
        ),
        "mesh2d_face_x_bnd": _da(
            "mesh2d_face_x_bnd",
            ugrid_arrays["face_x_bnd"],
            ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
        ),
        "mesh2d_face_y_bnd": _da(
            "mesh2d_face_y_bnd",
            ugrid_arrays["face_y_bnd"],
            ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
        ),
    }

    # Copy non-mesh variables from ds_ori (e.g. wgs84, mesh2d topology variable)
    mesh_dims = {
        "mesh2d_nNodes",
        "mesh2d_nEdges",
        "mesh2d_nFaces",
        "mesh2d_nMax_face_nodes",
    }
    for k in ds_ori.variables:
        if k in coords or k in data_vars:
            continue
        v = ds_ori.variables[k]
        if set(v.dims) & mesh_dims:
            continue
        data_vars[k] = xr.DataArray(
            data=ds_ori[k].values.copy(),
            dims=v.dims,
            attrs=dict(v.attrs),
        )

    ds_final = xr.Dataset(data_vars=data_vars, coords=coords, attrs=dict(ds_ori.attrs))
    return ds_final


def merge_circumcenters(
    ds_ori,
    removesmalllinkstrsh=0.1,
    jsferic=1,
):
    """Merge triangle pairs with small circumcenter links into quads.

    Identifies short dual flow links, merges each adjacent triangle pair into
    one quadrilateral, and rebuilds a Delft3D-FM UGRID dataset with the same
    structure as ``ds_ori``.

    Parameters
    ----------
    ds_ori : xarray.Dataset
        Input UGRID Delft3D-FM mesh.
    removesmalllinkstrsh : float, optional
        Threshold for small flow links. Default is 0.1.
    jsferic : int, optional
        Coordinate mode: ``1`` treats node coordinates as lon/lat degrees;
        ``0`` treats them as planar ``x/y``. Default is 1.

    Returns
    -------
    ds_final : xarray.Dataset
        Rebuilt mesh with small links replaced by quads.
        ``mesh2d_nMax_face_nodes`` is 4; triangles are padded in the fourth
        node column.
    """
    node_x = np.asarray(ds_ori["mesh2d_node_x"].values, dtype=np.float64)
    node_y = np.asarray(ds_ori["mesh2d_node_y"].values, dtype=np.float64)
    node_z = np.asarray(ds_ori["mesh2d_node_z"].values, dtype=np.float64)
    vert = np.column_stack([node_x, node_y])

    faces_in = _faces_list_from_ds(ds_ori)

    # Triangulate mixed faces to a *triangle proxy* using the same diagonal
    # (v1, v2) logic as the rest of the pipeline/export.
    tria = _mixed_faces_to_triangles(vert, faces_in)
    tria = np.asarray(tria, dtype=np.int32)

    # IMPORTANT: use the same small-link metric as ortho_merge.orthogonalize,
    # so our dual checks and merge decisions are consistent.
    from ..ortho_merge import orthogonalize as ortho
    from ..ortho_merge.geometry import build_edges_from_tria

    edge_nodes, edge_faces = build_edges_from_tria(tria)
    nlinktoosmall, small_edge_indices = ortho.compute_small_links_from_arrays(
        node_x=node_x,
        node_y=node_y,
        face_nodes=tria,
        edge_nodes=edge_nodes,
        edge_faces=edge_faces,
        removesmalllinkstrsh=float(removesmalllinkstrsh),
        jsferic=jsferic,
    )

    if nlinktoosmall > 0:
        # Build edge_cc compatible with `_merge_small_links_into_faces`:
        # [v1, v2, t1, t2] where t1/t2 are adjacent face indices in `tria`.
        edge_cc = np.column_stack(
            [
                edge_nodes[:, 0],
                edge_nodes[:, 1],
                edge_faces[:, 0],
                edge_faces[:, 1],
            ]
        ).astype(np.int32, copy=False)

        faces_list = _merge_small_links_into_faces(
            tria, edge_cc, small_edge_indices, vert
        )
    else:
        # No small links detected: keep the existing mixed topology.
        faces_list = [np.asarray(f, dtype=np.int32).copy() for f in faces_in]

    NODE = np.column_stack([vert[:, 0], vert[:, 1], node_z])
    ugrid_arrays = build_ugrid_arrays_mixed(NODE, faces_list)
    ds_final = _rebuild_ds_from_form(ds_ori, ugrid_arrays)
    return ds_final
