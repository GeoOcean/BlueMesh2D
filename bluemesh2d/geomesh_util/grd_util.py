"""
Delft3D-FM UGRID (xarray) builders, mixed tri/quad connectivity, and ADCIRC helpers.

Used by :mod:`bluemesh2d.smood` and ``merge_circumcenters``.

``xarray`` is imported lazily so the array-level helpers (``build_ugrid_arrays``,
``calculate_edges``, ...) stay usable without it; only the ``xr.Dataset``
builders (``adcirc2DFlowFM*``) require it.
"""
from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

try:
    import xarray as xr
except ImportError:  # optional: only needed for the xr.Dataset builders
    xr = None


def _signed_area_tri_xy(xy: np.ndarray, i: int, j: int, k: int) -> float:
    """Twice the signed triangle area in the x–y plane (CCW > 0)."""
    p, q, r = xy[i], xy[j], xy[k]
    return (q[0] - p[0]) * (r[1] - p[1]) - (r[0] - p[0]) * (q[1] - p[1])


def triangulate_mixed_face_row_to_tris(
    node_xy: np.ndarray, nodes_valid: np.ndarray
) -> list[tuple[int, int, int]]:
    """
    Build triangle connectivity for one polygon face (valid node indices only).

    - 3 nodes: one triangle.
    - 4 nodes: ``merge_circumcenters`` stores quads as ``[a, v1, b, v2]`` where
      ``(v1, v2)`` was the merged small-link edge. Triangulate by splitting along
      that diagonal: ``(a, v1, v2)`` and ``(b, …)`` with winding chosen so both
      triangles have the same signed area sign as the quad half (Delft3D-FM dual
      geometry is consistent with this choice; fan-from-``a`` uses diagonal ``(a, v2)``
      and can yield ``|cosφ| → 1`` on export).
    - 5+ nodes: fan around the first node.
    """
    nodes = np.asarray(nodes_valid, dtype=np.int64).reshape(-1)
    n = int(nodes.size)
    if n < 3:
        return []
    xy = np.asarray(node_xy, dtype=np.float64)
    if n == 3:
        return [(int(nodes[0]), int(nodes[1]), int(nodes[2]))]
    if n == 4:
        a, v1, b, v2 = int(nodes[0]), int(nodes[1]), int(nodes[2]), int(nodes[3])

        def sa(ii: int, jj: int, kk: int) -> float:
            return _signed_area_tri_xy(xy, ii, jj, kk)

        t1 = (a, v1, v2)
        if sa(*t1) < 0:
            t1 = (a, v2, v1)
        o1 = sa(*t1)
        t2 = None
        for cand in ((b, v1, v2), (b, v2, v1)):
            if o1 != 0.0 and sa(*cand) * o1 > 0:
                t2 = cand
                break
        if t2 is None:
            t2 = (b, v2, v1)
        return [t1, t2]
    tris: list[tuple[int, int, int]] = []
    n0 = int(nodes[0])
    for i in range(1, n - 1):
        tris.append((n0, int(nodes[i]), int(nodes[i + 1])))
    return tris


def face_nodes_0b_to_faces_list(face_nodes_0b: np.ndarray) -> list:
    """
    Convert a (F, 4) face-node array (0-based, ``-1`` padding) to a list of
    variable-length node index arrays (triangles length 3, quads length 4).
    """
    out: list = []
    for row in np.asarray(face_nodes_0b, dtype=np.int64):
        nodes = row[row >= 0]
        if nodes.size >= 3:
            out.append(nodes.copy())
    return out


def validate_mixed_export_matches_smood_tria(
    vert_xy: np.ndarray,
    face_nodes_0b: np.ndarray,
    tria_smood: np.ndarray,
) -> tuple[int, int]:
    """
    Check that mixed ``face_nodes_0b`` (final ortho+merge topology) expands to the **same**
    set of triangles as ``tria_smood`` returned by :func:`bluemesh2d.smood.smood` (same node
    coordinates). Catches accidental export of the wrong connectivity or a silent fallback.

    Returns
    -------
    n_tri_faces, n_quad_faces
    """
    vert_xy = np.asarray(vert_xy, dtype=np.float64)
    if vert_xy.ndim != 2 or vert_xy.shape[1] < 2:
        raise ValueError("vert_xy must have shape (N, 2+)")
    xy = vert_xy[:, :2]
    fn = np.asarray(face_nodes_0b, dtype=np.int64)
    tria_smood = np.asarray(tria_smood, dtype=np.int64)
    if tria_smood.ndim != 2 or tria_smood.shape[1] != 3:
        raise ValueError("tria_smood must have shape (T, 3)")

    expanded: list[tuple[int, int, int]] = []
    n_tri_f = n_quad_f = 0
    for row in fn:
        nodes = row[row >= 0]
        if nodes.size < 3:
            continue
        if int(nodes.min()) < 0 or int(nodes.max()) >= vert_xy.shape[0]:
            raise ValueError(
                f"face node index out of range [0, {vert_xy.shape[0]}): {nodes!r}"
            )
        if nodes.size == 4:
            n_quad_f += 1
        else:
            n_tri_f += 1
        expanded.extend(triangulate_mixed_face_row_to_tris(xy, nodes))

    exp = np.asarray(expanded, dtype=np.int64)
    if exp.shape != tria_smood.shape:
        raise ValueError(
            "Export topology mismatch: mixed faces expand to "
            f"{exp.shape[0]} triangles, but smood returned {tria_smood.shape[0]}. "
            "You may be exporting triangle-only (quads re-split) or a stale ``face_nodes``."
        )

    def _sort_rows(t: np.ndarray) -> np.ndarray:
        return np.sort(np.asarray(t, dtype=np.int64), axis=1)

    es = _sort_rows(exp)
    ts = _sort_rows(tria_smood)
    order_e = np.lexsort((es[:, 2], es[:, 1], es[:, 0]))
    order_t = np.lexsort((ts[:, 2], ts[:, 1], ts[:, 0]))
    if not np.array_equal(es[order_e], ts[order_t]):
        raise ValueError(
            "Triangle set from mixed faces ≠ smood triangle output (winding/diagonal mismatch)."
        )
    return int(n_tri_f), int(n_quad_f)


def _xr_dataset_from_ugrid_dict(ugrid: dict) -> xr.Dataset:
    """Build the standard Delft3D-FM-style xarray Dataset from a ``build_ugrid_arrays*`` dict."""
    if xr is None:
        raise ImportError(
            "xarray is required for the xr.Dataset builders (adcirc2DFlowFM*). "
            "Either install xarray, or use build_ugrid_arrays() and write the "
            "NetCDF yourself (e.g. with netCDF4)."
        )
    node_z_out = -ugrid["node_z"]
    _WGS84_FILL = np.int32(-2147483647)
    _UGRID_FILL = np.int32(-999)

    coords = {
        "mesh2d_node_x": xr.DataArray(
            ugrid["node_x"],
            dims=("mesh2d_nNodes",),
            attrs={
                "standard_name": "longitude",
                "long_name": "x-coordinate of mesh nodes",
                "units": "degrees_east",
            },
        ),
        "mesh2d_node_y": xr.DataArray(
            ugrid["node_y"],
            dims=("mesh2d_nNodes",),
            attrs={
                "standard_name": "latitude",
                "long_name": "y-coordinate of mesh nodes",
                "units": "degrees_north",
            },
        ),
    }

    data_vars = {
        "mesh2d_node_z": xr.DataArray(
            node_z_out,
            dims=("mesh2d_nNodes",),
            attrs={
                "mesh": "mesh2d",
                "location": "node",
                "units": "m",
                "standard_name": "altitude",
                "long_name": "z-coordinate of mesh nodes",
                "grid_mapping": "wgs84",
            },
        ),
        "mesh2d_edge_x": xr.DataArray(
            ugrid["edge_x"],
            dims=("mesh2d_nEdges",),
            attrs={
                "standard_name": "projection_x_coordinate",
                "long_name": "characteristic x-coordinate of the mesh edge (e.g. midpoint)",
                "units": "degrees_east",
                # Keep standard_name as in GUI (longitude) instead of projection_x_coordinate.
                "standard_name": "longitude",
            },
        ),
        "mesh2d_edge_y": xr.DataArray(
            ugrid["edge_y"],
            dims=("mesh2d_nEdges",),
            attrs={
                "standard_name": "projection_y_coordinate",
                "long_name": "characteristic y-coordinate of the mesh edge (e.g. midpoint)",
                "units": "degrees_north",
                "standard_name": "latitude",
            },
        ),
        "mesh2d_edge_nodes": xr.DataArray(
            ugrid["edge_nodes"],
            dims=("mesh2d_nEdges", "Two"),
            attrs={
                "cf_role": "edge_node_connectivity",
                "long_name": "Start and end nodes of mesh edges",
                "start_index": 1,
            },
        ),
        "mesh2d_edge_faces": xr.DataArray(
            ugrid["edge_faces"],
            dims=("mesh2d_nEdges", "Two"),
            attrs={
                "cf_role": "edge_face_connectivity",
                "long_name": "Neighboring faces of mesh edges",
                "start_index": np.int32(1),
            },
        ).assign_attrs(_FillValue=_UGRID_FILL),
        "mesh2d_face_nodes": xr.DataArray(
            ugrid["face_nodes"],
            dims=("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
            attrs={
                "cf_role": "face_node_connectivity",
                "long_name": "Vertex nodes of mesh faces (counterclockwise)",
                "start_index": np.int32(1),
                "coordinates": "mesh2d_node_x mesh2d_node_y",
            },
        ).assign_attrs(_FillValue=_UGRID_FILL),
        "mesh2d_face_x": xr.DataArray(
            ugrid["face_x"],
            dims=("mesh2d_nFaces",),
            attrs={
                "units": "degrees_east",
                "standard_name": "longitude",
                "long_name": "Characteristic x-coordinate of mesh face",
                "bounds": "mesh2d_face_x_bnd",
            },
        ),
        "mesh2d_face_y": xr.DataArray(
            ugrid["face_y"],
            dims=("mesh2d_nFaces",),
            attrs={
                "units": "degrees_north",
                "standard_name": "latitude",
                "long_name": "Characteristic y-coordinate of mesh face",
                "bounds": "mesh2d_face_y_bnd",
            },
        ),
        "mesh2d_face_x_bnd": xr.DataArray(
            ugrid["face_x_bnd"],
            dims=("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
            attrs={
                "long_name": "x-coordinate bounds of mesh faces (i.e. corner coordinates)",
                "units": "degrees_east",
                "standard_name": "longitude",
            },
        ),
        "mesh2d_face_y_bnd": xr.DataArray(
            ugrid["face_y_bnd"],
            dims=("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
            attrs={
                "long_name": "y-coordinate bounds of mesh faces (i.e. corner coordinates)",
                "units": "degrees_north",
                "standard_name": "latitude",
            },
        ),
        "wgs84": xr.DataArray(
            np.int32(4326),
            dims=(),
            attrs={
                "name": "WGS 84",
                "epsg": np.int32(4326),
                "grid_mapping_name": "latitude_longitude",
                "longitude_of_prime_meridian": 0.0,
                "semi_major_axis": 6378137.0,
                "semi_minor_axis": 6356752.314245,
                "inverse_flattening": 298.257223563,
                "EPSG_code": "",
                "value": "value is equal to EPSG code",
                "proj_string": "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
            },
        ).assign_attrs(_FillValue=_WGS84_FILL),
        "mesh2d": xr.DataArray(
            _WGS84_FILL,
            dims=(),
            attrs={
                "cf_role": "mesh_topology",
                "long_name": "Topology data of 2D mesh",
                "topology_dimension": 2,
                "node_coordinates": "mesh2d_node_x mesh2d_node_y",
                "node_dimension": "mesh2d_nNodes",
                "edge_node_connectivity": "mesh2d_edge_nodes",
                "edge_dimension": "mesh2d_nEdges",
                "edge_coordinates": "mesh2d_edge_x mesh2d_edge_y",
                "face_node_connectivity": "mesh2d_face_nodes",
                "face_dimension": "mesh2d_nFaces",
                "face_coordinates": "mesh2d_face_x mesh2d_face_y",
                "max_face_nodes_dimension": "mesh2d_nMax_face_nodes",
                "edge_face_connectivity": "mesh2d_edge_faces",
            },
        ),
    }

    attrs = {
        "institution": "GeoOcean",
        "references": "https://github.com/GeoOcean/BlueMath_tk",
        "source": f"BlueMath tk {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "history": "Created with OCSmesh",
        "Conventions": "CF-1.8 UGRID-1.0 Deltares-0.10",
    }

    return xr.Dataset(data_vars=data_vars, coords=coords, attrs=attrs)


def adcirc2DFlowFM_mixed(NODE: np.ndarray, face_nodes_0b: np.ndarray) -> xr.Dataset:
    """
    Build a UGRID dataset that keeps **quads + triangles** (e.g. after ``merge_circumcenters``).

    If you merge two triangles into a quad to remove a short dual link, then export
    a **triangle-only** mesh by re-splitting the quad, you recover the **same two
    triangles** and the same two triangle circumcenters — small-link checks still
    fail. Call this with the mixed ``face_nodes`` from the merge pipeline instead.
    """
    NODE = np.asarray(NODE, dtype=np.float64)
    if NODE.ndim != 2 or NODE.shape[1] < 3:
        raise ValueError("NODE must have shape (n_nodes, 3) with x, y, z")
    faces_list = face_nodes_0b_to_faces_list(face_nodes_0b)
    ugrid = build_ugrid_arrays_mixed(NODE, faces_list)
    ds = _xr_dataset_from_ugrid_dict(ugrid)
    # Defensive re-encoding:
    # Delft3D-FM / GUI is sensitive to the raw NetCDF encoding of UGRID connectivity
    # variables. In particular, `mesh2d_face_nodes` / `mesh2d_edge_faces` must be
    # int32 with `_FillValue=-999`.
    #
    # Note: `_xr_dataset_from_ugrid_dict` already sets `_FillValue` in the variable *attrs*.
    # Setting `_FillValue` again via `encoding` makes xarray error out with:
    # "failed to prevent overwriting existing key _FillValue in attrs".
    _UGRID_FILL = np.int32(-999)

    if "mesh2d_face_nodes" in ds:
        da = ds["mesh2d_face_nodes"]
        vals = da.values
        if np.issubdtype(vals.dtype, np.floating):
            # Replace NaN padding with GUI-like -999 fill, then cast to int32.
            vals = np.where(np.isnan(vals), _UGRID_FILL, vals)
            ds["mesh2d_face_nodes"] = da.copy(data=np.asarray(vals, dtype=np.int32))
        # Ensure dtype is int32 (but do not touch encoding _FillValue).
        if ds["mesh2d_face_nodes"].dtype != np.int32:
            ds["mesh2d_face_nodes"] = ds["mesh2d_face_nodes"].astype(np.int32)

    if "mesh2d_edge_faces" in ds:
        da = ds["mesh2d_edge_faces"]
        vals = da.values
        if np.issubdtype(vals.dtype, np.floating):
            vals = np.where(np.isnan(vals), _UGRID_FILL, vals)
            ds["mesh2d_edge_faces"] = da.copy(data=np.asarray(vals, dtype=np.int32))
        if ds["mesh2d_edge_faces"].dtype != np.int32:
            ds["mesh2d_edge_faces"] = ds["mesh2d_edge_faces"].astype(np.int32)

    # Match GUI type for the WGS84 grid mapping scalar (dtype only).
    if "wgs84" in ds and ds["wgs84"].dtype != np.int32:
        ds["wgs84"] = ds["wgs84"].astype(np.int32)

    n_quad = sum(1 for f in faces_list if len(f) == 4)
    n_tri = sum(1 for f in faces_list if len(f) == 3)
    ds.attrs["bluemesh2d_export"] = "adcirc2DFlowFM_mixed"
    ds.attrs["bluemesh2d_n_faces"] = str(len(faces_list))
    ds.attrs["bluemesh2d_n_triangle_faces"] = str(n_tri)
    ds.attrs["bluemesh2d_n_quad_faces"] = str(n_quad)
    return ds


def adcirc2DFlowFM(NODE: np.ndarray, EDGE: np.ndarray) -> xr.Dataset:
    """
    Build a Delft3D FM UGRID mesh Dataset from ADCIRC-style node and triangle data.

    Parameters
    ----------
    NODE : np.ndarray
        Array of shape (n_nodes, 3) containing node coordinates (x, y, z).
    EDGE : np.ndarray
        Either:
        - (n_faces, 3) triangle connectivity (0-based node indices), OR
        - (n_faces, 4) face_nodes (0-based) with fill = -1 (triangles padded to 4).

    Returns
    -------
    xr.Dataset
        UGRID mesh Dataset (mesh2d_node_x/y/z, mesh2d_face_nodes, etc.).
        Use ds.to_netcdf(path) to write to file.
    """
    # Shape-based export selection:
    # - (T,3) int : triangle connectivity => export triangle-only mesh.
    # - (F,4) int with negative padding (typically -1) => face_nodes_0b => export mixed
    #   tri+quad mesh using adcirc2DFlowFM_mixed.
    # - (F,4) int without negative padding => assume legacy "face nodes" encoding and
    #   triangulate them (triangle-only export) for backward compatibility.
    EDGE = np.asarray(EDGE)
    if EDGE.ndim == 2 and EDGE.shape[1] == 4 and np.any(EDGE < 0):
        # face_nodes_0b with -1 padding => keep mixed topology.
        return adcirc2DFlowFM_mixed(NODE, EDGE)

    if EDGE.ndim == 2 and EDGE.shape[1] == 4:
        # Legacy: triangulate face rows to triangle-only.
        node_xy = np.asarray(NODE[:, :2], dtype=np.float64)
        tri: list[tuple[int, int, int]] = []
        for row in EDGE:
            nodes = row[row >= 0]
            if nodes.size >= 3:
                tri.extend(triangulate_mixed_face_row_to_tris(node_xy, nodes))
        EDGE = np.asarray(tri, dtype=np.int64)

    ugrid = build_ugrid_arrays(NODE, EDGE)
    return _xr_dataset_from_ugrid_dict(ugrid)


def calculate_edges(Elmts: np.ndarray) -> np.ndarray:
    """
    Calculates the unique edges from the given triangle elements.

    Parameters
    ----------
    Elmts : np.ndarray
        A 2D array of shape (nelmts, 3) containing the node indices for each triangle element.

    Returns
    -------
    np.ndarray
        A 2D array of shape (n_edges, 2) containing the unique edges,
        each represented by a pair of node indices.
    """

    Links = np.zeros((len(Elmts) * 3, 2), dtype=int)
    tel = 0
    for elmt in Elmts:
        Links[tel] = [elmt[0], elmt[1]]
        tel += 1
        Links[tel] = [elmt[1], elmt[2]]
        tel += 1
        Links[tel] = [elmt[2], elmt[0]]
        tel += 1

    Links_sorted = np.sort(Links, axis=1)
    Links_unique = np.unique(Links_sorted, axis=0)

    return Links_unique


def build_ugrid_arrays(NODE: np.ndarray, EDGE: np.ndarray) -> dict:
    """
    Build UGRID mesh arrays from node coordinates and triangle connectivity.
    Same logic as adcirc2DFlowFM but returns arrays instead of writing to file.
    Used to rebuild ds_final from (vert, z, tria) after edge flips.

    Parameters
    ----------
    NODE : np.ndarray
        Array of shape (n_nodes, 3) containing node coordinates (x, y, z).
    EDGE : np.ndarray
        Array of shape (n_faces, 3) containing triangle connectivity (0-based node indices).

    Returns
    -------
    dict
        Keys: node_x, node_y, node_z, face_nodes (1-based), edge_nodes (1-based),
        edge_faces (1-based), face_x, face_y, edge_x, edge_y, face_x_bnd, face_y_bnd.
        Also num_nodes, num_faces, num_edges for dimension sizes.
    """
    edges = calculate_edges(EDGE) + 1
    EDGE_S = np.sort(EDGE, axis=1)
    EDGE_S = EDGE_S[EDGE_S[:, 2].argsort()]
    EDGE_S = EDGE_S[EDGE_S[:, 1].argsort()]
    face_node = np.array(EDGE_S[EDGE_S[:, 0].argsort()], dtype=np.int32)
    edge_node = np.zeros([len(edges), 2], dtype="i4")
    edge_face = np.zeros([len(edges), 2], dtype=np.double)
    edge_x = np.zeros(len(edges))
    edge_y = np.zeros(len(edges))

    face_x = (
        NODE[EDGE[:, 0].astype(int), 0]
        + NODE[EDGE[:, 1].astype(int), 0]
        + NODE[EDGE[:, 2].astype(int), 0]
    ) / 3
    face_y = (
        NODE[EDGE[:, 0].astype(int), 1]
        + NODE[EDGE[:, 1].astype(int), 1]
        + NODE[EDGE[:, 2].astype(int), 1]
    ) / 3

    edge_x = (NODE[edges[:, 0] - 1, 0] + NODE[edges[:, 1] - 1, 0]) / 2
    edge_y = (NODE[edges[:, 0] - 1, 1] + NODE[edges[:, 1] - 1, 1]) / 2

    face_node_dict = {}
    for idx, face in enumerate(face_node):
        for node in face:
            if node not in face_node_dict:
                face_node_dict[node] = []
            face_node_dict[node].append(idx)

    for i, edge in enumerate(edges):
        node1, node2 = map(int, edge)
        edge_node[i, 0] = node1
        edge_node[i, 1] = node2
        faces_node1 = face_node_dict.get(node1 - 1, [])
        faces_node2 = face_node_dict.get(node2 - 1, [])
        faces = list(set(faces_node1) & set(faces_node2))
        if len(faces) < 2:
            edge_face[i, 0] = faces[0] + 1 if faces else 0
            edge_face[i, 1] = 0
        else:
            edge_face[i, 0] = faces[0] + 1
            edge_face[i, 1] = faces[1] + 1

    face_x = np.asarray(face_x, dtype=np.float64)
    face_y = np.asarray(face_y, dtype=np.float64)
    node_x = np.asarray(NODE[:, 0], dtype=np.float64)
    node_y = np.asarray(NODE[:, 1], dtype=np.float64)
    node_z = np.asarray(NODE[:, 2], dtype=np.float64)
    face_x_bnd = np.asarray(node_x[face_node], dtype=np.float64)
    face_y_bnd = np.asarray(node_y[face_node], dtype=np.float64)

    return {
        "node_x": node_x,
        "node_y": node_y,
        "node_z": node_z,
        "face_nodes": face_node + 1,
        "edge_nodes": edge_node,
        "edge_faces": edge_face,
        "face_x": face_x,
        "face_y": face_y,
        "edge_x": edge_x,
        "edge_y": edge_y,
        "face_x_bnd": face_x_bnd,
        "face_y_bnd": face_y_bnd,
        "num_nodes": NODE.shape[0],
        "num_faces": EDGE.shape[0],
        "num_edges": edges.shape[0],
    }


def build_ugrid_arrays_mixed(NODE: np.ndarray, faces_list: list) -> dict:
    """
    Build UGRID mesh arrays from node coordinates and mixed faces (triangles and quads).
    Each element of faces_list is an array of 3 or 4 node indices (0-based).

    Parameters
    ----------
    NODE : np.ndarray
        Array of shape (n_nodes, 3) containing node coordinates (x, y, z).
    faces_list : list of np.ndarray
        Each array has shape (3,) or (4,) with 0-based node indices.

    Returns
    -------
    dict
        Same keys as build_ugrid_arrays, with mesh2d_nMax_face_nodes = 4.
    face_nodes has shape (n_faces, 4) with padding encoded like the GUI exports:
        triangles padded with `-999` in 4th column, with `_FillValue=-999`.
    """
    n_nodes = NODE.shape[0]
    n_faces = len(faces_list)
    node_x = np.asarray(NODE[:, 0], dtype=np.float64)
    node_y = np.asarray(NODE[:, 1], dtype=np.float64)
    node_z = np.asarray(NODE[:, 2], dtype=np.float64)
    # GUI-like connectivity encoding (raw NetCDF):
    # - face_nodes / edge_faces are int32 with `_FillValue=-999`.
    #   Xarray will typically decode -999 -> NaN depending on how the file is read.
    FILL = np.int32(-999)

    face_nodes = np.full((n_faces, 4), FILL, dtype=np.int32)
    face_x = np.zeros(n_faces, dtype=np.float64)
    face_y = np.zeros(n_faces, dtype=np.float64)
    face_x_bnd = np.zeros((n_faces, 4), dtype=np.float64)
    face_y_bnd = np.zeros((n_faces, 4), dtype=np.float64)

    faces_norm = [np.asarray(f, dtype=np.int32).reshape(-1) for f in faces_list]
    nv_arr = np.fromiter((f.size for f in faces_norm), dtype=np.int64, count=n_faces)

    # Half-edges of all faces (built CCW), in face-then-edge traversal order.
    he_v1_parts: list[np.ndarray] = []
    he_v2_parts: list[np.ndarray] = []
    he_fi_parts: list[np.ndarray] = []
    he_k_parts: list[np.ndarray] = []

    # Vectorized per group of equal vertex count (3 or 4).
    for nv in np.unique(nv_arr):
        nv = int(nv)
        gidx = np.where(nv_arr == nv)[0]
        fv = np.vstack([faces_norm[i] for i in gidx])
        x = node_x[fv]
        y = node_y[fv]
        kp1 = (np.arange(nv) + 1) % nv
        if nv >= 3:
            # Signed area in current vertex order; build faces in CCW order to
            # match UGRID expectations and Delft3D-FM polygon orientation
            # conventions. For convex polygons this reliably detects CW/CCW.
            d1 = np.zeros(gidx.size, dtype=np.float64)
            d2 = np.zeros(gidx.size, dtype=np.float64)
            for k in range(nv):
                d1 += x[:, k] * y[:, kp1[k]]
                d2 += y[:, k] * x[:, kp1[k]]
            flip = 0.5 * (d1 - d2) < 0.0
            if np.any(flip):
                fv[flip] = fv[flip, ::-1]
                x = node_x[fv]
                y = node_y[fv]

        face_nodes[gidx[:, None], np.arange(nv)] = fv + 1
        # Area-weighted polygon centroid (shoelace formula), with a fallback
        # to the vertex mean for near-degenerate polygons.
        cross = np.empty_like(x)
        for k in range(nv):
            cross[:, k] = x[:, k] * y[:, kp1[k]] - x[:, kp1[k]] * y[:, k]
        area2 = np.zeros(gidx.size, dtype=np.float64)
        sx = np.zeros(gidx.size, dtype=np.float64)
        sy = np.zeros(gidx.size, dtype=np.float64)
        for k in range(nv):
            area2 += cross[:, k]
            sx += (x[:, k] + x[:, kp1[k]]) * cross[:, k]
            sy += (y[:, k] + y[:, kp1[k]]) * cross[:, k]
        degenerate = np.abs(area2) < 1e-30
        den = np.where(degenerate, 1.0, 3.0 * area2)
        face_x[gidx] = np.where(degenerate, np.mean(x, axis=1), sx / den)
        face_y[gidx] = np.where(degenerate, np.mean(y, axis=1), sy / den)
        face_x_bnd[gidx[:, None], np.arange(nv)] = x
        face_y_bnd[gidx[:, None], np.arange(nv)] = y
        if nv == 3:
            face_x_bnd[gidx, 3] = np.nan
            face_y_bnd[gidx, 3] = np.nan

        he_v1_parts.append(fv.ravel())
        he_v2_parts.append(fv[:, kp1].ravel())
        he_fi_parts.append(np.repeat(gidx, nv))
        he_k_parts.append(np.tile(np.arange(nv), gidx.size))

    # Build unique edges and edge->face mapping. Edges are numbered by first
    # occurrence in face-then-edge traversal order, and each edge's faces are
    # kept in traversal order (same as the historical dict-based loop).
    if not he_v1_parts:
        he_v1_parts = [np.zeros(0, dtype=np.int64)]
        he_v2_parts = [np.zeros(0, dtype=np.int64)]
        he_fi_parts = [np.zeros(0, dtype=np.int64)]
        he_k_parts = [np.zeros(0, dtype=np.int64)]
    he_v1 = np.concatenate(he_v1_parts).astype(np.int64)
    he_v2 = np.concatenate(he_v2_parts).astype(np.int64)
    he_fi = np.concatenate(he_fi_parts)
    he_k = np.concatenate(he_k_parts)
    order = np.lexsort((he_k, he_fi))
    he_v1, he_v2, he_fi = he_v1[order], he_v2[order], he_fi[order]
    lo = np.minimum(he_v1, he_v2)
    hi = np.maximum(he_v1, he_v2)
    key = lo * np.int64(n_nodes + 1) + hi
    _, first_pos, inverse = np.unique(key, return_index=True, return_inverse=True)
    insertion = np.argsort(first_pos, kind="stable")
    rank = np.empty(insertion.size, dtype=np.int64)
    rank[insertion] = np.arange(insertion.size)
    edge_id = rank[inverse]

    edges = np.column_stack(
        [lo[first_pos][insertion], hi[first_pos][insertion]]
    ).astype(np.int32)
    edge_node = edges + 1
    # GUI padding for boundary edges:
    # In the working Delft3D-FM GUI export, missing neighbor faces in
    # `mesh2d_edge_faces(:, 1)` are encoded as 0 (NOT as `_FillValue`).
    # Using 0 instead of `_FillValue` makes UGRID import validation pass.
    edge_face = np.zeros((len(edges), 2), dtype=np.int32)
    pos = np.argsort(edge_id, kind="stable")
    sid = edge_id[pos]
    is_first = np.r_[True, sid[1:] != sid[:-1]]
    starts = np.flatnonzero(is_first)
    edge_face[sid[starts], 0] = he_fi[pos[starts]] + 1
    seconds = starts + 1
    seconds = seconds[seconds < sid.size]
    seconds = seconds[~is_first[seconds]]
    edge_face[sid[seconds], 1] = he_fi[pos[seconds]] + 1

    edge_x = (node_x[edges[:, 0]] + node_x[edges[:, 1]]) / 2
    edge_y = (node_y[edges[:, 0]] + node_y[edges[:, 1]]) / 2

    return {
        "node_x": node_x,
        "node_y": node_y,
        "node_z": node_z,
        "face_nodes": face_nodes,
        "edge_nodes": edge_node,
        "edge_faces": edge_face,
        "face_x": face_x,
        "face_y": face_y,
        "edge_x": edge_x,
        "edge_y": edge_y,
        "face_x_bnd": face_x_bnd,
        "face_y_bnd": face_y_bnd,
        "num_nodes": n_nodes,
        "num_faces": n_faces,
        "num_edges": edges.shape[0],
    }


def build_loops(edges):
    """
    Build closed loops from a list of edges.

    Parameters
    ----------
    edges : (N, 2) array
        List of edges defined by pairs of node indices.

    Returns
    -------
    loops : list of lists
        Each sublist contains node indices forming a closed loop.
    """
    edges = edges.tolist()
    loops = []
    while edges:
        start, end = edges.pop(0)
        loop = [start, end]
        closed = False
        while not closed:
            found = False
            for i, (a, b) in enumerate(edges):
                if a == loop[-1]:
                    loop.append(b)
                    edges.pop(i)
                    found = True
                    break
                elif b == loop[-1]:
                    loop.append(a)
                    edges.pop(i)
                    found = True
                    break
            if not found:
                break
            if loop[-1] == loop[0]:
                closed = True
        loops.append(loop)
    return loops


def export_to_grd(
    filename, vert, tria, z, crs, edge_tag, edge_open=None, edge_land=None,
    open_contours=None, land_contours=None,
):
    """
    Export mesh to ADCIRC .grd format with boundaries.
    Prefer open_contours / land_contours from identify_boundary (list of 1D arrays
    of node indices, one per contour). Discontinuity between contours is preserved.

    Parameters
    ----------
    filename : str
        Path to output .grd file.
    vert : (N, 2) array
        Node coordinates (x, y).
    tria : (M, 3) array
        Triangle connectivity (node indices).
    z : (N,) array
        Node depth/elevation values.
    crs : str
        Coordinate reference system string.
    edge_tag : (K, 3) array
        Edge tags (node1, node2, tag).
    edge_open : (L, 2) array, optional
        Open boundary edges (flat). Used if open_contours is None.
    edge_land : (P, 2) array, optional
        Land boundary edges (flat). Used if land_contours is None.
    open_contours : list of 1D arrays, optional
        One ordered contour per open boundary (node indices). If provided, used instead of edge_open.
    land_contours : list of 1D arrays, optional
        One ordered contour per land boundary (node indices). If provided, used instead of edge_land.
    """
    if open_contours is not None:
        open_loops = [np.asarray(c, dtype=int).tolist() if np.ndim(c) > 0 else [int(c)] for c in open_contours]
    else:
        if edge_open is None:
            edge_open = edge_tag[edge_tag[:, 2] == 1, :2].astype(int)
        open_loops = build_loops(edge_open) if edge_open.size > 0 else []

    if land_contours is not None:
        land_loops = [np.asarray(c, dtype=int).tolist() if np.ndim(c) > 0 else [int(c)] for c in land_contours]
    else:
        if edge_land is None:
            edge_land = edge_tag[edge_tag[:, 2] == 2, :2].astype(int)
        land_loops = build_loops(edge_land) if edge_land.size > 0 else []

    # --- 3. Write to file
    with open(filename, "w") as f:
        # --- Header
        f.write(f"{crs}\n")
        f.write(f"{tria.shape[0]} {vert.shape[0]}\n")

        # --- Nodes
        for i, (x, y, zi) in enumerate(zip(vert[:, 0], vert[:, 1], z), start=1):
            if np.isnan(zi):
                f.write(f"{i} {x:.15f} {y:.15f} NAN\n")
            else:
                f.write(f"{i} {x:.15f} {y:.15f} {zi:.15f}\n")
        # --- Triangles
        for i, tri in enumerate(tria, start=1):
            f.write(f"{i} 3 {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")

        # --- Open boundaries
        total_open_nodes = sum(len(loop) for loop in open_loops)
        f.write(f"{len(open_loops)} ! total number of open boundaries\n")
        f.write(f"{total_open_nodes} ! total number of open boundary nodes\n")

        for ib, loop in enumerate(open_loops):
            f.write(f"{len(loop)} ! number of nodes for open_boundary_{ib}\n")
            for nid in loop:
                f.write(f"{nid + 1}\n")

        # ---- Land boundaries
        total_land_nodes = sum(len(loop) for loop in land_loops)
        f.write(f"{len(land_loops)}  ! total number of land boundaries\n")
        f.write(f"{total_land_nodes} ! Total number of land boundary nodes\n")

        for i, loop in enumerate(land_loops):
            f.write(f"{len(loop)} 1 ! boundary 1:{i}\n")
            for nid in loop:
                f.write(f"{nid + 1}\n")


def plot_grd(filename, ax=None, show_boundaries=True):
    """
    draw a preview of an ADCIRC .grd mesh file with optional boundaries.

    Parameters
    ----------
    filename : str
        Path to the .grd file.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    show_boundaries : bool
        Whether to plot open and land boundaries.
    """

    if ax is None:
        fig, ax = plt.subplots()

    with open(filename, "r") as f:
        lines = f.readlines()

    # --- search header
    for i, line in enumerate(lines):
        if (
            len(line.split()) == 2
            and line.strip().replace(".", "", 1).replace("-", "", 1).isdigit() is False
        ):
            try:
                nelem, nnode = map(int, line.split())
                header_idx = i
                break
            except Exception:
                continue

    # --- Reading nodes
    node_lines = lines[header_idx + 1 : header_idx + 1 + nnode]
    vert = np.zeros((nnode, 3))
    for i, ln in enumerate(node_lines):
        parts = ln.split()
        vert[i, 0] = float(parts[1])  # lon
        vert[i, 1] = float(parts[2])  # lat
        vert[i, 2] = float(parts[3])  # z

    # --- Reading elements
    elem_lines = lines[header_idx + 1 + nnode : header_idx + 1 + nnode + nelem]
    tria = np.zeros((nelem, 3), dtype=int)
    for i, ln in enumerate(elem_lines):
        parts = ln.split()
        tria[i, :] = np.array(parts[2:5], dtype=int) - 1  # indices 0-based

    # --- Reading boundaries
    open_boundaries = []
    land_boundaries = []

    if show_boundaries:
        idx = header_idx + 1 + nnode + nelem
        for j in range(idx, len(lines)):
            line = lines[j].strip()
            if "! total number of open boundaries" in line:
                n_open = int(line.split()[0])
                j += 1
                n_open_nodes = int(lines[j].split()[0])
                j += 1
                for _ in range(n_open):
                    n_nodes = int(lines[j].split()[0])
                    j += 1
                    ids = []
                    for _ in range(n_nodes):
                        ids.append(int(lines[j].strip()) - 1)
                        j += 1
                    open_boundaries.append(ids)
            if "! total number of land boundaries" in line:
                n_land = int(line.split()[0])
                j += 1
                n_land_nodes = int(lines[j].split()[0])
                j += 1
                for _ in range(n_land):
                    parts = lines[j].split()
                    n_nodes = int(parts[0])
                    j += 1
                    ids = []
                    for _ in range(n_nodes):
                        ids.append(int(lines[j].strip()) - 1)
                        j += 1
                    land_boundaries.append(ids)
            # --- break outer loop
            if j >= len(lines):
                break

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 8))

    facecolors = np.mean(vert[:, 2][tria], axis=1)
    pm = ax.tripcolor(
        vert[:, 0],
        vert[:, 1],
        tria,
        facecolors=facecolors,
        edgecolors="k",
        lw=0.2,
        cmap="summer",
    )
    plt.colorbar(pm, ax=ax, label="Depth/Elevation")


    for i, b in enumerate(open_boundaries):
        ax.plot(
            vert[b, 0],
            vert[b, 1],
            "r-",
            lw=1.2,
            label="Open boundary" if i == 0 else None,
        )
    for i, b in enumerate(land_boundaries):
        ax.plot(
            vert[b, 0],
            vert[b, 1],
            "k-",
            lw=1.0,
            label="Land boundary" if i == 0 else None,
        )
    ax.set_aspect("equal")
    ax.set_xlabel("Longitude [°]")
    ax.set_ylabel("Latitude [°]")
    ax.set_title(f"Mesh preview: {filename}")
    ax.legend(loc="best", frameon=True)
