"""UGRID NetCDF writing/reading for Delft3D-FM meshes."""
from __future__ import annotations


from ..feedback import _NullFeedback, _check
from ..geom_util.proj_util import _raster_crs


_UGRID_FILL = -999
_WGS84_FILL = -2147483647


def apply_nc_metadata(nc_path, metadata):
    """Overwrite/add global attributes of an existing NetCDF file.

    Parameters
    ----------
    nc_path : str
        Path to the NetCDF file (modified in place).
    metadata : dict or None
        Global attributes to set, e.g. ``{"institution": "...", "title":
        "..."}``. Existing attributes with the same name are overwritten;
        others are left untouched. ``None`` or empty is a no-op.
    """
    if not metadata:
        return
    from netCDF4 import Dataset

    with Dataset(nc_path, "a") as nc:
        for key, value in metadata.items():
            setattr(nc, str(key), str(value))


def _write_ugrid_netcdf(ugrid: dict, path: str, crs=None, metadata=None):
    """Write a ``build_ugrid_arrays`` dict as a Delft3D-FM-style UGRID NetCDF.

    Mirrors ``grd_util._xr_dataset_from_ugrid_dict`` (variables, attributes,
    conventions) but uses ``netCDF4`` directly, so ``xarray`` is not required.

    Parameters
    ----------
    ugrid : dict
        UGRID arrays as returned by ``build_ugrid_arrays``.
    path : str
        Output NetCDF path.
    crs : pyproj.CRS or None, optional
        Controls the coordinate metadata: geographic (default) writes degree
        units and a ``wgs84`` grid mapping; a projected CRS writes metre
        units and a ``projected_coordinate_system`` grid mapping.
    metadata : dict or None, optional
        Extra/override global attributes; when ``None`` (default) the
        standard attributes are written unchanged, otherwise each given
        key overwrites (or adds to) them.
    """
    import numpy as np
    from datetime import datetime
    from netCDF4 import Dataset

    geographic = crs is None or getattr(crs, "is_geographic", True)
    if geographic:
        x_units, y_units = "degrees_east", "degrees_north"
        x_std, y_std = "longitude", "latitude"
        crs_name = "wgs84"
        epsg = 4326 if crs is None else (crs.to_epsg() or 4326)
        crs_attrs = {
            "name": "WGS 84", "epsg": np.int32(epsg),
            "grid_mapping_name": "latitude_longitude",
            "longitude_of_prime_meridian": 0.0,
            "semi_major_axis": 6378137.0,
            "semi_minor_axis": 6356752.314245,
            "inverse_flattening": 298.257223563,
            "EPSG_code": "", "value": "value is equal to EPSG code",
            "proj_string": "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs",
        }
    else:
        x_units, y_units = "m", "m"
        x_std, y_std = "projection_x_coordinate", "projection_y_coordinate"
        crs_name = "projected_coordinate_system"
        epsg = crs.to_epsg() or 0
        crs_attrs = {
            "name": crs.name, "epsg": np.int32(epsg),
            "grid_mapping_name": "Unknown projected",
            "EPSG_code": f"EPSG:{epsg}" if epsg else "",
            "value": "value is equal to EPSG code",
            "proj_string": crs.to_proj4() or "",
            "wkt": crs.to_wkt(),
        }

    face_nodes = np.asarray(ugrid["face_nodes"])
    n_max_face_nodes = int(face_nodes.shape[1])

    with Dataset(path, "w", format="NETCDF4") as nc:
        # ------------------------------------------------------------- dims
        nc.createDimension("mesh2d_nNodes", int(ugrid["num_nodes"]))
        nc.createDimension("mesh2d_nEdges", int(ugrid["num_edges"]))
        nc.createDimension("mesh2d_nFaces", int(ugrid["num_faces"]))
        nc.createDimension("mesh2d_nMax_face_nodes", n_max_face_nodes)
        nc.createDimension("Two", 2)

        def var(name, dtype, dims, data, attrs, fill_value=None):
            v = nc.createVariable(name, dtype, dims, fill_value=fill_value)
            for k, val in attrs.items():
                v.setncattr(k, val)
            v[...] = data
            return v

        i4 = np.int32
        # ------------------------------------------------------------ nodes
        var("mesh2d_node_x", "f8", ("mesh2d_nNodes",), ugrid["node_x"], {
            "standard_name": x_std,
            "long_name": "x-coordinate of mesh nodes",
            "units": x_units})
        var("mesh2d_node_y", "f8", ("mesh2d_nNodes",), ugrid["node_y"], {
            "standard_name": y_std,
            "long_name": "y-coordinate of mesh nodes",
            "units": y_units})
        var("mesh2d_node_z", "f8", ("mesh2d_nNodes",), -np.asarray(ugrid["node_z"]), {
            "mesh": "mesh2d", "location": "node", "units": "m",
            "standard_name": "altitude",
            "long_name": "z-coordinate of mesh nodes",
            "grid_mapping": crs_name,
            "coordinates": "mesh2d_node_x mesh2d_node_y"})
        # ------------------------------------------------------------ edges
        var("mesh2d_edge_x", "f8", ("mesh2d_nEdges",), ugrid["edge_x"], {
            "long_name": "characteristic x-coordinate of the mesh edge (e.g. midpoint)",
            "units": x_units, "standard_name": x_std})
        var("mesh2d_edge_y", "f8", ("mesh2d_nEdges",), ugrid["edge_y"], {
            "long_name": "characteristic y-coordinate of the mesh edge (e.g. midpoint)",
            "units": y_units, "standard_name": y_std})
        var("mesh2d_edge_nodes", "i4", ("mesh2d_nEdges", "Two"),
            np.asarray(ugrid["edge_nodes"], dtype=i4), {
            "cf_role": "edge_node_connectivity",
            "long_name": "Start and end nodes of mesh edges",
            "start_index": i4(1)})
        var("mesh2d_edge_faces", "i4", ("mesh2d_nEdges", "Two"),
            np.asarray(ugrid["edge_faces"], dtype=i4), {
            "cf_role": "edge_face_connectivity",
            "long_name": "Neighboring faces of mesh edges",
            "start_index": i4(1)}, fill_value=i4(_UGRID_FILL))
        # ------------------------------------------------------------ faces
        var("mesh2d_face_nodes", "i4", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
            np.asarray(face_nodes, dtype=i4), {
            "cf_role": "face_node_connectivity",
            "long_name": "Vertex nodes of mesh faces (counterclockwise)",
            "start_index": i4(1),
            "coordinates": "mesh2d_node_x mesh2d_node_y"},
            fill_value=i4(_UGRID_FILL))
        var("mesh2d_face_x", "f8", ("mesh2d_nFaces",), ugrid["face_x"], {
            "units": x_units, "standard_name": x_std,
            "long_name": "Characteristic x-coordinate of mesh face",
            "bounds": "mesh2d_face_x_bnd"})
        var("mesh2d_face_y", "f8", ("mesh2d_nFaces",), ugrid["face_y"], {
            "units": y_units, "standard_name": y_std,
            "long_name": "Characteristic y-coordinate of mesh face",
            "bounds": "mesh2d_face_y_bnd"})
        var("mesh2d_face_x_bnd", "f8", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
            ugrid["face_x_bnd"], {
            "long_name": "x-coordinate bounds of mesh faces (i.e. corner coordinates)",
            "units": x_units, "standard_name": x_std})
        var("mesh2d_face_y_bnd", "f8", ("mesh2d_nFaces", "mesh2d_nMax_face_nodes"),
            ugrid["face_y_bnd"], {
            "long_name": "y-coordinate bounds of mesh faces (i.e. corner coordinates)",
            "units": y_units, "standard_name": y_std})
        # -------------------------------------------------- CRS + topology
        var(crs_name, "i4", (), i4(epsg if epsg else _WGS84_FILL), crs_attrs,
            fill_value=i4(_WGS84_FILL))
        var("mesh2d", "i4", (), i4(_WGS84_FILL), {
            "cf_role": "mesh_topology",
            "long_name": "Topology data of 2D mesh",
            "topology_dimension": i4(2),
            "node_coordinates": "mesh2d_node_x mesh2d_node_y",
            "node_dimension": "mesh2d_nNodes",
            "edge_node_connectivity": "mesh2d_edge_nodes",
            "edge_dimension": "mesh2d_nEdges",
            "edge_coordinates": "mesh2d_edge_x mesh2d_edge_y",
            "face_node_connectivity": "mesh2d_face_nodes",
            "face_dimension": "mesh2d_nFaces",
            "face_coordinates": "mesh2d_face_x mesh2d_face_y",
            "max_face_nodes_dimension": "mesh2d_nMax_face_nodes",
            "edge_face_connectivity": "mesh2d_edge_faces"})
        # ------------------------------------------------------ global attrs
        nc.institution = "GeoOcean"
        nc.references = "https://github.com/GeoOcean/BlueMesh2D"
        nc.source = f"BlueMesh2D QGIS plugin {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        nc.history = "Created with BlueMesh2D"
        nc.Conventions = "CF-1.8 UGRID-1.0 Deltares-0.10"
        if metadata:
            for key, value in metadata.items():
                setattr(nc, str(key), str(value))


def export_ugrid(vert, tria, raster_path, utm_crs, output_path,
                 interp_order=3, metadata=None, invert_z=False,
                 nodata_value=None, feedback=None):
    """Sample bathymetry onto the mesh nodes and write the UGRID NetCDF.

    `vert` is in `utm_crs`; nodes are reprojected to the bathymetry raster's
    CRS for both the z-sampling and the output coordinates.

    Parameters
    ----------
    vert : ndarray of shape (N, 2)
        Mesh vertex coordinates in `utm_crs`.
    tria : ndarray of shape (T, 3)
        Triangle connectivity (0-based vertex indices).
    raster_path : str
        Path to the bathymetry raster used for node-depth sampling.
    utm_crs : pyproj.CRS
        CRS `vert` is expressed in.
    output_path : str
        Output UGRID NetCDF path.
    interp_order : int, optional
        Interpolation order passed to ``interpolate_from_tiff``
        (0=nearest, 1=bilinear, 3=bicubic). Default is 3.
    metadata : dict or None, optional
        Global-attribute overrides for the written NetCDF (see
        :func:`_write_ugrid_netcdf`). ``None`` keeps the defaults.
    invert_z : bool, optional
        Reverse the raster Z sign when sampling node depths (see
        :func:`interpolate_from_tiff`). Default ``False``.
    nodata_value : float or None, optional
        Elevation assigned to raster nodata when sampling node depths;
        ``None`` fills from the nearest valid pixel. Default ``None``.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    output_path : str
        Path to the written NetCDF (same as the input `output_path`).
    """
    feedback = feedback or _NullFeedback()
    import numpy as np
    import pyproj
    import rasterio

    from bluemesh2d.geom_util.proj_util import reproject_node
    from bluemesh2d.geomesh_util.interpolation_mesh import interpolate_from_tiff
    from bluemesh2d.geomesh_util.grd_util import build_ugrid_arrays

    with rasterio.open(raster_path) as src:
        raster_crs = _raster_crs(src)

    feedback.pushInfo("Interpolating bathymetry onto mesh nodes ...")
    if utm_crs == raster_crs:
        vert_geo = vert  # projected input: mesh is already in the tif CRS
    else:
        vert_geo = reproject_node(vert, utm_crs, raster_crs)
    z = interpolate_from_tiff(raster_path, vert_geo, order=interp_order,
                              invert_z=invert_z, nodata_value=nodata_value)
    _check(feedback)

    feedback.pushInfo(f"Writing UGRID NetCDF -> {output_path}")
    ugrid = build_ugrid_arrays(np.column_stack((vert_geo, z)),
                               np.asarray(tria, dtype=int))
    _write_ugrid_netcdf(ugrid, output_path, crs=raster_crs, metadata=metadata)
    return output_path


# ===========================================================================
# Stage 5: exports from a generated mesh (.nc)
# ===========================================================================

def read_ugrid_mesh(nc_path):
    """Read a stage-4 UGRID NetCDF back into vertex/triangle/depth arrays.

    Parameters
    ----------
    nc_path : str
        Path to a UGRID NetCDF written by :func:`export_ugrid`.

    Returns
    -------
    vert : ndarray of shape (N, 2)
        Node x/y coordinates.
    tria : ndarray of shape (M, 3)
        Triangle connectivity (0-based node indices).
    z_depth : ndarray of shape (N,)
        Node depth (positive down), matching the notebook's
        ``interpolate_from_tiff`` convention that ``identify_boundary``
        expects.

    Raises
    ------
    RuntimeError
        If the mesh contains quad faces (only triangle-only meshes are
        supported).
    """
    import numpy as np
    from netCDF4 import Dataset

    with Dataset(nc_path) as nc:
        x = np.asarray(nc["mesh2d_node_x"][:], dtype=float)
        y = np.asarray(nc["mesh2d_node_y"][:], dtype=float)
        z_elev = np.asarray(nc["mesh2d_node_z"][:], dtype=float)
        fn_var = nc["mesh2d_face_nodes"]
        fn = np.ma.filled(fn_var[:], _UGRID_FILL).astype(int)
        start = int(getattr(fn_var, "start_index", 1))

    if fn.shape[1] > 3 and (fn[:, 3] >= start).any():
        raise RuntimeError(
            "The mesh contains quad faces; .grd / boundary export supports "
            "triangle-only meshes (disable quad-preserving smood output).")
    tria = fn[:, :3] - start
    vert = np.column_stack([x, y])
    return vert, tria, -z_elev  # file stores elevation; return depth


