"""Staged meshing pipeline: single import point for the Python interface.

Re-exports every stage of the raster -> coastline -> size function ->
boundary -> refine/smooth -> UGRID NetCDF workflow from its home module,
plus the all-in-one facade (:class:`MeshConfig` -> :func:`generate_mesh`
-> :class:`MeshResult`) defined here.

Stage homes
-----------
1. ``bluemesh2d.geomesh_util.water_polygon``  -- extract_water_polygon
2. ``bluemesh2d.hfun_util.build_hfun``        -- build_hfun_raster & co.
3. ``bluemesh2d.geom_util.boundary_util``     -- resample_boundary, PSLG
4. ``bluemesh2d.meshgen``                     -- mesh_pslg
5. ``bluemesh2d.mesh_file.ugrid``             -- export_ugrid, read_ugrid_mesh
6. ``bluemesh2d.mesh_file.bnd_util``          -- boundary conditions, .grd/.pli
Support: ``bluemesh2d.feedback`` (progress/cancel), ``bluemesh2d.dependencies``.
"""
from __future__ import annotations

from dataclasses import dataclass

from .dependencies import (  # noqa: F401
    check_dependencies,
    optional_dependencies,
    smood_dependencies,
)
from .feedback import (  # noqa: F401
    MeshCanceled,
    _LogWriter,
    _NullFeedback,
    _SubProgress,
    _available_ram_bytes,
    _check,
    _warn_if_ram_risk,
)
from .geom_util.boundary_util import (  # noqa: F401
    _fixed_part_from_z,
    pslg_from_segments,
    resample_boundary,
)
from .geom_util.proj_util import _raster_crs  # noqa: F401
from .geomesh_util.water_polygon import (  # noqa: F401
    _flag_fixed_vertices,
    _prune_nonoriginal_fixed,
    extract_water_polygon,
)
from .hfun_util.build_hfun import (  # noqa: F401
    _compile_custom_hfun,
    _make_depth_hfun,
    build_hfun_constant_raster,
    build_hfun_raster,
    load_hfun_raster,
)
from .mesh_file.bnd_util import (  # noqa: F401
    _boundary_loops,
    boundary_lines_from_points,
    classify_boundary_lines,
    classify_boundary_points,
    export_boundary_conditions,
    export_grd,
    export_grd_from_lines,
    generate_boundary_condition_points,
    generate_boundary_conditions,
    write_open_boundary_files,
    write_open_boundary_pli,
)
from .mesh_file.ugrid import (  # noqa: F401
    _UGRID_FILL,
    _WGS84_FILL,
    _write_ugrid_netcdf,
    export_ugrid,
    read_ugrid_mesh,
)
from .meshgen import (  # noqa: F401
    _locate_fixed,
    _warn_if_mesh_too_big,
    mesh_pslg,
)


@dataclass
class MeshConfig:
    """All tunable parameters of the full meshing pipeline (all-in-one run).

    Attributes
    ----------
    raster_path : str
        Path to the bathymetry raster (elevation, positive up).
    output_path : str
        Output UGRID NetCDF path.
    coast_zmax : float
        Wet threshold (m), see :func:`extract_water_polygon`. Default 2.0.
    domain_buffer : float
        Domain buffer factor, see :func:`extract_water_polygon`.
        Default -0.05.
    keep_largest : bool
        Keep only the largest water polygon. Default ``True``.
    detail_geom : shapely.geometry.base.BaseGeometry or None
        Detail-region polygon in the raster CRS. Default ``None``.
    detail_hmin : float
        Element-size floor (m) inside `detail_geom`. Default 30.0.
    a, b : float
        Depth-polynomial sizing coefficients, see
        :func:`_make_depth_hfun`. Defaults 0.14 and 28.0.
    hmin, hmax : float
        Element-size floor and cap (m). Defaults 100.0 and 10000.0.
    max_gradient : float
        Maximum allowed size gradient (m/m). Default 0.1.
    min_angle_deg : float
        Minimum boundary angle (deg), see :func:`resample_boundary`.
        Default 25.0.
    min_hole_vertices : int
        Minimum hole vertex count, see :func:`resample_boundary`. Default 15.
    kind : {'delaunay', 'delfront'}
        Refinement scheme. Default ``'delaunay'``.
    do_smooth : bool
        Run mesh smoothing after refinement. Default ``True``.
    do_smood : bool
        Run smood orthogonalization after smoothing. Default ``False``.
    smood_merge_small_links : bool
        Enable small-link merging inside smood, see :func:`mesh_pslg`.
        Default ``False``.
    interp_order : int
        Bathymetry sampling order (0=nearest, 1=bilinear, 3=bicubic).
        Default 3.
    """

    raster_path: str
    output_path: str

    coast_zmax: float = 2.0
    domain_buffer: float = -0.05
    keep_largest: bool = True

    detail_geom: object = None       # shapely polygon in the raster CRS
    detail_hmin: float = 30.0

    a: float = 0.14
    b: float = 28.0
    hmin: float = 100.0
    hmax: float = 10000.0
    max_gradient: float = 0.1

    min_angle_deg: float = 25.0
    min_hole_vertices: int = 15

    kind: str = "delaunay"           # refine: 'delaunay' | 'delfront'
    do_smooth: bool = True
    do_smood: bool = False
    smood_merge_small_links: bool = False

    interp_order: int = 3            # bathy sampling: 0=nearest,1=bilinear,3=bicubic


@dataclass
class MeshResult:
    """Result of a :func:`generate_mesh` run.

    Attributes
    ----------
    output_path : str
        Path to the written UGRID NetCDF.
    n_nodes : int
        Number of mesh nodes.
    n_triangles : int
        Number of mesh triangles.
    utm_crs : str
        String representation of the working CRS used.
    """

    output_path: str
    n_nodes: int
    n_triangles: int
    utm_crs: str


def generate_mesh(config: MeshConfig, feedback=None) -> MeshResult:
    """Run the full raster -> UGRID-NetCDF meshing pipeline headlessly.

    Composes stages 1-4 (:func:`extract_water_polygon`, an in-memory
    equivalent of :func:`build_hfun_raster`, :func:`resample_boundary`,
    :func:`mesh_pslg`) and :func:`export_ugrid` in one call.

    Parameters
    ----------
    config : MeshConfig
        Pipeline parameters.
    feedback : object or None, optional
        Feedback sink, see :func:`extract_water_polygon`.

    Returns
    -------
    result : MeshResult
        Summary of the generated mesh.
    """
    feedback = feedback or _NullFeedback()

    missing = check_dependencies()
    if missing:
        raise RuntimeError(
            "Missing Python packages required by BlueMesh2D: "
            + ", ".join(missing)
            + ". Install them into this interpreter (see the plugin README).")
    for mod in optional_dependencies():
        feedback.pushInfo(
            f"Optional package '{mod}' not installed - using the built-in "
            "fallback (slower triangulation; install it for best performance).")

    import pyproj
    from bluemesh2d.geom_util.proj_util import reproject_geometry
    from bluemesh2d.geomesh_util.depth_field import depth_field_from_tif
    from bluemesh2d.hfun_util.smooth_and_precomput import smooth_precomput_hfun

    feedback.setProgress(2)
    poly, utm_crs, raster_crs = extract_water_polygon(
        config.raster_path, coast_zmax=config.coast_zmax,
        domain_buffer=config.domain_buffer,
        keep_largest=config.keep_largest,
        feedback=_SubProgress(feedback, 2, 25))
    feedback.pushInfo(f"Working CRS: {utm_crs.to_string() if hasattr(utm_crs, 'to_string') else utm_crs}")

    # In-memory hfun (no raster round-trip needed for the all-in-one run)
    feedback.pushInfo("Building depth-based size function ...")
    feedback.setProgress(25)
    depth_field = depth_field_from_tif(config.raster_path, output_crs=utm_crs)
    detail_u = None
    if config.detail_geom is not None:
        detail_u = reproject_geometry(config.detail_geom, raster_crs, utm_crs)
    hfun = _make_depth_hfun(
        depth_field, a=config.a, b=config.b,
        hmin=config.hmin, hmax=config.hmax,
        detail=detail_u,
        detail_hmin=(config.detail_hmin if detail_u is not None else None))

    feedback.pushInfo("Gradient-limiting the size function (this can take a moment) ...")
    feedback.setProgress(35)
    # limit the gradient-limiting grid to the water domain (poly is in utm_crs)
    hfuns = smooth_precomput_hfun(hfun, domain=poly, max_gradient=config.max_gradient,
                                  plot=False)
    _check(feedback)

    feedback.pushInfo("Resampling boundary to the size function ...")
    feedback.setProgress(45)
    poly_comput, node, edge = resample_boundary(
        poly, hfuns, config.min_angle_deg, config.min_hole_vertices, feedback)

    feedback.setProgress(55)
    vert, tria = mesh_pslg(node, edge, hfuns, kind=config.kind,
                           do_smooth=config.do_smooth,
                           do_smood=config.do_smood,
                           smood_merge_small_links=config.smood_merge_small_links,
                           feedback=_SubProgress(feedback, 55, 88))

    feedback.setProgress(88)
    export_ugrid(vert, tria, config.raster_path, utm_crs,
                 config.output_path, config.interp_order, feedback)
    feedback.setProgress(100)

    return MeshResult(
        output_path=config.output_path,
        n_nodes=int(len(vert)),
        n_triangles=int(len(tria)),
        utm_crs=utm_crs.to_string() if hasattr(utm_crs, "to_string") else str(utm_crs),
    )
