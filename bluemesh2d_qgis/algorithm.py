"""QGIS Processing algorithms wrapping the staged BlueMesh2D pipeline.

Algorithms
----------
``ExtractWaterPolygonAlgorithm``
    Raster -> water-domain polygon layer (stage 1).
``BuildHfunPolynomialAlgorithm``, ``BuildHfunWavelengthAlgorithm``,
``BuildHfunCustomAlgorithm``
    Raster -> element-size (hfun) raster, one algorithm per sizing method
    (stage 2).
``ResampleBoundaryAlgorithm``
    Polygon + hfun -> resampled boundary edges layer (stage 3).
``GenerateMeshFromBoundaryAlgorithm``
    Boundary + hfun -> refine/smooth[/smood] -> UGRID mesh (stage 4).
``GenerateBoundaryConditionsAlgorithm``
    Mesh -> editable open/closed/island boundary point layer (stage 5).
``ExportUgridAlgorithm``, ``ExportUgridBoundaryAlgorithm``, ``ExportGrdAlgorithm``
    Mesh (+ boundary conditions) -> plain UGRID NetCDF, UGRID NetCDF + open
    boundary condition, or ADCIRC ``.grd`` (stage 6, group "6 - Export").
``GenerateMeshAlgorithm``
    Stages 1-4 run in a single algorithm.

Notes
-----
Intermediate results live in ordinary QGIS layers (polygons, points, lines,
rasters), so every stage can be inspected -- and edited -- before the next.
"""

import os

from qgis.PyQt.QtCore import QVariant
from qgis.PyQt.QtGui import QIcon
from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFeatureSink,
    QgsProcessingParameterFeatureSource,
    QgsProcessingParameterDefinition,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterMeshLayer,
    QgsProcessingParameterNumber,
    QgsProcessingParameterRasterDestination,
    QgsProcessingParameterRasterLayer,
    QgsProcessingParameterString,
    QgsProject,
    QgsWkbTypes,
)

from .pipeline import (
    MeshCanceled,
    MeshConfig,
    _corner_vertices,
    _flag_fixed_vertices,
    _prune_nonoriginal_fixed,
    _valid_parts,
    boundary_lines_from_points,
    build_hfun_constant_raster,
    build_hfun_raster,
    check_dependencies,
    export_grd_from_lines,
    export_ugrid,
    extract_water_polygon,
    generate_boundary_condition_points,
    generate_mesh,
    load_hfun_raster,
    mesh_pslg,
    pslg_from_segments,
    resample_boundary,
    smood_dependencies,
    write_open_boundary_files,
    write_open_boundary_pli,
)

# No sub-group: algorithms sit directly under the BlueMesh2D provider node.
GROUP = ""
GROUP_ID = ""

ICON_PATH = os.path.join(os.path.dirname(__file__), "icon.png")


def plugin_icon():
    return QIcon(ICON_PATH)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------

def _require_deps():
    missing = check_dependencies()
    if missing:
        raise QgsProcessingException(
            "Missing Python packages in the QGIS environment: "
            + ", ".join(missing) + ". Install them via Plugins > BlueMesh2D "
            "> 'Check / install dependencies', then restart QGIS "
            "(manual commands: see the plugin README).")


def _check_smood_deps(do_smood):
    """Fail fast (before running refine/smooth) if smood needs xarray."""
    if not do_smood:
        return
    missing = smood_dependencies()
    if missing:
        raise QgsProcessingException(
            "'Apply smood' requires: " + ", ".join(missing)
            + ". Install it, or disable the smood option. See the plugin "
            "README.")


def _num(alg, name, label, default, minv=-1e9, maxv=1e9, integer=False,
         advanced=False):
    p = QgsProcessingParameterNumber(
        name, label,
        QgsProcessingParameterNumber.Integer if integer
        else QgsProcessingParameterNumber.Double,
        defaultValue=default, minValue=minv, maxValue=maxv)
    if advanced:
        p.setFlags(p.flags() | QgsProcessingParameterDefinition.FlagAdvanced)
    alg.addParameter(p)


def _shapely_from_qgis(g):
    """Convert a QgsGeometry to shapely, tolerating QGIS's Z/M WKT spelling.

    QGIS writes ``PolygonZ (...)`` while shapely's reader needs
    ``Polygon Z (...)``; insert the missing space in the type token.
    """
    import re

    from shapely import wkt as shapely_wkt

    s = re.sub(r"^\s*([A-Za-z]+?)(ZM|Z|M)\s*\(", r"\1 \2 (", g.asWkt(), count=1)
    return shapely_wkt.loads(s)


def _source_to_shapely(source, target_crs=None, densify=None):
    """Union all polygon features of a source into one shapely geometry.

    Parameters
    ----------
    source : QgsProcessingFeatureSource
        Polygon feature source.
    target_crs : QgsCoordinateReferenceSystem or None, optional
        If given, features are reprojected into it before conversion.
        Default is ``None``.
    densify : int or None, optional
        If given, add this many extra vertices per segment *before*
        reprojecting. Long straight edges reprojected vertex-by-vertex
        become chords that deviate from the true (displayed) edge; the
        extra vertices keep the reprojected outline on it. Default is
        ``None`` (no densification).

    Returns
    -------
    geom : shapely.geometry.base.BaseGeometry or None
        Union of all feature geometries, or ``None`` if `source` has no
        usable geometry.
    """
    from shapely.ops import unary_union

    xform = None
    src_crs = source.sourceCrs()
    if target_crs is not None and src_crs.isValid() and src_crs != target_crs:
        xform = QgsCoordinateTransform(src_crs, target_crs, QgsProject.instance())

    geoms = []
    for feat in source.getFeatures():
        g = feat.geometry()
        if g is None or g.isEmpty():
            continue
        if densify:
            g = g.densifyByCount(int(densify))
        if xform is not None:
            g = QgsGeometry(g)
            g.transform(xform)
        sg = _shapely_from_qgis(g)
        if not sg.is_valid:
            # make_valid keeps Z (fixed-vertex flags) on surviving vertices;
            # buffer(0) is the 2D-only fallback
            try:
                from shapely.validation import make_valid
                sg = make_valid(sg)
            except Exception:
                sg = sg.buffer(0)
        geoms.append(sg)
    if not geoms:
        return None
    return unary_union(geoms)


def _source_to_segments(source):
    """Collect every polyline of a line source as a coordinate array.

    Parameters
    ----------
    source : QgsProcessingFeatureSource
        Line feature source.

    Returns
    -------
    segments : list of list of tuple
        One ``[(x, y), ...]`` list per ``LineString`` feature/part.
    """

    segments = []
    for feat in source.getFeatures():
        g = feat.geometry()
        if g is None or g.isEmpty():
            continue
        geom = _shapely_from_qgis(g)
        lines = geom.geoms if hasattr(geom, "geoms") else [geom]
        for line in lines:
            if line.geom_type == "LineString" and len(line.coords) >= 2:
                segments.append([(p[0], p[1]) for p in line.coords])
    return segments


def _source_to_points(source):
    """Collect every point of a point source as ``(x, y)`` tuples.

    Parameters
    ----------
    source : QgsProcessingFeatureSource
        Point feature source (Point or MultiPoint geometries).

    Returns
    -------
    points : list of tuple
        One ``(x, y)`` tuple per point.
    """

    points = []
    for feat in source.getFeatures():
        g = feat.geometry()
        if g is None or g.isEmpty():
            continue
        geom = _shapely_from_qgis(g)
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for p in parts:
            if p.geom_type == "Point":
                points.append((p.x, p.y))
    return points


class _FeedbackAdapter:
    """Wrap QgsProcessingFeedback so pushWarning exists on older QGIS.

    Parameters
    ----------
    fb : QgsProcessingFeedback
        Feedback object to wrap.
    """

    def __init__(self, fb):
        self._fb = fb

    def isCanceled(self):
        return self._fb.isCanceled()

    def pushInfo(self, msg):
        self._fb.pushInfo(str(msg))

    def pushWarning(self, msg):
        if hasattr(self._fb, "pushWarning"):
            self._fb.pushWarning(str(msg))
        else:
            self._fb.pushInfo("WARNING: " + str(msg))

    def setProgress(self, pct):
        self._fb.setProgress(float(pct))


class _BaseAlg(QgsProcessingAlgorithm):
    def group(self):
        return GROUP

    def groupId(self):
        return GROUP_ID

    def icon(self):
        return plugin_icon()

    @staticmethod
    def _accept_invalid_geometries(context):
        """Deliver input features even when QGIS flags them invalid.

        The Z-flagged water polygon (stage 1) can be reported invalid by
        QGIS's strict check; the pipeline repairs geometries itself
        (``_valid_parts`` / ``_source_to_shapely``), so refusing the feature
        up front only breaks the workflow.
        """
        try:
            from qgis.core import QgsFeatureRequest
            context.setInvalidGeometryCheck(QgsFeatureRequest.GeometryNoCheck)
        except Exception:
            pass

    def flags(self):
        # Force execution on the MAIN thread. Processing runs algorithms on a
        # background QThread by default, but pyproj / PROJ can hard-crash
        # (Windows access violation) when its WKT parser and CRS database are
        # used off the main thread. NoThreading avoids that entirely.
        f = super().flags()
        try:
            return f | QgsProcessingAlgorithm.FlagNoThreading
        except AttributeError:  # QGIS >= 3.36 enum location
            from qgis.core import Qgis
            return f | Qgis.ProcessingAlgorithmFlag.NoThreading


# ---------------------------------------------------------------------------
# 1. Extract water polygon
# ---------------------------------------------------------------------------

def _reproject_polygon_keep_z(p, crs_from, crs_to):
    """Reproject a polygon's XY coordinates, preserving vertex order and Z.

    Unlike ``reproject_geometry`` (which drops Z and may rebuild rings via
    ``buffer(0)``), this transforms each vertex in place so per-vertex flags
    stored in Z survive.
    """
    import pyproj
    from shapely.geometry import Polygon

    tr = pyproj.Transformer.from_crs(crs_from, crs_to, always_xy=True)

    def ring(coords):
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        zs = [(c[2] if len(c) > 2 else 0.0) for c in coords]
        X, Y = tr.transform(xs, ys)
        return list(zip(X, Y, zs))

    return Polygon(ring(p.exterior.coords),
                   [ring(r.coords) for r in p.interiors])


class ExtractWaterPolygonAlgorithm(_BaseAlg):
    RASTER = "RASTER"
    COAST_ZMAX = "COAST_ZMAX"
    DEEP_ZMAX = "DEEP_ZMAX"
    EXTENT = "EXTENT"
    FIX_EXTENT_VERTS = "FIX_EXTENT_VERTS"
    DOMAIN_BUFFER = "DOMAIN_BUFFER"
    KEEP_LARGEST = "KEEP_LARGEST"
    OUTPUT = "OUTPUT"

    def group(self):
        return "1 - Extract water polygon"

    def groupId(self):
        return "extract_water_polygon"

    def name(self):
        return "extract_water_polygon"

    def displayName(self):
        return "1 - Extract water polygon"

    def shortHelpString(self):
        return ("Extract the water domain from a bathymetry raster: the region "
                "z <= coastline level, optionally limited offshore by a deep "
                "level (keeps only the band deep level < z <= coastline level, "
                "e.g. coastline 2 and deep -300 keeps water shallower than "
                "300 m). The domain extent is an optional extent polygon layer; "
                "when set, the raster is clipped to it before the contour "
                "extraction (much faster on large rasters) and the buffer is "
                "ignored. Without it, the raster's data extent is used, "
                "grown/shrunk by the domain buffer factor. With 'Fix polygon "
                "vertices' (default on), vertices lying on the extent "
                "boundary are marked inside the polygon (Z=1) and stage 3 "
                "keeps them exactly while resampling. Output feeds "
                "'3 - Resample boundary'.")

    def createInstance(self):
        return ExtractWaterPolygonAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.RASTER, "Bathymetry raster (elevation, positive up)"))
        _num(self, self.COAST_ZMAX, "Coastline level / wet threshold (m)", 0.0, -1e4, 1e4)
        self.addParameter(QgsProcessingParameterNumber(
            self.DEEP_ZMAX, "Deep level (m, optional; e.g. -300 keeps z > -300)",
            QgsProcessingParameterNumber.Double, optional=True,
            minValue=-1e5, maxValue=1e4))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.EXTENT, "Extent polygon (optional; default = raster extent)",
            [QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(QgsProcessingParameterBoolean(
            self.FIX_EXTENT_VERTS,
            "Fix polygon vertices on the domain boundary (extent polygon or "
            "buffered raster extent; kept exactly by stage 3 resampling)",
            defaultValue=True))
        _num(self, self.DOMAIN_BUFFER,
             "Domain buffer factor (negative shrinks; ignored if an extent "
             "polygon is set)", -0.05, -10, 10)
        self.addParameter(QgsProcessingParameterBoolean(
            self.KEEP_LARGEST, "Keep only the largest water region",
            defaultValue=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "1 - Water polygon", QgsProcessing.TypeVectorPolygon))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        fb = _FeedbackAdapter(feedback)
        raster = self.parameterAsRasterLayer(parameters, self.RASTER, context)

        deep_zmax = None
        if parameters.get(self.DEEP_ZMAX) is not None:
            deep_zmax = self.parameterAsDouble(parameters, self.DEEP_ZMAX, context)

        extent_geom = None
        extent_src = self.parameterAsSource(parameters, self.EXTENT, context)
        if extent_src is not None:
            # densify before reprojection: without it, long extent edges
            # become straight chords in the working CRS that deviate from
            # the true (displayed) edge, shifting the cut line and the
            # coast-intersection points sideways
            extent_geom = _source_to_shapely(extent_src, raster.crs(),
                                             densify=20)
            # original vertices (no densification): the only extent points
            # that may stay as fixed vertices in the output polygon
            extent_orig = _source_to_shapely(extent_src, raster.crs())
            if extent_geom is None:
                fb.pushWarning("Extent layer has no usable geometry; using the raster extent.")

        try:
            poly, utm_crs, raster_crs, domain_u = extract_water_polygon(
                raster.source(),
                self.parameterAsDouble(parameters, self.COAST_ZMAX, context),
                self.parameterAsDouble(parameters, self.DOMAIN_BUFFER, context),
                deep_zmax=deep_zmax,
                extent_geom=extent_geom,
                keep_largest=self.parameterAsBool(parameters, self.KEEP_LARGEST, context),
                return_domain=True,
                feedback=fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f"Extraction failed: {exc}")

        # Deliver the layer in the raster's CRS (a CRS QGIS resolves natively);
        # the metric local-UTM CRS stays internal to the pipeline.
        from bluemesh2d.geom_util.proj_util import reproject_geometry

        # fixed vertices work with or without an extent polygon: the clip
        # domain (extent polygon, or raster extent grown/shrunk by the
        # domain buffer) is where the cut and the coastline junctions lie
        fix_verts = self.parameterAsBool(
            parameters, self.FIX_EXTENT_VERTS, context)

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("area_km2", QVariant.Double))
        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields,
            QgsWkbTypes.MultiPolygonZ if fix_verts else QgsWkbTypes.MultiPolygon,
            raster.crs())

        if fix_verts:
            # Flag in the working (UTM) CRS, where the clip was computed: the
            # cut vertices AND the coast/domain intersection points lie
            # exactly on the clip domain's boundary there, so a small metric
            # tolerance catches them all before any reprojection noise.
            ext_boundary = domain_u.boundary
            tol = 1e-3  # metres

            import numpy as np
            if extent_geom is not None:
                # original extent vertices in the working CRS: these stay
                # fixed in the output (plus the coastline junctions)
                import pyproj as _pyproj
                _geoms = (extent_orig.geoms if hasattr(extent_orig, "geoms")
                          else [extent_orig])
                _pts = []
                for _g in _geoms:
                    _pts += [(c[0], c[1]) for c in _g.exterior.coords[:-1]]
                    for _r in _g.interiors:
                        _pts += [(c[0], c[1]) for c in _r.coords[:-1]]
                orig_xy = np.asarray(_pts, dtype=float)
                if utm_crs != raster_crs and orig_xy.size:
                    _tr = _pyproj.Transformer.from_crs(raster_crs, utm_crs,
                                                       always_xy=True)
                    _x, _y = _tr.transform(orig_xy[:, 0], orig_xy[:, 1])
                    orig_xy = np.column_stack([_x, _y])
            else:
                # generated domain (buffered raster extent): fix its
                # geometric corners plus the coastline intersection points
                orig_xy = _corner_vertices(domain_u)

        parts = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
        # repair invalid parts (self-touching rings from the clip) BEFORE
        # attaching Z flags: downstream stages reject invalid geometries
        parts = _valid_parts(parts)
        n_fixed = 0
        for i, p in enumerate(parts):
            area_km2 = p.area / 1e6  # metric area, computed in UTM
            if fix_verts:
                p = _flag_fixed_vertices(p, ext_boundary, tol)
                # extent polygon: keep original vertices + junctions, drop
                # densification points. Generated (buffered) domain: keep
                # the cut geometry but leave only the junctions fixed.
                p = _prune_nonoriginal_fixed(
                    p, orig_xy, unflag_only=(extent_geom is None))
                n_fixed += sum(int(c[2] > 0.5) for c in p.exterior.coords[:-1])
                # plain per-vertex transform: keeps ring order and Z flags
                p = _reproject_polygon_keep_z(p, utm_crs, raster_crs)
            else:
                p = reproject_geometry(p, utm_crs, raster_crs)
            f = QgsFeature(fields)
            f.setGeometry(QgsGeometry.fromWkt(p.wkt))
            f.setAttributes([i, area_km2])
            sink.addFeature(f)
        if fix_verts:
            fb.pushInfo(f"Fixed vertices on the domain boundary: {n_fixed} "
                        "(stored as Z=1; kept exactly by stage 3). Edit the "
                        "flags with the Vertex Tool's Vertex Editor panel "
                        "(Z column: 1 = fixed, 0 = free).")
        self._dest = dest_id
        self._show_verts = fix_verts
        return {self.OUTPUT: dest_id}

    def postProcessAlgorithm(self, context, feedback):
        if getattr(self, "_show_verts", False):
            _style_water_polygon(getattr(self, "_dest", None), context, feedback)
        return {self.OUTPUT: getattr(self, "_dest", None)}


# ---------------------------------------------------------------------------
# 2. Build hfun (element-size) raster
# ---------------------------------------------------------------------------

class _BuildHfunBase(_BaseAlg):
    """Base class sharing inputs, limits and run logic for the hfun methods.

    Each concrete subclass exposes only its own method's parameters, so
    choosing the algorithm chooses the sizing function and shows just its
    inputs -- no method selector, no irrelevant fields.
    """
    RASTER = "RASTER"
    DOMAIN = "DOMAIN"
    DETAIL = "DETAIL"
    HMIN = "HMIN"
    HMAX = "HMAX"
    DETAIL_HMIN = "DETAIL_HMIN"
    SLOPE = "SLOPE"
    SLOPE_NCELLS = "SLOPE_NCELLS"
    SLOPE_HMIN = "SLOPE_HMIN"
    MAX_GRADIENT = "MAX_GRADIENT"
    EXTENT_BUFFER = "EXTENT_BUFFER"
    OUTPUT = "OUTPUT"

    METHOD = None  # subclass: 'polynomial' | 'wavelength' | 'custom'

    def group(self):
        return "2 - Build element-size raster (hfun)"

    def groupId(self):
        return "build_hfun"

    def createInstance(self):
        return type(self)()

    def _add_inputs(self):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.RASTER, "Bathymetry raster (elevation, positive up)"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.DOMAIN, "Water polygon (optional; limits hfun to this area "
            "+ buffer, from stage 1)",
            [QgsProcessing.TypeVectorPolygon], optional=True))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.DETAIL, "Detail region (optional, polygons)",
            [QgsProcessing.TypeVectorPolygon], optional=True))

    def _add_limits(self):
        _num(self, self.HMIN, "Min element size (m)", 500.0, 0.1)
        _num(self, self.HMAX, "Max element size (m)", 10000.0, 1.0)
        _num(self, self.DETAIL_HMIN, "Detail min element size (m)", 100.0, 0.1)
        self.addParameter(QgsProcessingParameterBoolean(
            self.SLOPE,
            "Refine on bathymetric slope (shelf break)",
            defaultValue=False))
        _num(self, self.SLOPE_NCELLS,
             "Slope cells N (cells across a slope feature)", 15.0, 1.0, 1000.0)
        _num(self, self.SLOPE_HMIN,
             "Slope min element size (m; 0 = use Min element size)",
             0.0, 0.0, 1e7)
        _num(self, self.MAX_GRADIENT, "Max size gradient (m/m)", 0.1, 1e-3, 10.0)
        _num(self, self.EXTENT_BUFFER,
             "Buffer around the computed area (m; -1 = automatic)",
             -1.0, -1.0, 1e7, advanced=True)
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, "2 - Element-size raster (hfun)"))

    def _method_kwargs(self, parameters, context):
        return {}  # subclass: method-specific kwargs for build_hfun_raster

    def postProcessAlgorithm(self, context, feedback):
        _style_hfun_raster(getattr(self, "_dest", None), context, feedback)
        return {self.OUTPUT: getattr(self, "_dest", None)}

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        fb = _FeedbackAdapter(feedback)
        raster = self.parameterAsRasterLayer(parameters, self.RASTER, context)
        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        domain_geom = None
        dom_src = self.parameterAsSource(parameters, self.DOMAIN, context)
        if dom_src is not None:
            domain_geom = _source_to_shapely(dom_src, raster.crs())
            if domain_geom is None:
                fb.pushWarning("Water polygon has no usable geometry; "
                               "computing over the whole raster.")

        detail_geom = None
        source = self.parameterAsSource(parameters, self.DETAIL, context)
        if source is not None:
            detail_geom = _source_to_shapely(source, raster.crs())
            if detail_geom is None:
                fb.pushWarning("Detail layer has no usable geometry; ignoring it.")

        slope_ncells = None
        slope_hmin = None
        if self.parameterAsBoolean(parameters, self.SLOPE, context):
            slope_ncells = self.parameterAsDouble(
                parameters, self.SLOPE_NCELLS, context)
            v = self.parameterAsDouble(parameters, self.SLOPE_HMIN, context)
            slope_hmin = v if v > 0 else None

        try:
            build_hfun_raster(
                raster.source(), out_path,
                method=self.METHOD,
                hmin=self.parameterAsDouble(parameters, self.HMIN, context),
                hmax=self.parameterAsDouble(parameters, self.HMAX, context),
                detail_geom=detail_geom,
                detail_hmin=self.parameterAsDouble(parameters, self.DETAIL_HMIN, context),
                domain_geom=domain_geom,
                slope_ncells=slope_ncells,
                slope_hmin=slope_hmin,
                max_gradient=self.parameterAsDouble(parameters, self.MAX_GRADIENT, context),
                extent_buffer=self.parameterAsDouble(parameters, self.EXTENT_BUFFER, context),
                feedback=fb,
                **self._method_kwargs(parameters, context))
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f"hfun raster failed: {exc}")
        self._dest = out_path
        return {self.OUTPUT: out_path}


class BuildHfunPolynomialAlgorithm(_BuildHfunBase):
    METHOD = "polynomial"
    A = "A"
    B = "B"

    def name(self):
        return "build_hfun_polynomial"

    def displayName(self):
        return "2a - Depth polynomial"

    def shortHelpString(self):
        return ("Element-size raster from a depth polynomial:  h = a*d^2 + b*d  "
                "(d = depth in m, >= 0). Optionally also refine on the "
                "bathymetric slope:  h_slope = 2*pi*d / (N*|grad d|)  puts "
                "~N cells across steep features (shelf break) and has no "
                "effect where the seabed is flat. The result is floored at "
                "Min element size (Detail min size inside the detail "
                "polygons), capped at Max element size, then "
                "gradient-limited. Saved as a GeoTIFF in the working CRS; "
                "feeds stages 3 and 4.")

    def initAlgorithm(self, config=None):
        self._add_inputs()
        _num(self, self.A, "Coefficient a  (a*d^2)", 0.14, 0.0)
        _num(self, self.B, "Coefficient b  (b*d)", 28.0, 0.0)
        self._add_limits()

    def _method_kwargs(self, parameters, context):
        return dict(
            a=self.parameterAsDouble(parameters, self.A, context),
            b=self.parameterAsDouble(parameters, self.B, context))


class BuildHfunWavelengthAlgorithm(_BuildHfunBase):
    METHOD = "wavelength"
    WAVE_PERIOD = "WAVE_PERIOD"
    CELLS_PER_WL = "CELLS_PER_WL"
    ZMIN = "ZMIN"

    def name(self):
        return "build_hfun_wavelength"

    def displayName(self):
        return "2b - Wavelength (Hunt 1979)"

    def shortHelpString(self):
        return ("Element-size raster from the local wavelength:  h = L(T, d)/N, "
                "with L from the Hunt (1979) dispersion relation (wave period T, "
                "N cells per wavelength, depth floored at the min depth). "
                "Optionally also refine on the bathymetric slope (shelf "
                "break), see '2a - Depth polynomial'. The result is floored "
                "at Min element size, capped at Max, then gradient-limited. "
                "Feeds stages 3 and 4.")

    def initAlgorithm(self, config=None):
        self._add_inputs()
        _num(self, self.WAVE_PERIOD, "Wave period T (s)", 12.0, 0.1)
        _num(self, self.CELLS_PER_WL, "Cells per wavelength N", 20, 1, 10000, integer=True)
        _num(self, self.ZMIN, "Min depth for dispersion (m)", 1.0, 0.01, advanced=True)
        self._add_limits()

    def checkParameterValues(self, parameters, context):
        ok, msg = super().checkParameterValues(parameters, context)
        if not ok:
            return ok, msg
        if self.parameterAsDouble(parameters, self.WAVE_PERIOD, context) <= 0:
            return False, "Wave period T must be > 0."
        return True, ""

    def _method_kwargs(self, parameters, context):
        return dict(
            wave_period=self.parameterAsDouble(parameters, self.WAVE_PERIOD, context),
            cells_per_wavelength=self.parameterAsInt(parameters, self.CELLS_PER_WL, context),
            zmin=self.parameterAsDouble(parameters, self.ZMIN, context))


class BuildHfunCustomAlgorithm(_BuildHfunBase):
    METHOD = "custom"
    CUSTOM = "CUSTOM"

    def name(self):
        return "build_hfun_custom"

    def displayName(self):
        return "2c - Custom Python"

    def shortHelpString(self):
        return ("Element-size raster from custom Python: an expression using d "
                "(depth m, >= 0), x, y (UTM m) and np -- e.g. np.sqrt(9.81*d)*60 "
                "-- or a block defining  def hfun(d, x, y): return ... . "
                "Optionally also refine on the bathymetric slope (shelf "
                "break), see '2a - Depth polynomial'. The result is floored "
                "at Min element size, capped at Max, then gradient-limited. "
                "Feeds stages 3 and 4.")

    def initAlgorithm(self, config=None):
        self._add_inputs()
        self.addParameter(QgsProcessingParameterString(
            self.CUSTOM, "Python expression, or def hfun(d, x, y)",
            defaultValue="0.14*d**2 + 28*d", multiLine=True))
        self._add_limits()

    def checkParameterValues(self, parameters, context):
        ok, msg = super().checkParameterValues(parameters, context)
        if not ok:
            return ok, msg
        code = self.parameterAsString(parameters, self.CUSTOM, context)
        if not (code and code.strip()):
            return False, "The custom Python code field is empty."
        return True, ""

    def _method_kwargs(self, parameters, context):
        return dict(custom_code=self.parameterAsString(parameters, self.CUSTOM, context))


# ---------------------------------------------------------------------------
# 3. Resample boundary
# ---------------------------------------------------------------------------

class BuildHfunConstantAlgorithm(_BaseAlg):
    """Uniform element size over the water polygon; no bathymetry needed."""
    WATER = "WATER"
    DETAIL = "DETAIL"
    H_DOMAIN = "H_DOMAIN"
    H_DETAIL = "H_DETAIL"
    MAX_GRADIENT = "MAX_GRADIENT"
    EXTENT_BUFFER = "EXTENT_BUFFER"
    OUTPUT = "OUTPUT"

    def group(self):
        return "2 - Build element-size raster (hfun)"

    def groupId(self):
        return "build_hfun"

    def createInstance(self):
        return BuildHfunConstantAlgorithm()

    def name(self):
        return "build_hfun_constant"

    def displayName(self):
        return "2d - Constant value"

    def shortHelpString(self):
        return ("Element-size raster with one target size over the whole "
                "domain and another inside the detail region, without any "
                "bathymetry raster: the computed extent comes from the "
                "water polygon (stage 1). The transition between the two "
                "sizes is gradient-limited. Saved as a GeoTIFF in the "
                "working CRS; feeds stages 3 and 4.")

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.WATER, "Water polygon (from stage 1)",
            [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.DETAIL, "Detail region (optional, polygons)",
            [QgsProcessing.TypeVectorPolygon], optional=True))
        _num(self, self.H_DOMAIN, "Element size in the domain (m)",
             1000.0, 0.1)
        _num(self, self.H_DETAIL, "Element size in the detail region (m)",
             250.0, 0.1)
        _num(self, self.MAX_GRADIENT, "Max size gradient (m/m)",
             0.1, 1e-3, 10.0)
        _num(self, self.EXTENT_BUFFER,
             "Buffer around the computed area (m; -1 = automatic)",
             -1.0, -1.0, 1e7, advanced=True)
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, "2 - Element-size raster (hfun)"))

    def postProcessAlgorithm(self, context, feedback):
        _style_hfun_raster(getattr(self, "_dest", None), context, feedback)
        return {self.OUTPUT: getattr(self, "_dest", None)}

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        import pyproj
        fb = _FeedbackAdapter(feedback)
        out_path = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)

        water_src = self.parameterAsSource(parameters, self.WATER, context)
        water_geom = _source_to_shapely(water_src)
        if water_geom is None:
            raise QgsProcessingException("Water layer has no polygon geometry.")
        layer_crs = pyproj.CRS.from_wkt(water_src.sourceCrs().toWkt())

        detail_geom = None
        detail_src = self.parameterAsSource(parameters, self.DETAIL, context)
        if detail_src is not None:
            detail_geom = _source_to_shapely(detail_src, water_src.sourceCrs())
            if detail_geom is None:
                fb.pushWarning("Detail layer has no usable geometry; "
                               "ignoring it.")

        try:
            build_hfun_constant_raster(
                water_geom, out_path,
                h_domain=self.parameterAsDouble(parameters, self.H_DOMAIN, context),
                detail_geom=detail_geom,
                h_detail=self.parameterAsDouble(parameters, self.H_DETAIL, context),
                max_gradient=self.parameterAsDouble(parameters, self.MAX_GRADIENT, context),
                extent_buffer=self.parameterAsDouble(parameters, self.EXTENT_BUFFER, context),
                layer_crs=layer_crs,
                feedback=fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f"hfun raster failed: {exc}")
        self._dest = out_path
        return {self.OUTPUT: out_path}


class ResampleBoundaryAlgorithm(_BaseAlg):
    WATER = "WATER"
    HFUN = "HFUN"
    MIN_ANGLE = "MIN_ANGLE"
    MIN_HOLE_VERTS = "MIN_HOLE_VERTS"
    EDGES = "EDGES"

    def group(self):
        return "3 - Resample boundary to element size"

    def groupId(self):
        return "resample_boundary"

    def name(self):
        return "resample_boundary"

    def displayName(self):
        return "3 - Resample boundary to element size"

    def shortHelpString(self):
        return ("Resample the water polygon (stage 1) so boundary vertices are "
                "spaced by the element-size raster (stage 2), removing sharp "
                "spikes and small holes. Outputs the boundary as one continuous "
                "closed line per contour (styled with visible vertex points). "
                "Edit it with the Vertex Tool -- move / add / delete vertices, "
                "or delete whole rings -- without breaking the contour, then "
                "run '4 - Generate mesh from boundary'.")

    def createInstance(self):
        return ResampleBoundaryAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.WATER, "Water polygon (from stage 1)",
            [QgsProcessing.TypeVectorPolygon]))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.HFUN, "Element-size raster (from stage 2)"))
        _num(self, self.MIN_ANGLE, "Min boundary angle (deg)", 25.0, 0.0, 180.0)
        _num(self, self.MIN_HOLE_VERTS, "Min hole vertices", 15, 0, 10000, integer=True)
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.EDGES, "3 - Boundary edges", QgsProcessing.TypeVectorLine))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        import pyproj
        fb = _FeedbackAdapter(feedback)
        source = self.parameterAsSource(parameters, self.WATER, context)
        hfun_layer = self.parameterAsRasterLayer(parameters, self.HFUN, context)
        crs = source.sourceCrs()  # layers are delivered back in this CRS

        poly = _source_to_shapely(source)
        if poly is None:
            raise QgsProcessingException("Water layer has no polygon geometry.")

        from bluemesh2d.geom_util.proj_util import reproject_geometry, reproject_node
        try:
            hfuns = load_hfun_raster(hfun_layer.source())
            if not hfuns.crs_wkt:
                raise RuntimeError("The element-size raster has no CRS.")
            # the hfun raster carries the metric working CRS (local UTM, or
            # the tif CRS itself when the input raster is already projected)
            utm_crs = pyproj.CRS.from_wkt(hfuns.crs_wkt)
            layer_crs = pyproj.CRS.from_wkt(crs.toWkt())
            poly_utm = poly if layer_crs == utm_crs else \
                reproject_geometry(poly, layer_crs, utm_crs)
            poly_comput, node, edge = resample_boundary(
                poly_utm, hfuns,
                self.parameterAsDouble(parameters, self.MIN_ANGLE, context),
                self.parameterAsInt(parameters, self.MIN_HOLE_VERTS, context),
                fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f"Boundary resampling failed: {exc}")

        # One continuous closed line per boundary ring (exterior / hole), so
        # vertices can be moved with the vertex tool without breaking the
        # contour; stage 4 re-closes each ring automatically.
        import numpy as np

        efields = QgsFields()
        efields.append(QgsField("part", QVariant.Int))
        efields.append(QgsField("ring", QVariant.String))
        efields.append(QgsField("vertices", QVariant.Int))
        esink, edest = self.parameterAsSink(
            parameters, self.EDGES, context, efields, QgsWkbTypes.LineString, crs)

        def add_ring(part_id, ring_name, coords_utm):
            coords = np.asarray(coords_utm, dtype=float)
            if layer_crs != utm_crs:
                coords = reproject_node(coords, utm_crs, layer_crs)
            wkt = "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in coords) + ")"
            f = QgsFeature(efields)
            f.setGeometry(QgsGeometry.fromWkt(wkt))
            f.setAttributes([part_id, ring_name, len(coords) - 1])
            esink.addFeature(f)

        parts = list(poly_comput.geoms) if hasattr(poly_comput, "geoms") else [poly_comput]
        n_rings = 0
        for pi, part in enumerate(parts):
            add_ring(pi, "exterior", part.exterior.coords)
            n_rings += 1
            for hi, interior in enumerate(part.interiors):
                add_ring(pi, f"hole {hi}", interior.coords)
                n_rings += 1
        fb.pushInfo(f"Boundary rings written: {n_rings}")

        self._edges_dest = edest
        return {self.EDGES: edest}

    def postProcessAlgorithm(self, context, feedback):
        # Style the edges layer: line with visible vertex markers, so boundary
        # points can be seen and grabbed while editing.
        dest = getattr(self, "_edges_dest", None)
        if dest:
            try:
                from qgis.core import (
                    QgsLineSymbol, QgsMarkerLineSymbolLayer, QgsMarkerSymbol,
                    QgsProcessingUtils, QgsSingleSymbolRenderer,
                )
                layer = QgsProcessingUtils.mapLayerFromString(dest, context)
                if layer is not None:
                    sym = QgsLineSymbol.createSimple(
                        {"line_color": "31,120,180,255", "line_width": "0.5"})
                    marker_line = QgsMarkerLineSymbolLayer()
                    try:
                        from qgis.core import Qgis
                        marker_line.setPlacement(Qgis.MarkerLinePlacement.Vertex)
                    except Exception:
                        marker_line.setPlacement(QgsMarkerLineSymbolLayer.Vertex)
                    marker_line.setSubSymbol(QgsMarkerSymbol.createSimple(
                        {"name": "circle", "color": "227,26,28,255",
                         "outline_style": "no", "size": "1.4"}))
                    sym.appendSymbolLayer(marker_line)
                    layer.setRenderer(QgsSingleSymbolRenderer(sym))
                    layer.triggerRepaint()
            except Exception as exc:
                feedback.pushInfo(f"Could not style edges layer: {exc}")
        return {self.EDGES: dest}


# ---------------------------------------------------------------------------
# 4. Generate mesh from boundary
# ---------------------------------------------------------------------------

class GenerateMeshFromBoundaryAlgorithm(_BaseAlg):
    EDGES = "EDGES"
    HFUN = "HFUN"
    RASTER = "RASTER"
    FIXED_POINTS = "FIXED_POINTS"
    KIND = "KIND"
    DO_SMOOTH = "DO_SMOOTH"
    DO_SMOOD = "DO_SMOOD"
    SMOOD_MERGE = "SMOOD_MERGE"
    INTERP_ORDER = "INTERP_ORDER"
    OUTPUT = "OUTPUT"

    _KIND_OPTS = ["delaunay", "delfront"]
    _INTERP_OPTS = ["nearest", "bilinear", "bicubic"]
    _INTERP_VALS = [0, 1, 3]

    def group(self):
        return "4 - Generate mesh from boundary"

    def groupId(self):
        return "generate_mesh_from_boundary"

    def name(self):
        return "generate_mesh_from_boundary"

    def displayName(self):
        return "4 - Generate mesh from boundary"

    def shortHelpString(self):
        return ("Triangulate the boundary lines (stage 3, editable with the "
                "Vertex Tool) driven by the element-size raster (stage 2): "
                "Delaunay refinement (delaunay | delfront), optional smoothing, "
                "and an optional smood pass (orthogonalization for Delft3D-FM). "
                "Each line feature is treated as a closed boundary ring (closed "
                "automatically if its ends differ); vertices closer than 1 mm "
                "are snapped together. An optional point layer gives fixed "
                "points: each becomes a mesh node at exactly that position, "
                "pinned through smoothing and orthogonalization. Bathymetry "
                "is sampled from the raster onto the nodes; output is a "
                "UGRID NetCDF loaded as a mesh layer.")

    def createInstance(self):
        return GenerateMeshFromBoundaryAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.EDGES, "Boundary edges (from stage 3, possibly edited)",
            [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.HFUN, "Element-size raster (from stage 2)"))
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.RASTER, "Bathymetry raster (for node depths)"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.FIXED_POINTS,
            "Fixed points (optional; forced mesh nodes, never moved)",
            [QgsProcessing.TypeVectorPoint], optional=True))
        self.addParameter(QgsProcessingParameterEnum(
            self.KIND, "Refinement kind", options=self._KIND_OPTS, defaultValue=0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.DO_SMOOTH, "Smooth mesh after refinement", defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(
            self.DO_SMOOD, "Apply smood (orthogonalization, Delft3D-FM)",
            defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(
            self.SMOOD_MERGE,
            "smood: merge small links (only if triangle-only smood is not "
            "enough to remove the remaining small flow links)",
            defaultValue=False))
        self.addParameter(QgsProcessingParameterEnum(
            self.INTERP_ORDER, "Bathymetry interpolation",
            options=self._INTERP_OPTS, defaultValue=2))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "4 - Output mesh (UGRID NetCDF)", fileFilter="NetCDF (*.nc)"))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        _check_smood_deps(self.parameterAsBool(parameters, self.DO_SMOOD, context))
        import pyproj
        fb = _FeedbackAdapter(feedback)

        source = self.parameterAsSource(parameters, self.EDGES, context)
        hfun_layer = self.parameterAsRasterLayer(parameters, self.HFUN, context)
        bathy = self.parameterAsRasterLayer(parameters, self.RASTER, context)
        fixed_source = self.parameterAsSource(parameters, self.FIXED_POINTS, context)
        out_path = self.parameterAsFileOutput(parameters, self.OUTPUT, context)

        segments = _source_to_segments(source)
        if not segments:
            raise QgsProcessingException("Boundary edges layer has no line geometry.")

        from bluemesh2d.geom_util.proj_util import reproject_node
        try:
            hfuns = load_hfun_raster(hfun_layer.source())
            if not hfuns.crs_wkt:
                raise RuntimeError("The element-size raster has no CRS.")
            # the hfun raster carries the metric working CRS (local UTM)
            utm_crs = pyproj.CRS.from_wkt(hfuns.crs_wkt)
            layer_crs = pyproj.CRS.from_wkt(source.sourceCrs().toWkt())
            # reproject to metres BEFORE snapping, so the mm node-merge
            # tolerance is metric (in degrees it would wrongly fuse nodes)
            if layer_crs != utm_crs:
                import numpy as np
                segments = [reproject_node(np.asarray(s, dtype=float),
                                           layer_crs, utm_crs)
                            for s in segments]
            node, edge = pslg_from_segments(segments)
            fb.pushInfo(f"Boundary from edges layer: {len(node)} nodes, {len(edge)} edges")

            fixed_points = None
            if fixed_source is not None:
                import numpy as np
                pts = _source_to_points(fixed_source)
                if pts:
                    fixed_points = np.asarray(pts, dtype=float)
                    fixed_crs = pyproj.CRS.from_wkt(
                        fixed_source.sourceCrs().toWkt())
                    if fixed_crs != utm_crs:
                        fixed_points = reproject_node(
                            fixed_points, fixed_crs, utm_crs)

            vert, tria = mesh_pslg(
                node, edge, hfuns,
                kind=self._KIND_OPTS[self.parameterAsEnum(parameters, self.KIND, context)],
                do_smooth=self.parameterAsBool(parameters, self.DO_SMOOTH, context),
                do_smood=self.parameterAsBool(parameters, self.DO_SMOOD, context),
                smood_merge_small_links=self.parameterAsBool(parameters, self.SMOOD_MERGE, context),
                fixed_points=fixed_points,
                feedback=fb)
            interp_idx = self.parameterAsEnum(parameters, self.INTERP_ORDER, context)
            export_ugrid(vert, tria, bathy.source(), utm_crs, out_path,
                         self._INTERP_VALS[interp_idx], fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except QgsProcessingException:
            raise
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f"Mesh generation failed: {exc}")

        fb.pushInfo(f"Done: {len(vert)} nodes, {len(tria)} faces.")
        self._output_path = out_path
        return {self.OUTPUT: out_path}

    def postProcessAlgorithm(self, context, feedback):
        _load_mesh_layer(getattr(self, "_output_path", None), feedback)
        return {self.OUTPUT: getattr(self, "_output_path", None)}


# ---------------------------------------------------------------------------
# 5. Generate boundary conditions (editable open / closed / island lines)
# ---------------------------------------------------------------------------

def _mesh_source_path(layer):
    if layer is None:
        raise QgsProcessingException("Invalid mesh layer.")
    return layer.source().split("?")[0]


def _boundary_lines_by_type(source, target_crs):
    """Return ``{btype: [ (M,2) arrays ]}`` from a boundary conditions source.

    Accepts either the point layer produced by stage 5 (one feature per
    boundary node, with ``loop``/``seq``/``btype`` attributes -- polylines
    are rebuilt from consecutive nodes of the same type) or a legacy line
    layer with one ``btype`` per feature.

    Geometries are reprojected to `target_crs` (a
    ``QgsCoordinateReferenceSystem``) so they line up with the mesh nodes.
    Features without a ``btype`` attribute are treated as ``'closed'``.
    """
    import numpy as np

    xform = None
    src_crs = source.sourceCrs()
    if target_crs is not None and src_crs.isValid() and src_crs != target_crs:
        xform = QgsCoordinateTransform(src_crs, target_crs, QgsProject.instance())

    field_names = [f.name() for f in source.fields()]
    has_btype = "btype" in field_names

    if QgsWkbTypes.geometryType(source.wkbType()) == QgsWkbTypes.PointGeometry:
        if "loop" not in field_names or "seq" not in field_names:
            raise QgsProcessingException(
                "The boundary point layer must have 'loop' and 'seq' "
                "attributes (as produced by '5 - Generate boundary "
                "conditions').")
        loops = {}
        for feat in source.getFeatures():
            g = feat.geometry()
            if g is None or g.isEmpty():
                continue
            if xform is not None:
                g = QgsGeometry(g)
                g.transform(xform)
            p = g.asPoint()
            bt = str(feat["btype"]) if has_btype else "closed"
            loops.setdefault(int(feat["loop"]), []).append(
                (int(feat["seq"]), (p.x(), p.y()), bt))
        ordered = []
        for pts in loops.values():
            pts.sort(key=lambda t: t[0])
            ordered.append(
                (np.asarray([c for _, c, _ in pts], dtype=float),
                 [b for _, _, b in pts]))
        return boundary_lines_from_points(ordered)

    out = {}
    for feat in source.getFeatures():
        g = feat.geometry()
        if g is None or g.isEmpty():
            continue
        if xform is not None:
            g = QgsGeometry(g)
            g.transform(xform)
        bt = str(feat["btype"]) if has_btype else "closed"
        geom = _shapely_from_qgis(g)
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for line in parts:
            if line.geom_type == "LineString" and len(line.coords) >= 2:
                out.setdefault(bt, []).append(
                    np.asarray([(p[0], p[1]) for p in line.coords], dtype=float))
    return out


def _style_water_polygon(dest, context, feedback):
    """Show the water polygon's vertices, colored by their fixed flag.

    Adds two geometry-generator marker layers on top of the fill: red for
    fixed vertices (Z=1, on the extent boundary) and green for free ones.
    The markers read Z live, so edits in the Vertex Editor recolor at once.
    """
    try:
        from qgis.core import (
            QgsFillSymbol, QgsGeometryGeneratorSymbolLayer, QgsMarkerSymbol,
            QgsProcessingUtils, QgsSingleSymbolRenderer,
        )
        try:
            from qgis.core import Qgis
            marker_type = Qgis.SymbolType.Marker
        except (ImportError, AttributeError):
            from qgis.core import QgsSymbol
            marker_type = QgsSymbol.Marker

        layer = QgsProcessingUtils.mapLayerFromString(dest, context)
        if layer is None:
            return

        fill = QgsFillSymbol.createSimple(
            {"color": "166,206,227,110", "outline_color": "31,120,180,255",
             "outline_width": "0.35"})

        vertices = ("array_foreach(generate_series(1, num_points($geometry)), "
                    "point_n($geometry, @element))")

        def gen(zfilter, color, outline):
            glayer = QgsGeometryGeneratorSymbolLayer.create(
                {"geometryModifier":
                 f"collect_geometries(array_filter({vertices}, {zfilter}))"})
            glayer.setSymbolType(marker_type)
            glayer.setSubSymbol(QgsMarkerSymbol.createSimple(
                {"name": "circle", "color": color, "outline_color": outline,
                 "outline_width": "0.2", "size": "1.6"}))
            return glayer

        # free vertices (Z != 1, incl. missing Z) in green, fixed in red
        fill.appendSymbolLayer(
            gen("coalesce(z(@element), 0) <> 1",
                "51,160,44,255", "20,90,20,255"))
        fill.appendSymbolLayer(
            gen("coalesce(z(@element), 0) = 1",
                "227,26,28,255", "153,0,0,255"))

        layer.setRenderer(QgsSingleSymbolRenderer(fill))
        layer.triggerRepaint()
    except Exception as exc:
        feedback.pushInfo(f"Could not style water polygon: {exc}")


def _style_hfun_raster(dest, context, feedback):
    """Style the hfun raster as singleband pseudocolor, linear, Blues ramp."""
    try:
        from qgis.core import (
            QgsColorRampShader, QgsProcessingUtils, QgsRasterShader,
            QgsSingleBandPseudoColorRenderer, QgsStyle,
        )
        layer = QgsProcessingUtils.mapLayerFromString(dest, context)
        if layer is None:
            return

        ramp = QgsStyle.defaultStyle().colorRamp("Blues")
        if ramp is None:
            return

        provider = layer.dataProvider()
        stats = provider.bandStatistics(1)
        vmin, vmax = stats.minimumValue, stats.maximumValue

        shader = QgsRasterShader()
        renderer = QgsSingleBandPseudoColorRenderer(provider, 1, shader)
        renderer.setClassificationMin(vmin)
        renderer.setClassificationMax(vmax)
        renderer.createShader(
            ramp, QgsColorRampShader.Interpolated, QgsColorRampShader.Continuous)

        layer.setRenderer(renderer)
        layer.triggerRepaint()
    except Exception as exc:
        feedback.pushInfo(f"Could not style hfun raster: {exc}")


def _style_boundary_conditions(dest, context, feedback):
    """Categorize the boundary-condition point layer by ``btype``."""
    try:
        from qgis.core import (
            QgsCategorizedSymbolRenderer, QgsMarkerSymbol,
            QgsProcessingUtils, QgsRendererCategory,
        )
        layer = QgsProcessingUtils.mapLayerFromString(dest, context)
        if layer is None:
            return

        def sym(color, outline_color):
            return QgsMarkerSymbol.createSimple(
                {"name": "circle", "color": color,
                 "outline_color": outline_color, "outline_width": "0.2",
                 "size": "1.8"})

        cats = [
            QgsRendererCategory("open", sym("227,26,28,255", "153,0,0,255"),
                                "open boundary"),
            QgsRendererCategory("closed", sym("51,160,44,255", "20,90,20,255"),
                                "closed boundary"),
            QgsRendererCategory("island", sym("31,120,180,255", "10,60,110,255"),
                                "island"),
        ]
        layer.setRenderer(QgsCategorizedSymbolRenderer("btype", cats))

        # edit 'btype' with a drop-down list instead of free text
        idx = layer.fields().indexOf("btype")
        if idx >= 0:
            from qgis.core import QgsEditorWidgetSetup
            layer.setEditorWidgetSetup(idx, QgsEditorWidgetSetup(
                "ValueMap",
                {"map": {"open": "open", "closed": "closed",
                         "island": "island"}}))
        layer.triggerRepaint()
    except Exception as exc:
        feedback.pushInfo(f"Could not style boundary layer: {exc}")


class GenerateBoundaryConditionsAlgorithm(_BaseAlg):
    MESH = "MESH"
    ZLIM = "ZLIM"
    OUTPUT = "OUTPUT"

    def group(self):
        return "5 - Generate boundary conditions"

    def groupId(self):
        return "generate_boundary_conditions"

    def name(self):
        return "generate_boundary_conditions"

    def displayName(self):
        return "5 - Generate boundary conditions"

    def shortHelpString(self):
        return ("Classify each mesh (stage 4) boundary node into an editable "
                "point feature: open boundary (offshore, depth above the "
                "threshold), closed boundary (land on the outer boundary) and "
                "islands (interior coastline loops). The output is one point "
                "layer with a 'btype' attribute (open / closed / island), "
                "colored by type. Select points on the map and change their "
                "'btype' in the attribute table (drop-down list) to "
                "reclassify them, then feed the layer to '6 - Export' -- the "
                "export rebuilds continuous boundary lines from consecutive "
                "points of the same type.")

    def createInstance(self):
        return GenerateBoundaryConditionsAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMeshLayer(
            self.MESH, "Mesh layer (from stage 4)"))
        _num(self, self.ZLIM, "Open boundary depth threshold (m)", 0.0, -1e4, 1e5)
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "5 - Boundary conditions", QgsProcessing.TypeVectorPoint))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        fb = _FeedbackAdapter(feedback)
        layer = self.parameterAsMeshLayer(parameters, self.MESH, context)
        src_path = _mesh_source_path(layer)
        crs = layer.crs()

        try:
            loops = generate_boundary_condition_points(
                src_path,
                zlim=self.parameterAsDouble(parameters, self.ZLIM, context),
                feedback=fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f"Boundary classification failed: {exc}")

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("loop", QVariant.Int))
        fields.append(QgsField("seq", QVariant.Int))
        fields.append(QgsField("btype", QVariant.String))
        fields.append(QgsField("depth", QVariant.Double))
        sink, dest = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields, QgsWkbTypes.Point, crs)

        fid = 0
        for loop_id, loop in enumerate(loops):
            for seq, ((x, y), btype, depth) in enumerate(
                    zip(loop["coords"], loop["btype"], loop["depth"])):
                f = QgsFeature(fields)
                f.setGeometry(QgsGeometry.fromWkt(f"POINT({x} {y})"))
                f.setAttributes([fid, loop_id, seq, btype, depth])
                sink.addFeature(f)
                fid += 1
        fb.pushInfo(f"Boundary points written: {fid}")

        self._dest = dest
        return {self.OUTPUT: dest}

    def postProcessAlgorithm(self, context, feedback):
        _style_boundary_conditions(getattr(self, "_dest", None), context, feedback)
        return {self.OUTPUT: getattr(self, "_dest", None)}


# ---------------------------------------------------------------------------
# 6. Export (mesh + boundary conditions -> files)
# ---------------------------------------------------------------------------

class _ExportBase(_BaseAlg):
    def group(self):
        return "6 - Export"

    def groupId(self):
        return "export"


class ExportUgridAlgorithm(_ExportBase):
    MESH = "MESH"
    OUTPUT = "OUTPUT"

    def name(self):
        return "export_ugrid"

    def displayName(self):
        return "6a - Export UGRID (.nc)"

    def shortHelpString(self):
        return ("Save the mesh (stage 4) as a UGRID NetCDF -- the simple "
                "default export. For Delft3D-FM open-boundary files "
                "(.pli/.bc/.ext), use "
                "'6b - Export open boundary condition (.pli / .bc)' instead.")

    def createInstance(self):
        return ExportUgridAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMeshLayer(
            self.MESH, "Mesh layer (from stage 4)"))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output mesh (UGRID NetCDF)", fileFilter="NetCDF (*.nc)"))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        import os
        import shutil
        fb = _FeedbackAdapter(feedback)

        layer = self.parameterAsMeshLayer(parameters, self.MESH, context)
        src_path = _mesh_source_path(layer)
        out_path = self.parameterAsFileOutput(parameters, self.OUTPUT, context)

        try:
            if os.path.abspath(src_path) != os.path.abspath(out_path):
                shutil.copyfile(src_path, out_path)
            fb.pushInfo(f"UGRID NetCDF -> {out_path}")
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f"Export failed: {exc}")
        return {self.OUTPUT: out_path}


class ExportUgridBoundaryAlgorithm(_ExportBase):
    BOUNDARY = "BOUNDARY"
    WRITE_BC = "WRITE_BC"
    OUTPUT = "OUTPUT"

    def name(self):
        return "export_ugrid_boundary"

    def displayName(self):
        return "6b - Export open boundary condition (.pli / .bc)"

    def shortHelpString(self):
        return ("From the boundary conditions layer (stage 5), write the "
                "Delft3D-FM open-boundary files: a .pli polyline file (one "
                "polyline per 'open' feature) and, optionally, the matching "
                ".bc (Riemann time-series stanza per point) and .ext files. "
                "For the mesh NetCDF export, use '6a - Export UGRID (.nc)'.")

    def createInstance(self):
        return ExportUgridBoundaryAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.BOUNDARY, "Boundary conditions (from stage 5)",
            [QgsProcessing.TypeVectorPoint, QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterBoolean(
            self.WRITE_BC,
            "Also write .bc / .ext (Riemann boundary condition)",
            defaultValue=True))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output open boundary (.pli)", fileFilter="PLI (*.pli)"))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        import os
        fb = _FeedbackAdapter(feedback)

        boundary = self.parameterAsSource(parameters, self.BOUNDARY, context)
        if boundary is None:
            raise QgsProcessingException("A boundary conditions layer is required.")
        out_path = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        write_bc = self.parameterAsBoolean(parameters, self.WRITE_BC, context)

        out_dir = os.path.dirname(out_path) or "."
        pli_name = os.path.splitext(os.path.basename(out_path))[0]

        try:
            by_type = _boundary_lines_by_type(boundary, None)
            open_lines = by_type.get("open", [])
            if not open_lines:
                raise QgsProcessingException(
                    "No 'open' features in the boundary layer; classify one "
                    "in '5 - Generate boundary conditions' first.")

            if write_bc:
                pli, bc, ext = write_open_boundary_files(
                    out_dir, open_lines, pli_name=pli_name, feedback=fb)
                fb.pushInfo(f"Open boundary files: {pli}, {bc}, {ext}")
            else:
                pli, _ids = write_open_boundary_pli(
                    out_dir, open_lines, pli_name=pli_name, feedback=fb)
                fb.pushInfo(f"Open boundary file: {pli}")
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f"Export failed: {exc}")
        return {self.OUTPUT: out_path}


class ExportGrdAlgorithm(_ExportBase):
    MESH = "MESH"
    BOUNDARY = "BOUNDARY"
    OUTPUT = "OUTPUT"

    def name(self):
        return "export_grd"

    def displayName(self):
        return "6c - Export ADCIRC .grd (with boundaries)"

    def shortHelpString(self):
        return ("Save the mesh (stage 4) as an ADCIRC-style .grd file with "
                "open/land boundary loops taken from the boundary conditions "
                "(stage 5): 'open' features become open boundaries, 'closed' "
                "and 'island' features become land boundaries. Each boundary "
                "vertex is snapped to the nearest mesh node, so edits made in "
                "stage 5 are honoured.")

    def createInstance(self):
        return ExportGrdAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMeshLayer(
            self.MESH, "Mesh layer (from stage 4)"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.BOUNDARY, "Boundary conditions (from stage 5)",
            [QgsProcessing.TypeVectorPoint, QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output grid (.grd)", fileFilter="ADCIRC grid (*.grd)"))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        fb = _FeedbackAdapter(feedback)

        layer = self.parameterAsMeshLayer(parameters, self.MESH, context)
        src_path = _mesh_source_path(layer)
        out_path = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        boundary = self.parameterAsSource(parameters, self.BOUNDARY, context)
        if boundary is None:
            raise QgsProcessingException("A boundary conditions layer is required.")
        crs = layer.crs().authid() if layer.crs().isValid() else "EPSG:4326"

        try:
            by_type = _boundary_lines_by_type(boundary, layer.crs())
            open_lines = by_type.get("open", [])
            land_lines = by_type.get("closed", []) + by_type.get("island", [])
            export_grd_from_lines(src_path, out_path, open_lines, land_lines,
                                  crs=crs, feedback=fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f".grd export failed: {exc}")
        return {self.OUTPUT: out_path}


# ---------------------------------------------------------------------------
# 6. All-in-one
# ---------------------------------------------------------------------------

class GenerateMeshAlgorithm(_BaseAlg):
    RASTER = "RASTER"
    DETAIL = "DETAIL"
    COAST_ZMAX = "COAST_ZMAX"
    KEEP_LARGEST = "KEEP_LARGEST"
    HMIN = "HMIN"
    HMAX = "HMAX"
    DETAIL_HMIN = "DETAIL_HMIN"
    A = "A"
    B = "B"
    MAX_GRADIENT = "MAX_GRADIENT"
    MIN_ANGLE = "MIN_ANGLE"
    MIN_HOLE_VERTS = "MIN_HOLE_VERTS"
    KIND = "KIND"
    DO_SMOOTH = "DO_SMOOTH"
    DO_SMOOD = "DO_SMOOD"
    SMOOD_MERGE = "SMOOD_MERGE"
    INTERP_ORDER = "INTERP_ORDER"
    OUTPUT = "OUTPUT"

    _KIND_OPTS = ["delaunay", "delfront"]
    _INTERP_OPTS = ["nearest", "bilinear", "bicubic"]
    _INTERP_VALS = [0, 1, 3]

    def name(self):
        return "generate_mesh"

    def displayName(self):
        return "Generate mesh from bathymetry (all steps)"

    def shortHelpString(self):
        return ("Run the whole BlueMesh2D pipeline in one go: water polygon, "
                "gradient-limited size function, boundary resampling, "
                "refinement (delaunay | delfront), optional smooth / smood, "
                "bathymetry sampling and UGRID NetCDF export. For step-by-step "
                "control with inspectable layers, use algorithms 1-4 instead.")

    def createInstance(self):
        return GenerateMeshAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.RASTER, "Bathymetry raster (elevation, positive up)"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.DETAIL, "Detail region (optional, polygons)",
            [QgsProcessing.TypeVectorPolygon], optional=True))
        _num(self, self.COAST_ZMAX, "Coastline level / wet threshold (m)", 0.0, -1e4, 1e4)
        self.addParameter(QgsProcessingParameterBoolean(
            self.KEEP_LARGEST, "Keep only the largest water region",
            defaultValue=True))
        _num(self, self.HMIN, "Min element size (m)", 500.0, 0.1)
        _num(self, self.HMAX, "Max element size (m)", 10000.0, 1.0)
        _num(self, self.DETAIL_HMIN, "Detail min element size (m)", 100.0, 0.1)
        _num(self, self.A, "Depth coefficient a (a*d^2)", 0.14, 0.0)
        _num(self, self.B, "Depth coefficient b (b*d)", 28.0, 0.0)
        _num(self, self.MAX_GRADIENT, "Max size gradient (m/m)", 0.1, 1e-3, 10.0)
        _num(self, self.MIN_ANGLE, "Min boundary angle (deg)", 25.0, 0.0, 180.0)
        _num(self, self.MIN_HOLE_VERTS, "Min hole vertices", 15, 0, 10000, integer=True)
        self.addParameter(QgsProcessingParameterEnum(
            self.KIND, "Refinement kind", options=self._KIND_OPTS, defaultValue=0))
        self.addParameter(QgsProcessingParameterBoolean(
            self.DO_SMOOTH, "Smooth mesh after refinement", defaultValue=True))
        self.addParameter(QgsProcessingParameterBoolean(
            self.DO_SMOOD, "Apply smood (orthogonalization, Delft3D-FM)",
            defaultValue=False))
        self.addParameter(QgsProcessingParameterBoolean(
            self.SMOOD_MERGE,
            "smood: merge small links (only if triangle-only smood is not "
            "enough to remove the remaining small flow links)",
            defaultValue=False))
        self.addParameter(QgsProcessingParameterEnum(
            self.INTERP_ORDER, "Bathymetry interpolation",
            options=self._INTERP_OPTS, defaultValue=2))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output mesh (UGRID NetCDF)", fileFilter="NetCDF (*.nc)"))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        self._accept_invalid_geometries(context)
        _check_smood_deps(self.parameterAsBool(parameters, self.DO_SMOOD, context))
        fb = _FeedbackAdapter(feedback)
        raster_layer = self.parameterAsRasterLayer(parameters, self.RASTER, context)
        if raster_layer is None:
            raise QgsProcessingException("Invalid raster layer.")

        detail_geom = None
        source = self.parameterAsSource(parameters, self.DETAIL, context)
        if source is not None:
            detail_geom = _source_to_shapely(source, raster_layer.crs())
            if detail_geom is None:
                fb.pushWarning("Detail layer has no usable geometry; ignoring it.")

        interp_idx = self.parameterAsEnum(parameters, self.INTERP_ORDER, context)
        cfg = MeshConfig(
            raster_path=raster_layer.source(),
            output_path=self.parameterAsFileOutput(parameters, self.OUTPUT, context),
            coast_zmax=self.parameterAsDouble(parameters, self.COAST_ZMAX, context),
            keep_largest=self.parameterAsBool(parameters, self.KEEP_LARGEST, context),
            detail_geom=detail_geom,
            detail_hmin=self.parameterAsDouble(parameters, self.DETAIL_HMIN, context),
            a=self.parameterAsDouble(parameters, self.A, context),
            b=self.parameterAsDouble(parameters, self.B, context),
            hmin=self.parameterAsDouble(parameters, self.HMIN, context),
            hmax=self.parameterAsDouble(parameters, self.HMAX, context),
            max_gradient=self.parameterAsDouble(parameters, self.MAX_GRADIENT, context),
            min_angle_deg=self.parameterAsDouble(parameters, self.MIN_ANGLE, context),
            min_hole_vertices=self.parameterAsInt(parameters, self.MIN_HOLE_VERTS, context),
            kind=self._KIND_OPTS[self.parameterAsEnum(parameters, self.KIND, context)],
            do_smooth=self.parameterAsBool(parameters, self.DO_SMOOTH, context),
            do_smood=self.parameterAsBool(parameters, self.DO_SMOOD, context),
            smood_merge_small_links=self.parameterAsBool(parameters, self.SMOOD_MERGE, context),
            interp_order=self._INTERP_VALS[interp_idx],
        )

        try:
            result = generate_mesh(cfg, fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except QgsProcessingException:
            raise
        except Exception as exc:
            import traceback
            fb.pushInfo(traceback.format_exc())
            raise QgsProcessingException(f"Mesh generation failed: {exc}")

        fb.pushInfo(f"Done: {result.n_nodes} nodes, {result.n_triangles} triangles.")
        self._output_path = result.output_path
        return {self.OUTPUT: result.output_path}

    def postProcessAlgorithm(self, context, feedback):
        _load_mesh_layer(getattr(self, "_output_path", None), feedback)
        return {self.OUTPUT: getattr(self, "_output_path", None)}


def _load_mesh_layer(path, feedback):
    """Load a UGRID NetCDF as a mesh layer and add it to the project.

    Processing does not auto-load mesh outputs, so this is called explicitly
    from ``postProcessAlgorithm``.

    Parameters
    ----------
    path : str or None
        Path to the UGRID NetCDF; no-op if falsy.
    feedback : object
        Feedback sink exposing ``pushInfo``.
    """
    if not path:
        return
    try:
        from qgis.core import QgsMeshLayer
        layer = QgsMeshLayer(path, "4 - BlueMesh2D mesh", "mdal")
        if layer.isValid():
            _enable_native_mesh(layer, feedback)
            # Add without auto-inserting into the tree, then insert at the
            # very top so it sits above every layer already produced by
            # earlier pipeline stages.
            QgsProject.instance().addMapLayer(layer, False)
            QgsProject.instance().layerTreeRoot().insertLayer(0, layer)
        else:
            feedback.pushInfo(
                "Mesh written but could not be loaded as a layer; open it "
                "manually via Layer > Add Mesh Layer.")
    except Exception as exc:
        feedback.pushInfo(f"Could not add mesh layer: {exc}")


def _enable_native_mesh(layer, feedback):
    """Turn on native-mesh (triangle frame) rendering, off by default in QGIS.

    Parameters
    ----------
    layer : QgsMeshLayer
        Mesh layer to configure.
    feedback : object
        Feedback sink exposing ``pushInfo``.
    """
    try:
        settings = layer.rendererSettings()
        native = settings.nativeMeshSettings()
        native.setEnabled(True)
        settings.setNativeMeshSettings(native)
        layer.setRendererSettings(settings)
        layer.triggerRepaint()
    except Exception as exc:
        feedback.pushInfo(f"Could not enable native mesh rendering: {exc}")


ALL_ALGORITHMS = (
    ExtractWaterPolygonAlgorithm,
    BuildHfunPolynomialAlgorithm,
    BuildHfunWavelengthAlgorithm,
    BuildHfunCustomAlgorithm,
    BuildHfunConstantAlgorithm,
    ResampleBoundaryAlgorithm,
    GenerateMeshFromBoundaryAlgorithm,
    GenerateBoundaryConditionsAlgorithm,
    ExportUgridAlgorithm,
    ExportUgridBoundaryAlgorithm,
    ExportGrdAlgorithm,
    # GenerateMeshAlgorithm ("all steps") intentionally not registered:
    # the step-by-step workflow is the supported path (the class and the
    # bluemesh2d.pipeline.generate_mesh facade remain for Python scripting)
)
