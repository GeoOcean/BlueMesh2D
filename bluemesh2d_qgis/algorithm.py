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
    Mesh -> editable open/closed/island boundary line layer (stage 5).
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
    build_hfun_raster,
    check_dependencies,
    export_grd_from_lines,
    export_ugrid,
    extract_water_polygon,
    generate_boundary_conditions,
    generate_mesh,
    load_hfun_raster,
    mesh_pslg,
    pslg_from_segments,
    resample_boundary,
    smood_dependencies,
    write_open_boundary_files,
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
            + ", ".join(missing) + ". See the plugin README.")


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


def _source_to_shapely(source, target_crs=None):
    """Union all polygon features of a source into one shapely geometry.

    Parameters
    ----------
    source : QgsProcessingFeatureSource
        Polygon feature source.
    target_crs : QgsCoordinateReferenceSystem or None, optional
        If given, features are reprojected into it before conversion.
        Default is ``None``.

    Returns
    -------
    geom : shapely.geometry.base.BaseGeometry or None
        Union of all feature geometries, or ``None`` if `source` has no
        usable geometry.
    """
    from shapely import wkt as shapely_wkt
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
        if xform is not None:
            g = QgsGeometry(g)
            g.transform(xform)
        geoms.append(shapely_wkt.loads(g.asWkt()))
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
    from shapely import wkt as shapely_wkt

    segments = []
    for feat in source.getFeatures():
        g = feat.geometry()
        if g is None or g.isEmpty():
            continue
        geom = shapely_wkt.loads(g.asWkt())
        lines = geom.geoms if hasattr(geom, "geoms") else [geom]
        for line in lines:
            if line.geom_type == "LineString" and len(line.coords) >= 2:
                segments.append([(p[0], p[1]) for p in line.coords])
    return segments


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

class ExtractWaterPolygonAlgorithm(_BaseAlg):
    RASTER = "RASTER"
    COAST_ZMAX = "COAST_ZMAX"
    DEEP_ZMAX = "DEEP_ZMAX"
    EXTENT = "EXTENT"
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
                "grown/shrunk by the domain buffer factor. Output feeds "
                "'3 - Resample boundary'.")

    def createInstance(self):
        return ExtractWaterPolygonAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterRasterLayer(
            self.RASTER, "Bathymetry raster (elevation, positive up)"))
        _num(self, self.COAST_ZMAX, "Coastline level / wet threshold (m)", 2.0, -1e4, 1e4)
        self.addParameter(QgsProcessingParameterNumber(
            self.DEEP_ZMAX, "Deep level (m, optional; e.g. -300 keeps z > -300)",
            QgsProcessingParameterNumber.Double, optional=True,
            minValue=-1e5, maxValue=1e4))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.EXTENT, "Extent polygon (optional; default = raster extent)",
            [QgsProcessing.TypeVectorPolygon], optional=True))
        _num(self, self.DOMAIN_BUFFER,
             "Domain buffer factor (negative shrinks; ignored if an extent "
             "polygon is set)", -0.05, -10, 10)
        self.addParameter(QgsProcessingParameterBoolean(
            self.KEEP_LARGEST, "Keep only the largest water region",
            defaultValue=True))
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Water polygon", QgsProcessing.TypeVectorPolygon))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        fb = _FeedbackAdapter(feedback)
        raster = self.parameterAsRasterLayer(parameters, self.RASTER, context)

        deep_zmax = None
        if parameters.get(self.DEEP_ZMAX) is not None:
            deep_zmax = self.parameterAsDouble(parameters, self.DEEP_ZMAX, context)

        extent_geom = None
        extent_src = self.parameterAsSource(parameters, self.EXTENT, context)
        if extent_src is not None:
            extent_geom = _source_to_shapely(extent_src, raster.crs())
            if extent_geom is None:
                fb.pushWarning("Extent layer has no usable geometry; using the raster extent.")

        try:
            poly, utm_crs, raster_crs = extract_water_polygon(
                raster.source(),
                self.parameterAsDouble(parameters, self.COAST_ZMAX, context),
                self.parameterAsDouble(parameters, self.DOMAIN_BUFFER, context),
                deep_zmax=deep_zmax,
                extent_geom=extent_geom,
                keep_largest=self.parameterAsBool(parameters, self.KEEP_LARGEST, context),
                feedback=fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            raise QgsProcessingException(f"Extraction failed: {exc}")

        # Deliver the layer in the raster's CRS (a CRS QGIS resolves natively);
        # the metric local-UTM CRS stays internal to the pipeline.
        from bluemesh2d.geom_util.proj_util import reproject_geometry

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("area_km2", QVariant.Double))
        sink, dest_id = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields,
            QgsWkbTypes.MultiPolygon, raster.crs())

        parts = list(poly.geoms) if hasattr(poly, "geoms") else [poly]
        for i, p in enumerate(parts):
            area_km2 = p.area / 1e6  # metric area, computed in UTM
            p = reproject_geometry(p, utm_crs, raster_crs)
            f = QgsFeature(fields)
            f.setGeometry(QgsGeometry.fromWkt(p.wkt))
            f.setAttributes([i, area_km2])
            sink.addFeature(f)
        return {self.OUTPUT: dest_id}


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
        _num(self, self.HMIN, "Min element size (m)", 100.0, 0.1)
        _num(self, self.HMAX, "Max element size (m)", 10000.0, 1.0)
        _num(self, self.DETAIL_HMIN, "Detail min element size (m)", 30.0, 0.1)
        _num(self, self.MAX_GRADIENT, "Max size gradient (m/m)", 0.1, 1e-3, 10.0)
        _num(self, self.EXTENT_BUFFER,
             "Buffer around the computed area (m; -1 = automatic)",
             -1.0, -1.0, 1e7, advanced=True)
        self.addParameter(QgsProcessingParameterRasterDestination(
            self.OUTPUT, "Element-size raster (hfun)"))

    def _method_kwargs(self, parameters, context):
        return {}  # subclass: method-specific kwargs for build_hfun_raster

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
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

        try:
            build_hfun_raster(
                raster.source(), out_path,
                method=self.METHOD,
                hmin=self.parameterAsDouble(parameters, self.HMIN, context),
                hmax=self.parameterAsDouble(parameters, self.HMAX, context),
                detail_geom=detail_geom,
                detail_hmin=self.parameterAsDouble(parameters, self.DETAIL_HMIN, context),
                domain_geom=domain_geom,
                max_gradient=self.parameterAsDouble(parameters, self.MAX_GRADIENT, context),
                extent_buffer=self.parameterAsDouble(parameters, self.EXTENT_BUFFER, context),
                feedback=fb,
                **self._method_kwargs(parameters, context))
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            raise QgsProcessingException(f"hfun raster failed: {exc}")
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
                "(d = depth in m, >= 0). The result is floored at Min element "
                "size (Detail min size inside the detail polygons), capped at "
                "Max element size, then gradient-limited. Saved as a GeoTIFF in "
                "the working CRS; feeds stages 3 and 4.")

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
                "N cells per wavelength, depth floored at the min depth). The "
                "result is floored at Min element size, capped at Max, then "
                "gradient-limited. Feeds stages 3 and 4.")

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
                "-- or a block defining  def hfun(d, x, y): return ... . The "
                "result is floored at Min element size, capped at Max, then "
                "gradient-limited. Feeds stages 3 and 4.")

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
            self.EDGES, "Boundary edges", QgsProcessing.TypeVectorLine))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
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
                "are snapped together. Bathymetry is sampled from the raster "
                "onto the nodes; output is a UGRID NetCDF loaded as a mesh "
                "layer.")

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
        _check_smood_deps(self.parameterAsBool(parameters, self.DO_SMOOD, context))
        import pyproj
        fb = _FeedbackAdapter(feedback)

        source = self.parameterAsSource(parameters, self.EDGES, context)
        hfun_layer = self.parameterAsRasterLayer(parameters, self.HFUN, context)
        bathy = self.parameterAsRasterLayer(parameters, self.RASTER, context)
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
            vert, tria = mesh_pslg(
                node, edge, hfuns,
                kind=self._KIND_OPTS[self.parameterAsEnum(parameters, self.KIND, context)],
                do_smooth=self.parameterAsBool(parameters, self.DO_SMOOTH, context),
                do_smood=self.parameterAsBool(parameters, self.DO_SMOOD, context),
                smood_merge_small_links=self.parameterAsBool(parameters, self.SMOOD_MERGE, context),
                feedback=fb)
            interp_idx = self.parameterAsEnum(parameters, self.INTERP_ORDER, context)
            export_ugrid(vert, tria, bathy.source(), utm_crs, out_path,
                         self._INTERP_VALS[interp_idx], fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except QgsProcessingException:
            raise
        except Exception as exc:
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
    """Return ``{btype: [ (M,2) arrays ]}`` from a boundary line source.

    Geometries are reprojected to `target_crs` (a
    ``QgsCoordinateReferenceSystem``) so they line up with the mesh nodes.
    Features without a ``btype`` attribute are treated as ``'closed'``.
    """
    import numpy as np
    from shapely import wkt as shapely_wkt

    xform = None
    src_crs = source.sourceCrs()
    if target_crs is not None and src_crs.isValid() and src_crs != target_crs:
        xform = QgsCoordinateTransform(src_crs, target_crs, QgsProject.instance())

    has_btype = "btype" in [f.name() for f in source.fields()]
    out = {}
    for feat in source.getFeatures():
        g = feat.geometry()
        if g is None or g.isEmpty():
            continue
        if xform is not None:
            g = QgsGeometry(g)
            g.transform(xform)
        bt = str(feat["btype"]) if has_btype else "closed"
        geom = shapely_wkt.loads(g.asWkt())
        parts = geom.geoms if hasattr(geom, "geoms") else [geom]
        for line in parts:
            if line.geom_type == "LineString" and len(line.coords) >= 2:
                out.setdefault(bt, []).append(
                    np.asarray([(p[0], p[1]) for p in line.coords], dtype=float))
    return out


def _style_boundary_conditions(dest, context, feedback):
    """Categorize the boundary-condition layer by ``btype`` with vertex dots."""
    try:
        from qgis.core import (
            QgsCategorizedSymbolRenderer, QgsLineSymbol,
            QgsMarkerLineSymbolLayer, QgsMarkerSymbol, QgsProcessingUtils,
            QgsRendererCategory,
        )
        layer = QgsProcessingUtils.mapLayerFromString(dest, context)
        if layer is None:
            return

        def sym(line_color, dot_color):
            s = QgsLineSymbol.createSimple(
                {"line_color": line_color, "line_width": "0.6"})
            ml = QgsMarkerLineSymbolLayer()
            try:
                from qgis.core import Qgis
                ml.setPlacement(Qgis.MarkerLinePlacement.Vertex)
            except Exception:
                ml.setPlacement(QgsMarkerLineSymbolLayer.Vertex)
            ml.setSubSymbol(QgsMarkerSymbol.createSimple(
                {"name": "circle", "color": dot_color,
                 "outline_style": "no", "size": "1.4"}))
            s.appendSymbolLayer(ml)
            return s

        cats = [
            QgsRendererCategory("open", sym("227,26,28,255", "153,0,0,255"),
                                "open boundary"),
            QgsRendererCategory("closed", sym("51,160,44,255", "20,90,20,255"),
                                "closed boundary"),
            QgsRendererCategory("island", sym("31,120,180,255", "10,60,110,255"),
                                "island"),
        ]
        layer.setRenderer(QgsCategorizedSymbolRenderer("btype", cats))
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
        return ("Classify the mesh (stage 4) boundary into editable line "
                "features: open boundary (offshore, mean depth above the "
                "threshold), closed boundary (land on the outer boundary) and "
                "islands (interior coastline loops). The output is one line "
                "layer with a 'btype' attribute (open / closed / island), "
                "styled by type with visible vertex dots. Edit it with the "
                "Vertex Tool and by changing 'btype' (e.g. reclassify a segment "
                "from closed to open), then feed it to '6 - Export'.")

    def createInstance(self):
        return GenerateBoundaryConditionsAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMeshLayer(
            self.MESH, "Mesh layer (from stage 4)"))
        _num(self, self.ZLIM, "Open boundary depth threshold (m)", 20.0, -1e4, 1e5)
        self.addParameter(QgsProcessingParameterFeatureSink(
            self.OUTPUT, "Boundary conditions", QgsProcessing.TypeVectorLine))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        import numpy as np
        fb = _FeedbackAdapter(feedback)
        layer = self.parameterAsMeshLayer(parameters, self.MESH, context)
        src_path = _mesh_source_path(layer)
        crs = layer.crs()

        try:
            lines = generate_boundary_conditions(
                src_path,
                zlim=self.parameterAsDouble(parameters, self.ZLIM, context),
                feedback=fb)
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
            raise QgsProcessingException(f"Boundary classification failed: {exc}")

        fields = QgsFields()
        fields.append(QgsField("id", QVariant.Int))
        fields.append(QgsField("btype", QVariant.String))
        fields.append(QgsField("npoints", QVariant.Int))
        sink, dest = self.parameterAsSink(
            parameters, self.OUTPUT, context, fields, QgsWkbTypes.LineString, crs)

        fid = 0
        for btype in ("open", "closed", "island"):
            for coords in lines.get(btype, []):
                coords = np.asarray(coords, dtype=float)
                wkt = "LINESTRING(" + ", ".join(f"{x} {y}" for x, y in coords) + ")"
                f = QgsFeature(fields)
                f.setGeometry(QgsGeometry.fromWkt(wkt))
                f.setAttributes([fid, btype, len(coords)])
                sink.addFeature(f)
                fid += 1
        fb.pushInfo(f"Boundary features written: {fid}")

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
        return ("Save the mesh (stage 4) as a UGRID NetCDF, with no boundary "
                "condition files -- the simple default export. For Delft3D-FM "
                "open-boundary files (.pli/.bc/.ext), use "
                "'6b - Export UGRID + open boundary condition' instead.")

    def createInstance(self):
        return ExportUgridAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMeshLayer(
            self.MESH, "Mesh layer (from stage 4)"))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output mesh (UGRID NetCDF)", fileFilter="NetCDF (*.nc)"))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
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
            raise QgsProcessingException(f"Export failed: {exc}")
        return {self.OUTPUT: out_path}


class ExportUgridBoundaryAlgorithm(_ExportBase):
    MESH = "MESH"
    BOUNDARY = "BOUNDARY"
    OUTPUT = "OUTPUT"

    def name(self):
        return "export_ugrid_boundary"

    def displayName(self):
        return "6b - Export UGRID (.nc) + open boundary condition"

    def shortHelpString(self):
        return ("Save the mesh (stage 4) as a UGRID NetCDF and, from the "
                "boundary conditions (stage 5), write the Delft3D-FM "
                "open-boundary files next to it: Boundary01.pli (one polyline "
                "per 'open' feature), Riemann.bc (a Riemann time-series stanza "
                "per point) and FlowFM_bnd.ext. For a plain mesh export with "
                "no boundary files, use '6a - Export UGRID (.nc)' instead.")

    def createInstance(self):
        return ExportUgridBoundaryAlgorithm()

    def initAlgorithm(self, config=None):
        self.addParameter(QgsProcessingParameterMeshLayer(
            self.MESH, "Mesh layer (from stage 4)"))
        self.addParameter(QgsProcessingParameterFeatureSource(
            self.BOUNDARY, "Boundary conditions (from stage 5)",
            [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output mesh (UGRID NetCDF)", fileFilter="NetCDF (*.nc)"))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
        import os
        import shutil
        fb = _FeedbackAdapter(feedback)

        layer = self.parameterAsMeshLayer(parameters, self.MESH, context)
        src_path = _mesh_source_path(layer)
        out_path = self.parameterAsFileOutput(parameters, self.OUTPUT, context)
        boundary = self.parameterAsSource(parameters, self.BOUNDARY, context)
        if boundary is None:
            raise QgsProcessingException(
                "A boundary conditions layer is required (use "
                "'6a - Export UGRID (.nc)' if you don't need boundary files).")

        try:
            if os.path.abspath(src_path) != os.path.abspath(out_path):
                shutil.copyfile(src_path, out_path)
                fb.pushInfo(f"UGRID NetCDF -> {out_path}")

            by_type = _boundary_lines_by_type(boundary, layer.crs())
            open_lines = by_type.get("open", [])
            if not open_lines:
                fb.pushWarning(
                    "No 'open' features in the boundary layer; no boundary "
                    "condition files were written.")
            else:
                pli, bc, ext = write_open_boundary_files(
                    os.path.dirname(out_path) or ".", open_lines, feedback=fb)
                fb.pushInfo(f"Open boundary files: {pli}, {bc}, {ext}")
        except MeshCanceled:
            raise QgsProcessingException("Canceled by user.")
        except Exception as exc:
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
            [QgsProcessing.TypeVectorLine]))
        self.addParameter(QgsProcessingParameterFileDestination(
            self.OUTPUT, "Output grid (.grd)", fileFilter="ADCIRC grid (*.grd)"))

    def processAlgorithm(self, parameters, context, feedback):
        _require_deps()
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

    def group(self):
        return "7 - All steps"

    def groupId(self):
        return "all_steps"

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
        _num(self, self.COAST_ZMAX, "Coastline level / wet threshold (m)", 2.0, -1e4, 1e4)
        self.addParameter(QgsProcessingParameterBoolean(
            self.KEEP_LARGEST, "Keep only the largest water region",
            defaultValue=True))
        _num(self, self.HMIN, "Min element size (m)", 100.0, 0.1)
        _num(self, self.HMAX, "Max element size (m)", 10000.0, 1.0)
        _num(self, self.DETAIL_HMIN, "Detail min element size (m)", 30.0, 0.1)
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
        layer = QgsMeshLayer(path, "BlueMesh2D mesh", "mdal")
        if layer.isValid():
            _enable_native_mesh(layer, feedback)
            QgsProject.instance().addMapLayer(layer)
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
    ResampleBoundaryAlgorithm,
    GenerateMeshFromBoundaryAlgorithm,
    GenerateBoundaryConditionsAlgorithm,
    ExportUgridAlgorithm,
    ExportUgridBoundaryAlgorithm,
    ExportGrdAlgorithm,
    GenerateMeshAlgorithm,
)
