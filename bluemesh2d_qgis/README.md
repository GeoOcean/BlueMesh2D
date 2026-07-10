# BlueMesh2D — QGIS plugin

A self-contained QGIS **Processing** plugin that generates an unstructured
triangular mesh from a bathymetry GeoTIFF, wrapping a bundled copy of the
[BlueMesh2D](https://github.com/GeoOcean/BlueMesh2D) library. The output is a
UGRID NetCDF that loads directly as a QGIS **mesh layer**.

## About BlueMesh2D

`BlueMesh2D` (successor of `PyMesh2D`, part of the BlueMath ecosystem) is a
pure-Python unstructured mesh generator for two-dimensional polygonal
geometries. It is a Python translation of
[`MESH2D`](https://github.com/dengwirda/mesh2d) (Darren Engwirda's MATLAB/
Octave tool) and a simplified 2D sibling of his C++
[`JIGSAW`](https://github.com/dengwirda/jigsaw-matlab) mesh generator.

It implements:
- **Delaunay refinement** and a **Frontal-Delaunay ("delfront")** scheme for
  generating triangulations from a boundary (PSLG) and a user-defined
  mesh-size function;
- **non-linear mesh optimisation** (`smooth`), plus `smood`, an
  orthogonalization / small-link-merging pass geared at Delft3D-FM
  flexible-mesh quality requirements;
- mesh-spacing ("hfun") utilities: gradient limiting, depth/wavelength-based
  sizing, local-feature-size estimation;
- geometry and bathymetry pre/post-processing: coastline extraction from
  raster elevation, polygon resampling, CRS-aware raster/point interpolation,
  boundary classification (open/land), and export to ADCIRC `.grd` /
  Delft3D-FM UGRID NetCDF.

The algorithms are "probably-good": they guarantee termination, geometric and
topological correctness, and worst-case element-quality bounds, while
supporting user-defined mesh-size functions and multi-part domains. See the
[BlueMesh2D repository](https://github.com/GeoOcean/BlueMesh2D) for the full
library, its demos (`python -m bluemesh2d.tridemo <n>`), and references.

This plugin exposes the raster-to-mesh subset of that library — coastline
extraction, sizing, refinement/smoothing, and export — as QGIS Processing
algorithms, so the workflow can run without Python scripting and its
intermediate results (water polygon, size raster, boundary, mesh) become
ordinary QGIS layers.

## What it does

```
bathymetry.tif
   → extract coastline + domain (depth threshold)
   → depth-based element size  h = clip(a·d² + b·d, hmin, hmax)
     (+ optional finer 'detail' region)
   → gradient-limit the size field (smooth, Lipschitz)
   → refine (delaunay | delfront) → smooth [→ smood orthogonalization]
   → sample bathymetry onto nodes
   → write UGRID NetCDF  → load as a QgsMeshLayer
```

## The algorithms (Processing Toolbox ▸ BlueMesh2D)

The pipeline is available **split into four stages** — each result lands in an
ordinary (temporary or saved) QGIS layer you can inspect before the next step —
plus an all-in-one:

| algorithm | inputs | outputs |
|---|---|---|
| **1 – Extract water polygon** | bathy raster; coastline level; optional **deep level** (keeps the band `deep < z ≤ coast`, e.g. water shallower than 300 m); optional **extent polygon** (clips the raster *before* extraction — much faster; buffer then ignored); buffer factor (default −0.05) | water polygon layer |
| **2 – Build element-size raster (hfun)** *(folder)* | one algorithm **per sizing method** (pick the algorithm, see only its parameters): **2a** depth polynomial `a·d²+b·d`; **2b** wavelength `L(T,d)/N` (Hunt 1979, `hfun_wavenumhunt`); **2c** custom Python (`d`,`x`,`y`,`np`). All share: optional **Water polygon (stage 1)** to limit computation to that area + buffer (much faster than the whole raster), detail polygons, min/max/detail size, gradient and **buffer** (m, −1 = automatic) | GeoTIFF, pixel = element size (m) |
| **3 – Resample boundary to element size** | water polygon (1), hfun raster (2) | boundary **edges** line layer, styled with visible vertex markers (editable) |
| **4 – Generate mesh from boundary** | **edges layer (3) — editable: move/delete/add segments first**, hfun raster (2), bathy raster; kind = delaunay/delfront; smooth; optional **smood** (+ *merge small links*, only if triangle-only smood can't remove the last small flow links) | UGRID NetCDF → mesh layer |
| **5 – Generate boundary conditions** | mesh layer (4); depth threshold (default 20 m) | one **line layer** with a `btype` attribute — **open** / **closed** / **island** — styled by type with visible vertex dots, **editable** (move vertices, or change `btype` to reclassify a segment) |
| **6 – Export** *(folder)* — **6a** plain UGRID, **6b** UGRID + open BC, **6c** ADCIRC `.grd` | **6a** (default): mesh layer (4) only → `.nc`, no boundary files, no boundary layer needed. **6b**: mesh (4) **+ boundary conditions (5, required)** → `.nc` and, from the `open` features, `Boundary01.pli` / `Riemann.bc` / `FlowFM_bnd.ext`. **6c**: mesh (4) + boundary conditions (5, required) → `.grd` with open/land loops (`open`→open, `closed`+`island`→land), snapping each boundary vertex back to the nearest mesh node so stage-5 edits are honoured | `.nc` / `.nc` + Delft3D-FM BC / `.grd` |
| **Generate mesh from bathymetry (all steps)** | stages 1–4 in one dialog | UGRID NetCDF → mesh layer |

Stage 4 rebuilds the PSLG from the (possibly edited) boundary lines, so you can
reshape the domain before meshing; likewise stage 6 rebuilds the boundary
classification from the (possibly edited) stage-5 lines, so you can hand-correct
which segments are open / closed / island before exporting. The stages also
chain in the **Graphical Modeler**.

Every step is its own numbered folder under **BlueMesh2D** (`1 - Extract water
polygon`, `2 - Build element-size raster (hfun)` with 2a/2b/2c inside, `3 -
Resample boundary...`, ..., `6 - Export` with 6a/6b/6c inside, `7 - All steps`).
QGIS's Processing toolbox sorts grouped and ungrouped algorithms in two
separate buckets rather than one merged alphabetical list, so giving every
step its own group is what keeps them in numeric order 1→7 instead of the
folders (2, 6) drifting away from the single algorithms.

**CRS handling**: vector layers are always delivered in the input tif's CRS.
If the tif is *geographic* (e.g. EPSG:4326), a local metric UTM CRS is built
internally (and embedded in the hfun raster). If the tif is already
*projected* (e.g. UTM), the whole pipeline runs natively in that CRS — no
reprojection at all — and the output NetCDF carries metre units with a
`projected_coordinate_system` grid mapping.

## Layout

```
bluemesh2d_qgis/
├── __init__.py        classFactory (QGIS entry point)
├── metadata.txt       plugin metadata (hasProcessingProvider=yes)
├── plugin.py          registers the Processing provider
├── provider.py        BlueMesh2DProvider
├── algorithm.py       Processing algorithms
├── pipeline.py        headless orchestration facade (no QGIS dependency)
└── bluemesh2d/        bundled copy of the meshing library
```

`pipeline.py` is the integration seam: it puts the plugin directory on
`sys.path` so the bundled `bluemesh2d` imports as a top-level package, forces
the non-interactive `Agg` matplotlib backend (the contour extraction needs it),
routes progress/cancellation through the algorithm's
`QgsProcessingFeedback`, and captures the mesher's stdout into the log.

## Install

1. Copy this whole `bluemesh2d_qgis/` folder into your QGIS plugins directory:
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
2. Restart QGIS → **Plugins ▸ Manage and Install Plugins ▸ Installed** → enable
   **BlueMesh2D**.
3. The algorithm appears in the **Processing Toolbox** under
   *BlueMesh2D ▸ Generate mesh from bathymetry (all steps)*.

### Python dependencies (important)

Required in **QGIS's own Python** (all commonly present already):

```
numpy  scipy  shapely  pyproj  matplotlib  rasterio  netCDF4
```

Optional:

- **`triangle`** — Shewchuk's Triangle, for fast *constrained* Delaunay
  triangulation. Without it the plugin automatically falls back to a
  pure-`scipy` *conforming* Delaunay (same mesh character, somewhat slower).
- **`xarray`** — required only by the **"Apply smood"** option (stage 4 /
  all-in-one). `bluemesh2d.smood` always builds an in-memory
  `xarray.Dataset` internally, regardless of the *merge small links*
  sub-option. It is **not** needed for anything else — in particular every
  UGRID NetCDF export (stage 6a and the all-in-one output) is written
  directly with `netCDF4`. If it's missing, the algorithm now fails
  immediately with a clear message instead of partway through refinement;
  either install `xarray` or leave "Apply smood" unchecked.

If something is missing, the algorithm names it up front. To install into the
interpreter QGIS uses:

- **Windows** — open the *OSGeo4W Shell* and run:
  ```
  python -m pip install rasterio netCDF4 xarray        # + triangle (optional)
  ```
- **Linux / macOS** — install into the Python QGIS runs on (Debian/Ubuntu system
  Python needs `--break-system-packages` with `--user`):
  ```
  python3 -m pip install --user rasterio netCDF4 xarray   # + triangle (optional)
  ```

If `rasterio` clashes with QGIS's bundled GDAL on your platform, run the
pipeline out-of-process instead (see below).

## Usage

- **Bathymetry raster** — GeoTIFF, elevation positive up (depth = −elevation).
- **Detail region** *(optional)* — a polygon layer; its union is refined down to
  *Detail min element size*. Digitize it in QGIS or load a file; any CRS (it is
  reprojected to the raster CRS automatically).
- **Sizing** — *Min/Max element size*, depth coefficients *a*, *b*, and
  *Max size gradient* (m/m; lower = smoother, wider fine-to-coarse transitions).
- **Output** — choose a `.nc` path. On success the mesh is added to the project.

> Element count scales as 1/h². A small *Min size* over a large *Detail region*
> can produce millions of triangles and exhaust memory — keep the detail area
> tight and the floor sensible.

## Run without QGIS (headless / CLI)

`pipeline.py` has no QGIS dependency, so you can script it:

```python
import sys
sys.path.insert(0, "/path/to/bluemesh2d_qgis")
from pipeline import MeshConfig, generate_mesh

res = generate_mesh(MeshConfig(
    raster_path="Bati_10m_ohau_4326.tif",
    output_path="mesh.nc",
    hmin=100.0, hmax=10000.0, max_gradient=0.1,
    # detail_geom=<a shapely polygon in the raster CRS>, detail_hmin=30.0,
))
print(res)   # MeshResult(n_nodes=..., n_triangles=..., ...)
```

This is also the recommended path for an **out-of-process** setup: keep
`rasterio`/`netCDF4`/`triangle` in a dedicated env and have a thin QGIS plugin
call this in a subprocess if in-process installation is troublesome.

## Notes

- The bundled `bluemesh2d` is a **copy** — update it by re-syncing the upstream
  `bluemesh2d/` package into the plugin root (`rsync -a --exclude=__pycache__
  --exclude=poly_data <repo>/bluemesh2d/ bluemesh2d_qgis/bluemesh2d/`).
- Currently exposes the core raster→mesh path. ADCIRC `.grd`, Delft3D-FM forcing
  (`.pli`/`.bc`/`.ext`) and quality reports exist in the library and can be added
  as further algorithms.
