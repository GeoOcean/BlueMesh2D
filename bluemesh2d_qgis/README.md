# BlueMesh2D — QGIS plugin

A QGIS **Processing** plugin that generates an unstructured triangular mesh
from a bathymetry GeoTIFF, wrapping the
[BlueMesh2D](https://github.com/GeoOcean/BlueMesh2D) library (installed from
PyPI on first run, see below). The output is a UGRID NetCDF that loads directly
as a QGIS **mesh layer**.

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

The algorithms are "provably-good": they guarantee termination, geometric and
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

The pipeline is available **split into six stages** — each result lands in an
ordinary (temporary or saved) QGIS layer you can inspect before the next step:

| algorithm | inputs | outputs |
|---|---|---|
| **1 – Extract water polygon** | bathy raster; coastline level; optional **deep level** (keeps the band `deep < z ≤ coast`, e.g. water shallower than 300 m); optional **extent polygon** (clips the raster *before* extraction — much faster; buffer then ignored); buffer factor (default −0.05) | water polygon layer |
| **2 – Build element-size raster (hfun)** *(folder)* | one algorithm **per sizing method** (pick the algorithm, see only its parameters): **2a** depth polynomial `a·d²+b·d`; **2b** wavelength `L(T,d)/N` (Hunt 1979, `hfun_wavenumhunt`); **2c** custom Python (`d`,`x`,`y`,`np`). All share: optional **Water polygon (stage 1)** to limit computation to that area + buffer (much faster than the whole raster), detail polygons, min/max/detail size, gradient and **buffer** (m, −1 = automatic) | GeoTIFF, pixel = element size (m) |
| **3 – Resample boundary to element size** | water polygon (1), hfun raster (2) | boundary **edges** line layer, styled with visible vertex markers (editable) |
| **4 – Generate mesh from boundary** | **edges layer (3) — editable: move/delete/add segments first**, hfun raster (2), bathy raster; kind = delaunay/delfront; smooth; optional **smood** (+ *merge small links*, only if triangle-only smood can't remove the last small flow links; + *merge the remaining problematic elements during recovery*, see below) | UGRID NetCDF → mesh layer |
| **5 – Generate boundary conditions** | mesh layer (4); depth threshold (default 20 m) | one **line layer** with a `btype` attribute — **open** / **closed** / **island** — styled by type with visible vertex dots, **editable** (move vertices, or change `btype` to reclassify a segment) |
| **6 – Export** *(folder)* — **6a** plain UGRID, **6b** UGRID + open BC, **6c** ADCIRC `.grd` | **6a** (default): mesh layer (4) only → `.nc`, no boundary files, no boundary layer needed. **6b**: mesh (4) **+ boundary conditions (5, required)** → `.nc` and, from the `open` features, `Boundary01.pli` / `Riemann.bc` / `FlowFM_bnd.ext`. **6c**: mesh (4) + boundary conditions (5, required) → `.grd` with open/land loops (`open`→open, `closed`+`island`→land), snapping each boundary vertex back to the nearest mesh node so stage-5 edits are honoured | `.nc` / `.nc` + Delft3D-FM BC / `.grd` |

Stage 4 rebuilds the PSLG from the (possibly edited) boundary lines, so you can
reshape the domain before meshing; likewise stage 6 rebuilds the boundary
classification from the (possibly edited) stage-5 lines, so you can hand-correct
which segments are open / closed / island before exporting. The stages also
chain in the **Graphical Modeler**.

Every step is its own numbered folder under **BlueMesh2D** (`1 - Extract water
polygon`, `2 - Build element-size raster (hfun)` with 2a/2b/2c inside, `3 -
Resample boundary...`, ..., `6 - Export` with 6a/6b/6c inside).
QGIS's Processing toolbox sorts grouped and ungrouped algorithms in two
separate buckets rather than one merged alphabetical list, so giving every
step its own group is what keeps them in numeric order 1→6 instead of the
folders (2, 6) drifting away from the single algorithms.

### When smood fails: "mesh still violates dual criteria"

smood accepts a mesh only when **both** criteria hold: `max|cosφ|` under the
orthogonality threshold (0.49) **and** zero small flow links. It prints a
progress table per cycle — `MAX|COS(PHI)|`, `N_SMALL`, `N_ZONES` and
`N_MERGED` — first the `outer=N` cycles, then `recovery=N` cycles that
re-smooth an ever-widening neighbourhood around whatever is left.

If the numbers stop moving and the run ends in an error, read the two metric
columns: they say which criterion is blocking. `MAX|COS(PHI)|` at, say,
0.4898 already passes — a stuck `N_SMALL` of 1 or 2 is then the whole problem.
Small flow links are removed by the **merge** step, and in the default
triangle-only mode that step is off, so the orthogonalizer has to clear them by
moving nodes and flipping edges alone. When the last ones sit on nodes it may
not move (fixed points, boundary vertices), no amount of extra iterations will
help — which is what a table of identical `recovery=N` rows means.

This is handled by **"smood: merge the remaining problematic elements during
recovery"**, which is **on by default**: from recovery cycle 2 onwards the
recovery cycles also run the merge step, so those few elements are merged
instead of failing the run, and `N_MERGED` shows it happening. The mesh stays
triangle-only — each merged quad is re-split on its other diagonal, and that
re-split is what removes the small link.

The advanced **"recovery step the merge starts at"** decides when it kicks in:
the default `2` lets the triangle-only pass try twice before merging (it often
succeeds on its own, as `recovery=0` and `recovery=1` in the table above), `0`
merges from the very first recovery cycle. Untick the option to get the old
behaviour: triangle-only recovery that fails rather than merging anything.

**CRS handling**: vector layers are always delivered in the input tif's CRS.
If the tif is *geographic* (e.g. EPSG:4326), a local metric UTM CRS is built
internally (and embedded in the hfun raster). If the tif is already
*projected* (e.g. UTM), the whole pipeline runs natively in that CRS — no
reprojection at all — and the output NetCDF carries metre units with a
`projected_coordinate_system` grid mapping.

## Install

1. Copy this whole `bluemesh2d_qgis/` folder into your QGIS plugins directory:
   - **Linux**: `~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/`
   - **Windows**: `%APPDATA%\QGIS\QGIS3\profiles\default\python\plugins\`
   - **macOS**: `~/Library/Application Support/QGIS/QGIS3/profiles/default/python/plugins/`
2. Restart QGIS → **Plugins ▸ Manage and Install Plugins ▸ Installed** → enable
   **BlueMesh2D**.
3. The algorithms appear in the **Processing Toolbox** under
   *BlueMesh2D ▸ 1 - Extract water polygon* (and the following stages).

### The BlueMesh2D library (important)

The plugin ships **no copy of the meshing library**: it installs the
published [`bluemesh2d`](https://pypi.org/project/bluemesh2d/) package from
PyPI, which pulls in numpy, scipy, shapely, rasterio, matplotlib, netCDF4,
xarray and triangle through its own dependency metadata. `pyproj` is
installed alongside it (the library imports it without declaring it).

**Automatic (recommended):** on first load the plugin checks QGIS's Python
and, if the library is missing or older than the version this plugin needs,
opens a dialog with an **Install now** button — one `pip install bluemesh2d
pyproj`. pip runs inside QGIS's own interpreter — the right one on every
platform — with `--user`. On Debian/Ubuntu system Python (PEP 668,
"externally managed"), packages go into a small plugin-managed virtual
environment (`.../profiles/<profile>/python/bluemesh2d_deps`, created with
`--system-site-packages`) that the plugin adds to `sys.path` on load — the
system Python is never modified and `--break-system-packages` is not needed.
If that Python lacks pip entirely (no `python3-pip`), pip is bootstrapped
into the venv automatically with the official `get-pip.py`. On conda-based
QGIS the dialog shows the manual command instead of running pip.
Restart QGIS after installing. The same dialog can be reopened anytime from
**Plugins ▸ BlueMesh2D ▸ Check / install dependencies**. The commands below
are the manual fallback if the dialog's pip fails.

- **Windows** — open the *OSGeo4W Shell* and run:
  ```
  python -m pip install bluemesh2d pyproj
  ```
- **Linux** — install into the Python QGIS runs on:
  ```
  python3 -m pip install --user bluemesh2d pyproj
  ```
  On Debian/Ubuntu system Python (PEP 668) use the plugin's venv instead of
  `--break-system-packages`:
  ```
  ~/.local/share/QGIS/QGIS3/profiles/default/python/bluemesh2d_deps/bin/python -m pip install bluemesh2d pyproj
  ```
- **macOS** — the official QGIS.app bundles its **own** Python, so a plain
  `python3 -m pip install` (Homebrew/system Python) installs into the wrong
  interpreter. The reliable way — it works on every bundle layout — is to run
  pip from inside the **QGIS Python console** (Plugins > Python Console):
  ```python
  import pip
  pip.main(["install", "--user", "bluemesh2d", "pyproj"])
  ```
  then **restart QGIS**. `--user` installs into `~/Library/Python/3.x/...`,
  which QGIS has on its path.

  Notes on the Terminal route: on **recent QGIS bundles** (vcpkg-based,
  ~3.40+) the bundled interpreter
  (`/Applications/QGIS.app/Contents/MacOS/python3.12`) **cannot run
  standalone** — it aborts with `ModuleNotFoundError: No module named
  'encodings'` because its library prefix is baked to the build machine's
  path — so use the Python-console method above. On **older bundles** the
  Terminal command was:
  ```
  /Applications/QGIS.app/Contents/MacOS/bin/python3 -m pip install bluemesh2d pyproj
  ```
  (Adjust the app name if yours is e.g. `QGIS-LTR.app`.) Do **not** use
  `subprocess` with `sys.executable` from the QGIS console on macOS: there
  `sys.executable` points to the QGIS application binary, not to Python.

- **conda** (`conda install -c conda-forge qgis`) — pip wheels would fight
  the conda stack, so take the dependencies from conda-forge and only the
  library from pip:
  ```
  conda install -c conda-forge numpy scipy shapely rasterio pyproj matplotlib netcdf4 xarray
  python -m pip install --no-deps bluemesh2d
  ```
  `triangle` has no conda-forge package; without it the plugin falls back to
  a pure-`scipy` *conforming* Delaunay (same mesh character, somewhat slower).

### Development (working from a checkout)

When the plugin folder is run straight from a clone of this repository —
typically by symlinking it into the profile's `plugins/` directory:

```
ln -s <repo>/bluemesh2d_qgis ~/.local/share/QGIS/QGIS3/profiles/default/python/plugins/bluemesh2d_qgis
```

the dependency dialog grows an extra checkbox: **"Development: install this
source checkout editable"**. Ticking it runs `pip install -e <repo>` instead
of pulling the release from PyPI, so edits to `bluemesh2d/` take effect in
QGIS after a restart, with no reinstall. The checkout is found by resolving
the plugin folder's real path; set `BLUEMESH2D_DEV_PATH=<repo>` to point at a
different one. The checkbox never appears in a plugin installed from a zip.

## Layout

```
bluemesh2d_qgis/
├── __init__.py        classFactory (QGIS entry point)
├── metadata.txt       plugin metadata (hasProcessingProvider=yes)
├── LICENSE            GPL-3.0-only (copy of the repository LICENSE;
│                      the QGIS plugin repository requires it in the zip)
├── plugin.py          registers the Processing provider
├── provider.py        BlueMesh2DProvider
├── algorithm.py       Processing algorithms
├── deps_installer.py  dependency check + guided pip install
└── pipeline.py        headless orchestration facade (no QGIS dependency)
```

`pipeline.py` is the integration seam: it imports the installed `bluemesh2d`
package and re-exports the historical names, loads pyproj before rasterio can
bring its own libproj into the process, forces the non-interactive `Agg`
matplotlib backend (the contour extraction needs it), routes
progress/cancellation through the algorithm's `QgsProcessingFeedback`, and
captures the mesher's stdout into the log.

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

## Notes

- Enums are written in the **scoped** Qt6 form
  (`QgsProcessing.SourceType.TypeVectorPolygon`, not
  `QgsProcessing.TypeVectorPolygon`) — required by the plugin repository's
  Qt6-compatibility check, and the reason `qgisMinimumVersion` is 3.34: older
  releases do not expose every enum under its scope name. Verified against
  QGIS 3.34 LTR (PyQt5); the scoped and unscoped values are identical.
- To build the zip for the QGIS plugin repository, from the repo root:
  ```
  zip -r bluemesh2d_qgis.zip bluemesh2d_qgis -x '*__pycache__*' '*.pyc'
  ```
  The upload validator requires `metadata.txt` and `LICENSE` inside the
  plugin folder — both are tracked here, so nothing has to be copied in
  first.
- The plugin and the library version independently: `metadata.txt` carries the
  plugin version, `MIN_VERSION` in `deps_installer.py` is the oldest
  `bluemesh2d` release it works against. Bump the latter when an algorithm
  starts relying on a new library feature.
- Currently exposes the core raster→mesh path. ADCIRC `.grd`, Delft3D-FM forcing
  (`.pli`/`.bc`/`.ext`) and quality reports exist in the library and can be added
  as further algorithms.
