<p align="center">
  <img src="https://raw.githubusercontent.com/GeoOcean/BlueMesh2D/develop/assets/mesh_geocean.webp" alt="BlueMesh2D logo" width="120">
</p>

> **BlueMesh2D** is the successor of **PyMesh2D**. The repository has been recreated under its new name as part of the BlueMath ecosystem. While the repository history has changed, the codebase and its development continue from the original project.

## `BlueMesh2D: Delaunay-based mesh generation in Python`

`BlueMesh2D` is a `Python`-based unstructured mesh-generator for two-dimensional polygonal geometries, providing a range of relatively simple, yet effective two-dimensional meshing algorithms. `BlueMesh2D` includes variations on the "classical" Delaunay refinement technique, a new "Frontal"-Delaunay refinement scheme, a non-linear mesh optimisation method, and auxiliary mesh and geometry pre- and post-processing facilities. 

`BlueMesh2D` is a `Python` translation of <a href="https://github.com/dengwirda/mesh2d">`MESH2D`</a>, a `MATLAB` / `OCTAVE`-based tool developed by Darren Engwirda, extended here with a bathymetry-to-mesh workflow and a QGIS plugin. The meshing algorithms and their design are his work; see the [references](#references) below.

Algorithms implemented in `BlueMesh2D` are "provably-good" - ensuring convergence, geometrical and topological correctness, and providing guarantees on algorithm termination and worst-case element quality bounds. Support for user-defined "mesh-spacing" functions and "multi-part" geometry definitions is also provided, allowing `BlueMesh2D` to handle a wide range of complex domain types and user-defined constraints. `BlueMesh2D` typically generates very high-quality output, appropriate for a variety of finite-volume/element type applications.

`MESH2D`, and therefore `BlueMesh2D`, is a simplified two-dimensional counterpart of Engwirda's <a href="https://github.com/dengwirda/jigsaw-matlab/">`JIGSAW`</a> mesh generator (a `C++` code). `BlueMesh2D` aims to provide a straightforward `Python` implementation of these Delaunay-based triangulation and mesh optimisation techniques.

### `Code Structure`

`BlueMesh2D` is a pure `Python` package, consisting of a core library + associated utilities:

    BlueMesh2D:
    ├── bluemesh2d            -- core BlueMesh2D library functions. See refine, smooth, tridemo etc.
    │   ├── aabb_tree         -- support for fast spatial indexing, via tree-based data-structures.
    │   ├── geom_util         -- geometry processing, repair, reprojection, etc.
    │   ├── geomesh_util      -- mesh management, export, interpolation, etc.
    │   ├── hfun_util         -- mesh-spacing definitions, limiters, etc.
    │   ├── hjac_util         -- solver for Hamilton-Jacobi eqn's.
    │   ├── mesh_ball         -- circumscribing balls, orthogonal balls etc.
    │   ├── mesh_cost         -- mesh cost/quality functions, etc.
    │   ├── mesh_file         -- mesh i/o via ASCII serialisation, UGRID/ADCIRC, boundary conditions.
    │   ├── mesh_util         -- meshing/triangulation utility functions.
    │   ├── ortho_merge       -- mesh orthogonalisation & merging (Delft3D-FM).
    │   ├── poly_data         -- polygon definitions for demo problems, etc.
    │   └── poly_test         -- fast inclusion test for polygons.
    ├── bluemesh2d_qgis       -- QGIS Processing plugin wrapping the raster-to-mesh workflow.
    │   ├── __init__.py       -- classFactory -- the entry point QGIS calls to load the plugin.
    │   ├── metadata.txt      -- plugin metadata: version, external_deps, changelog.
    │   ├── plugin.py         -- bootstrap: dependency gate, menu, provider registration.
    │   ├── provider.py       -- BlueMesh2DProvider -- registers the algorithms with Processing.
    │   ├── algorithm.py      -- the Processing algorithms: stages 1-6 plus the all-in-one run.
    │   ├── deps_installer.py -- dependency check + guided pip install dialog.
    │   ├── pipeline.py       -- seam to the library: headless facade, matplotlib/pyproj set-up.
    │   ├── icon.png          -- toolbox icon.
    │   ├── README.md         -- install and usage documentation for the plugin.
    │   └── LICENSE           -- GPL-3.0, required for the QGIS plugin repository.
    ├── tests                 -- pytest suite; tests/reference holds the golden meshes.
    ├── docs                  -- tutorial material (Santander workshop).
    └── assets                -- images used by the READMEs.

On top of the meshing primitives, the top level of `bluemesh2d` also carries the
staged bathymetry-to-mesh workflow the QGIS plugin drives: `pipeline` (the
raster → water polygon → hfun → boundary → mesh → NetCDF orchestration, with
`MeshConfig` / `generate_mesh`), `meshgen` (refine / smooth / smood for a given
PSLG), `smood` (Delft3D-FM orthogonalisation), plus `dependencies` and
`feedback` (optional-import checks and progress/cancellation plumbing, so the
same code runs headless or inside QGIS).

### `Quickstart`

#### Installation

You can install **`bluemesh2d`** directly from PyPI:

```bash
pip install bluemesh2d
```

If you want to install it in *developer mode* (for local development and contribution):

```bash
pip install -e .
```

#### Examples

    python -m bluemesh2d.tridemo  0  # a very simple example to get everything started.
    python -m bluemesh2d.tridemo  1  # investigate the impact of the "radius-edge" threshold.
    python -m bluemesh2d.tridemo  2  # Frontal-Delaunay vs. Delaunay-refinement refinement.
    python -m bluemesh2d.tridemo  3  # explore impact of user-defined mesh-size constraints.
    python -m bluemesh2d.tridemo  4  # explore impact of "hill-climbing" mesh optimisations.
    python -m bluemesh2d.tridemo  5  # assemble triangulations for "multi-part" geometries.
    python -m bluemesh2d.tridemo  6  # assemble triangulations with "internal" constraints.
    python -m bluemesh2d.tridemo  7  # investigate the use of "quadtree"-type refinement.
    python -m bluemesh2d.tridemo  8  # explore use of custom, user-defined mesh-size functions.
    python -m bluemesh2d.tridemo  9  # larger-scale problem, mesh refinement + optimisation.
    python -m bluemesh2d.tridemo 10  # medium-scale problem, mesh refinement + optimisation.

More examples available in [`BlueMath`](https://github.com/GeoOcean/BlueMath/tree/main/toolkit/mesh).

### `QGIS Plugin`

A **QGIS Processing plugin** exposes the raster-to-mesh part of `BlueMesh2D` as a
set of Processing algorithms, so a bathymetry-to-mesh workflow can be run
without writing any Python. It is published on the QGIS plugin repository —
**<https://plugins.qgis.org/plugins/BlueMesh2D>** — so it installs straight from
*Plugins ▸ Manage and Install Plugins* by searching for **BlueMesh2D**. On first
run it installs `bluemesh2d` from PyPI into QGIS's Python (Plugins ▸ BlueMesh2D ▸
*Check / install dependencies*), and covers:

1. **Extract water polygon** from a bathymetry raster (coastline / domain
   extraction).
2. **Build element-size raster (hfun)** — depth-polynomial, wavelength
   (Hunt 1979), or a custom Python sizing function, gradient-limited.
3. **Resample boundary** to the element size, producing an editable boundary
   line layer.
4. **Generate mesh from boundary** — Delaunay / Frontal-Delaunay refinement,
   optional smoothing and `smood` orthogonalization (Delft3D-FM).
5. **Generate boundary conditions** — classify the mesh boundary into
   open / closed / island lines, editable directly in QGIS.
6. **Export** — UGRID NetCDF (with or without Delft3D-FM open-boundary
   `.pli`/`.bc`/`.ext` files) or ADCIRC `.grd`.

Each stage's output is an ordinary QGIS layer (polygon, raster, or line), so
intermediate results can be inspected and edited before the next step; an
all-in-one algorithm is also available for a single-dialog run. See the
plugin's own README (`bluemesh2d_qgis/README.md`) for installation and usage
details.

### `Contact`

For questions, bug reports, or feature requests, please open an issue on GitHub.

For direct contact:

**Etienne**  
faugeree@unican.es

### `License`

`BlueMesh2D` is released under the **GPL-3.0-only** license; see [LICENSE](LICENSE). The QGIS plugin carries the same license (`bluemesh2d_qgis/LICENSE`).

### `References`

The algorithms in `BlueMesh2D` come from `MESH2D`, so if you make use of `BlueMesh2D` in published work, please cite Darren Engwirda's papers below. For a much more advanced and fully three-dimensional mesh-generation library, see the <a href="https://github.com/dengwirda/jigsaw-matlab/">`JIGSAW`</a> package. Like `MESH2D`, `BlueMesh2D` computes efficient spatial queries and intersection tests with translations of the <a href="https://github.com/dengwirda/aabb-tree">`AABBTREE`</a> and <a href="https://github.com/dengwirda/find-tria">`FINDTRIA`</a> packages (`bluemesh2d/aabb_tree`, `bluemesh2d/poly_test`).

`[1]` - Darren Engwirda, <a href="http://hdl.handle.net/2123/13148">Locally-optimal Delaunay-refinement and optimisation-based mesh generation</a>, Ph.D. Thesis, School of Mathematics and Statistics, The University of Sydney, September 2014.

`[2]` - Darren Engwirda, Unstructured mesh methods for the Navier-Stokes equations, Honours Thesis, School of Aerospace, Mechanical and Mechatronic Engineering, The University of Sydney, November 2005.
