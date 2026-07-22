# BlueMesh2D tests

Regression and validity tests for the core meshing pipeline, built from a
head-less subset of [`bluemesh2d/tridemo.py`](../bluemesh2d/tridemo.py).

## Running

```bash
pip install pytest
pytest                 # everything
pytest -m "not slow"   # skip the >100k-triangle islands case (~1 s total)
```

`tests/conftest.py` forces the non-interactive matplotlib `Agg` backend, so
no windows open and nothing blocks on `plt.show`.

## What is covered

- **Regression** (`test_case_matches_reference`) — the small cases in
  [`_cases.py`](_cases.py) (`demo0_basic`, `demo0_hfun`, `demo6_internal`,
  `smood_square`) are compared array-for-array against the golden `.npz`
  files in [`reference/`](reference). Triangle connectivity must match
  exactly; coordinates are compared with a small tolerance so a different
  BLAS/scipy build does not cause spurious failures.
- **Validity** — every produced mesh must have finite nodes, in-range
  connectivity and non-degenerate (non-zero, non-inverted) triangle areas.
  The large `islands.msh` geometry is checked this way under the `slow` mark.
- **smood** (flow orthogonalisation) — the `fixed` vertices stay put, the
  node numbering is preserved, interior nodes actually move, and the
  worst-case dual-mesh non-orthogonality (`max|cos φ|`) does not get worse.

## Updating the golden references

Regenerate only when the meshing output changes **on purpose**, and review
the diff (e.g. triangle counts) before committing:

```bash
python -m tests.regenerate_references
```
