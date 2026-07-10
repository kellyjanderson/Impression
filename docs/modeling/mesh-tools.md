# Modeling — Mesh Analysis Tools

Imported from `impression.mesh`:

```python
from impression.mesh import analyze_mesh, repair_mesh, section_mesh_with_plane
```

These tools are retained as part of Impression's mesh analysis and repair
toolchain. They are explicit downstream tools. They are not canonical modeling
truth.

## Intended Use

Retained mesh analysis is useful for:

- watertightness and manifold checks
- mesh QA and defect inspection
- plane sectioning / slicing for analysis
- loft and surface regression workflows
- foreign-mesh inspection before repair or reconstruction

## analyze_mesh(mesh)

Returns a `MeshAnalysis` report with:

- vertex and face counts
- degenerate-face count
- boundary-edge count
- non-manifold-edge count
- invalid-vertex count

```python
from impression.mesh import analyze_mesh
from impression.modeling import make_box

mesh = make_box(size=(2.0, 2.0, 2.0), backend="mesh")
report = analyze_mesh(mesh)

assert report.is_watertight
assert report.is_manifold
```

## section_mesh_with_plane(mesh, origin, normal)

Intersects a triangle mesh with a plane and returns stitched section polylines.

```python
from impression.mesh import section_mesh_with_plane
from impression.modeling import make_cylinder

mesh = make_cylinder(radius=1.0, height=2.0, backend="mesh")
section = section_mesh_with_plane(mesh, origin=(0.0, 0.0, 0.0), normal=(0.0, 0.0, 1.0))

assert section.closed_count == 1
```

The result is intended for analysis and debugging workflows such as:

- validating loft correspondence by slicing generated output
- recovering section contours near a damaged band in a foreign mesh
- comparing reconstructed sections against expected topology

Current scope:

- triangle meshes only
- plane/triangle intersection with stitched 3D polylines
- coplanar triangle spans are skipped in this first analysis slice

This tool does not mutate the mesh and does not silently promote mesh analysis
into canonical modeling behavior.

## repair_mesh(mesh)

Applies bounded explicit cleanup intended for downstream repair workflows.

Current bounded cleanup includes:

- removing faces that reference invalid or non-finite vertices
- removing degenerate triangle faces
- removing unreferenced vertices

```python
from impression.mesh import repair_mesh

repaired_mesh, report = repair_mesh(mesh)
assert report.changed
```

This is repair tooling, not canonical modeling. It is useful when salvaging or
cleaning foreign mesh inputs before deeper analysis or reconstruction.

## Standalone Mesh Utilities

Some mesh-only helpers remain valuable as explicit tools even though mesh is no
longer canonical modeling truth.

Current retained standalone utilities include:

- `union_meshes(...)`
  - explicit mesh union tool for analysis, repair, and debugging workflows
- `hull([...])` for mesh inputs
  - explicit mesh convex-hull utility

These utilities should be treated as tools, not as the preferred path for new
modeled geometry.
