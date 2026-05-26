# Surface Spec 159 Implementation: Mesh Execution Inventory And Classification Report (v1.0)

## Purpose

This report is the durable implementation artifact for:

- [Surface Spec 159: Mesh Execution Inventory And Classification](surface-159-mesh-execution-inventory-and-classification-v1_0.md)
- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

It records the mesh-producing and mesh-consuming paths found by static source
search before the migration specs remove hidden mesh execution from authored
modeling paths.

## Reproducible Source Search

Run these commands from the repository root to reproduce the inventory basis:

```bash
rg -n "from impression\\.mesh import|import impression\\.mesh|\\bMesh\\b|combine_meshes|tessellate_surface|to_mesh|mesh_from|backend=\\\"mesh\\\"|backend:.*mesh|return Mesh|-> Mesh" src/impression/modeling src/impression/io src/impression/cli.py src/impression/preview.py src/impression/mesh.py
rg -n "def make_|def .*mesh|_mesh\\(|Mesh\\(" src/impression/modeling/primitives.py src/impression/modeling/loft.py src/impression/modeling/csg.py src/impression/modeling/text.py src/impression/modeling/drafting.py src/impression/modeling/heightmap.py src/impression/modeling/group.py src/impression/modeling/transform.py src/impression/modeling/ops.py src/impression/modeling/_ops_mesh.py src/impression/modeling/tessellation.py
rg -n "backend=\\\"mesh\\\"|Literal\\[.*mesh|Mesh \\| SurfaceBody|SurfaceBody \\| Mesh|-> Mesh|return .*Mesh|MeshGroup|combine_meshes" src/impression/modeling tests docs project/release-0.1.0a
```

## Classification Vocabulary

- `tessellation-boundary`: allowed mesh production after `SurfaceBody` or `SurfacePatch` input.
- `mesh-consumer`: allowed downstream preview, export, rendering, or analysis consumption.
- `legacy-compatibility`: allowed only when API names or arguments explicitly say mesh.
- `explicit-mesh-tool`: allowed standalone mesh tooling, analysis, repair, or utility behavior.
- `invalid-modeling-fallback`: authored modeling path still produces mesh by default or as hidden truth.

## Inventory

| Area | Symbols / Paths | Classification | Owner Specs | Required Migration Target |
| --- | --- | --- | --- | --- |
| Surface tessellation | `tessellate_surface_patch`, `tessellate_surface_shell`, `tessellate_surface_body`, `_patch_mesh`, `_rectangular_grid_patch_mesh`, `_subdivision_patch_mesh`, `_implicit_patch_mesh`, `SurfaceMeshAdapter`, `mesh_from_surface_body` in `src/impression/modeling/tessellation.py` | `tessellation-boundary` | Surface Specs 151, 166, 188 | Keep mesh creation here. `mesh_from_surface_body` remains explicit compatibility only. |
| Preview and CLI export | `PyVistaPreviewer.collect_datasets`, `combine_to_mesh`, `_log_mesh_analysis`, CLI export merge in `src/impression/preview.py` and `src/impression/cli.py` | `mesh-consumer` | Surface Specs 184, 188 | Ensure callers tessellate or pass explicit mesh payloads; do not treat preview as authored modeling. |
| STL and mesh IO | `write_stl` in `src/impression/io/stl.py` | `mesh-consumer` | Surface Specs 166, 188 | Keep as mesh export boundary. Surface export must tessellate before STL. |
| Mesh core tools | `Mesh`, `combine_meshes`, `analyze_mesh`, `section_mesh_with_plane`, `repair_mesh`, `mesh_to_pyvista` in `src/impression/mesh.py` | `explicit-mesh-tool` | Surface Specs 122-125, 187 | Retain as mesh tools, repair, analysis, and consumer helpers. |
| Primitive public constructors | `make_box`, `make_cylinder`, `make_ngon`, `make_polyhedron`, `make_nhedron`, `make_sphere`, `make_torus`, `make_cone`, `make_prism` in `src/impression/modeling/primitives.py` | `invalid-modeling-fallback` while defaulting to `backend="mesh"` | Surface Specs 160-163 | Make authored defaults return `SurfaceBody`; keep mesh only through explicit mesh compatibility names or tessellation. |
| Primitive private mesh helpers | `_orient_mesh`, `_box_mesh`, `_sphere_mesh`, `_torus_mesh`, `_circular_frustum_mesh`, `_rectangular_frustum_mesh` in `primitives.py` | `legacy-compatibility` or `tessellation-boundary` after relocation | Surface Specs 164-166 | Move behind tessellation or explicit legacy mesh helper module; no surfaced modeling imports. |
| Loft mesh executor | `validate_mesh_executor_correspondence_input`, `emit_mesh_faces_from_sample_correspondence`, `loft_execute_plan`, other `-> Mesh` plan/executor paths in `src/impression/modeling/loft.py` | `invalid-modeling-fallback` unless exposed as debug tessellation | Surface Specs 167-168 | Surface executor consumes topology correspondence; mesh emission moves to tessellation/debug boundary or retires. |
| Boolean CSG | `_mesh_from_manifold`, `_manifold_from_mesh`, `_flatten_meshes`, `_check_mesh`, `_boolean_meshes`, `boolean_union`, `boolean_difference`, `boolean_intersection`, `union_meshes` in `src/impression/modeling/csg.py` | mixed `invalid-modeling-fallback`, `legacy-compatibility`, and `explicit-mesh-tool` | Surface Specs 153, 187, 188 | Surface booleans are canonical; manifold/mesh stays explicit mesh utility or compatibility only. |
| Text modeling | `make_text`, `text_to_section`, `_mesh_text_extrude`, `_mesh_extrude_region` in `src/impression/modeling/text.py` | `invalid-modeling-fallback` while defaulting to `backend="mesh"` | Surface Specs 169-170 | Public text defaults to surface; mesh text remains explicit compatibility and handles empty text intentionally. |
| Drafting modeling | `make_line`, `make_plane`, `make_arrow`, `make_dimension` in `src/impression/modeling/drafting.py` | `invalid-modeling-fallback` while defaulting to `backend="mesh"` | Surface Spec 171 | Public drafting defaults to surface; mesh drafting remains explicit compatibility only. |
| Heightmap and displacement | `_heightmap_mesh_impl`, `heightmap`, `_triangle_surface_body_from_mesh`, `_displace_heightmap_mesh_impl`, `displace_heightmap`, `_mask_faces`, `_vertex_normals`, `_displace_direction` in `src/impression/modeling/heightmap.py` | mixed `invalid-modeling-fallback` and `legacy-compatibility` | Surface Specs 172-176 | Surface heightmap payload owns sampled/displacement truth; mesh generation moves to tessellation or explicit compatibility. |
| Grouping | `MeshGroup`, `group`, `to_meshes`, `to_mesh` in `src/impression/modeling/group.py` | `legacy-compatibility` / `mesh-consumer` | Surface Specs 183-186 | Add surfaced composition type; quarantine `MeshGroup` as explicit compatibility/consumer helper. |
| Transforms | `translate`, `rotate`, `scale`, `resize`, `mirror`, `multmatrix`, `_bounds` in `src/impression/modeling/transform.py` | `legacy-compatibility` while accepting only `Mesh | MeshGroup` | Surface Specs 182-184 | Public authored transform defaults operate on surface/composition objects; mesh transforms stay explicit compatibility. |
| Hull and mesh ops | `hull`, `hull_mesh`, `manifold_from_mesh_group`, `mesh_from_manifold` in `src/impression/modeling/ops.py` and `_ops_mesh.py` | `explicit-mesh-tool` when inputs are mesh; invalid if used as authored surface fallback | Surface Specs 187-188 | Keep mesh hull as explicit utility; surface/topology hull must not route through mesh silently. |
| Color helpers | `set_mesh_color`, `get_mesh_color`, `get_mesh_rgba`, `transfer_mesh_color`, `set_cell_colors` in `src/impression/modeling/_color.py` | `mesh-consumer` / `legacy-compatibility` | Surface Specs 184-188 | Keep for mesh consumers; surface color metadata must use surface metadata paths. |
| Threading and hinges shims | `src/impression/modeling/threading.py`, `src/impression/modeling/hinges.py` now load sibling projects | `legacy-compatibility` until sibling implementations are updated | Surface Specs 177-181 | Sibling libraries own updates to surface-first defaults and explicit mesh compatibility. |

## Public API Risk List

These are the highest-risk public or commonly imported paths because callers can
still receive `Mesh` from an authored modeling request:

- `impression.modeling.make_box`
- `impression.modeling.make_cylinder`
- `impression.modeling.make_ngon`
- `impression.modeling.make_polyhedron`
- `impression.modeling.make_nhedron`
- `impression.modeling.make_sphere`
- `impression.modeling.make_torus`
- `impression.modeling.make_cone`
- `impression.modeling.make_prism`
- `impression.modeling.text.make_text`
- `impression.modeling.text.text_to_section`
- `impression.modeling.drafting.make_line`
- `impression.modeling.drafting.make_plane`
- `impression.modeling.drafting.make_arrow`
- `impression.modeling.drafting.make_dimension`
- `impression.modeling.heightmap.heightmap`
- `impression.modeling.heightmap.displace_heightmap`
- `impression.modeling.csg.boolean_union`
- `impression.modeling.csg.boolean_difference`
- `impression.modeling.csg.boolean_intersection`
- `impression.modeling.loft.loft_execute_plan`
- `impression.modeling.loft.emit_mesh_faces_from_sample_correspondence`
- `impression.modeling.group.MeshGroup`
- `impression.modeling.transform.translate`
- `impression.modeling.transform.rotate`
- `impression.modeling.transform.scale`
- `impression.modeling.transform.resize`
- `impression.modeling.transform.mirror`
- `impression.modeling.transform.multmatrix`

## Downstream Allowed Mesh Boundaries

These paths are allowed to produce or consume mesh after the tessellation
boundary, provided callers do not treat them as authored modeling truth:

- `src/impression/modeling/tessellation.py`
- `src/impression/mesh.py`
- `src/impression/io/stl.py`
- `src/impression/preview.py`
- mesh-only docs and tests under `docs/modeling/mesh-tools.md` and
  `tests/test_mesh_tools.py`

## Verification Notes

- Static source search was run with the commands listed above.
- The report classifies symbols rather than only modules, so downstream specs
  have concrete migration targets.
- No production code was changed by this inventory leaf.
