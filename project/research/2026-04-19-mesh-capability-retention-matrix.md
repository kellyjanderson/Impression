# Mesh Capability Retention Matrix

This matrix records the intended end state for mesh-era capability in
Impression.

It exists to support the retained mesh analysis/repair architecture and to make
deletion candidates explicit instead of hiding them behind vague "legacy"
labels.

## Inventory Fields

Each row should capture:

- `area`
- `file`
- `symbols`
- `role`
- `retain_or_delete`
- `target_state`
- `notes`

## Matrix

| area | file | symbols | role | retain_or_delete | target_state | notes |
| --- | --- | --- | --- | --- | --- | --- |
| mesh-core | `src/impression/mesh.py` | `Mesh`, `MeshAnalysis`, `analyze_mesh`, `section_mesh_with_plane`, `repair_mesh`, `mesh_to_pyvista` | analysis / repair / boundary conversion | retain | explicit mesh toolchain module | mesh remains downstream of surfaced modeling |
| mesh-quality | `src/impression/mesh_quality.py` | `MeshQuality`, `apply_lod`, `downshift_quality` | mesh budget / tessellation support | retain | shared downstream quality tool | used by legacy mesh generators and mesh-facing tooling |
| preview-boundary | `src/impression/preview.py` | previewer mesh conversion and rendering paths | rendering / inspection boundary | retain | explicit rendering pipeline | mesh remains valid in preview/export pipelines |
| export-boundary | `src/impression/io/stl.py` | STL export helpers | export boundary | retain | explicit export tool | export remains mesh payload |
| mesh-ops | `src/impression/modeling/_ops_mesh.py` | mesh hull/offset-style helpers | standalone mesh utility candidates | retain for now | review under standalone mesh utility branch | should not be treated as canonical surfaced modeling |
| mesh-consumer-bridge | `src/impression/modeling/tessellation.py` | `mesh_from_surface_body`, `SurfaceMeshAdapter` | consumer compatibility bridge | retain for now | boundary bridge pending later deletion review | bridge remains explicit and lossy |
| mesh-group-transform | `src/impression/modeling/group.py`, `src/impression/modeling/transform.py`, `src/impression/modeling/ops.py` | mesh-centric composition and transforms | legacy support / possible tool extraction | review | split retained tool behavior from deletable modeling behavior | current state is mixed |
| mesh-primitives | `src/impression/modeling/primitives.py` mesh backend paths | `make_box`, `make_cylinder`, etc. with `backend=\"mesh\"` | deprecated modeling capability | delete | remove after surfaced migration fully closes | canonical truth must be SurfaceBody |
| mesh-extrude | `src/impression/modeling/extrude.py` mesh backend paths | `linear_extrude`, `rotate_extrude` mesh lane | deprecated modeling capability | delete | remove after surfaced migration fully closes | surface-first replacement already exists |
| mesh-loft | `src/impression/modeling/loft.py` mesh-returning legacy lanes | `loft`, `loft_sections`, `loft_profiles`, mesh execution helpers | deprecated modeling capability | delete | remove after surfaced loft migration fully closes | loft remains valuable as reconstruction logic, not mesh truth |
| mesh-text | `src/impression/modeling/text.py` mesh lanes | `make_text`, `text` mesh-primary behavior | deprecated modeling capability | delete | remove once surfaced text is fully promoted | keep only surfaced path |
| mesh-drafting | `src/impression/modeling/drafting.py` mesh lanes | drafting mesh-primary behavior | deprecated modeling capability | delete | remove once surfaced drafting is fully promoted | retain only analysis/preview mesh downstream |
| mesh-threading | `src/impression/modeling/threading.py` mesh-returning geometry | thread meshes and mesh convenience builders | deprecated geometry capability | delete later | replace with surfaced thread outputs and explicit downstream tessellation | standalone mesh analysis of thread output may remain |
| mesh-hinges | `src/impression/modeling/hinges.py` mesh-returning geometry | hinge mesh builders | deprecated geometry capability | delete later | replace with surfaced hinge outputs and explicit downstream tessellation | mesh remains valid only at boundary/tooling |
| mesh-heightfield | `src/impression/modeling/heightmap.py` mesh-primary geometry | `heightmap`, `displace_heightmap` mesh lane | deprecated geometry capability | delete later | replace with surfaced heightfield/displacement outputs | mesh remains downstream for analysis/export |
| mesh-csg-modeling | `src/impression/modeling/csg.py` mesh execution lane | `boolean_union`, `boolean_difference`, `boolean_intersection` mesh execution | temporary executable fallback | review | eventually separate retained mesh utility posture from surfaced modeling API | current public posture is transitional |

## Notes

- `retain` here means retain as explicit toolchain or boundary behavior, not as
  canonical modeling truth.
- `delete` here means remove the mesh-primary modeling role after the surfaced
  replacement is complete and migration evidence is sufficient.
- `review` means the capability likely survives in some form, but its exact
  retained shape is still being narrowed by the open mesh-toolchain specs.
