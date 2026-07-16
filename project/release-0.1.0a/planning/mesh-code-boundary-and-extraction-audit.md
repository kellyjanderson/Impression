# Mesh Code Boundary And Extraction Audit

Date: 2026-07-11

This audit defines where mesh code is allowed in Impression and lists current
code that falls outside that boundary. It also records the sibling mesh-library
staging area created to preserve mesh-world code before cleanup.

Sibling mesh library:

- `/Users/k/Documents/Projects/impression-mesh-library`

## Boundary Decision

Impression's authored modeling kernel is `SurfaceBody` / `SurfaceShell` /
`SurfacePatch` first. Mesh is allowed only when a consumer or foreign data
source actually requires triangles.

### Mesh Allowed In Impression

- Tessellation:
  - `src/impression/modeling/tessellation.py`
  - Purpose: produce triangle output from `SurfaceBody`, `SurfaceShell`, or
    `SurfacePatch` for preview, export, reference artifacts, and downstream
    mesh-only consumers.
- Export and loading/reading boundaries:
  - `src/impression/io/stl.py`
  - `src/impression/io/impress.py` only insofar as it rejects mesh wrappers and
    loads/saves surface-native data.
  - CLI/export paths that tessellate or merge explicit mesh boundary products.
- Preview and reference review:
  - `src/impression/preview.py`
  - `src/impression/preview_qt.py`
  - `src/impression/devtools/reference_review/**`
  - Purpose: render mesh datasets after a tessellation/export/preview boundary,
    not to define authored model truth.
- Mesh container and narrow utilities needed by allowed boundaries:
  - `src/impression/mesh.py`
  - Keep only the minimal `Mesh`/`Polyline` data carriers plus analysis,
    sectioning, repair, and PyVista adapter while they are needed by preview,
    STL, reference review, or the future foreign-mesh-to-loft workflow.
- Future small mesh world:
  - importing a foreign mesh;
  - inspecting, repairing, and slicing/sectioning it;
  - extracting curves/profiles for loft or reconstruction;
  - handing reconstructed geometry back into surface-body modeling.

### Mesh Not Allowed In Impression

- Mesh CSG as canonical CSG or fallback implementation.
- Mesh primitive constructors as public authored-modeling defaults.
- `MeshGroup` as authored composition truth.
- Mesh transforms as the primary transform API for authored modeling objects.
- Mesh hull as an authored 3D modeling feature.
- Loft mesh executor output as canonical loft output.
- Text, drafting, heightmap, threading, morph, or extrusion features whose
  canonical result is mesh rather than surface body.
- Any hidden path that creates mesh because surface execution is unsupported.

## Code Outside The Boundary

- [ ] Public mesh primitive compatibility constructors
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/primitives.py`
  - Symbols: `make_box_mesh`, `make_cylinder_mesh`, `make_ngon_mesh`,
    `make_polyhedron_mesh`, `make_nhedron_mesh`, `make_sphere_mesh`,
    `make_torus_mesh`, `make_cone_mesh`, `make_prism_mesh`
  - Boundary issue: public authored-modeling namespace still exposes primitive
    mesh constructors.
  - Cleanup: move to the sibling mesh library or a clearly private fixture/test
    compatibility module. Remove public docs/examples that present these as a
    normal modeling path.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/primitives_mesh_helpers_excerpt.py`
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/_legacy_mesh_primitives.py`

- [ ] Legacy primitive mesh generator module
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/_legacy_mesh_primitives.py`
  - Symbols: `box_mesh`, `circular_frustum_mesh`, `rectangular_frustum_mesh`,
    `sphere_mesh`, `torus_mesh`, `orient_mesh`
  - Boundary issue: useful mesh-world code, but it should not remain in
    Impression's authored modeling package except as temporary compatibility.
  - Cleanup: move to sibling mesh library; replace Impression callers with
    surface primitives plus tessellation or explicit test fixtures.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/_legacy_mesh_primitives.py`

- [ ] MeshGroup compatibility container
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/group.py`
  - Symbols: `MeshGroup`, `group`, `to_meshes`, `to_mesh`
  - Boundary issue: this is a mesh-world composition type. It must not be an
    authored surface composition API.
  - Cleanup: move or retire from Impression after any remaining preview/export
    consumers are converted to surface composition or explicit tessellation
    collections.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/group.py`

- [ ] Mesh transforms in public transform helpers
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/transform.py`
  - Symbols: mesh and `MeshGroup` branches inside `translate`, `rotate`,
    `rotate_euler`, `scale`, `resize`, `mirror`, `multmatrix`
  - Boundary issue: public transform helpers still mutate mesh-world objects.
  - Cleanup: keep surface-body transforms as the authored path. Move mesh
    transform helpers to the sibling mesh library or private compatibility code.

- [ ] Mesh hull dispatch in public ops
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/ops.py`
  - Symbols: `hull` dispatch to `hull_mesh`
  - Boundary issue: public `hull` still exposes mesh hull behavior beside
    topology-native planar hull.
  - Cleanup: keep planar/topology hull in Impression. Move mesh hull to sibling
    mesh library or require an explicitly mesh-named import.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/_ops_mesh.py`

- [ ] Mesh ops module
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/_ops_mesh.py`
  - Symbols: `hull_mesh`, `manifold_from_mesh_group`, `mesh_from_manifold`
  - Boundary issue: explicit mesh utility code belongs in the small mesh world,
    not the authored surface modeling package.
  - Cleanup: move to sibling mesh library. Keep no Impression runtime reference
    unless there is a temporary explicit compatibility path.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/_ops_mesh.py`

- [ ] Mesh utility facade
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/mesh_tools/__init__.py`
  - Symbols: `analyze_mesh`, `repair_mesh`, `section_mesh_with_plane`,
    `hull_mesh`, `manifold_from_mesh_group`, `mesh_from_manifold`
  - Boundary issue: analysis/repair/sectioning is valid mesh-world behavior, but
    the facade lives under `impression.modeling`, which suggests authored
    modeling support.
  - Cleanup: move mesh tools out of `impression.modeling` or keep only a
    temporary quarantine import path with strong deprecation language.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/mesh_tools.py`

- [ ] Mesh boolean implementation inside CSG module
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/csg.py`
  - Symbols: `_manifold_from_mesh`, `_mesh_from_manifold`, `_flatten_meshes`,
    `_check_mesh`, `_apply_boolean`, `union_meshes`
  - Boundary issue: mesh boolean code lives inside the canonical CSG module. It
    is valuable as a mesh utility, but not as Impression's authored CSG path.
  - Cleanup: move mesh boolean implementation to sibling mesh library. Keep
    Impression CSG surface-body-only and make unsupported cases explicit.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/csg_mesh_boolean_excerpt.py`
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/csg_public_boundary_excerpt.py`

- [ ] Boolean public signatures still accept mesh objects
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/csg.py`
  - Symbols: `boolean_union`, `boolean_difference`, `boolean_intersection`
  - Boundary issue: type signatures still include `Mesh | MeshGroup` and local
    variable names still use `meshes`.
  - Cleanup: restrict public boolean APIs to `SurfaceBody` operands. If mesh
    boolean is retained, expose it only through the sibling mesh library.

- [ ] Text mesh compatibility helpers
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/text.py`
  - Symbols: `TextMeshCompatibilityResult`, `_make_text_mesh_impl`,
    `make_text_mesh`, `make_text_mesh_result`, `_mesh_text_extrude`,
    `_extrude_region_loops`
  - Boundary issue: text should be surface-body authored output. Mesh text
    extrusion is useful compatibility code but belongs outside Impression's
    canonical modeling path.
  - Cleanup: move mesh text extrusion to sibling mesh library or tests. Keep
    `make_text` as surface body only.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/text_mesh_compatibility_excerpt.py`

- [ ] Heightmap mesh compatibility helpers
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/heightmap.py`
  - Symbols: `_triangle_surface_body_from_mesh`, `_heightmap_mesh_impl`,
    `_vertex_normals`, `_displace_direction`, `_mask_faces`,
    `_displace_heightmap_mesh_impl`
  - Boundary issue: heightmap/displacement authored output is now surface-body.
    Mesh displacement helpers are useful mesh-world salvage code but should not
    live as latent authored behavior in Impression.
  - Cleanup: move mesh heightmap/displacement helpers to sibling mesh library.
    Keep native heightmap and displacement surface patch paths in Impression.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/heightmap_mesh_compatibility_excerpt.py`

- [ ] Loft debug mesh executor and mesh endcaps
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/loft.py`
  - Symbols: `validate_mesh_executor_correspondence_input`,
    `emit_mesh_faces_from_sample_correspondence`, `loft_execute_plan_debug_mesh`,
    `loft_execute_plan_debug_mesh_result`, `loft_endcaps`
  - Boundary issue: debug mesh emission may be useful for diagnostics, but it
    must not sit beside canonical loft execution as a route agents can select
    for implementation. `loft_endcaps` still returns `Mesh`.
  - Cleanup: move debug mesh emission and mesh endcap experiment code to the
    sibling mesh library. Keep `loft_execute_plan -> SurfaceBody` only in
    Impression.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/modeling/loft_mesh_debug_excerpt.py`

- [ ] CAD mesh shim
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/cad/__init__.py`
  - Symbols: `shape_to_polydata`
  - Boundary issue: disabled mesh-producing CAD shim is not currently useful in
    Impression.
  - Cleanup: move to sibling mesh library or delete from Impression if no
    active code imports it.
  - Preserved in mesh library:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_from_impression/cad.py`

- [ ] Public modeling exports of mesh-world symbols
  - Path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/__init__.py`
  - Symbols: `make_*_mesh`, `union_meshes`, `MeshGroup`,
    `loft_execute_plan_debug_mesh`, `make_text_mesh`, `mesh_from_surface_body`
  - Boundary issue: exporting these symbols from the top-level modeling package
    teaches callers and agents that mesh-world APIs are normal modeling APIs.
  - Cleanup: remove top-level public exports or move them behind explicitly
    transitional/private compatibility imports until deletion.

## Deleted Mesh-Era Code Recovered Into Sibling Library

- [x] Fastener threading mesh/surface hybrid code
  - Original path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/threading.py`
  - Deleted by commit: `d4ff4861f0edd5e335d78fd29a5c8bde32c244b2`
  - Recovered from parent into:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_history/2026-05-26-before-threading-removal/src/impression/modeling/threading.py`
  - Notes: also recovered deleted threading docs and examples.

- [x] Legacy extrusion code
  - Original path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/extrude.py`
  - Deleted by commit: `4b3acd8d8dfa207bba56e719936d01621ad63fa3`
  - Recovered from parent into:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_history/2026-04-22-before-surface-first-removal/src/impression/modeling/extrude.py`

- [x] Legacy morph code
  - Original path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/morph.py`
  - Deleted by commit: `4b3acd8d8dfa207bba56e719936d01621ad63fa3`
  - Recovered from parent into:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_history/2026-04-22-before-surface-first-removal/src/impression/modeling/morph.py`

- [x] Legacy 2D profile code
  - Original path: `/Users/k/Documents/Projects/Impression/src/impression/modeling/_profile2d.py`
  - Deleted by commit: `979f4cc54fdf35c4ba9a09d25417dd446e5077ff`
  - Recovered from parent into:
    - `/Users/k/Documents/Projects/impression-mesh-library/src/impression_mesh_library/_history/2026-03-14-before-profile2d-removal/src/impression/modeling/_profile2d.py`

## Cleanup Sequencing

1. Keep `src/impression/mesh.py` temporarily as the boundary mesh carrier until
   preview, STL, reference review, and foreign-mesh import workflows have a
   narrower home.
2. Remove mesh-world exports from `impression.modeling.__init__`.
3. Move mesh CSG, mesh primitive constructors, mesh hull, mesh group, mesh text,
   mesh heightmap, and debug loft mesh code out of Impression.
4. Rewrite docs and examples so `SurfaceBody` is the only authored modeling
   path, and mesh appears only at named consumer/tool boundaries.
5. Add static tests that fail if authored modeling modules import mesh-world
   extraction code or introduce hidden mesh fallback.
