# 2026-04-18 Legacy Mesh Audit

## Purpose

Identify every public area where mesh remains the primary modeling product, mark the first deprecation wave in code, and define a removal order for the remaining legacy stack.

## Deprecated In This Pass

### Mesh-primary backend entrypoints

- `impression.modeling.primitives`
  - `make_box`
  - `make_cylinder`
  - `make_ngon`
  - `make_polyhedron`
  - `make_nhedron`
  - `make_sphere`
  - `make_torus`
  - `make_cone`
  - `make_prism`
- `impression.modeling.extrude`
  - `linear_extrude`
  - `rotate_extrude`

These now emit `DeprecationWarning` when the legacy `backend="mesh"` path is used.

### Mesh-primary public loft entrypoints

- `loft`
- `loft_profiles`
- `loft_sections`
- `loft_execute_plan`
- `loft_endcaps`

These still return `Mesh` today, but now explicitly warn that the surfaced loft path is the intended replacement.

### Mesh-only public modeling APIs

- `text.make_text`
- `text.text`
- `csg.boolean_union`
- `csg.boolean_difference`
- `csg.boolean_intersection`
- `csg.union_meshes`
- `drafting.make_line`
- `drafting.make_plane`
- `drafting.make_arrow`
- `drafting.make_dimension`
- `threading.make_external_thread`
- `threading.make_internal_thread`
- `threading.make_threaded_rod`
- `threading.make_tapped_hole_cutter`
- `threading.make_hex_nut`
- `threading.make_round_nut`
- `threading.make_runout_relief`
- `hinges.make_traditional_hinge_leaf`
- `hinges.make_traditional_hinge_pair`
- `hinges.make_living_hinge`
- `hinges.make_bistable_hinge`
- `heightmap.heightmap`
- `heightmap.displace_heightmap`

These now emit `DeprecationWarning` on use.

### Legacy mesh bridge

- `SurfaceMeshAdapter`
- `mesh_from_surface_body`

These are still allowed as migration bridges, but now warn so callers stop treating them as primary modeling outputs.

## Canonical Inventory

This section is the durable deletion inventory for mesh-era code that is still
expected to disappear as SurfaceBody becomes canonical.

### Deprecated Public Mesh Capability

These are public or user-facing capabilities that still expose legacy mesh
truth and are on the deprecation/removal path.

- `src/impression/modeling/primitives.py`
  - mesh primitive construction internals at lines `7`, `154`, `438`, `469`,
    `550`, `605`
  - legacy compatibility path still needed for `backend="mesh"`
  - intended end state: surface-first defaults with mesh only as explicit
    compatibility or consumer output

- `src/impression/modeling/extrude.py`
  - legacy mesh backend warnings and mesh triangulation path at lines `11`,
    `86`, `109`, `206`, `233`
  - intended end state: surfaced extrude canonical path, mesh only at explicit
    consumer boundary

- `src/impression/modeling/csg.py`
  - legacy mesh boolean execution and deprecation posture at lines `14`, `263`,
    `355`, `374`, `397`, `409`
  - intended end state: true SurfaceBody CSG execution replacing mesh boolean
    truth

- `src/impression/modeling/drafting.py`
  - mesh drafting and deprecation path at lines `8`, `11`, `80`, `100`, `145`,
    `159`, `211`, `235`, `326`
  - intended end state: surfaced drafting only

- `src/impression/modeling/text.py`
  - legacy mesh text warnings at lines `23`, `88`, `132`
  - intended end state: surfaced text bodies only

- `src/impression/modeling/heightmap.py`
  - legacy public mesh entrypoints at lines `13`, `229`, `379`
  - transitional mesh bridge into surfaced triangles at line `80`
  - intended end state: surface-native heightfield/displacement without mesh as
    authored truth

- `src/impression/modeling/loft.py`
  - legacy mesh loft execution still exists in the legacy public loft family
  - notable mesh triangulation/endcap execution points at lines `1539`, `1554`,
    `1574`, `1595`, `1608`, `1612`, `2934`, `4565`
  - intended end state: surfaced loft canonical path only

- `src/impression/modeling/threading.py`
  - threading convenience helpers still depend on public legacy booleans at
    lines `16`, `671`, `736`, `801`
  - intended end state: surfaced thread assemblies and surfaced CSG

- `src/impression/modeling/hinges.py`
  - hinge composition still depends on public legacy booleans at lines `13`,
    `75`, `86`
  - intended end state: surfaced hinge assembly and surfaced CSG

- `src/impression/modeling/tessellation.py`
  - legacy mesh bridge/deprecation path at lines `10`, `504`, `513`, `729`
  - intended end state: keep only explicit consumer-boundary tessellation, not
    “surface back to primary mesh” helpers

### Mesh-Centric Support And Tooling Still On The Removal Path

These are not necessarily deprecated public capability leaves, but they are
still part of the mesh-era stack that should be converted or deleted.

- `src/impression/modeling/_ops_mesh.py`
  - mesh/manifold hull helpers at lines `7`, `8`, `11`, `24`, `38`
  - intended end state: surfaced hull program or explicit legacy compatibility
    deletion

- `src/impression/modeling/group.py`
  - `MeshGroup` composition container at lines `8`, `109` through `164`
  - intended end state: surfaced composition/group truth only

- `src/impression/modeling/transform.py`
  - mesh/group transform helpers at lines `7`, `8`, `11` through `185`
  - intended end state: transform helpers centered on surfaced objects and
    surfaced collections

- `src/impression/modeling/ops.py`
  - mixed planar/mesh hull dispatch at lines `5`, `7`, `14`, `42`
  - intended end state: planar + surfaced dispatch, not planar + mesh

- `src/impression/mesh.py`
  - canonical mesh container and analysis helpers at lines `10`, `48`, `157`,
    `170`, `197`, `233`
  - mesh remains valid as a boundary artifact, but this file should not be
    treated as modeling truth once migration is complete

- `src/impression/mesh_quality.py`
  - mesh-only runtime/tessellation quality knobs at lines `6`, `10`, `22`, `38`
  - intended end state: either explicit boundary-only usage or surfaced quality
    contracts replacing these semantics

- `src/impression/preview.py`
  - preview stack is still mesh/polyline-centric at lines `21`, `24`, `69`,
    `87`, `92`, `136`, `480`, `501`, `728`, `739`
  - intended end state: preview consumes surfaced collections more directly and
    uses mesh only as final render payload

- `src/impression/cli.py`
  - export command still converts to merged mesh + STL at lines `26`, `504`,
    `516`, `522`, `542`, `548`
  - this is currently allowed as a consumer boundary, but it is still part of
    the mesh-era path

- `src/impression/io/stl.py`
  - STL output is still mesh-boundary-only at line `30`
  - this should remain as a consumer boundary, not modeling truth

### Missing Or Unported Analysis Utilities

These are useful mesh-era or analysis-era tools that do not yet have surfaced
equivalents and should be tracked explicitly.

- plane intersection / sectioning of a `SurfaceBody`
  - **not currently present as a surfaced analysis utility**
  - high-value use:
    - loft reconstruction checks
    - correspondence regression
    - station-by-station twist detection
  - recommended direction:
    - add surfaced plane-section capability first
    - allow mesh-based fallback only as temporary analysis tooling, not
      canonical geometry

- hull on surfaced bodies
  - current hull implementation is mesh-only through
    `src/impression/modeling/_ops_mesh.py`
  - no surfaced hull replacement exists yet

- surfaced boolean inspection fixtures
  - current surfaced boolean lane stops at preparation/result envelopes
  - no true surfaced execution analyzer exists yet

## Remaining Legacy Mesh Surface Area

These areas were identified but not deprecated in this pass because they are support layers rather than primary modeling entrypoints:

- `impression.modeling.group`
  - `MeshGroup`
  - `group`
- `impression.modeling.transform`
  - mesh/group transform helpers
- mesh-centric preview / CLI / downstream consumer layers
- mesh analysis / printability helpers

These should be retired after the public modeling constructors and surfaced-loft flows are removed or redirected.

## Loft Testing Opportunity: Plane-Section Reconstruction

A high-value surfaced testing tool is:

- intersect a body with a plane at known loft progression positions
- recover section loops from each slice
- compare those recovered loops to the topology used to author the loft

This is especially valuable for:

- square-to-square or rectangle station progressions that reveal twist
- cylindrical or polygonal cylinder progressions with phase offsets
- correspondence-sensitive hole layouts
- detecting unexpected region permutation across progression

The expected result should not be exact identity after tessellation or
surface-body execution, but it should be close enough for durable similarity
matching.

Recommended comparison signals:

- loop count and region count
- outer/inner classification
- normalized area and perimeter
- centroid drift
- winding/orientation agreement
- best-match contour similarity after rigid alignment

This should become a dedicated loft verification lane once surfaced plane
sectioning exists.

## Recommended Removal Order

1. Flip public primitives and public modeling ops to surface-first defaults with an explicit legacy escape hatch only if still needed.
2. Replace public mesh-returning loft entrypoints with `SurfaceBody` or surface-consumer collection returns.
3. Land surface-native replacements for:
   - text
   - booleans
   - threading
   - hinges
   - heightfields / displacement
4. Remove `SurfaceMeshAdapter` and `mesh_from_surface_body` once all known consumers operate on `SurfaceBody` or standardized tessellation requests.
5. Delete mesh-only support layers that exist only to prop up the retired public APIs.

## Notes

- This audit is intentionally about public API posture, not every internal mesh utility.
- The goal is to make mesh a boundary artifact rather than a modeling truth.
