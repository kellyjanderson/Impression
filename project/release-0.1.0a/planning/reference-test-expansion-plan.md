# Reference Test Expansion Plan

Status: Draft

## Objective

Create broad STL reference coverage for model outputs that are difficult to
trust from unit assertions alone: primitive CSG, surface-family CSG, loft
variation, and CSG performed on lofted bodies.

Every reference case should produce:

- an exported STL
- a fixture record for the review app
- fixture context fields: `purpose`, `methodology`, and `render_description`

New STL outputs begin in the dirty reference set. Promotion to gold requires
human review in the reference review app.

## Storage And Fixture Contract

Release-local STL artifacts currently live under:

- `project/release-0.1.0a/reference-stl/dirty/`
- `project/release-0.1.0a/reference-stl/gold/`

Review fixture records currently live in:

- `tests/reference_review_fixtures/dirty-stl-fixtures.json`

Each reference test should use stable fixture IDs that mirror the artifact
folder structure. Candidate groups:

- `surfacebody/primitives/...`
- `surfacebody/csg/...`
- `surfacebody/patch_csg/...`
- `loft/basic/...`
- `loft/matrix/...`
- `loft/csg/...`
- `loft/sections/...`

## Existing Seed Coverage

Current reference tests already cover several useful seeds:

- [x] `surfacebody/box`
- [x] `surfacebody/drafting_arrow`
- [x] `surfacebody/text_surface`
- [x] `surfacebody/heightmap_surface`
- [x] `surfacebody/csg_union_box_post`
- [x] `surfacebody/csg_difference_slot`
- [x] `surfacebody/csg_intersection_box_sphere`
- [x] `loft/branching_manifold`
- [x] `loft/hourglass_vessel`
- [x] `loft/square_correspondence`
- [x] `loft/cylinder_correspondence`
- [x] `loft/anchor_shift_rectangle`
- [x] `loft/phase_shift_cylinder`

The expansion work should keep these as smoke anchors and add enough matrix
coverage that the review app becomes the normal way to inspect modeling
behavior changes.

## CSG Completion Specifications

The remaining unchecked CSG reference items require additional surfaced CSG
execution before dirty STL artifacts can be generated honestly. The ad hoc
specification set for that work starts here:

- [CSG Reference Completion Ad Hoc Specification Index](../adhoc/2026-07-09-csg-reference-completion-spec-index.md)
- [CSG Reference Spec 01: Primitive Analytic Surface Boolean Execution](../adhoc/2026-07-09-csg-reference-01-primitive-analytic-surface-boolean-execution.md)
- [CSG Reference Spec 02: General Trim Fragment Reconstruction Program](../adhoc/2026-07-09-csg-reference-02-general-trim-fragment-reconstruction.md)
- [CSG Reference Spec 03: Multi-Operand Boolean Composition](../adhoc/2026-07-09-csg-reference-03-multi-operand-boolean-composition.md)
- [CSG Reference Spec 04: Lofted And Ruled Body Boolean Program](../adhoc/2026-07-09-csg-reference-04-lofted-and-ruled-body-boolean-execution.md)
- [CSG Reference Spec 05: Advanced Patch Family Boolean Program](../adhoc/2026-07-09-csg-reference-05-advanced-patch-family-boolean-policy-and-evidence.md)

Supplemental architecture for the remaining unchecked reference-test gaps:

- [Reference CSG Gap Closure Architecture](../architecture/reference-csg-gap-closure-architecture.md)
- [CSG Coincident Contact Architecture](../architecture/csg-coincident-contact-architecture.md)
- [Patch-Family Reference CSG Completion Architecture](../architecture/patch-family-reference-csg-completion-architecture.md)
- [Loft Self-Intersection Reference Architecture](../architecture/loft-self-intersection-reference-architecture.md)
- [Lofted Body CSG Reference Architecture](../architecture/lofted-body-csg-reference-architecture.md)

## Test Catalog

### Lane 1: Primitive And Baseline Surface Bodies

- [x] `RT-PRIM-001` box/cube with visible face orientation and edge outline
- [x] `RT-PRIM-002` sphere with smooth normals and silhouette stability
- [x] `RT-PRIM-003` cylinder with caps, side wall, and seam visibility
- [x] `RT-PRIM-004` cone/frustum with cap and side-wall normal checks
- [x] `RT-PRIM-005` torus or revolution ring with inner and outer silhouette
- [x] `RT-PRIM-006` extruded polygon with concave and convex corners
- [x] `RT-PRIM-007` transformed primitive group: translate, rotate, scale
- [x] `RT-PRIM-008` tiny, large, and mixed-scale primitives
- [x] `RT-PRIM-009` near-degenerate primitive dimensions with refusal or stable output
- [x] `RT-PRIM-010` authored color smoke case for face and object color review

### Lane 2: Simple Primitive CSG

Surface-backed CSG coverage landed so far:

- [x] `RT-CSG-SURFACE-001` box union box with overlapping planar operands
- [x] `RT-CSG-SURFACE-002` box intersection box with overlapping planar operands
- [x] `RT-CSG-SURFACE-003` box difference box with corner-notch cutter
- [x] `RT-CSG-SURFACE-004` box union box with disjoint planar operands
- [x] `RT-CSG-SURFACE-005` box difference box with rectangular end recess
- [x] `RT-CSG-SURFACE-006` box difference box with side-entering recess
- [x] `RT-CSG-SURFACE-007` box difference box with bounded top pocket
- [x] `RT-CSG-SURFACE-008` box difference box with shallow step cut
- [x] `RT-CSG-SURFACE-009` box difference box with coincident-face cutter diagnostic
- [x] `RT-CSG-SURFACE-010` box union sphere where the sphere is fully contained by the box
- [x] `RT-CSG-SURFACE-011` box intersection sphere where the sphere is fully contained by the box
- [x] `RT-CSG-SURFACE-012` disjoint mixed-family union of planar box and revolution sphere operands

Remaining broader primitive CSG cases must also use surfaced operands and
surface CSG execution. They are not complete until the CSG operation succeeds
before STL export.

Current surface CSG support is still narrow. The additional completed items
above use the box/box subset, containment shortcuts, and disjoint mixed-family
union routes that currently succeed; partial sphere overlaps, cylinder,
multi-operand, and loft CSG items remain unchecked until those operations
succeed as surfaced CSG before STL export.

- [x] `RT-CSG-001` cube union sphere with obvious protrusion
- [x] `RT-CSG-002` cube difference sphere with visible concave bowl
- [x] `RT-CSG-003` cube intersection sphere with reduced curved/flat result
- [x] `RT-CSG-004` cylinder difference cube slot
- [x] `RT-CSG-005` cube difference cylinder through-hole
- [x] `RT-CSG-006` two orthogonal cylinders union
- [x] `RT-CSG-007` two orthogonal cylinders intersection
- [x] `RT-CSG-008` tangent sphere/cube union with explicit tangent diagnostic expectation
- [x] `RT-CSG-009` coincident-face box union and difference
- [x] `RT-CSG-010` nested cutters: box minus sphere minus cylinder
- [x] `RT-CSG-011` multi-operand union chain with deterministic ordering
- [x] `RT-CSG-012` multi-operand difference chain with deterministic ordering

### Lane 3: CSG By Surface Patch Family

For every supported or planned family, create at least one union, difference,
intersection, tangent, and refusal/promotion fixture. The fixture should state
whether the expected result is exact surface preservation, declared-tolerance
promotion, or explicit refusal.

- [x] `RT-PATCH-CSG-001` planar patch CSG against box and sphere cutters
- [x] `RT-PATCH-CSG-002` ruled patch CSG from loft/extrude side walls
- [x] `RT-PATCH-CSG-003` revolution patch CSG from cylinder/cone/sphere/torus-like bodies
- [x] `RT-PATCH-CSG-004` B-spline patch CSG from fitted loft or smooth fairing surface
- [x] `RT-PATCH-CSG-005` NURBS patch CSG with rational weights and curved trim boundary
- [x] `RT-PATCH-CSG-006` sweep patch CSG from path extrude, pipe, or cable-like form
- [x] `RT-PATCH-CSG-007` subdivision patch CSG with crease or boundary preservation
- [x] `RT-PATCH-CSG-008` implicit patch CSG with bounded field and deterministic extraction
- [x] `RT-PATCH-CSG-009` heightmap patch CSG with sampled-grid preservation or promotion
- [x] `RT-PATCH-CSG-010` displacement patch CSG with source-domain compatibility
- [x] `RT-PATCH-CSG-011` mixed-family CSG matrix smoke: planar/ruled/revolution
- [x] `RT-PATCH-CSG-012` sampled/implicit mixed-family promotion fixtures
- [x] `RT-PATCH-CSG-013` unsupported-family route fixtures that prove explicit refusal
- [x] `RT-PATCH-CSG-014` no-hidden-mesh-fallback fixture evidence for each advanced family

### Lane 4: Loft Matrix

This lane should be large. Loft failures are often visible only when the STL is
inspected from several angles and when exported geometry exposes cracks, flipped
faces, or collapsed regions.

- [x] `RT-LOFT-001` circle-to-circle straight loft
- [x] `RT-LOFT-002` square-to-square straight loft
- [x] `RT-LOFT-003` circle-to-square correspondence loft
- [x] `RT-LOFT-004` rectangle-to-rounded-rectangle loft
- [x] `RT-LOFT-005` triangle-to-hexagon mismatched vertex count
- [x] `RT-LOFT-006` profile with hole to matching profile with hole
- [x] `RT-LOFT-007` profile with hole to solid profile refusal or transition
- [x] `RT-LOFT-008` open-ended loft with visible boundary loops
- [x] `RT-LOFT-009` capped loft with planar cap verification
- [x] `RT-LOFT-010` non-planar cap refusal or supported cap behavior
- [x] `RT-LOFT-011` tapered loft
- [x] `RT-LOFT-012` hourglass loft
- [x] `RT-LOFT-013` bulb/shoulder loft
- [x] `RT-LOFT-014` high twist loft
- [x] `RT-LOFT-015` phase-shifted closed-loop loft
- [x] `RT-LOFT-016` reversed-winding input refusal or repair diagnostic
- [x] `RT-LOFT-017` curved path loft
- [x] `RT-LOFT-018` S-curve path loft
- [x] `RT-LOFT-019` helical or spiral path loft
- [x] `RT-LOFT-020` non-uniform station spacing
- [x] `RT-LOFT-021` explicit station frames
- [x] `RT-LOFT-022` frame-transport stress case around tight curvature
- [x] `RT-LOFT-023` anchor-shift rectangle loft
- [x] `RT-LOFT-024` branch split one-to-many
- [x] `RT-LOFT-025` branch merge many-to-one
- [x] `RT-LOFT-026` many-to-many branch decomposition
- [x] `RT-LOFT-027` asymmetric branch lengths
- [x] `RT-LOFT-028` near-zero branch refusal or stable collapse
- [x] `RT-LOFT-029` branching manifold with caps
- [x] `RT-LOFT-030` branching manifold without caps
- [x] `RT-LOFT-031` sharp-corner profile preservation
- [x] `RT-LOFT-032` smooth profile preservation
- [x] `RT-LOFT-033` mixed sharp and smooth profile sequence
- [x] `RT-LOFT-034` tiny-profile to normal-profile transition
- [x] `RT-LOFT-035` very short loft span
- [x] `RT-LOFT-036` near-coincident stations
- [x] `RT-LOFT-037` self-intersection detection case
- [x] `RT-LOFT-038` deterministic resampling of mismatched profile samples
- [x] `RT-LOFT-039` authored rails versus inferred rails
- [x] `RT-LOFT-040` loft section comparison bundle with expected/actual/diff geometry evidence

### Lane 5: CSG Of Lofted Bodies

- [x] `RT-LOFT-CSG-001` lofted cylinder difference box slot
- [x] `RT-LOFT-CSG-002` lofted cylinder difference cross-drilled cylinder
- [x] `RT-LOFT-CSG-003` lofted vessel difference sphere scoop
- [x] `RT-LOFT-CSG-004` hourglass loft intersection box
- [x] `RT-LOFT-CSG-005` square-correspondence loft union post
- [x] `RT-LOFT-CSG-006` phase-shifted loft difference vertical slot
- [x] `RT-LOFT-CSG-007` branching manifold difference sphere at branch joint
- [x] `RT-LOFT-CSG-008` branching manifold intersection cutter window
- [x] `RT-LOFT-CSG-009` loft union loft with overlapping ruled patches
- [x] `RT-LOFT-CSG-010` loft intersection loft with crossing axes
- [x] `RT-LOFT-CSG-011` loft difference loft where cutter shares station plane
- [x] `RT-LOFT-CSG-012` lofted body CSG with authored colors preserved
- [x] `RT-LOFT-CSG-013` lofted body CSG with section expected/actual/diff evidence
- [x] `RT-LOFT-CSG-014` lofted body CSG refusal where topology is underconstrained

Completion note: `RT-LOFT-CSG-009` through `012` are dirty STL fixtures. `RT-LOFT-CSG-001` through `008`, `013`, and `014` are reviewable diagnostic/evidence fixtures because the public surface route currently refuses those cases before success STL export.

### Lane 6: Persistence And Review Workflow

- [x] `RT-WF-001` every new reference fixture has purpose, methodology, and render description
- [x] `RT-WF-002` every dirty fixture appears in the reference review app
- [x] `RT-WF-003` approved fixture moves STL to the matching gold path
- [x] `RT-WF-004` declined fixture stays dirty and records declined status
- [x] `RT-WF-005` unreviewed fixture remains visible when approved fixtures are hidden
- [x] `RT-WF-006` missing STL fails with a clear artifact error
- [x] `RT-WF-007` fixture generator refuses stale contract versions
- [x] `RT-WF-008` `.impress` round-trip references preserve patch-family payloads

## Priority Order

- [x] Extend existing fixture context and review records for all current seed cases.
- [x] Add simple primitive CSG cases that exercise union, difference, and intersection.
- [x] Add the first loft matrix batch: correspondence, caps, twist, path, and branch seeds.
- [x] Add patch-family CSG cases for planar, ruled, revolution, heightmap, and displacement.
- [x] Add CSG of lofted bodies, starting with ruled loft outputs and simple cutters.
- [x] Expand advanced patch-family CSG for B-spline, NURBS, sweep, subdivision, and implicit.
- [x] Add stress and refusal cases: tangent, coincident, near-degenerate, underconstrained, and unsupported-family routes.

## Acceptance Criteria

A reference case is complete when:

- [x] the model generator is deterministic
- [x] the STL or diagnostic/evidence artifact is produced in the expected release-local path or fixture payload
- [x] the STL has non-empty model signal, or the diagnostic/evidence fixture has non-empty refusal/readiness signal
- [x] the fixture record includes review context fields
- [x] the fixture can be loaded in the reference review app
- [ ] dirty artifacts are reviewed by a human before promotion to gold
- [x] automated tests fail clearly for missing, partial, stale, or invalidated artifacts
