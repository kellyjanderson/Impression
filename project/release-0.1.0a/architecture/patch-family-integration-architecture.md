# Patch Family Integration Architecture

## Overview

This architecture defines what it means for a surface body patch family to be
fully functional and fully integrated in Impression.

The existing [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
defines the patch families and their family-local geometry contracts. This
document defines the cross-system integration contract:

- every patch family must be accepted by the in-memory `SurfaceBody` model
- every patch family must round-trip through `.impress`
- every patch family must tessellate only at the tessellation boundary
- authored modeling producers must emit the most appropriate patch family
- unsupported consumers must refuse with diagnostics rather than mesh fallback
- the capability matrix must only mark a family `available` when the full
  integration contract is true

The target is not merely to have patch classes. The target is for patch
families to be usable modeling truth across construction, storage, traversal,
persistence, inspection, and downstream consumer handoff.

## Relationship To Existing Architecture

This document extends:

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)
- [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)
- [Loft Evolution System Architecture](loft-evolution-system.md)
- [Loft Topology Point Correspondence Architecture](loft-topology-point-correspondence-architecture.md)

It clarifies the release target for work that grew out of removing deferred
patch-family exclusions: patch families must become integrated surface-native
capabilities, not isolated records.

## Integration Definition

A patch family is fully integrated only when all of the following are true.

### Runtime Patch Contract

The family has a concrete `SurfacePatch` implementation that supports:

- validation of family-local payloads
- stable identity
- deterministic `canonical_payload`
- transform attachment
- parameter-domain ownership
- evaluation or an explicit family-appropriate evaluation contract
- trim-loop semantics where the family supports trims
- seam and boundary participation, or an explicit diagnostic when the family
  cannot participate in a requested seam mode

### SurfaceBody Store Contract

The in-memory store of authored objects must accept the family without special
escape hatches.

This includes:

- `SurfacePatch`
- `SurfaceShell`
- `SurfaceBody`
- `SurfaceComposition`
- traversal to `SurfaceConsumerCollection`
- transform helpers
- stable identity and cache keys
- metadata split between kernel and consumer namespaces

No family may require conversion to `Mesh` to be stored as authored modeling
truth.

### Tessellation Boundary Contract

Each family must have a tessellation adapter or a structured refusal diagnostic
at the explicit tessellation boundary.

The adapter must:

- consume the family-native patch payload
- respect `TessellationRequest`
- preserve source patch identity in mesh metadata
- report approximation/lossiness where relevant
- avoid changing authored surface truth

Mesh creation remains legal only at explicit consumer boundaries such as
preview, export, analysis, repair, or explicitly named mesh compatibility APIs.

### `.impress` Contract

Every integrated family must round-trip through `.impress`.

The persistence contract requires:

- deterministic encoding
- schema-versioned geometry payload
- deterministic decoding to the same patch family
- stable identity preservation after round trip
- validation diagnostics for malformed payloads
- no mesh-derived wrapper serialization as surface truth

The current `.impress` family map must include every family that the capability
matrix marks `available`.

### Producer Contract

Authoring functions must emit the most appropriate patch family for the
geometry being authored.

This rule is intentionally conservative:

- use `PlanarSurfacePatch` for exact planar faces and caps
- use `RuledSurfacePatch` for exact linear transitions and linear extrusions
- use `RevolutionSurfacePatch` for exact surfaces of revolution
- use `BSplineSurfacePatch` for smooth non-rational lofts and fitted smooth
  surfaces where exact ruled/revolution patches are not the authored intent
- use `NURBSSurfacePatch` for rational or conic-exact surfaces when weights are
  needed to preserve authored geometry
- use `SweepSurfacePatch` when the authored object is a profile transported
  along a path and the path/profile relationship is part of the model intent
- use `SubdivisionSurfacePatch` for authored cage-and-crease surfaces
- use `ImplicitSurfacePatch` for declarative field-defined surfaces
- use `HeightmapSurfacePatch` for sampled heightfield surfaces
- use `DisplacementSurfacePatch` for sampled displacement applied to an
  existing source surface

Producers should not use a more complex family merely to exercise support. A
box should stay planar. A cylinder should stay revolution. A simple two-station
linear loft should stay ruled.

### Consumer Contract

Consumers must either support a family directly or refuse explicitly.

Examples:

- CSG may support only certain family pairs at first, but unsupported pairs
  must return family-aware unsupported diagnostics.
- Loft execution may emit B-spline or sweep surfaces only when the input
  intent requires those families.
- `.impress` must refuse unknown or malformed family payloads rather than
  accepting partial surface truth.
- Analysis and export may tessellate, but they may not mutate the stored
  `SurfaceBody`.

## Current Gap Assessment

The current codebase has partial integration:

- `SurfaceBody` can structurally contain the major patch classes.
- `PlanarSurfacePatch`, `RuledSurfacePatch`, and `RevolutionSurfacePatch` are
  the mature authored primitive families.
- Advanced patch classes exist for B-spline, NURBS, sweep, subdivision,
  implicit, heightmap, and displacement.
- `.impress` already has coverage for several advanced families, but the file
  format map must be audited against the complete family list.
- Heightmap and displacement need explicit `.impress` round-trip verification
  before they can be considered fully integrated.
- Loft still primarily emits ruled side patches and planar caps; smooth
  multi-station lofts and path/profile intent need family selection for
  B-spline, NURBS, and sweep output.
- The capability matrix still distinguishes `available` and `planned`; this is
  correct until each family satisfies this integration contract.

## Family Availability Gates

The `PATCH_FAMILY_CAPABILITY_MATRIX` may mark a family `available` only when
these gates pass:

- patch payload validation
- point/evaluation behavior or explicit family evaluation contract
- tessellation adapter
- SurfaceBody/Shell storage
- transform preservation
- seam/boundary policy
- `.impress` encode/decode round trip
- at least one appropriate producer or an explicitly documented externally
  authored construction path
- no hidden mesh fallback tests
- unsupported consumer diagnostics

If any gate is missing, the family remains `planned` or operation-scoped rather
than globally available.

## Loft Patch Family Selection

Loft is the most important producer for advanced patch families.

Loft execution should choose patch families by authored intent and topology:

- two compatible linear stations:
  - emit `RuledSurfacePatch`
- multiple smooth compatible stations without rational requirements:
  - emit `BSplineSurfacePatch`
- conic/rational station or rail intent:
  - emit `NURBSSurfacePatch`
- profile transported along a path where the path is the primary authored
  guide:
  - emit `SweepSurfacePatch`
- topology transitions with births, deaths, splits, or merges:
  - decompose into supported patches according to the topology correspondence
    architecture
  - refuse unresolved ambiguity with diagnostics
- caps:
  - emit `PlanarSurfacePatch` when cap loops are planar
  - use another family only when cap geometry is explicitly non-planar

Loft must not use mesh sampling as its canonical executor. Mesh output from a
loft remains debug, preview, export, analysis, or explicit compatibility
output.

## Primitive Patch Family Selection

Primitive builders should not be rewritten to use all families. They should use
the simplest exact family.

Required primitive posture:

- box:
  - planar faces
- polyhedron:
  - planar faces
- prism and n-gon prism:
  - ruled sidewalls and planar caps
- cylinder and cone:
  - revolution sidewalls and planar caps
- sphere and torus:
  - revolution surfaces
- heightmap:
  - heightmap patch
- displaced surface:
  - displacement patch over source surface

Future primitives that are naturally advanced-family producers should be added
as such, for example:

- swept profile primitive -> `SweepSurfacePatch`
- authored freeform cage primitive -> `SubdivisionSurfacePatch`
- declarative field primitive -> `ImplicitSurfacePatch`
- smooth fitted surface primitive -> `BSplineSurfacePatch` or
  `NURBSSurfacePatch`

## `.impress` Integration Requirements

The `.impress` architecture must be extended or audited so the patch-kind map
and geometry payload codecs cover every available family.

Required codec behavior:

- `PlanarSurfacePatch`
- `RuledSurfacePatch`
- `RevolutionSurfacePatch`
- `BSplineSurfacePatch`
- `NURBSSurfacePatch`
- `SweepSurfacePatch`
- `SubdivisionSurfacePatch`
- `ImplicitSurfacePatch`
- `HeightmapSurfacePatch`
- `DisplacementSurfacePatch`

Each family needs:

- payload version
- encode function
- decode function
- malformed payload tests
- round-trip stable identity tests
- no mesh wrapper acceptance

## Internal Store Requirements

The internal object store is the `SurfaceBody` graph plus surfaced composition:

```text
SurfacePatch
-> SurfaceShell
-> SurfaceBody
-> SurfaceComposition / SurfaceScene
-> SurfaceConsumerCollection
-> explicit tessellation boundary
```

This store must preserve family-native payloads. It may attach transforms,
metadata, seams, adjacency, and composition traversal records, but it may not
replace a family-native patch with mesh-derived triangle wrappers.

## Diagnostics

All unsupported states must be explicit. Diagnostics should include:

- subsystem
- patch family
- operation
- reason
- whether the operation is unsupported, malformed, ambiguous, or outside the
  tessellation boundary
- suggested supported route when one exists

Diagnostics are required for:

- unsupported CSG family pairs
- unsupported loft topology transitions
- malformed `.impress` payloads
- unsupported seam participation
- unsafe implicit field payloads
- attempts to store mesh as surface truth
- hidden mesh fallback detection tests

## Data Flow

```text
Authoring Input
-> producer-specific intent classification
-> family-appropriate SurfacePatch records
-> SurfaceShell / SurfaceBody
-> SurfaceComposition when grouping is needed
-> .impress save/load for persistence
-> tessellation only for explicit consumers
```

The reverse path for `.impress` is:

```text
.impress JSON
-> root/schema validation
-> SurfaceBodyStore
-> patch payload decode
-> shell/body reconstruction
-> stable identity validation
-> SurfaceBody runtime graph
```

No data flow may insert mesh execution between authoring input and stored
surface truth.

## Specification Manifest For Discovery

### Candidate Spec: Patch Family Availability Gate And Capability Matrix

Discovery purpose:
- Define the testable gates that allow a patch family to move from `planned`
  to `available` without overstating partial support.

Responsibilities:
- Functions/methods:
  - capability gate validator
  - capability matrix assertion helper
- Data structures/models:
  - availability gate record
  - family capability record
- Dependencies/services:
  - `src/impression/modeling/surface.py`
  - test suite
- Returns/outputs/signals:
  - availability diagnostic
  - updated `PATCH_FAMILY_CAPABILITY_MATRIX`
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `PATCH_FAMILY_CAPABILITY_MATRIX`
  - Additions to existing reusable library/module: `surface.py` capability helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes capability declarations and tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded static/test checks over the family matrix
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - a family is `available` only when store, tessellation, `.impress`, producer,
    diagnostics, and no-hidden-fallback gates pass
- Test strategy:
  - matrix tests fail when an `available` family lacks required integration
    evidence
- Data ownership:
  - `PATCH_FAMILY_CAPABILITY_MATRIX` owns family availability truth
- Routes:
  - test helper reads matrix and integration evidence
- Reuse/extraction decision:
  - add to existing `surface.py`; do not create a separate registry yet
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- CSG support may remain operation-scoped; the gate should distinguish family
  availability from universal consumer support.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one matrix policy boundary and one validator
  route.

### Candidate Spec: Patch Family Availability Promotion Pass

Discovery purpose:
- Promote families from `planned` to `available` only after their integration
  evidence exists, and leave explicit diagnostics for any operation-scoped
  gaps.

Responsibilities:
- Functions/methods:
  - availability promotion helper
  - operation coverage assertion
- Data structures/models:
  - promotion evidence record
  - operation-scoped support record
- Dependencies/services:
  - `src/impression/modeling/surface.py`
  - integration test evidence
- Returns/outputs/signals:
  - promoted capability matrix entries
  - unpromoted family diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: availability gate helper and capability matrix
  - Additions to existing reusable library/module: surface capability tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - updates capability declarations
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded static/test evidence checks
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - promotion is evidence-driven and may leave operation-level support notes
- Test strategy:
  - failing test if a promoted family lacks required integration evidence
- Data ownership:
  - capability matrix owns public availability; tests own evidence check
- Routes:
  - availability gate helper to capability matrix update
- Reuse/extraction decision:
  - reuse availability gate helper
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- A family can be available for storage/persistence/tessellation while CSG
  remains operation-scoped with diagnostics.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because promotion is one final capability declaration pass.

### Candidate Spec: `.impress` Patch Codec Coverage Inventory

Discovery purpose:
- Establish the authoritative `.impress` family codec coverage matrix before
  adding missing codecs.

Responsibilities:
- Functions/methods:
  - codec inventory scanner
  - family map assertion
- Data structures/models:
  - codec coverage record
  - patch family list
- Dependencies/services:
  - `src/impression/io/impress.py`
  - `src/impression/modeling/surface.py`
- Returns/outputs/signals:
  - missing codec diagnostic
  - coverage matrix test result
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `.impress` codec helpers
  - Additions to existing reusable library/module: test helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only unless inventory reveals stale declarations
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded static inspection of codec maps
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/` plus `src/impression/io/impress.py` if stale map entries are found
- Chosen defaults / parameters:
  - the codec matrix must include every family marked `available`
- Test strategy:
  - static test comparing codec map against patch-family capability matrix
- Data ownership:
  - `.impress` owns persistence codec truth; `surface.py` owns family list truth
- Routes:
  - test helper reads public constants and codec module maps
- Reuse/extraction decision:
  - add test helper; do not move codec tables unless duplication appears
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- The codec inventory should not require codecs for `planned` families unless
  the architecture decides `.impress` leads availability.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is a bounded inventory and assertion spec, not the
  implementation of every missing codec.

### Candidate Spec: `.impress` All-Family Round-Trip Identity Coverage

Discovery purpose:
- Prove every codec-covered family round-trips through `.impress` while
  preserving stable identity and family-native payloads.

Responsibilities:
- Functions/methods:
  - all-family round-trip fixture builder
  - stable identity assertion helper
- Data structures/models:
  - all-family `.impress` fixture
  - round-trip identity record
- Dependencies/services:
  - `src/impression/io/impress.py`
  - all codec-covered patch families
- Returns/outputs/signals:
  - loaded `SurfaceBodyStore`
  - stable identity preservation result
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `.impress` save/load framework
  - Additions to existing reusable library/module: test fixture helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only
- Security/privacy-sensitive behavior:
  - implicit fixtures use safe declarative payloads and sampled fixtures avoid
    source paths
- Performance-sensitive behavior:
  - bounded sample grids, cages, and patch counts
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests for `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - one minimal non-degenerate fixture per codec-covered family
- Test strategy:
  - save/load round-trip preserves body identity, patch family list, and patch
    identities
- Data ownership:
  - `.impress` owns persistence; runtime patches own validation after load
- Routes:
  - `make_impress_document_payload`, `dumps/loads`, `save/load`
- Reuse/extraction decision:
  - reuse all-family fixture helpers when available
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This spec depends on missing family codecs landing before the full matrix can
  pass.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one persistence acceptance fixture matrix.

### Candidate Spec: `.impress` Heightmap Surface Patch Codec

Discovery purpose:
- Persist and restore `HeightmapSurfacePatch` payloads without source-image or
  mesh-wrapper fallback.

Responsibilities:
- Functions/methods:
  - heightmap patch encoder
  - heightmap patch decoder
- Data structures/models:
  - `HeightmapSurfacePatch` payload
  - sampled grid payload version
- Dependencies/services:
  - `src/impression/io/impress.py`
  - `src/impression/modeling/surface.py`
- Returns/outputs/signals:
  - round-tripped heightmap patch
  - `ImpressFormatError` diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `.impress` patch codec framework
  - Additions to existing reusable library/module: `impress.py` heightmap codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - extends persisted `.impress` payload support
- Security/privacy-sensitive behavior:
  - source image paths are not serialized
- Performance-sensitive behavior:
  - bounded sample grid fixtures
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - encode height samples, alpha mask, alpha mode, height scale, and xy scale
- Test strategy:
  - stable identity round-trip and malformed grid/mask tests
- Data ownership:
  - persisted sampled arrays belong to `.impress`; runtime validation belongs to
    `HeightmapSurfacePatch`
- Routes:
  - `encode_surface_patch_payload` and `decode_surface_patch_payload`
- Reuse/extraction decision:
  - add to existing codec helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- None; source image paths are explicitly outside persisted surface truth.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because heightmap encode/decode and validation are one payload
  boundary.

### Candidate Spec: `.impress` Displacement Surface Patch Codec

Discovery purpose:
- Persist and restore `DisplacementSurfacePatch` payloads while preserving the
  nested source surface and sampled displacement truth.

Responsibilities:
- Functions/methods:
  - displacement patch encoder
  - displacement patch decoder
- Data structures/models:
  - `DisplacementSurfacePatch` payload
  - source patch payload
  - sampled displacement payload version
- Dependencies/services:
  - `src/impression/io/impress.py`
  - `src/impression/modeling/surface.py`
- Returns/outputs/signals:
  - round-tripped displacement patch
  - `ImpressFormatError` diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `.impress` patch codec framework
  - Additions to existing reusable library/module: `impress.py` displacement codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - extends persisted `.impress` payload support
- Security/privacy-sensitive behavior:
  - no external source references for displacement samples
- Performance-sensitive behavior:
  - bounded sample grid fixtures
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - encode source patch inline with displacement grid, alpha policy, direction,
    projection plane, and bounds
- Test strategy:
  - stable identity round-trip and malformed source/grid/direction tests
- Data ownership:
  - `.impress` owns persisted source patch plus displacement payload
- Routes:
  - `encode_surface_patch_payload` and `decode_surface_patch_payload`
- Reuse/extraction decision:
  - add to existing codec helpers; reuse heightmap array helpers if created
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Nested source patch identity must be preserved without requiring document-level
  patch reference ordering.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because displacement encode/decode and validation are one
  payload boundary.

### Candidate Spec: SurfaceBody All-Family Store Fixture Coverage

Discovery purpose:
- Prove `SurfacePatch -> SurfaceShell -> SurfaceBody` preserves every patch
  family as surface truth.

Responsibilities:
- Functions/methods:
  - all-family patch fixture builder
  - store assertion helper
- Data structures/models:
  - all-family `SurfaceBody` fixture
  - family fixture matrix
- Dependencies/services:
  - `src/impression/modeling/surface.py`
  - tests
- Returns/outputs/signals:
  - preserved patch family list
  - stable identities
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `SurfacePatch`, `SurfaceShell`, `SurfaceBody`
  - Additions to existing reusable library/module: test fixture helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only
- Security/privacy-sensitive behavior:
  - implicit field fixture uses safe declarative nodes only
- Performance-sensitive behavior:
  - bounded low-resolution fixtures for sampled/subdivision/implicit families
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests, with fixtures near existing surface tests
- Chosen defaults / parameters:
  - fixture grids and cages use minimal non-degenerate dimensions
- Test strategy:
  - one body containing one patch from each family, asserting construction and
    stable identity
- Data ownership:
  - test fixtures own sample payloads; runtime `SurfaceBody` owns stored truth
- Routes:
  - direct constructors into `make_surface_shell` and `make_surface_body`
- Reuse/extraction decision:
  - add reusable test fixture helper if needed by later specs
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Implicit fixtures must remain declarative and safe so store tests do not
  become evaluator security tests.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one store acceptance fixture and assertion
  boundary.

### Candidate Spec: SurfaceComposition All-Family Traversal Coverage

Discovery purpose:
- Prove surfaced composition preserves every family through traversal to
  consumer handoff without tessellation.

Responsibilities:
- Functions/methods:
  - all-family composition fixture builder
  - traversal assertion helper
- Data structures/models:
  - `SurfaceComposition`
  - `SurfaceConsumerCollection`
  - all-family traversal records
- Dependencies/services:
  - `src/impression/modeling/surface_scene.py`
  - `src/impression/modeling/surface.py`
- Returns/outputs/signals:
  - family-native bodies
  - ordered consumer records
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `SurfaceComposition`, store fixture helpers
  - Additions to existing reusable library/module: tests and fixture helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only
- Security/privacy-sensitive behavior:
  - none beyond safe implicit fixture inherited from store spec
- Performance-sensitive behavior:
  - bounded family fixture count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests for `surface_scene.py`
- Chosen defaults / parameters:
  - use one composition containing every available patch family
- Test strategy:
  - assert family order, identity/family preservation, and no mesh output
- Data ownership:
  - composition owns grouping; patches retain family-native payloads
- Routes:
  - `surface_group` to `handoff_surface_composition`
- Reuse/extraction decision:
  - reuse store fixture helper from all-family store spec
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Bounds checks remain outside this traversal-only spec.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this spec is now only the composition traversal path.

### Candidate Spec: SurfaceComposition Transform And Identity Preservation Coverage

Discovery purpose:
- Prove composition transforms preserve all-family patch identity, family-native
  payloads, and family-appropriate bounds behavior without tessellation.

Responsibilities:
- Functions/methods:
  - transformed composition fixture builder
  - transform preservation assertion helper
- Data structures/models:
  - `SurfaceComposition`
  - transform records
  - transformed traversal records
- Dependencies/services:
  - `src/impression/modeling/surface_scene.py`
  - `src/impression/modeling/transform.py`
  - `src/impression/modeling/surface.py`
- Returns/outputs/signals:
  - transformed family-native bodies
  - preserved identity and transform metadata
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `SurfaceComposition`, transform helpers
  - Additions to existing reusable library/module: transform preservation tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only
- Security/privacy-sensitive behavior:
  - none beyond safe implicit fixture inherited from store spec
- Performance-sensitive behavior:
  - bounded transform fixture count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests for `surface_scene.py` and `transform.py`
- Chosen defaults / parameters:
  - use one identity and one non-identity transform across the fixture
- Test strategy:
  - assert transform attachment, stable identity, family preservation, and no
    mesh output; bounds checks apply only where the family exposes
    bounded/evaluable geometry
- Data ownership:
  - transforms own placement; patches retain family-native payloads
- Routes:
  - `surface_group` to `handoff_surface_composition`
- Reuse/extraction decision:
  - reuse store fixture helper and composition traversal fixture
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Exact transformed bounds are intentionally family-dependent; implicit,
  subdivision, and approximation-heavy families may assert transform metadata
  rather than exact bounds.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is now only the transform/identity path for
  composition handoff.

### Candidate Spec: Tessellation Adapter Capability Inventory And Metadata Gate

Discovery purpose:
- Establish the authoritative tessellation adapter coverage matrix and require
  traceable metadata before families are considered tessellation-ready.

Responsibilities:
- Functions/methods:
  - adapter inventory scanner
  - adapter metadata assertion helper
- Data structures/models:
  - adapter coverage record
  - lossiness metadata
- Dependencies/services:
  - `src/impression/modeling/tessellation.py`
  - patch family capability matrix
- Returns/outputs/signals:
  - missing adapter diagnostic
  - metadata coverage test result
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation adapter framework
  - Additions to existing reusable library/module: adapter metadata tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes tessellation metadata/tests only unless stale adapter declarations
    are found
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded static/test inspection of adapter maps
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - every available family must have an adapter or explicit unsupported
    diagnostic route
- Test strategy:
  - static test compares available families against adapter metadata coverage
- Data ownership:
  - tessellation owns adapter support truth; capability matrix owns public
    family availability
- Routes:
  - adapter registry to `tessellate_surface_patch` and
    `tessellate_surface_body`
- Reuse/extraction decision:
  - add metadata assertions to existing adapter framework
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This spec inventories readiness only; family-specific adapter behavior lands
  in the adapter specs below.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is only the tessellation coverage inventory and
  metadata gate.

### Candidate Spec: Sampled Surface Tessellation Adapters

Discovery purpose:
- Tessellate heightmap and displacement patches at the explicit boundary while
  preserving authored source payloads and lossiness metadata.

Responsibilities:
- Functions/methods:
  - heightmap tessellation adapter
  - displacement tessellation adapter
- Data structures/models:
  - sampled grid fixture
  - displacement source patch fixture
  - lossiness metadata
- Dependencies/services:
  - `src/impression/modeling/tessellation.py`
  - sampled patch families
- Returns/outputs/signals:
  - `SurfaceTessellationResult`
  - sampled-family tessellation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation adapter framework
  - Additions to existing reusable library/module: sampled adapters/tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds tessellation adapter behavior and tests
- Security/privacy-sensitive behavior:
  - no external image or source-file reads during tessellation
- Performance-sensitive behavior:
  - bounded sampled grids and preview-quality tessellation requests
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - preview-quality tessellation for minimal sampled fixtures
- Test strategy:
  - heightmap and displacement smoke tests assert mesh output metadata, source
    family identity, and no mutation of stored patch payloads
- Data ownership:
  - tessellation owns mesh output; sampled patches retain authored truth
- Routes:
  - `tessellate_surface_patch`, `tessellate_surface_body`
- Reuse/extraction decision:
  - reuse shared grid sampling helper if the codec specs create one
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Displacement tessellation must preserve source patch identity even when the
  generated mesh samples displaced positions.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because heightmap and displacement share sampled-grid adapter
  mechanics.

### Candidate Spec: Spline Surface Tessellation Adapters

Discovery purpose:
- Tessellate B-spline, NURBS, and sweep surface patches from family-native
  surface records rather than mesh-producing loft executors.

Responsibilities:
- Functions/methods:
  - B-spline tessellation adapter
  - NURBS tessellation adapter
  - sweep tessellation adapter
- Data structures/models:
  - spline tessellation fixture
  - rational weight fixture
  - sweep rail/section fixture
- Dependencies/services:
  - `src/impression/modeling/tessellation.py`
  - spline patch families
- Returns/outputs/signals:
  - `SurfaceTessellationResult`
  - spline-family tessellation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation adapter framework
  - Additions to existing reusable library/module: spline adapters/tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds tessellation adapter behavior and tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded knot grids, section counts, and preview tessellation requests
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - preview-quality tessellation with explicit parameter-domain sampling
- Test strategy:
  - one smoke per spline-like family asserts metadata, source identity, and no
    mutation of control nets or weight grids
- Data ownership:
  - tessellation owns mesh output; spline patches retain authored truth
- Routes:
  - `tessellate_surface_patch`, `tessellate_surface_body`
- Reuse/extraction decision:
  - reuse shared spline basis utilities from loft specs
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Sweep tessellation should consume the stored sweep patch, not the loft
  planner that produced it.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because these adapters share parameter-domain sampling and
  spline basis evaluation.

### Candidate Spec: Subdivision Surface Tessellation Adapter

Discovery purpose:
- Tessellate subdivision patches at the tessellation boundary with explicit
  approximation metadata and without mutating the authored control cage.

Responsibilities:
- Functions/methods:
  - subdivision tessellation adapter
  - subdivision approximation assertion helper
- Data structures/models:
  - subdivision cage fixture
  - approximation metadata
- Dependencies/services:
  - `src/impression/modeling/tessellation.py`
  - `SubdivisionSurfacePatch`
- Returns/outputs/signals:
  - `SurfaceTessellationResult`
  - subdivision approximation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation adapter framework
  - Additions to existing reusable library/module: subdivision adapter/tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds tessellation adapter behavior and tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded subdivision level for preview fixtures
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - preview subdivision level is bounded and recorded in metadata
- Test strategy:
  - subdivision smoke test asserts approximation metadata, source identity, and
    unchanged control cage
- Data ownership:
  - tessellation owns mesh output; subdivision patch owns control cage truth
- Routes:
  - `tessellate_surface_patch`, `tessellate_surface_body`
- Reuse/extraction decision:
  - reuse adapter framework; do not share with implicit sampling unless a common
    approximation metadata helper emerges
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Approximation is acceptable only when the tessellation result records it.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one subdivision adapter boundary.

### Candidate Spec: Implicit Surface Tessellation Adapter Safety And Sampling

Discovery purpose:
- Tessellate implicit patches only from bounded safe declarative fields, with
  explicit sampling metadata and diagnostics for unsafe or unbounded states.

Responsibilities:
- Functions/methods:
  - implicit tessellation adapter
  - implicit sampling safety assertion helper
- Data structures/models:
  - safe implicit field fixture
  - bounded sampling volume
  - sampling metadata
- Dependencies/services:
  - `src/impression/modeling/tessellation.py`
  - `ImplicitSurfacePatch`
- Returns/outputs/signals:
  - `SurfaceTessellationResult`
  - unsafe or unbounded implicit diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation adapter framework
  - Additions to existing reusable library/module: implicit adapter/tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds tessellation adapter behavior and tests
- Security/privacy-sensitive behavior:
  - samples declarative safe field nodes only and refuses unsafe payloads
- Performance-sensitive behavior:
  - bounded sampling volume and preview resolution
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - require finite bounds and preview sampling resolution for implicit
    tessellation
- Test strategy:
  - safe bounded field tessellates with metadata; unsafe or unbounded fields
    refuse before mesh creation
- Data ownership:
  - tessellation owns sampled mesh output; implicit patch owns field truth
- Routes:
  - `tessellate_surface_patch`, `tessellate_surface_body`
- Reuse/extraction decision:
  - reuse implicit safety validation from surface helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- The adapter should refuse rather than guess bounds or evaluate arbitrary
  expressions.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because implicit tessellation safety and sampling are one
  inseparable adapter boundary.

### Candidate Spec: Loft Patch Family Selection Record

Discovery purpose:
- Add a deterministic record that explains why loft selected ruled, B-spline,
  NURBS, or sweep output for a planned transition.

Responsibilities:
- Functions/methods:
  - loft family classifier
  - family selection diagnostic helper
- Data structures/models:
  - loft family selection record
  - intent evidence record
- Dependencies/services:
  - loft planner/executor
  - topology correspondence records
- Returns/outputs/signals:
  - selected patch family
  - refusal reason for unsupported intent
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft plan and correspondence structures
  - Additions to existing reusable library/module: loft family selection helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes loft plan/executor metadata
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - classifier must be linear in station/transition count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - simple two-station compatible transitions stay ruled unless explicit smooth
    or sweep intent is present
- Test strategy:
  - classifier tests for ruled, B-spline, NURBS, sweep, and unsupported cases
- Data ownership:
  - loft plan owns selection evidence; executor consumes it
- Routes:
  - loft planner to loft executor
- Reuse/extraction decision:
  - add to existing loft planning records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- NURBS selection requires an explicit rational/conic intent signal until
  automatic conic detection is specified.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because classifier and selection diagnostics are one planner
  contract.

### Candidate Spec: Shared Spline Basis And Loft Control-Net Utilities

Discovery purpose:
- Separate reusable B-spline/NURBS basis, knot, and control-net utility work
  from loft producer behavior so later producers do not hide math
  infrastructure.

Responsibilities:
- Functions/methods:
  - basis evaluation helper
  - knot validation helper
  - clamped interpolation control-net helper
- Data structures/models:
  - basis evaluation record
  - knot policy record
  - control-net construction record
- Dependencies/services:
  - `src/impression/modeling/surface.py`
  - `src/impression/modeling/bspline.py`
  - loft executor
- Returns/outputs/signals:
  - validated basis/control-net data
  - numerical validation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: B-spline/NURBS patch records
  - Additions to existing reusable library/module: spline utility helpers
  - New reusable library/module to create: optional `surface_spline.py`
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds reusable numerical utility code
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by degree, knot count, and control-net dimensions
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - new helper module near `surface.py` or `bspline.py`
- Chosen defaults / parameters:
  - clamped knot vectors; finite positive weights for rational callers
- Test strategy:
  - basis partition, knot validation, interpolation shape, and invalid input
    tests
- Data ownership:
  - helper owns numerical basis/control-net construction; patch classes own
    final payload validation
- Routes:
  - loft producers call helper before creating B-spline/NURBS patches
- Reuse/extraction decision:
  - create a reusable helper module because both B-spline and NURBS producers
    need it
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This utility does not decide loft intent; it only provides bounded numerical
  support for specs that already selected a spline family.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 1 x 3 = 3
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one reusable numerical utility boundary shared
  by B-spline and NURBS producers.

### Candidate Spec: Loft B-Spline Output Contract

Discovery purpose:
- Define when smooth loft intent may select B-spline output and what metadata
  the output must carry, without implementing control-net generation.

Responsibilities:
- Functions/methods:
  - B-spline loft intent validator
  - B-spline output contract assertion
- Data structures/models:
  - B-spline loft intent record
  - loft fit metadata
- Dependencies/services:
  - loft family selection record
  - `BSplineSurfacePatch`
- Returns/outputs/signals:
  - B-spline output contract
  - intent refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft plan and B-spline patch record
  - Additions to existing reusable library/module: loft intent/metadata helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds loft selection metadata and tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - contract checks are linear in station/transition count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - B-spline requires explicit smooth intent or a later exact selector; initial
    contract records `interpolation` as the approximation posture
- Test strategy:
  - contract tests for valid smooth intent and refusal of under-defined input
- Data ownership:
  - loft plan owns intent evidence; output metadata owns fit posture
- Routes:
  - family selection record to B-spline output contract
- Reuse/extraction decision:
  - add to existing loft planning records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Actual control-net generation is intentionally split into the B-spline
  producer implementation spec.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one output contract, not the producer algorithm.

### Candidate Spec: Loft B-Spline Control-Net Producer

Discovery purpose:
- Build `BSplineSurfacePatch` payloads for smooth multi-station loft output
  using the shared spline utility boundary.

Responsibilities:
- Functions/methods:
  - smooth loft producer
  - station-to-control-net builder
- Data structures/models:
  - `BSplineSurfacePatch`
  - control net
  - knot vectors
- Dependencies/services:
  - loft executor
  - shared spline utilities
  - topology correspondence records
- Returns/outputs/signals:
  - B-spline `SurfaceBody`
  - fit/refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft plan and B-spline patch record
  - Additions to existing reusable library/module: loft executor producer
  - New reusable library/module to create: none beyond shared spline utility
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes loft output behavior for smooth intent
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by station count, section sample count, degree, and knot policy
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - use clamped interpolation in station and section dimensions
- Test strategy:
  - multi-station smooth loft emits B-spline, preserves metadata, and tessellates
    at boundary
- Data ownership:
  - loft executor owns conversion from correspondence samples to B-spline
    control net
- Routes:
  - B-spline output contract to producer to `BSplineSurfacePatch`
- Reuse/extraction decision:
  - consume shared spline utility; do not duplicate basis logic in loft
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Fitted approximation with tolerance is future work unless explicitly added by
  a later spec.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one producer algorithm after shared spline
  utilities exist.

### Candidate Spec: Loft NURBS Explicit Rational Intent Contract

Discovery purpose:
- Define the explicit authored rational/conic intent required before loft may
  emit NURBS output.

Responsibilities:
- Functions/methods:
  - rational intent validator
  - NURBS output contract assertion
- Data structures/models:
  - rational intent record
  - weight policy record
- Dependencies/services:
  - loft family selection record
  - `NURBSSurfacePatch`
- Returns/outputs/signals:
  - NURBS output contract
  - rational intent refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: NURBS patch record and loft plan
  - Additions to existing reusable library/module: loft intent/metadata helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds loft selection metadata and tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - contract checks are linear in station/transition count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - NURBS requires explicit rational/conic intent; automatic conic detection is
    out of scope
- Test strategy:
  - explicit rational intent accepts; missing or malformed weights refuse
- Data ownership:
  - authored loft input owns rational evidence; loft plan records it
- Routes:
  - family selection record to NURBS output contract
- Reuse/extraction decision:
  - add to existing loft planning records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Automatic conic detection is explicitly future work.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one explicit intent contract.

### Candidate Spec: Loft NURBS Weight-Grid Producer

Discovery purpose:
- Build `NURBSSurfacePatch` payloads from explicit rational loft intent and
  shared spline utilities.

Responsibilities:
- Functions/methods:
  - NURBS loft producer
  - weight-grid builder
- Data structures/models:
  - `NURBSSurfacePatch`
  - weight grid
  - rational control net
- Dependencies/services:
  - loft executor
  - shared spline utilities
  - topology correspondence records
- Returns/outputs/signals:
  - NURBS `SurfaceBody`
  - weight validation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: NURBS patch record and loft plan
  - Additions to existing reusable library/module: loft executor producer
  - New reusable library/module to create: none beyond shared spline utility
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds loft output path for rational intent
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by station count, section sample count, and weight grid size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - use explicit positive finite weights; do not infer conic weights
- Test strategy:
  - rational loft fixture emits NURBS with positive finite weights and
    tessellates at boundary
- Data ownership:
  - loft executor owns weight grid generation from authored rational evidence
- Routes:
  - NURBS output contract to producer to `NURBSSurfacePatch`
- Reuse/extraction decision:
  - consume shared spline utility; do not duplicate basis logic in loft
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Automatic conic detection remains outside this producer.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one producer algorithm after explicit NURBS
  intent exists.

### Candidate Spec: Loft Sweep Surface Producer

Discovery purpose:
- Emit `SweepSurfacePatch` output when authored path/profile intent is the
  primary loft guide.

Responsibilities:
- Functions/methods:
  - sweep loft producer
  - frame policy resolver
- Data structures/models:
  - `SweepSurfacePatch`
  - path/profile reference metadata
  - frame policy record
- Dependencies/services:
  - loft executor
  - `Path3D`
  - sweep surface patch
- Returns/outputs/signals:
  - sweep `SurfaceBody`
  - frame-policy diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `Path3D`, sweep patch record, loft plan
  - Additions to existing reusable library/module: loft executor producer
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds loft output path for path/profile intent
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by path sample count and profile sample count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - default frame policy is `parallel_transport`
- Test strategy:
  - path/profile loft emits sweep and preserves path/profile references
- Data ownership:
  - loft executor owns sweep construction; `SweepSurfacePatch` owns path/profile
    payload
- Routes:
  - family selection record to sweep producer
- Reuse/extraction decision:
  - reuse existing sweep patch and `Path3D`
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Sweep may also become a standalone public primitive; this spec only covers
  loft output.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one producer for one patch family.

### Candidate Spec: Primitive Family Appropriateness Audit

Discovery purpose:
- Confirm primitives use the simplest exact patch family and identify only
  natural advanced-family primitive additions.

Responsibilities:
- Functions/methods:
  - primitive family audit tests
  - missing advanced producer notes
- Data structures/models:
  - primitive-to-family matrix
  - producer gap record
- Dependencies/services:
  - `src/impression/modeling/primitives.py`
  - `src/impression/modeling/_surface_primitives.py`
  - heightmap/displacement builders
- Returns/outputs/signals:
  - primitive family assertions
  - unsupported/missing producer diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current primitive builders
  - Additions to existing reusable library/module: tests and fixture matrix
  - New reusable library/module to create: none by default
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only unless stale primitive output is found
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded primitive smoke fixtures
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests covering `primitives.py`, `_surface_primitives.py`, and heightmap
- Chosen defaults / parameters:
  - primitives use simplest exact family; no advanced family is used merely for
    coverage
- Test strategy:
  - assert box/polyhedron planar, prism ruled, cylinder/cone/sphere/torus
    revolution, heightmap/displacement native
- Data ownership:
  - primitive builders own producer family choice
- Routes:
  - public primitive APIs to surface primitive builders
- Reuse/extraction decision:
  - add audit tests; only add new primitive builders in later specs
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- A future sweep primitive should be separate from this audit because it is a
  new producer, not a correction to existing primitive choices.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is an audit and guardrail, not a family producer
  implementation.

### Candidate Spec: Subdivision Surface Public Authoring Helper

Discovery purpose:
- Provide a bounded public authoring route for `SubdivisionSurfacePatch` cage
  and crease payloads.

Responsibilities:
- Functions/methods:
  - subdivision surface helper
  - cage/crease validation wrapper
- Data structures/models:
  - `SubdivisionSurfacePatch`
  - control cage
  - crease payload
- Dependencies/services:
  - `src/impression/modeling/surface.py`
  - public modeling exports
- Returns/outputs/signals:
  - subdivision `SurfaceBody`
  - validation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `SubdivisionSurfacePatch`
  - Additions to existing reusable library/module: public modeling helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds public authoring API
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bound default subdivision level for helper fixtures
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py` or small public helper module
- Chosen defaults / parameters:
  - default scheme is `catmull_clark`, default subdivision level is bounded
- Test strategy:
  - valid cage returns `SurfaceBody`; invalid cage/crease refuses
- Data ownership:
  - authoring helper owns body wrapping; patch owns cage validation
- Routes:
  - public helper to `SubdivisionSurfacePatch` to `make_surface_body`
- Reuse/extraction decision:
  - add wrapper only; keep refinement logic in existing surface module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Limit-surface evaluation remains patch-family behavior; helper only creates
  authored payloads.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because the helper and validation wrapper are one public
  authoring route.

### Candidate Spec: Implicit Surface Public Authoring Helper And Safety Gate

Discovery purpose:
- Provide a bounded public authoring route for `ImplicitSurfacePatch` that uses
  safe declarative field nodes only.

Responsibilities:
- Functions/methods:
  - implicit surface helper
  - safety validation wrapper
- Data structures/models:
  - `ImplicitSurfacePatch`
  - `ImplicitFieldNode`
  - safety diagnostic
- Dependencies/services:
  - `src/impression/modeling/surface.py`
  - public modeling exports
- Returns/outputs/signals:
  - implicit `SurfaceBody`
  - unsafe field diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: implicit field validation functions
  - Additions to existing reusable library/module: public modeling helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - adds public authoring API
- Security/privacy-sensitive behavior:
  - field nodes must remain declarative and bounded
- Performance-sensitive behavior:
  - helper requires finite bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py` or small public helper module
- Chosen defaults / parameters:
  - only allow declarative field nodes accepted by current safety policy
- Test strategy:
  - safe field returns `SurfaceBody`; unsafe/malformed field refuses
- Data ownership:
  - helper owns body wrapping; patch owns field payload and bounds
- Routes:
  - public helper to safety validator to `ImplicitSurfacePatch`
- Reuse/extraction decision:
  - reuse existing implicit field safety functions
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This spec does not add arbitrary expression evaluation; it exposes only the
  safe node payload contract already in architecture.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because public helper and safety gate are one authoring route.

### Candidate Spec: CSG Unsupported Family Diagnostics

Discovery purpose:
- Ensure CSG family-pair gaps return explicit surface boolean diagnostics
  instead of falling back to mesh execution.

Responsibilities:
- Functions/methods:
  - CSG family support assertion
  - unsupported pair diagnostic helper
- Data structures/models:
  - surface boolean unsupported diagnostic
  - CSG family-pair support matrix
- Dependencies/services:
  - `src/impression/modeling/csg.py`
  - patch family capability matrix
- Returns/outputs/signals:
  - unsupported `SurfaceBooleanResult`
  - no-hidden-mesh-fallback assertion
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG family eligibility diagnostics
  - Additions to existing reusable library/module: targeted CSG tests/helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only unless a missing diagnostic is found
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded family-pair fixture matrix
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests plus `src/impression/modeling/csg.py` if diagnostics are missing
- Chosen defaults / parameters:
  - unsupported CSG pairs return `status=\"unsupported\"` with family-aware
    reason
- Test strategy:
  - unsupported advanced-family pair refuses without mesh output
- Data ownership:
  - CSG owns family-pair support matrix
- Routes:
  - `boolean_union`/`difference`/`intersection` surface backend
- Reuse/extraction decision:
  - reuse existing CSG diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- CSG breadth remains operation-scoped and does not block family availability.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one CSG support/diagnostic boundary.

### Candidate Spec: Loft Unsupported Family Diagnostics

Discovery purpose:
- Ensure unresolved loft topology or family-selection cases refuse explicitly
  rather than generating mesh fallback geometry.

Responsibilities:
- Functions/methods:
  - loft unsupported family diagnostic helper
  - loft refusal assertion helper
- Data structures/models:
  - loft unsupported family diagnostic
  - loft family support record
- Dependencies/services:
  - `src/impression/modeling/loft.py`
  - topology correspondence records
- Returns/outputs/signals:
  - loft planning/execution refusal
  - no-hidden-mesh-fallback assertion
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current loft ambiguity/refusal diagnostics
  - Additions to existing reusable library/module: targeted loft tests/helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only unless a missing diagnostic is found
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded loft fixtures
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests plus `src/impression/modeling/loft.py` if diagnostics are missing
- Chosen defaults / parameters:
  - unresolved family selection refuses before executor mesh/debug paths
- Test strategy:
  - unresolved smooth/rational/sweep intent fixtures refuse with diagnostics
- Data ownership:
  - loft planner owns family support/refusal evidence
- Routes:
  - loft planner to executor
- Reuse/extraction decision:
  - reuse current loft diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Debug mesh output remains allowed only after a canonical surfaced plan exists.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one loft refusal policy boundary.

### Candidate Spec: `.impress` Unsupported Family And Malformed Payload Diagnostics

Discovery purpose:
- Ensure `.impress` rejects unsupported, unknown, or malformed family payloads
  without accepting partial surface truth.

Responsibilities:
- Functions/methods:
  - unsupported family payload assertion
  - malformed family payload assertion
- Data structures/models:
  - `.impress` family diagnostic
  - malformed payload fixture matrix
- Dependencies/services:
  - `src/impression/io/impress.py`
  - all patch family payload codecs
- Returns/outputs/signals:
  - `ImpressFormatError`
  - no mesh-wrapper acceptance assertion
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `.impress` error handling
  - Additions to existing reusable library/module: malformed payload tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only unless missing diagnostics are found
- Security/privacy-sensitive behavior:
  - rejects unsafe implicit and external-reference payloads
- Performance-sensitive behavior:
  - bounded malformed payload fixture matrix
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests plus `src/impression/io/impress.py` if diagnostics are missing
- Chosen defaults / parameters:
  - unknown families and malformed known families raise `ImpressFormatError`
- Test strategy:
  - one unsupported or malformed fixture per codec family
- Data ownership:
  - `.impress` owns persisted payload validation
- Routes:
  - `decode_surface_patch_payload` and document load path
- Reuse/extraction decision:
  - reuse current error type; add helper only if fixtures duplicate heavily
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Unknown future schema versions remain schema-level errors, not family payload
  errors.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one persistence refusal boundary.

### Candidate Spec: Tessellation Unsupported Family Diagnostics

Discovery purpose:
- Ensure tessellation refuses unsupported family states explicitly while keeping
  all mesh generation at the tessellation boundary.

Responsibilities:
- Functions/methods:
  - tessellation unsupported family assertion
  - adapter refusal diagnostic helper
- Data structures/models:
  - tessellation unsupported diagnostic
  - adapter support record
- Dependencies/services:
  - `src/impression/modeling/tessellation.py`
  - all patch family fixtures
- Returns/outputs/signals:
  - tessellation refusal
  - no-hidden-mesh-fallback assertion
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: tessellation adapter framework
  - Additions to existing reusable library/module: adapter refusal tests/helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only unless missing diagnostics are found
- Security/privacy-sensitive behavior:
  - unsafe implicit tessellation states refuse
- Performance-sensitive behavior:
  - bounded tessellation fixtures
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests plus `src/impression/modeling/tessellation.py` if diagnostics are
    missing
- Chosen defaults / parameters:
  - unsupported adapter states raise or return family-aware diagnostics before
    mesh creation
- Test strategy:
  - unsafe implicit and malformed approximation fixtures refuse explicitly
- Data ownership:
  - tessellation owns adapter support/refusal truth
- Routes:
  - `tessellate_surface_patch` and `tessellate_surface_body`
- Reuse/extraction decision:
  - reuse adapter framework diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Approximate adapters are supported if approximation metadata is explicit;
  lack of exactness is not itself a refusal.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one tessellation refusal boundary.

### Candidate Spec: Advanced Patch Family Public API Documentation

Discovery purpose:
- Document the public authoring routes and examples for advanced patch families
  once they are integrated, so users do not rediscover raw patch payload rules.

Responsibilities:
- Functions/methods:
  - documentation examples
  - API inventory check
- Data structures/models:
  - public advanced-family API list
  - example fixture matrix
- Dependencies/services:
  - docs/modeling
  - public modeling exports
- Returns/outputs/signals:
  - updated docs
  - documentation coverage test
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current modeling docs
  - Additions to existing reusable library/module: docs tests
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - documentation updates
- Security/privacy-sensitive behavior:
  - implicit docs must state declarative safe-node-only policy
- Performance-sensitive behavior:
  - examples use bounded fixtures
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `docs/modeling/` and documentation tests
- Chosen defaults / parameters:
  - examples show surface-native outputs and explicit tessellation boundaries
- Test strategy:
  - docs coverage tests mention each public advanced-family route
- Data ownership:
  - docs own user-facing API guidance; code owns behavior
- Routes:
  - docs index to modeling patch-family page or section
- Reuse/extraction decision:
  - add to existing modeling docs rather than creating a separate process doc
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Documentation should land after API names are stable, but before release.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Keep together because this is one docs/API coverage pass.

## Open Decisions

- Whether B-spline loft output should be exact interpolation of stations or a
  fit with explicit tolerance metadata.
- Whether NURBS output is selected only by explicit rational intent or also by
  automatic conic detection.
- Whether sweep should be a loft output mode, a separate public primitive, or
  both.
- Whether subdivision and implicit families need first-class public authoring
  APIs in this release or can be available through direct patch construction
  plus `.impress`.
- How broad CSG support must be before a family is considered globally
  available.

## Change History

- 2026-05-26: Updated the manifest after critical review for hidden work. Split
  loft spline and NURBS production, split unsupported-consumer diagnostics,
  added shared spline utilities, round-trip identity coverage, availability
  promotion, and public API documentation candidates, and rescored all
  candidates.
- 2026-05-26: Re-reviewed the manifest for remaining hidden work. Split
  SurfaceComposition traversal from transform/identity preservation, and split
  tessellation coverage into inventory, sampled, spline, subdivision, and
  implicit adapter candidates.
- 2026-05-26: Added a scored Specification Manifest for Discovery covering
  availability gates, `.impress` codecs, all-family storage/traversal,
  tessellation, loft producers, primitive audit, public advanced-family
  authoring helpers, and unsupported-consumer diagnostics.
- 2026-05-26: Initial architecture document. Created to define full patch
  family integration across runtime storage, `.impress`, loft, primitives,
  tessellation, diagnostics, and capability gating.
