# Loft Primitive Trim-Fragment Execution Adapter Architectural Change Document

Date: 2026-07-15
Status: Manifesting
Canonical architecture targets:

- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/surface-csg-trim-fragment-reconstruction-architecture.md`
- `project/release-0.1.0a/architecture/surfacebody-csg-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Parent ACD, if any:
  `project/release-0.1.0a/architecture/acd-single-shell-loft-csg-operation-route.md`
- Triggering spec:
  `project/release-0.1.0a/specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md`

## Change Intent

Define the missing architecture between an eligible single-shell loft/primitive
route selection and a full trim-fragment CSG execution result.

This ACD exists because Surface Spec 406 names "surface CSG trim/fragment
reconstruction" as reusable infrastructure, but it does not define the adapter
that turns loft side/cap patches, station seams, and cap trim loops into the
fragment-classification inputs required by that infrastructure.

The exact no-cut and containment execution subset can return valid existing
`SurfaceBody` geometry. It does not satisfy the full cut-producing route where a
primitive actually intersects loft side or cap patches.

## Current Architecture

The current architecture has:

- loft-owned boundary, seam coverage, closure, and cap validity evidence
- loft-aware CSG route selection
- general surface CSG trim/fragment reconstruction architecture
- exact/no-cut Boolean result helpers that can reuse existing surface bodies

The missing architecture is the loft-specific adapter for cutting cases:

- mapping loft sidewall parameter boundaries to CSG trim curves
- mapping cap trim loops and station seams into fragment ownership records
- classifying loft fragments as survive, discard, or cut-cap contributors
- assembling a valid result shell from loft fragments plus generated primitive
  cut caps
- preserving enough fragment identity for later provenance/color work

## Target Architecture

Loft/primitive CSG execution is split into two execution families:

- Exact reuse execution: no-cut or containment cases where the returned
  geometry is exactly one existing input body or an exact disjoint union.
- Trim-fragment execution: intersecting cases where loft patches are adapted to
  the surface CSG trim/fragment pipeline and reassembled as a new result shell.

The trim-fragment execution architecture adds these components:

- `LoftPrimitiveTrimAdapterRecord`: maps one loft patch and primitive operand
  into trim curves and patch-local curve ownership.
- `LoftPrimitiveFragmentClassificationRecord`: records survive/discard/cut-cap
  decisions for loft side and cap fragments.
- `LoftPrimitiveCutShellAssemblyRecord`: records result-shell assembly,
  generated cap participation, seam rebuild inputs, and validity diagnostics.
- `LoftPrimitiveExecutionScopeRecord`: records whether an operation used exact
  reuse, trim-fragment execution, or refused before execution.

Surface Spec 406 should not be closed by exact reuse alone. Either it must be
split into the leaves in this ACD, or it must be updated so its acceptance
criteria explicitly distinguish exact reuse from full cut-producing execution.

## Non-Goals

- Branching loft decomposition and recomposition.
- Loft/loft CSG execution.
- Provenance/color ownership resolution after result assembly.
- Section artifact generation and review-fixture bundle display.
- Mesh fallback, tessellation-as-execution, or rasterized proof.

## Canonical Document Impact

- Architecture docs to update on closure:
  - `lofted-body-csg-reference-architecture.md` - distinguish exact reuse
    execution from trim-fragment cut execution.
  - `surface-csg-trim-fragment-reconstruction-architecture.md` - add the loft
    patch adapter contract and fragment classifier inputs.
  - `surfacebody-csg-architecture.md` - document loft/primitive execution as a
    surface-native route with explicit refusal for unadapted cut cases.
- Specs or plans affected:
  - Surface Spec 406 - must be split or revised before it is checked complete.
  - Surface Spec 407 - should consume only completed execution-family proof.
  - Surface Specs 411-412 - should consume fragment identity after this ACD
    resolves the adapter path.

## Readiness Blocker Resolution

- Blocker being resolved:
  - Surface Spec 406 is broader than the implementation and architecture can
    truthfully support because exact reuse execution and intersecting
    trim-fragment cut execution are different routes.
- Source artifact:
  - `project/release-0.1.0a/specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md`
- Resolution provided by this ACD:
  - Split the broad execution route into exact reuse execution, loft
    trim-fragment adapter, and cut-shell assembly/validity manifest candidates.
- Follow-on artifact:
  - Final canonical specifications derived from this ACD's manifest candidates.
- Resolution status:
  - resolved at architecture/manifest level; pending final specification
    promotion.

## Compatibility And Migration Strategy

- Exact reuse loft/primitive results may remain implemented and tested as a
  bounded execution subset.
- Intersecting cut cases must refuse with structured diagnostics until the
  trim-fragment adapter and cut-shell assembly leaves are complete.
- Existing primitive, B-spline/NURBS, sweep/subdivision, and ruled-affine-box
  routes must keep their current dispatch precedence unless an executor-authored
  loft boundary graph is present.
- No hidden mesh fallback is introduced.

## Application Integration Contract

- App type: library-only
- User/caller surface: public boolean API consumers calling loft/primitive CSG
- Invocation route: public boolean API to eligibility, route selection, exact
  reuse or trim-fragment execution, result validity gate
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: exact reuse cases return valid `SurfaceBody` results;
  intersecting cut cases return valid new `SurfaceBody` results only after the
  adapter and assembly leaves are implemented
- Integration validation: public boolean API tests for exact reuse, refusal for
  unimplemented cut cases, then cut-producing union/difference/intersection
  tests after trim-fragment execution lands

## Specification Manifest for Discovery

### Manifest Rule

Specification manifests and final specifications must not finish with open
readiness blockers. Sequencing dependencies are represented by executable
manifest candidates in this ACD.

### Five-Pass Manifest Review Notes - 2026-07-15

- Pass 1 - Template completion: added the missing ACD-level readiness blocker
  resolution and verified each manifest candidate has project readiness fields.
- Pass 2 - Prerequisite review: converted predecessor notes into explicit
  prerequisite handling; no unarchitected prerequisite remains.
- Pass 3 - Rescore: corrected performance scoring to the shared policy and
  added readiness-blocker and missing-prerequisite score lines.
- Pass 4 - Split review: confirmed Surface Spec 406 should split/supersede into
  three implementation leaves; exact reuse stays separate from cut-producing
  trim-fragment execution.
- Pass 5 - Final blocker audit: no candidate has unresolved readiness blockers;
  next action is final spec promotion and progression rewiring before further
  implementation.

### Candidate Spec: Loft Primitive Exact Reuse Execution

Discovery purpose:
- Preserve the exact no-cut/containment loft/primitive execution subset without
  pretending it completes cut-producing execution.

Responsibilities:
- Functions/methods:
  - exact loft/primitive execution selector
  - exact result metadata attachment
- Data structures/models:
  - `LoftPrimitiveExecutionScopeRecord`
  - `LoftCSGResultGeometryRecord`
- Dependencies/services:
  - loft route selection
  - exact body containment/no-cut relation helpers
- Returns/outputs/signals:
  - public boolean API `SurfaceBooleanResult`
  - no-hidden-mesh-fallback metadata
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: exact body result finalizer
  - Additions to existing reusable library/module: exact loft execution helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by operand count and body metadata
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers using loft/primitive operands
- Invocation route:
  - selected loft route to exact reuse execution
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - public boolean API returns exact existing-body results for no-cut and
    containment cases
- Integration validation:
  - public boolean API union, difference, and intersection tests for exact
    reuse cases
- Incomplete status risk:
  - complete only for exact reuse; not a substitute for trim-fragment cuts
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - refuse intersecting cut cases until trim-fragment execution is implemented
- Test strategy:
  - focused public API exact-result tests
- Data ownership:
  - CSG owns execution scope and result records
- Routes:
  - public boolean API to exact result finalizer
- Reuse/extraction decision:
  - reuse current exact result helpers
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Linked existing prerequisite: `Loft CSG Operation Route Selection` /
    `project/release-0.1.0a/specifications/surface-405-loft-csg-operation-route-selection-v1_0.md`

Predecessor candidates:
- `Loft CSG Operation Route Selection`

Open questions / nuance discovered:
- This leaf can be considered implemented by the current exact no-cut subset
  only after a final spec is created and progression points at it.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Additions to existing reusable modules: 1 x 1 = 1
- New reusable modules: 0 x 2 = 0
- Database work: 0 x 2 = 0
- Async/concurrency: 0 x 2 = 0
- Destructive/write behavior: 0 x 1 = 0
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Total: 13.5

Readiness blockers:
- none

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final specification from this manifest candidate
- Resolution artifact:
  - this ACD manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked existing prerequisite; prerequisite implementation must remain
    sequenced before this candidate
  - Artifact: `project/release-0.1.0a/specifications/surface-405-loft-csg-operation-route-selection-v1_0.md`

Split decision:
- Small. This remains a single exact-reuse leaf.

Manifest cleanup:
- Parent manifest candidate, if split: `Single-Shell Loft Primitive CSG Execution`
- Child manifest candidates:
  - `Loft Primitive Trim-Fragment Adapter`
  - `Loft Primitive Cut Shell Assembly And Validity`
- Parent candidate responsibilities still missing from children:
  - cut-producing trim-fragment execution
- Removal readiness: ready after final specs replace or supersede Surface Spec
  406.

### Candidate Spec: Loft Primitive Trim-Fragment Adapter

Discovery purpose:
- Adapt loft side/cap patches into the existing surface CSG trim/fragment
  reconstruction pipeline.

Responsibilities:
- Functions/methods:
  - loft patch trim adapter
  - primitive intersection request builder for loft patches
  - patch-local trim curve mapper
- Data structures/models:
  - `LoftPrimitiveTrimAdapterRecord`
  - `LoftPrimitiveFragmentClassificationRecord`
- Dependencies/services:
  - surface intersection request normalization
  - CSG curve mapping
  - loft boundary and cap evidence
- Returns/outputs/signals:
  - trim adapter records
  - fragment classification records
  - structured refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface intersection requests and curve
    mapping
  - Additions to existing reusable library/module: loft patch adapter helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by loft patch and primitive patch count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers using intersecting loft/primitive operands
- Invocation route:
  - selected loft route to trim-fragment adapter
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - adapter emits trim and fragment records or refuses before execution
- Integration validation:
  - public API tests for intersecting loft/box adapter records and refusal
    diagnostics
- Incomplete status risk:
  - helper-only implementation if records are not consumed by cut-shell assembly
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - no mesh fallback; unsupported patch/primitive intersections refuse
- Test strategy:
  - adapter unit tests plus public API refusal/adapter evidence tests
- Data ownership:
  - CSG owns adapter records; loft owns source patch role metadata
- Routes:
  - route selector to adapter to cut-shell assembly
- Reuse/extraction decision:
  - add adapter to existing CSG module
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Predecessor manifest candidate in this ACD: `Loft Primitive Exact Reuse
    Execution`

Predecessor candidates:
- `Loft Primitive Exact Reuse Execution`

Open questions / nuance discovered:
- Cap trim-loop orientation and station seam ownership must be preserved so
  later provenance/color work can distinguish side fragments from cap fragments.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Additions to existing reusable modules: 1 x 1 = 1
- New reusable modules: 0 x 2 = 0
- Database work: 0 x 2 = 0
- Async/concurrency: 0 x 2 = 0
- Destructive/write behavior: 0 x 1 = 0
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Total: 18

Readiness blockers:
- none

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final specification from this manifest candidate
- Resolution artifact:
  - this ACD manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked predecessor manifest candidate; exact reuse execution should be
    sequenced before adapter implementation
  - Artifact: `Loft Primitive Exact Reuse Execution`

Split decision:
- Review for split. It remains one leaf because the adapter record is not useful
  unless it also classifies the resulting loft fragments. Splitting trim
  creation from fragment classification would leave both leaves without a public
  route-level verification surface.

Manifest cleanup:
- Parent manifest candidate, if split: `Single-Shell Loft Primitive CSG Execution`
- Child manifest candidates:
  - `Loft Primitive Cut Shell Assembly And Validity`
- Parent candidate responsibilities still missing from children:
  - result shell assembly and validity
- Removal readiness: ready after final specs replace or supersede Surface Spec
  406.

### Candidate Spec: Loft Primitive Cut Shell Assembly And Validity

Discovery purpose:
- Assemble cut-producing loft/primitive results from classified loft fragments
  and generated primitive cap fragments.

Responsibilities:
- Functions/methods:
  - loft fragment shell assembler
  - generated cap seam rebuilder
  - result validity gate handoff
- Data structures/models:
  - `LoftPrimitiveCutShellAssemblyRecord`
  - `LoftCSGResultGeometryRecord`
- Dependencies/services:
  - trim-fragment adapter records
  - surface shell assembly helpers
  - CSG validity gate
- Returns/outputs/signals:
  - returned `SurfaceBody`
  - assembly diagnostics
  - no-hidden-mesh proof
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: shell assembly and validity gate helpers
  - Additions to existing reusable library/module: loft cut-shell assembler
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by classified fragment and seam counts
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers using cut-producing loft/primitive operands
- Invocation route:
  - selected loft route to trim adapter to cut-shell assembler to result
    validity gate
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - public boolean API returns a valid new `SurfaceBody` for intersecting
    loft/primitive union, difference, and intersection cases
- Integration validation:
  - public boolean API tests for cut-producing union, difference, and
    intersection
- Incomplete status risk:
  - incomplete if shell assembly exists but is not reachable from the public API
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - invalid result geometry refuses with diagnostics; no fallback
- Test strategy:
  - public API cut-producing operation tests and validity-gate assertions
- Data ownership:
  - CSG owns assembled result and diagnostics
- Routes:
  - adapter records to shell assembler to returned result body
- Reuse/extraction decision:
  - reuse existing shell assembly and validity gate
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Predecessor manifest candidate in this ACD: `Loft Primitive Trim-Fragment
    Adapter`

Predecessor candidates:
- `Loft Primitive Trim-Fragment Adapter`

Open questions / nuance discovered:
- This leaf is the point where Surface Spec 406's current acceptance criteria
  become true for intersecting cuts.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Additions to existing reusable modules: 1 x 1 = 1
- New reusable modules: 0 x 2 = 0
- Database work: 0 x 2 = 0
- Async/concurrency: 0 x 2 = 0
- Destructive/write behavior: 0 x 1 = 0
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Total: 18

Readiness blockers:
- none

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final specification from this manifest candidate
- Resolution artifact:
  - this ACD manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked predecessor manifest candidate; trim-fragment adapter implementation
    must be sequenced before cut-shell assembly
  - Artifact: `Loft Primitive Trim-Fragment Adapter`

Split decision:
- Review for split. It remains one leaf because assembly and validity handoff
  are one public API completion route. Splitting shell assembly from the
  validity gate would permit helper-only completion without proving the returned
  body through the public boolean API.

Manifest cleanup:
- Parent manifest candidate, if split: `Single-Shell Loft Primitive CSG Execution`
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none after this leaf lands
- Removal readiness: ready after final specs replace or supersede Surface Spec
  406.

## Specification Conformance

- Parent specs created or affected:
  - `../specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md` - split/supersession represented by Surface Specs 420-422; current parent should remain out of active implementation sequencing.
- Canonical child specs:
  - `../specifications/surface-420-loft-primitive-exact-reuse-execution-v1_0.md` - canonical child from `Loft Primitive Exact Reuse Execution`.
  - `../specifications/surface-421-loft-primitive-trim-fragment-adapter-v1_0.md` - canonical child from `Loft Primitive Trim-Fragment Adapter`.
  - `../specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md` - canonical child from `Loft Primitive Cut Shell Assembly And Validity`.
- Paired test specs:
  - none yet.

## Conformance Checklist

- [ ] Implementation conforms to the target architecture.
- [x] Surface Spec 406 is split, revised, or superseded by final canonical
  children.
- [x] Canonical child specs point to this ACD or updated canonical architecture.
- [ ] Paired test specs point to canonical child specs.
- [x] Progression and indexes point to canonical specs.
- [ ] Completed manifests are removed from active canonical architecture docs.
- [ ] Canonical architecture docs describe the conformed architecture.

## Closure Criteria

- Exact reuse execution is represented by a final spec or by a revised 406
  scope.
- Loft primitive trim-fragment adapter records are implemented and validated.
- Cut-producing public boolean API union, difference, and intersection return
  valid surface-native `SurfaceBody` results without mesh fallback.
- Surface Spec 406 is no longer broader than its implemented route.

## Closure Notes

- Canonical architecture updated:
  - none yet.
- Archived or removed scaffolding:
  - none yet.
- Follow-up ACDs:
  - none.

## Change History

- 2026-07-15 - Initial draft. Reason: implementation of Surface Spec 406
  exposed that exact reuse execution is specified, but the intersecting
  loft/primitive trim-fragment adapter and cut-shell assembly architecture was
  incomplete.
- 2026-07-15 - Five-pass manifest review, update, rescore, and split check.
  Reason: confirm Surface Spec 406 is covered by three smaller candidates with
  no unresolved readiness blockers before final spec promotion.
