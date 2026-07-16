# Single-Shell Loft CSG Operation Route Architectural Change Document

Date: 2026-07-15
Status: Proposed
Canonical architecture targets:

- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/surfacebody-csg-architecture.md`
- `project/release-0.1.0a/architecture/surface-csg-trim-fragment-reconstruction-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Release / plan / issue: `project/release-0.1.0a/planning/reference-test-expansion-plan.md`
- Parent ACD, if any:
  `project/release-0.1.0a/architecture/acd-loft-shell-connectivity-and-closure-evidence.md`

## Change Intent

Define the successful single-shell loft CSG operation route that downstream
provenance, color, and section-evidence work can rely on.

This ACD exists because earlier manifests named "successful loft CSG result
geometry" as a prerequisite without defining the architectural work that
produces that geometry. Specification manifests must not leave that as a
readiness blocker. The route is therefore captured here as explicit manifest
work.

## Current Architecture

Canonical architecture defines a CSG eligibility gate and expects loft-authored
`SurfaceBody` operands to use surface-native CSG. The active loft shell ACD
defines the connectivity and closure evidence needed for a loft body to reach
the gate.

The missing architecture is the operation route after an eligible single-shell
loft reaches CSG:

- how CSG selects a loft-compatible surface operation route
- how loft patches participate in fragment classification and shell assembly
- how the route produces a returned `SurfaceBody`, not a mesh fallback
- how reference tests prove that the returned geometry is usable by later
  provenance, color, and section evidence work

## Target Architecture

Single-shell loft CSG uses the existing surface-native CSG pipeline with a
loft-aware route selector and result proof.

The target architecture has these components:

- `LoftCSGOperationRouteRecord`: records selected operation, operand roles,
  accepted loft eligibility evidence, and selected solver path.
- `LoftPatchFragmentParticipationRecord`: records which loft side/cap patches
  enter trim classification and fragment reconstruction.
- `LoftCSGResultGeometryRecord`: records returned surface-body shell count,
  validity summary, fragment count, and no-hidden-mesh-fallback proof.
- `LoftCSGReferenceGeometryProof`: records the fixture-facing evidence that a
  successful route produced usable dirty STL/reference artifacts from the
  returned `SurfaceBody`.

The first conforming route is single-shell loft against already-supported
analytic primitive operands. Branching lofts, loft/loft decomposition, and
section evidence are separate ACD responsibilities.

## Non-Goals

- Implementing branch decomposition or recomposition.
- Implementing loft/loft CSG beyond the first single-shell route.
- Changing the shell connectivity and closure evidence contract.
- Rasterizing or mesh-executing the boolean operation.

## Canonical Document Impact

- Architecture docs to update on closure:
  - `lofted-body-csg-reference-architecture.md` - replace prerequisite wording
    with the conformed single-shell loft CSG operation route.
  - `surfacebody-csg-architecture.md` - add loft-authored operand routing as a
    surface-native CSG consumer.
  - `surface-csg-trim-fragment-reconstruction-architecture.md` - align loft
    patch fragment participation with existing trim/fragment reconstruction.
- Specs or plans affected:
  - Surface Specs 393-396 - derive executable single-shell loft CSG route specs
    from this ACD after shell evidence specs exist.
  - Surface Spec 399 - may consume `LoftCSGResultGeometryRecord` without
    declaring its own operation-route prerequisite.
  - Surface Spec 400 - may consume `LoftCSGReferenceGeometryProof` for section
    artifact generation.

## Compatibility And Migration Strategy

- Existing invalid or incomplete loft operands continue to refuse through the
  CSG eligibility gate.
- Eligible single-shell loft operands use surface-native CSG only; no mesh lane
  or hidden tessellation fallback is introduced.
- The first implementation may support a bounded primitive-pair subset as long
  as the route record refuses unsupported pairings with structured diagnostics.
- Dirty reference artifacts are evidence of route execution, not canonical
  completion until promoted through the reference artifact lifecycle.

## Application Integration Contract

- App type: library-only
- User/caller surface: public boolean API consumers calling loft/primitive CSG
- Invocation route: public boolean API call to CSG eligibility, route
  selection, surface-native fragment reconstruction, and returned `SurfaceBody`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: eligible single-shell loft and analytic primitive operands
  return valid surface-native CSG result geometry with no hidden mesh fallback
- Integration validation: focused public boolean API tests plus generated dirty
  STL/reference proof from the returned `SurfaceBody`

## Specification Manifest for Discovery

### Manifest Rule

Specification manifests and final specifications must not finish with open
readiness blockers. Sequencing dependencies are captured as predecessor
candidates or ACD references, and the predecessor work is itself represented by
an executable manifest candidate.

### Five-Pass Manifest Review Notes - 2026-07-15

- Pass 1: Rechecked the new ACD against downstream provenance, color, and
  section-evidence ACDs; this ACD owns only the successful single-shell route
  and reference proof.
- Pass 2: Confirmed route selection, route execution, and reference proof are
  independently failing leaves and should remain split.
- Pass 3: Rescored all three leaves against the active template; all remain
  below 25 and only reference proof is in the 16-24 split-review band.
- Pass 4: Added readiness blocker resolution records so predecessor sequencing
  is explicit and resolved.
- Pass 5: Final review found no further split needed; downstream ACDs can point
  to this ACD as the explicit route-geometry resolution artifact.

### Candidate Spec: Loft CSG Operation Route Selection

Discovery purpose:
- Select a surface-native CSG route for eligible single-shell loft operands.

Responsibilities:
- Functions/methods:
  - loft CSG route selector
  - unsupported loft pairing diagnostic builder
- Data structures/models:
  - `LoftCSGOperationRouteRecord`
- Dependencies/services:
  - loft CSG eligibility evidence
  - surface CSG route registry
- Returns/outputs/signals:
  - selected route record
  - unsupported pairing diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface CSG route registry
  - Additions to existing reusable library/module: loft route selector
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
  - bounded by operand patch and route count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers using single-shell loft operands
- Invocation route:
  - public boolean API call to route selector after CSG eligibility acceptance
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - supported loft/primitive pairings select a surface-native route; unsupported
    pairings refuse with structured diagnostics
- Integration validation:
  - public boolean API route-selection tests
- Incomplete status risk:
  - implemented in isolation if route records are not used by CSG execution
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - only eligible single-shell lofts may select an execution route
- Test strategy:
  - route-selection unit tests and public API refusal tests
- Data ownership:
  - CSG owns route records; loft owns eligibility evidence
- Routes:
  - eligibility gate to CSG route selector
- Reuse/extraction decision:
  - add to existing CSG module
- UI field/control inventory:
  - not applicable

Predecessor candidates:
- `Loft Boundary Graph And Seam Coverage Evidence`
- `Loft Closure And Cap Validity Evidence`

Open questions / nuance discovered:
- The first supported operand pair should be the smallest analytic primitive
  pairing that proves the route without adding branching behavior.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 12.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; sequencing is captured by predecessor candidates in
    `acd-loft-shell-connectivity-and-closure-evidence.md`
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidates `Loft Boundary Graph And Seam Coverage Evidence` and
    `Loft Closure And Cap Validity Evidence`
- Resolution status:
  - resolved

Split decision:
- Small.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Single-Shell Loft Primitive CSG Execution

Discovery purpose:
- Execute the first surface-native CSG route for an eligible single-shell loft
  and analytic primitive operand pair.

Responsibilities:
- Functions/methods:
  - loft primitive CSG executor
  - loft patch fragment classifier adapter
- Data structures/models:
  - `LoftPatchFragmentParticipationRecord`
  - `LoftCSGResultGeometryRecord`
- Dependencies/services:
  - loft CSG operation route record
  - surface CSG trim fragment reconstruction
  - shell assembly validator
- Returns/outputs/signals:
  - returned `SurfaceBody`
  - route execution diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface CSG trim/fragment reconstruction
  - Additions to existing reusable library/module: loft patch classification
    adapter
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
  - bounded by loft patch, primitive patch, and fragment counts
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public boolean API consumers using supported loft/primitive pairings
- Invocation route:
  - selected loft CSG route to surface-native CSG execution
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - public boolean API returns a valid `SurfaceBody` for the supported
    loft/primitive pairing
- Integration validation:
  - public boolean API tests covering union, difference, and intersection for
    the first supported loft/primitive case
- Incomplete status risk:
  - implemented in isolation if executor bypasses the public boolean API or
    returns tessellated mesh data
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - no hidden mesh fallback; unsupported pairings refuse before execution
- Test strategy:
  - operation tests plus no-hidden-mesh-fallback assertions
- Data ownership:
  - CSG owns result geometry; loft owns source patch role metadata
- Routes:
  - selected route record to CSG executor to returned `SurfaceBody`
- Reuse/extraction decision:
  - add adapter to existing surface CSG execution path
- UI field/control inventory:
  - not applicable

Predecessor candidates:
- `Loft CSG Operation Route Selection`

Open questions / nuance discovered:
- The route must preserve enough fragment identity for later provenance/color
  work, but provenance resolution itself belongs to the provenance ACD.

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
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 14.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; route selection is captured by predecessor candidate `Loft CSG
    Operation Route Selection`
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `Loft CSG Operation Route Selection`
- Resolution status:
  - resolved

Split decision:
- Small.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Loft CSG Result Geometry Reference Proof

Discovery purpose:
- Prove that the single-shell loft CSG route produces reusable result geometry
  for reference artifacts and downstream evidence work.

Responsibilities:
- Functions/methods:
  - loft CSG result proof builder
  - dirty STL/reference artifact signal checker
- Data structures/models:
  - `LoftCSGReferenceGeometryProof`
- Dependencies/services:
  - returned CSG `SurfaceBody`
  - reference artifact lifecycle
  - STL tessellation boundary
- Returns/outputs/signals:
  - result geometry proof record
  - dirty reference artifact path
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reference artifact lifecycle helpers
  - Additions to existing reusable library/module: loft CSG proof helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty reference artifacts during reference update workflow
- Security/privacy-sensitive behavior:
  - validates generated paths under dirty reference roots
- Performance-sensitive behavior:
  - bounded by tessellation output size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - workflow
- User/caller surface:
  - reference update workflow for loft CSG fixtures
- Invocation route:
  - returned `SurfaceBody` to reference artifact generation
- Wiring owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
  - `tests/reference_images.py`
- Observable result:
  - dirty STL/reference artifacts exist for the supported loft CSG route and
    identify the source result geometry
- Integration validation:
  - reference update test proving dirty artifact generation from the returned
    surface body
- Incomplete status risk:
  - implemented in isolation if proof records are generated from synthetic data
    instead of the public CSG result
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
  - `tests/reference_images.py`
- Chosen defaults / parameters:
  - dirty artifacts are proof inputs, not promoted gold completion evidence
- Test strategy:
  - reference artifact generation test with no-hidden-mesh-fallback assertion
- Data ownership:
  - CSG result owns geometry; reference workflow owns dirty artifacts
- Routes:
  - public boolean API result to dirty reference artifact
- Reuse/extraction decision:
  - add to existing reference fixture generation helpers
- UI field/control inventory:
  - not applicable

Predecessor candidates:
- `Single-Shell Loft Primitive CSG Execution`

Open questions / nuance discovered:
- Section evidence can consume this proof after it exists; section artifact
  generation remains owned by the multi-artifact evidence ACD.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
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
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; execution is captured by predecessor candidate `Single-Shell Loft
    Primitive CSG Execution`
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `Single-Shell Loft Primitive CSG Execution`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: proof construction, dirty artifact signal
  checking, and path reporting are one reference-generation route.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

## Specification Conformance

- Parent specs created or affected:
  - Surface Specs 393-396 - coverage pending spec promotion from this ACD.
- Canonical child specs:
  - `../specifications/surface-405-loft-csg-operation-route-selection-v1_0.md` - canonical child from `Loft CSG Operation Route Selection`.
  - `../specifications/surface-406-single-shell-loft-primitive-csg-execution-v1_0.md` - canonical child from `Single-Shell Loft Primitive CSG Execution`.
  - `../specifications/surface-407-loft-csg-result-geometry-reference-proof-v1_0.md` - canonical child from `Loft CSG Result Geometry Reference Proof`.
- Paired test specs:
  - none yet.

## Conformance Checklist

- [ ] Implementation conforms to the target architecture.
- [ ] Parent specs are 100% represented by canonical child specs.
- [ ] Superseded parent specs are archived.
- [ ] Canonical child specs point to architecture or active ACD as primary ancestor.
- [ ] Paired test specs point to canonical child specs.
- [ ] Progression and indexes point to canonical child specs.
- [ ] Completed manifests are removed from active canonical architecture docs.
- [ ] Canonical architecture docs describe the conformed architecture.

## Closure Criteria

- Final specs and paired test specs exist for the route selection, execution,
  and reference proof candidates.
- At least one supported single-shell loft/primitive pairing returns a valid
  surface-native `SurfaceBody` through the public boolean API.
- Dirty reference proof is generated from the returned `SurfaceBody`.
- Provenance/color and section-evidence ACDs refer to this route as an existing
  predecessor with resolved sequencing.

## Closure Notes

- Canonical architecture updated:
  - `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- Archived or removed scaffolding:
  - none yet.
- Follow-up ACDs:
  - `project/release-0.1.0a/architecture/acd-loft-primitive-trim-fragment-execution-adapter.md` - created after implementation of Surface Spec 406 exposed that exact reuse execution is architected separately from intersecting trim-fragment cut execution.

## Change History

- 2026-07-15 - Initial draft. Reason: successful loft CSG result geometry is
  represented by explicit architecture/manifest work rather than as a readiness
  blocker in downstream manifests.
- 2026-07-15 - Added follow-up ACD link for loft primitive trim-fragment
  execution adapter. Reason: Surface Spec 406 needs split/revision before
  intersecting cut-producing execution can be truthfully checked complete.
