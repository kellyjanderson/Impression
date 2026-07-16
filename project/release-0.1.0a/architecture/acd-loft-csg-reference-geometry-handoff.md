# Loft CSG Reference Geometry Handoff ACD

Date: 2026-07-16
Status: Manifesting
Canonical architecture targets:

- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/reference-csg-gap-closure-architecture.md`
- `project/release-0.1.0a/architecture/reference-artifact-promotion-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Parent ACD:
  `project/release-0.1.0a/architecture/acd-loft-primitive-cut-shell-geometric-kernel.md`
- Predecessor ACD:
  `project/release-0.1.0a/architecture/acd-loft-primitive-seam-shell-validity-execution.md`

## Change Intent

Define the architecture boundary between accepted loft CSG result geometry and
reference workflow generation.

This ACD exists because reference proof specs should not be blocked by vague
"CSG not ready" notes. They should declare exactly what accepted public result
geometry they require, how fixture source records consume it, and what they
must refuse when only adapter, synthetic, or unsupported payloads are present.

## Current Architecture

The reference workflow can generate and register many dirty STL artifacts.
Loft CSG fixture specs exist for result-geometry proof and section evidence,
but they cannot honestly complete until public loft CSG routes return accepted
surface-native result geometry.

The current architecture does not explicitly define:

- the accepted-result gate for loft CSG reference generation
- the prohibition on adapter-only or synthetic geometry as reference evidence
- the handoff from public result metadata to STL fixture source records
- the handoff from public result geometry to section evidence readiness

## Target Architecture

The target architecture introduces a reference handoff boundary:

- `LoftCsgReferenceGeometryHandoffRecord`: operation id, fixture id, source
  `.impress` path, accepted `SurfaceBooleanResult`, result body identity,
  operation metadata, and refusal diagnostics.
- `LoftCsgSectionEvidenceReadinessRecord`: accepted result body, section plane
  declaration, section artifact bundle readiness, and registry payload.
- `build_loft_csg_reference_geometry_handoff(...)`
- `build_loft_csg_section_evidence_handoff(...)`

Reference workflows may generate STL or section artifacts only from accepted
public results whose body is non-null and surface-native. Adapter-only
diagnostics, sampled preview geometry, tessellation buffers, and synthetic
payloads must produce structured readiness refusal instead of dirty reference
artifacts.

## Non-Goals

- Implementing the loft/primitive cut-shell kernel.
- Defining section record internals already covered by section evidence specs.
- Changing review UI status, notes, or fixture approval behavior.
- Writing reference artifacts from this ACD directly.

## Canonical Document Impact

- `lofted-body-csg-reference-architecture.md` should describe reference proof
  as downstream of accepted public CSG result geometry.
- `reference-csg-gap-closure-architecture.md` should identify loft result
  handoff as the gate for Surface Specs 407 and 418.
- `reference-artifact-promotion-architecture.md` should state that generated
  loft CSG artifacts require accepted public results, not adapter payloads.

## Readiness Blocker Resolution

- Blocker being resolved:
  - Surface Specs 407 and 418 need real loft CSG result geometry but the
    prerequisite was not modeled as an explicit workflow handoff.
- Source artifact:
  - `project/release-0.1.0a/specifications/surface-407-loft-csg-result-geometry-reference-proof-v1_0.md`
  - `project/release-0.1.0a/specifications/surface-418-loft-csg-section-artifact-generation-v1_0.md`
- Resolution provided by this ACD:
  - Defines accepted-result handoff records and refusal behavior.
- Follow-on artifact:
  - Final specs for result-geometry handoff proof and section evidence
    handoff readiness.
- Resolution status:
  - proposed; ready for manifest review.

## Compatibility And Migration Strategy

- Existing non-loft reference fixtures remain unchanged.
- Existing loft fixture records may stay declared but ungenerated until public
  result geometry is accepted.
- Dirty STL generation remains a workflow output, not a CSG execution fallback.
- Review app fixture schemas can consume new evidence fields without changing
  approval semantics.

## Application Integration Contract

- App type: workflow/tooling
- User/caller surface: reference generation and reference review workflows
- Invocation route: fixture source registry to public loft CSG operation to
  accepted-result handoff to STL or section evidence generation
- Wiring owner/module: `tests/reference_review_fixtures/stl_review_sources.py`,
  `tests/reference_images.py`
- Observable result: generated dirty STL and section artifact readiness only
  when public result geometry is accepted; structured refusal otherwise
- Integration validation: fixture-source tests proving accepted handoff,
  adapter-only refusal, section readiness, and no synthetic geometry output

## Specification Manifest for Discovery

### Candidate Spec: Loft CSG Reference Geometry Handoff Proof

Discovery purpose:
- Gate dirty STL generation for loft CSG references on accepted public result
  geometry.

Responsibilities:
- Functions/methods:
  - loft CSG reference handoff builder
  - accepted-result validator
  - adapter/synthetic refusal diagnostic builder
- Data structures/models:
  - `LoftCsgReferenceGeometryHandoffRecord`
  - accepted result body reference
  - reference handoff refusal payload
- Dependencies/services:
  - public `SurfaceBooleanResult`
  - fixture source registry
  - STL export workflow
- Returns/outputs/signals:
  - accepted handoff record
  - structured refusal
  - dirty STL source readiness signal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: fixture source registry and STL export workflow
  - Additions to existing reusable library/module: loft CSG accepted-result gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - dirty reference artifact creation through existing reference workflow
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - no duplicate CSG execution for metadata extraction
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - workflow/tooling
- User/caller surface:
  - reference generation workflow
- Invocation route:
  - fixture source registry to public CSG result to STL export
- Wiring owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Observable result:
  - dirty STL generated only from accepted public result body
- Integration validation:
  - accepted handoff test, adapter-only refusal test, no synthetic geometry test
- User-accessible surface:
  - dirty STL reference generation workflow and review fixtures
- Integration route:
  - fixture source registry to public CSG result to STL export
- App wiring owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Completion proof:
  - accepted handoff test, adapter-only refusal test, no-synthetic-geometry test, and dirty artifact smoke
- Unwired risk:
  - fixture records could generate dirty STLs from adapter diagnostics or synthetic geometry
- Incomplete status risk:
  - reference fixtures could bless adapter or synthetic output as real geometry
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - require `status == succeeded` and non-null surface-native body
- Test strategy:
  - focused fixture source tests plus dirty artifact generation smoke
- Data ownership:
  - fixture source registry owns handoff records; CSG owns result body
- Routes:
  - reference fixture source route
- Reuse/extraction decision:
  - reuse existing registry/export code; add gate only
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Loft Primitive Public Cut Executor Integration

Open questions / nuance discovered:
- Final spec should name the exact existing result fields used to identify an
  accepted body and avoid broad metadata scraping.

Refinement history:
- Iteration: 1
  - Date: 2026-07-16
  - Scope reviewed: Loft CSG Reference Geometry Handoff Proof candidate.
  - Files written before barrier:
    - `project/release-0.1.0a/architecture/acd-loft-csg-reference-geometry-handoff.md`
  - Updates made: added reachability fields and fixed-point manifest history.
  - Score before update: 22
  - Score after update: 22
  - Split decision: review for split; cohesive dirty-STL handoff gate.
  - Split artifacts: none.
  - Child re-review status: not applicable.
  - Next scope after readback: Loft CSG Section Evidence Readiness Handoff.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
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
- Missing prerequisites: 0 x 2 = 0
- Total: 22

Readiness blockers:
- none.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final spec after public cut executor spec exists
- Resolution artifact:
  - this manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked predecessor manifest candidate
  - Artifact: Loft Primitive Public Cut Executor Integration

Split decision:
- Review for split.
- Cohesion reason: accepted-result validation and dirty STL handoff are one
  workflow gate.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Loft CSG Section Evidence Readiness Handoff

Discovery purpose:
- Gate section artifact generation on accepted loft CSG result geometry and
  declared section evidence inputs.

Responsibilities:
- Functions/methods:
  - section evidence handoff builder
  - section readiness validator
  - section refusal diagnostic builder
- Data structures/models:
  - `LoftCsgSectionEvidenceReadinessRecord`
  - section plane declaration
  - section bundle readiness payload
- Dependencies/services:
  - public accepted CSG result body
  - section evidence contract records
  - reference fixture registry
- Returns/outputs/signals:
  - section evidence readiness record
  - structured refusal
  - fixture registry payload
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: section evidence contract records and fixture registry
  - Additions to existing reusable library/module: loft section readiness handoff
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - section artifact creation through existing reference workflow
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - avoid rerunning loft CSG for each section artifact
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - workflow/tooling
- User/caller surface:
  - section evidence generation workflow
- Invocation route:
  - fixture source registry to accepted CSG result to section evidence bundle
- Wiring owner/module:
  - `tests/reference_images.py`
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Observable result:
  - section readiness bundle only from accepted result geometry and declared
    section inputs
- Integration validation:
  - section readiness test, missing plane refusal test, adapter-only refusal test
- User-accessible surface:
  - reference section evidence workflow and review fixture artifacts tab
- Integration route:
  - fixture source registry to accepted CSG result to section evidence bundle
- App wiring owner/module:
  - `tests/reference_images.py`
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Completion proof:
  - section readiness test, missing plane refusal test, adapter-only refusal test, and fixture registry integration test
- Unwired risk:
  - section artifacts could be generated from detached evidence rather than accepted result geometry
- Incomplete status risk:
  - Surface Spec 418 could generate section evidence detached from real result
    geometry
- Implementation owner/module:
  - `tests/reference_images.py`
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - require accepted result body plus explicit section plane declaration
- Test strategy:
  - focused section handoff tests plus fixture registry integration test
- Data ownership:
  - section evidence workflow owns bundles; CSG owns result body
- Routes:
  - reference section evidence route
- Reuse/extraction decision:
  - reuse section evidence records; add handoff gate only
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Loft CSG Reference Geometry Handoff Proof
  - Surface Spec 417 section evidence contract records
  - Surface Spec 419 fixture registry integration for section bundles

Open questions / nuance discovered:
- Final spec should preserve section evidence as readiness/proof data, not as a
  second source of geometry truth.

Refinement history:
- Iteration: 1
  - Date: 2026-07-16
  - Scope reviewed: Loft CSG Section Evidence Readiness Handoff candidate.
  - Files written before barrier:
    - `project/release-0.1.0a/architecture/acd-loft-csg-reference-geometry-handoff.md`
  - Updates made: added reachability fields and fixed-point manifest history.
  - Score before update: 22
  - Score after update: 22
  - Split decision: review for split; cohesive section-evidence handoff gate.
  - Split artifacts: none.
  - Child re-review status: not applicable.
  - Next scope after readback: fixed-point ledger readback.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
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
- Missing prerequisites: 0 x 2 = 0
- Total: 22

Readiness blockers:
- none.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final spec after geometry handoff spec exists and section contracts
    conform
- Resolution artifact:
  - this manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked predecessor specs as prerequisites, not blockers
  - Artifact: Loft CSG Reference Geometry Handoff Proof
  - Artifact: Surface Spec 417
  - Artifact: Surface Spec 419

Split decision:
- Review for split.
- Cohesion reason: section readiness handoff is distinct from dirty STL handoff
  but still one reference workflow boundary.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

## Manifest Review History

- Pass 1 - Template/readiness review:
  - Added explicit reachability fields for both workflow candidates.
  - No unresolved readiness blockers remained.
- Pass 2 - Rescore:
  - Loft CSG Reference Geometry Handoff Proof: 22.
  - Loft CSG Section Evidence Readiness Handoff: 22.
  - No candidate reached the 25+ forced-split threshold.
- Pass 3 - Split review:
  - Both 16-24 candidates remain cohesive because dirty STL handoff and section
    evidence readiness are separate workflow gates with distinct artifacts.
- Pass 4 - Prerequisite review:
  - Result geometry handoff is sequenced after public cut executor integration.
  - Section evidence readiness is sequenced after result geometry handoff and
    the existing section contract/registry specs.
- Pass 5 - Final manifest readiness review:
  - No parent-only responsibilities, missing prerequisites, or unresolved
    blockers remain.
  - Both candidates are ready for final specification promotion in sequence.

## Specification Conformance

- Parent specs created or affected:
  - Surface Spec 407 - result geometry reference proof.
  - Surface Spec 418 - section artifact generation.
- Canonical child specs:
  - pending.
- Paired test specs:
  - pending.

## Conformance Checklist

- [ ] Result geometry handoff final spec exists.
- [ ] Section evidence readiness handoff final spec exists.
- [ ] Reference workflows refuse adapter-only, sampled, tessellated, or
  synthetic geometry payloads.
- [ ] Dirty STL generation consumes accepted public result geometry only.
- [ ] Section artifact generation consumes accepted result geometry plus
  declared section inputs only.
- [ ] Canonical architecture is updated after implementation conforms.

## Closure Criteria

- Surface Spec 407 can generate or refuse dirty STL artifacts based on an
  explicit accepted-result handoff.
- Surface Spec 418 can generate or refuse section artifacts based on accepted
  result geometry and declared section inputs.
- Reference review fixtures never treat adapter-only diagnostics or synthetic
  geometry as accepted loft CSG evidence.

## Closure Notes

- Canonical architecture updated:
  - pending
- Archived or removed scaffolding:
  - pending
- Follow-up ACDs:
  - none

## Change History

- 2026-07-16 - Initial split from cut-shell umbrella. Reason: reference
  workflow handoff is a downstream architecture boundary separate from the
  geometric kernel and public CSG executor.
