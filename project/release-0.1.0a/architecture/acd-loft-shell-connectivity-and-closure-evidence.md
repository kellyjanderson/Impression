# Loft Shell Connectivity And Closure Evidence Architectural Change Document

Date: 2026-07-14
Status: Proposed
Canonical architecture targets:

- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/loft-planner-executor-architecture.md`
- `project/release-0.1.0a/architecture/surfacebody-seam-adjacency-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Release / plan / issue: `project/release-0.1.0a/planning/reference-test-expansion-plan.md`
- Parent ACD, if any: none

## Change Intent

Make loft executor output eligible for downstream CSG only when the produced
`SurfaceBody` carries reliable connected, closed, cap, seam, adjacency, and
provenance evidence.

This change exists because current loft outputs expose `connected=False` through
the loft shell validity summary. The CSG eligibility gate is correctly refusing
those bodies, which blocks `Surface Spec 393` through `Surface Spec 396` and
all successful `RT-LOFT-CSG-001` through `RT-LOFT-CSG-011` fixture work.

## Current Architecture

Canonical architecture already says loft CSG eligibility must be explicit and
that CSG must not guess whether a loft is valid enough to cut. The current code
has a loft shell validity summary and CSG eligibility gate, but the loft
executor does not yet provide enough connected/closed shell evidence for actual
loft outputs to pass the gate.

The current architecture names this as a blocker, but it does not yet define:

- how loft side patches, cap patches, trim boundaries, and station seams produce
  a connected shell graph
- how closedness is proven without tessellating to mesh
- how cap validity and seam coverage are represented as reusable loft-owned
  records
- how invalid or incomplete evidence distinguishes implementation gaps from
  invalid authored input

## Target Architecture

The loft executor owns a first-class `LoftShellConnectivityRecord` and
`LoftClosureEvidenceRecord` produced as part of surface-body construction.

The target architecture has these components:

- `LoftBoundaryGraph`: a loft-owned graph of patch boundary refs, station
  seams, cap boundaries, and transition boundaries.
- `LoftShellConnectivityRecord`: records shell count, component count,
  connected component membership, seam coverage, open boundary refs, and source
  transition ids.
- `LoftClosureEvidenceRecord`: records whether side surfaces, caps, and station
  seams close every boundary loop needed for a valid single shell.
- `LoftCapValidityRecord`: records cap presence, cap loop ownership, cap
  orientation, and cap/body adjacency.
- `LoftCSGEligibilityEvidence`: combines connectivity, closure, cap validity,
  self-intersection validity, and loft provenance into the record consumed by
  CSG.

CSG remains the acceptance owner for boolean execution, but CSG consumes loft
evidence rather than re-deriving loft topology. A loft may enter CSG only when:

- it has exactly one shell
- the shell has exactly one connected component
- no required side, cap, or station boundary is open
- every seam references two valid boundaries
- cap loops are present and oriented for capped lofts
- loft provenance identifies transition count, station count, patch roles, and
  branch count
- self-intersection validity does not refuse the body

## Non-Goals

- Implementing loft primitive or loft/loft CSG routes.
- Implementing branching decomposition or recomposition.
- Promoting dirty reference STLs.
- Weakening the CSG eligibility gate to accept incomplete loft evidence.

## Canonical Document Impact

- Architecture docs to update on closure:
  - `lofted-body-csg-reference-architecture.md` - replace the current blocker
    language with the conformed loft shell evidence contract.
  - `loft-planner-executor-architecture.md` - describe boundary graph and
    closure evidence emitted by the executor.
  - `surfacebody-seam-adjacency-architecture.md` - align seam adjacency
    expectations with loft boundary graph records.
- Specs or plans affected:
  - `surface-392-loft-csg-eligibility-gate-v1_0.md` - may need follow-up
    implementation spec for full evidence conformance.
  - `surface-393` through `surface-396` - sequence after single-shell loft
    eligibility evidence passes.
  - `reference-test-expansion-plan.md` - sequence `RT-LOFT-CSG-001` through
    `RT-LOFT-CSG-006` and prerequisite portions of `RT-LOFT-CSG-009` through
    `RT-LOFT-CSG-011` after eligibility evidence exists.

## Compatibility And Migration Strategy

- Existing incomplete loft outputs continue to refuse through the CSG
  eligibility gate.
- New evidence records are additive and must be serializable in kernel metadata
  without changing public modeling signatures.
- Tests should first assert refusal with incomplete evidence, then assert
  successful eligibility for bounded single-shell loft fixtures.
- CSG route specs are represented by
  `acd-single-shell-loft-csg-operation-route.md` and sequence after at least
  one bounded actual loft output passes the evidence gate through the public
  loft executor.

## Application Integration Contract

- App type: library-only
- User/caller surface: CSG consumers calling `boolean_union`,
  `boolean_difference`, or `boolean_intersection` with loft-authored
  `SurfaceBody` operands
- Invocation route: public modeling API call
- Wiring owner/module: `src/impression/modeling/loft.py` and
  `src/impression/modeling/csg.py`
- Observable result: eligible single-shell loft operands pass the CSG
  eligibility gate; incomplete or invalid lofts refuse with structured
  diagnostics
- Integration validation: focused loft shell evidence tests plus CSG eligibility
  tests through the public boolean API

## Specification Manifest for Discovery

### Five-Pass Manifest Review Notes

- Pass 1: Added explicit app-type and invocation-route readiness fields for
  both candidates so future specs do not lose the public modeling API route.
- Pass 2: Rechecked scoring against the current manifest-entry template; totals
  remain accurate after counting no UI, database, async, or privacy surface.
- Pass 3: Reclassified the closure/cap candidate as review-for-split because
  its score is in the 16-24 band; kept it cohesive because cap validity and
  closure evidence are one eligibility contract.
- Pass 4: Added manifest cleanup fields so parent/child coverage is explicit
  when these candidates are promoted into final specs.
- Pass 5: Final rescore confirmed no candidate is 25+ and every readiness
  blocker resolution record is resolved; ordering is represented by
  predecessor candidates.

### Five-Pass Manifest Review Notes - 2026-07-15

- Pass 1: Rechecked both candidates after the new single-shell loft CSG route
  ACD entered the active sequence; no new responsibilities move into this ACD.
- Pass 2: Confirmed the boundary graph candidate remains the first executable
  leaf and has no predecessor inside this ACD.
- Pass 3: Confirmed closure/cap validity is sequenced by predecessor candidate,
  not by a readiness blocker.
- Pass 4: Added explicit readiness blocker resolution records to both
  candidates per the current manifest template.
- Pass 5: Rescore unchanged; no candidate is 25+ and no split is required.

### Candidate Spec: Loft Boundary Graph And Seam Coverage Evidence

Discovery purpose:
- Build the loft-owned boundary graph used to prove shell connectivity and seam
  coverage before CSG.

Responsibilities:
- Functions/methods:
  - loft boundary graph builder
  - seam coverage classifier
- Data structures/models:
  - `LoftBoundaryGraph`
  - `LoftSeamCoverageRecord`
- Dependencies/services:
  - loft executor patch role metadata
  - `SurfaceBoundaryRef`
  - `SurfaceSeam`
- Returns/outputs/signals:
  - boundary graph
  - missing seam diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `SurfaceSeam`, `SurfaceBoundaryRef`
  - Additions to existing reusable library/module: loft boundary graph helpers
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
  - bounded by loft patch and seam count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public modeling API consumers creating loft-authored `SurfaceBody` values
    for CSG operands
- Invocation route:
  - loft executor call during public modeling API construction
- Wiring owner/module:
  - `src/impression/modeling/loft.py`
- Observable result:
  - loft outputs include boundary graph and seam coverage evidence consumed by
    shell validity and CSG eligibility checks
- Integration validation:
  - focused loft executor tests plus public CSG eligibility probes
- Incomplete status risk:
  - implemented in isolation if the graph is not exposed through the shell
    validity summary consumed by CSG
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - every non-cap side boundary must be accounted for by a station seam,
    transition seam, cap seam, or explicit open-boundary diagnostic
- Test strategy:
  - unit tests for complete, missing, duplicate, and dangling seam coverage
- Data ownership:
  - loft owns boundary graph evidence
- Routes:
  - loft executor to boundary graph to shell validity summary
- Reuse/extraction decision:
  - add to existing loft module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Need confirm whether current `SurfaceShell.connected` should be derived from
  this graph or remain a lower-level storage fact.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 15

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - none
- Resolution artifact:
  - not applicable
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

### Candidate Spec: Loft Closure And Cap Validity Evidence

Discovery purpose:
- Prove capped loft bodies are closed before they can enter CSG.

Responsibilities:
- Functions/methods:
  - loft closure evidence builder
  - cap validity classifier
- Data structures/models:
  - `LoftClosureEvidenceRecord`
  - `LoftCapValidityRecord`
- Dependencies/services:
  - loft boundary graph
  - cap patch metadata
  - trim loop orientation
- Returns/outputs/signals:
  - closed-valid signal
  - cap diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: planar cap patches and trim loops
  - Additions to existing reusable library/module: cap/closure evidence helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG eligibility for loft outputs
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by cap loop and seam count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public modeling API consumers calling boolean operations with loft-authored
    `SurfaceBody` operands
- Invocation route:
  - loft executor evidence to CSG eligibility gate during public boolean API
    calls
- Wiring owner/module:
  - `src/impression/modeling/loft.py`
  - `src/impression/modeling/csg.py`
- Observable result:
  - capped single-shell loft bodies pass closure eligibility; uncapped or
    invalid cap cases refuse with structured diagnostics
- Integration validation:
  - loft closure tests plus public boolean API eligibility tests
- Incomplete status risk:
  - implemented in isolation if CSG still derives or guesses closedness instead
    of consuming loft evidence
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - capped lofts require cap evidence; uncapped lofts refuse for closed-body CSG
- Test strategy:
  - bounded single-shell loft eligibility tests and invalid cap refusal tests
- Data ownership:
  - loft owns closure evidence; CSG owns acceptance
- Routes:
  - loft closure evidence to CSG eligibility gate
- Reuse/extraction decision:
  - extend current loft shell validity summary
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Need determine whether uncapped lofts can later enter open-surface CSG, but
  this ACD owns only closed-body CSG eligibility.

Predecessor candidates:
- `Loft Boundary Graph And Seam Coverage Evidence`

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 18

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; sequencing is captured by predecessor candidate `Loft Boundary Graph
    And Seam Coverage Evidence`
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `Loft Boundary Graph And Seam Coverage Evidence`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: closure evidence and cap validity are the
  same closed-body eligibility contract; splitting would force duplicate
  diagnostics and leave the CSG gate guessing between partial records.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

## Specification Conformance

- Parent specs created or affected:
  - `surface-392-loft-csg-eligibility-gate-v1_0.md` - implemented partial
    gate; this ACD adds missing conformance evidence.
- Canonical child specs:
  - `../specifications/surface-403-loft-boundary-graph-and-seam-coverage-evidence-v1_0.md` - canonical child from `Loft Boundary Graph And Seam Coverage Evidence`.
  - `../specifications/surface-404-loft-closure-and-cap-validity-evidence-v1_0.md` - canonical child from `Loft Closure And Cap Validity Evidence`.
- Paired test specs:
  - none yet

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

- Actual loft executor outputs for bounded single-shell capped lofts expose
  connected and closed-valid evidence.
- CSG eligibility accepts at least one actual loft output through the public
  modeling API.
- Incomplete, open, disconnected, or invalid-cap loft outputs continue to
  refuse with structured no-mesh-fallback diagnostics.
- Canonical loft and lofted-body CSG architecture documents describe the
  conformed evidence model.

## Closure Notes

- Canonical architecture updated:
  - none yet
- Archived or removed scaffolding:
  - none yet
- Follow-up ACDs:
  - none

## Change History

- 2026-07-14 - Initial draft. Reason: successful loft CSG progression items
  require explicit loft shell connectivity and closure evidence.
