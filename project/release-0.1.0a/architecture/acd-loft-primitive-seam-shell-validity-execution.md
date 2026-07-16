# Loft Primitive Seam Shell Validity Execution ACD

Date: 2026-07-16
Status: Manifesting
Canonical architecture targets:

- `project/release-0.1.0a/architecture/surfacebody-csg-architecture.md`
- `project/release-0.1.0a/architecture/surface-csg-trim-fragment-reconstruction-architecture.md`
- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Parent ACD:
  `project/release-0.1.0a/architecture/acd-loft-primitive-cut-shell-geometric-kernel.md`
- Predecessor ACD:
  `project/release-0.1.0a/architecture/acd-loft-primitive-generated-cap-and-topology-policy.md`

## Change Intent

Define how selected loft fragments and generated primitive caps become a
durable, valid `SurfaceBody` result returned through the public Boolean API.

This ACD keeps durable shell construction, seam/use pairing, validity,
persistence proof, no-hidden-mesh proof, and public route wiring together as an
execution boundary after geometry/topology policy is already decided.

## Current Architecture

Current code can return exact reuse results for no-cut/containment cases and
adapter-only refusals for intersecting cases. It cannot:

- pair retained loft loops with generated cap loops
- rebuild seams and adjacency for cut shells
- validate closed shell/cavity/multi-shell output
- prove persistence and no-hidden-mesh readiness before return
- route true intersecting cut cases through `surface_boolean_result`

## Target Architecture

This ACD introduces:

- `LoftPrimitiveSeamUsePairRecord`: paired loop-use identity, orientation,
  continuity class, tolerance, and provenance.
- `LoftPrimitiveCutShellAssemblyRecord`: selected fragments, generated caps,
  seam-use pairs, candidate shell ids, topology class, diagnostics, and
  no-mesh proof.
- `LoftPrimitiveCutShellValidityRecord`: runtime validity, persistence
  readiness, tessellation-boundary readiness, and no-hidden-mesh evidence.
- `LoftPrimitiveExecutionScopeRecord`: exact reuse, trim-fragment cut, or
  structured refusal scope.
- `assemble_loft_primitive_cut_shell(...)`
- `validate_loft_primitive_cut_shell_result(...)`
- `execute_loft_primitive_trim_fragment_csg(...)`

## Non-Goals

- Source normalization and cut-loop construction.
- Generated cap representation and topology policy.
- Reference artifact writing.
- New public Boolean API surface.

## Canonical Document Impact

- `surfacebody-csg-architecture.md` should describe this stage as the durable
  result topology reconstructor for loft/primitive cuts.
- `surface-csg-trim-fragment-reconstruction-architecture.md` should include
  loft seam/use pairing and validity handoff.
- `lofted-body-csg-reference-architecture.md` should describe public cut route
  integration after exact reuse and adapter diagnostics.

## Readiness Blocker Resolution

- Blocker being resolved:
  - Surface Spec 422 cannot return a valid result because seam/shell assembly,
    validity, and public route wiring are not defined.
- Source artifact:
  - `project/release-0.1.0a/specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Resolution provided by this ACD:
  - Defines shell assembly, validity, persistence, no-mesh, and public executor
    architecture.
- Follow-on artifact:
  - Final specs for seam/shell assembly, runtime validity, and public executor
    integration.
- Resolution status:
  - proposed; ready for manifest review.

## Compatibility And Migration Strategy

- Exact reuse remains first in route precedence.
- Adapter-only refusal remains until all predecessor ACD-derived specs conform.
- Public API result envelopes stay `SurfaceBooleanResult`.
- Tessellation remains preview/export/proof boundary only.

## Application Integration Contract

- App type: library-only
- User/caller surface: public Boolean API consumers
- Invocation route: selected topology records to seam/shell assembly to
  validity gate to `surface_boolean_result`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: accepted cut cases return valid `SurfaceBody`; invalid or
  unsupported cases return structured `SurfaceBooleanResult` diagnostics
- Integration validation: public API tests for accepted cut cases, invalid seam
  refusal, no-hidden-mesh evidence, persistence readiness, and route precedence

## Specification Manifest for Discovery

### Candidate Spec: Loft Primitive Seam And Shell Assembly

Discovery purpose:
- Build candidate result shells and seam/use pair records from selected
  fragments and generated caps.

Responsibilities:
- Functions/methods:
  - seam/use pairing builder
  - cut-shell assembler
  - adjacency rebuild diagnostic builder
- Data structures/models:
  - `LoftPrimitiveCutShellAssemblyRecord`
  - `LoftPrimitiveSeamUsePairRecord`
  - adjacency rebuild diagnostic
- Dependencies/services:
  - topology selection records
  - generated cap records
  - SurfaceBody constructors and seam helpers
- Returns/outputs/signals:
  - candidate `SurfaceBody` shells
  - seam/adjacency records
  - assembly diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `make_surface_shell`, `make_surface_body`, seam/adjacency helpers
  - Additions to existing reusable library/module: loft primitive shell assembler in `src/impression/modeling/csg.py`
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
  - bounded by fragment, loop, and seam count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public Boolean API consumers
- Invocation route:
  - topology records to seam/use pairing to candidate `SurfaceBody`
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - candidate result shells or assembly diagnostics
- Integration validation:
  - tests for cap-loop pairing, unpaired seam refusal, cavity assembly, and no mesh fragments
- User-accessible surface:
  - public Boolean API result diagnostics
- Integration route:
  - topology selector to seam/use pairing to candidate `SurfaceBody`
- App wiring owner/module:
  - `src/impression/modeling/csg.py`
- Completion proof:
  - seam assembly tests plus public invalid-seam refusal tests
- Unwired risk:
  - candidate bodies could be constructed without public-route proof or with unpaired seams
- Incomplete status risk:
  - public executor could return structurally invalid shells
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - required generated cap loops pair exactly once with retained loft loops
- Test strategy:
  - seam assembly unit tests and public invalid-seam refusal tests
- Data ownership:
  - CSG owns assembly records; SurfaceBody constructors own durable shell representation
- Routes:
  - topology selector to shell assembler
- Reuse/extraction decision:
  - reuse SurfaceBody constructors and CSG seam helpers
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Loft Primitive Fragment Topology And Operation Selection

Open questions / nuance discovered:
- Continuity beyond C0/G0 must be recorded as future capability, not silently claimed.

Refinement history:
- Iteration: 1
  - Date: 2026-07-16
  - Scope reviewed: Loft Primitive Seam And Shell Assembly candidate.
  - Files written before barrier:
    - `project/release-0.1.0a/architecture/acd-loft-primitive-seam-shell-validity-execution.md`
  - Updates made: added reachability fields and fixed-point manifest history.
  - Score before update: 19
  - Score after update: 19
  - Split decision: review for split; cohesive seam/shell assembly boundary.
  - Split artifacts: none.
  - Child re-review status: not applicable.
  - Next scope after readback: Loft Primitive Runtime Validity And Persistence Gate.

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
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Total: 19

Readiness blockers:
- none.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final spec after review
- Resolution artifact:
  - this manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked predecessor ACD candidate
  - Artifact: Loft Primitive Fragment Topology And Operation Selection

Split decision:
- Review for split.
- Cohesion reason: seam pairing and shell assembly validate one candidate
  `SurfaceBody` boundary.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Loft Primitive Runtime Validity And Persistence Gate

Discovery purpose:
- Accept or reject assembled cut-shell candidates before public return,
  persistence, tessellation, or reference generation.

Responsibilities:
- Functions/methods:
  - cut-shell runtime validity checker
  - persistence/tessellation-boundary proof collector
  - no-hidden-mesh gate
- Data structures/models:
  - `LoftPrimitiveCutShellValidityRecord`
  - persistence evidence record
  - no-hidden-mesh proof record
- Dependencies/services:
  - candidate `SurfaceBody`
  - existing runtime validity gate
  - `.impress` persistence and tessellation boundary helpers
- Returns/outputs/signals:
  - accepted result body signal
  - invalid/unsupported diagnostics
  - persistence and no-mesh proof payload
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: runtime validity, persistence, and tessellation evidence helpers
  - Additions to existing reusable library/module: loft primitive validity payload builder
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
  - bounded by result shell, seam, trim, and patch counts
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public Boolean API consumers and reference workflows
- Invocation route:
  - assembled candidate body to validity/persistence gate
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - accepted body metadata or invalid/unsupported result
- Integration validation:
  - public API tests for accepted result, invalid seam refusal, persistence readiness, and no-hidden-mesh evidence
- User-accessible surface:
  - public Boolean API result metadata and downstream reference workflow readiness
- Integration route:
  - assembled candidate body to validity/persistence gate to result finalizer
- App wiring owner/module:
  - `src/impression/modeling/csg.py`
- Completion proof:
  - public API tests proving accepted body metadata, invalid refusal, persistence readiness, and no-hidden-mesh evidence
- Unwired risk:
  - reference workflows could receive bodies before durability and no-hidden-mesh proof are established
- Incomplete status risk:
  - result body could be returned before it is durable enough for references
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - tessellation is proof/export only; mesh-backed fragments refuse
- Test strategy:
  - validity tests plus public API route tests
- Data ownership:
  - CSG owns validity proof; persistence/tessellation helpers own boundary evidence
- Routes:
  - shell assembler to result finalizer
- Reuse/extraction decision:
  - reuse existing validity and persistence helpers
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Loft Primitive Seam And Shell Assembly

Open questions / nuance discovered:
- Final spec should choose a focused persistence proof that avoids broad
  filesystem churn.

Refinement history:
- Iteration: 1
  - Date: 2026-07-16
  - Scope reviewed: Loft Primitive Runtime Validity And Persistence Gate candidate.
  - Files written before barrier:
    - `project/release-0.1.0a/architecture/acd-loft-primitive-seam-shell-validity-execution.md`
  - Updates made: added reachability fields and fixed-point manifest history.
  - Score before update: 19.5
  - Score after update: 19.5
  - Split decision: review for split; cohesive result-acceptance gate.
  - Split artifacts: none.
  - Child re-review status: not applicable.
  - Next scope after readback: Loft Primitive Public Cut Executor Integration.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- none.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final spec after shell assembly spec exists
- Resolution artifact:
  - this manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked predecessor manifest candidate
  - Artifact: Loft Primitive Seam And Shell Assembly

Split decision:
- Review for split.
- Cohesion reason: runtime validity, persistence proof, and no-mesh proof are
  one result-acceptance gate.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Loft Primitive Public Cut Executor Integration

Discovery purpose:
- Wire the complete kernel into `surface_boolean_result` without disrupting
  exact reuse or other CSG routes.

Responsibilities:
- Functions/methods:
  - `execute_loft_primitive_trim_fragment_csg(...)`
  - public result metadata/failure payload builder
  - route precedence guard
- Data structures/models:
  - `LoftPrimitiveExecutionScopeRecord`
  - public executor diagnostic
  - result metadata payload
- Dependencies/services:
  - all predecessor cut-shell stages
  - `surface_boolean_result`
  - route selection and family gates
- Returns/outputs/signals:
  - succeeded `SurfaceBooleanResult`
  - unsupported/invalid `SurfaceBooleanResult`
  - result metadata consumed by references
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `surface_boolean_result`, exact reuse executor, adapter refusal path
  - Additions to existing reusable library/module: public loft primitive cut executor
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
  - bounded by prior stages; executor must not rerun CSG for metadata
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - library-only
- User/caller surface:
  - public Boolean API consumers and reference workflows
- Invocation route:
  - `surface_boolean_result` to exact reuse or trim-fragment cut executor
- Wiring owner/module:
  - `src/impression/modeling/csg.py`
- Observable result:
  - public API accepted/unsupported/invalid results with cut-shell metadata
- Integration validation:
  - public API tests for cavity difference, partial-overlap difference, union, intersection, invalid seam refusal, and route precedence
- User-accessible surface:
  - public Boolean API `SurfaceBooleanResult`
- Integration route:
  - `surface_boolean_result` to exact reuse or trim-fragment cut executor
- App wiring owner/module:
  - `src/impression/modeling/csg.py`
- Completion proof:
  - public API integration tests for accepted cut cases, structured refusals, and route precedence
- Unwired risk:
  - all helper stages could pass while the public API still returns adapter-only refusal
- Incomplete status risk:
  - helper stages could pass while public API still returns adapter-only refusal
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - exact reuse remains first; cut executor handles true intersecting cut cases
- Test strategy:
  - public API integration tests and route precedence tests
- Data ownership:
  - public CSG surface owns final `SurfaceBooleanResult`
- Routes:
  - public Boolean API route
- Reuse/extraction decision:
  - add to existing CSG public route; no parallel API
- UI field/control inventory:
  - not applicable
- Prerequisites:
  - Loft Primitive Runtime Validity And Persistence Gate

Open questions / nuance discovered:
- Result metadata must support reference handoff but not embed generated
  artifact paths.

Refinement history:
- Iteration: 1
  - Date: 2026-07-16
  - Scope reviewed: Loft Primitive Public Cut Executor Integration candidate.
  - Files written before barrier:
    - `project/release-0.1.0a/architecture/acd-loft-primitive-seam-shell-validity-execution.md`
  - Updates made: added reachability fields and fixed-point manifest history.
  - Score before update: 19.5
  - Score after update: 19.5
  - Split decision: review for split; cohesive public-route integration boundary.
  - Split artifacts: none.
  - Child re-review status: not applicable.
  - Next scope after readback: reference geometry handoff ACD candidates.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- none.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - create final spec after validity gate spec exists
- Resolution artifact:
  - this manifest candidate
- Resolution status:
  - resolved
- Prerequisite handling:
  - linked predecessor manifest candidate
  - Artifact: Loft Primitive Runtime Validity And Persistence Gate

Split decision:
- Review for split.
- Cohesion reason: route wiring, result envelope, and precedence are one
  public integration boundary.

Manifest cleanup:
- Parent manifest candidate, if split: none
- Child manifest candidates:
  - none
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

## Manifest Review History

- Pass 1 - Template/readiness review:
  - Added explicit reachability fields for all three candidates.
  - No unresolved readiness blockers remained.
- Pass 2 - Rescore:
  - Loft Primitive Seam And Shell Assembly: 19.
  - Loft Primitive Runtime Validity And Persistence Gate: 19.5.
  - Loft Primitive Public Cut Executor Integration: 19.5.
  - No candidate reached the 25+ forced-split threshold.
- Pass 3 - Split review:
  - All 16-24 candidates remain cohesive because each owns one execution-stage
    boundary: assembly, acceptance gate, or public route wiring.
- Pass 4 - Prerequisite review:
  - Seam/shell assembly is sequenced after topology selection.
  - Runtime validity is sequenced after seam/shell assembly.
  - Public executor integration is sequenced after runtime validity.
- Pass 5 - Final manifest readiness review:
  - No parent-only responsibilities, missing prerequisites, or unresolved
    blockers remain.
  - All candidates are ready for final specification promotion in sequence.

## Specification Conformance

- Parent specs created or affected:
  - Surface Spec 422 - blocked parent currently lacks execution/validity wiring.
- Canonical child specs:
  - pending.
- Paired test specs:
  - pending.

## Conformance Checklist

- [ ] Seam/shell assembly final spec exists.
- [ ] Runtime validity and persistence final spec exists.
- [ ] Public cut executor final spec exists.
- [ ] Public API route tests cover accepted and refused outcomes.
- [ ] Canonical architecture is updated after implementation conforms.

## Closure Criteria

- Public cut-producing loft/primitive CSG returns valid `SurfaceBody` results
  for supported cases and structured refusals for invalid/unsupported cases.
- Surface Spec 422 is superseded or fully represented by child specs.

## Closure Notes

- Canonical architecture updated:
  - pending
- Archived or removed scaffolding:
  - pending
- Follow-up ACDs:
  - none

## Change History

- 2026-07-16 - Initial split from cut-shell umbrella. Reason: public execution,
  seam/shell assembly, validity, and persistence are a distinct architecture
  boundary after cut-loop and cap/topology decisions.
