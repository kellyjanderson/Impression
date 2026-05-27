# Surface Body Completion Architecture

## Overview

This document defines the remaining architecture needed for Impression to make
surface bodies feel complete as an authored modeling kernel.

The target is not merely "no mesh fallback." The target is:

- all surface patch families are first-class authored data
- surface operations either execute natively or return precise diagnostics
- authored topology rails drive correspondence
- ambiguous authored lofts report every ambiguity and refuse execution until
  resolved
- `.impress` persists complete surface truth
- mesh appears only at the tessellation, preview, export, analysis, or explicit
  compatibility boundary

## Related Architecture

This document coordinates and supersedes the completion posture in:

- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [Patch Family Integration Architecture](patch-family-integration-architecture.md)
- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)
- [Loft Evolution System Architecture](loft-evolution-system.md)
- [Loft Ambiguity and Diagnostics Architecture](loft-ambiguity-and-diagnostics.md)
- [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)

## Ownership Boundaries

### Impression Owns

- surface-body storage and topology
- patch family records, evaluation, tessellation adapters, and `.impress`
  payloads
- surface-native CSG over `SurfaceBody`
- loft planning, diagnostics, and surface execution
- primitive and feature-builder integration boundaries that produce
  `SurfaceBody` or `SurfaceConsumerCollection`
- no-hidden-mesh-fallback enforcement

## Completion Streams

### 1. Patch Family Promotion

The current capability matrix distinguishes `available` and `planned` families.
Completion requires every family intended for authored modeling to pass a
family-specific promotion gate.

Each family needs:

- canonical in-memory patch record
- parameter domain and boundary model
- evaluation and derivative contract where meaningful
- trim and seam participation
- tessellation adapter with lossiness metadata when approximate
- `.impress` codec and round-trip identity tests
- diagnostic behavior for unsupported operations
- CSG and loft eligibility classification

The promotion target includes:

- planar
- ruled
- revolution
- B-spline
- NURBS
- sweep
- subdivision
- implicit
- heightmap
- displacement

### 2. Surface CSG Completion

Surface CSG must move from a bounded planar/box slice to a broad surface-body
boolean system.

The completed system needs:

- exact or declared-tolerance intersection records for supported family pairs
- operation-specific trim graph construction
- fragment classification that works on trimmed and curved surfaces
- shell reconstruction for multi-shell and nested-loop results
- cap construction for cut regions
- seam and adjacency rebuild for all result shells
- validity, healing, provenance, and diagnostics at the result boundary
- explicit refusal records for family pairs that genuinely remain outside the
  kernel scope

The priority order should be:

1. analytic planar, ruled, and revolution pair coverage
2. box/cylinder/sphere/cone/torus primitive boolean coverage
3. trimmed-surface fragment graph coverage
4. B-spline/NURBS curve-surface and surface-surface intersection boundary
5. subdivision, implicit, heightmap, and displacement operation policy

### 3. Loft Completion

Impression targets authored topologies.

That means the planner should rely on user-authored rails, named entities,
anchors, path order, lifecycle records, and topology paths before attempting
automatic ambiguity resolution.

Automatic resolution is allowed only when it is deterministic, high confidence,
and produces a better user experience without hiding ambiguity. It is not a
requirement for this version.

The core execution rule is:

- planning continues after ambiguity is discovered so all problems can be
  reported together
- each ambiguity is recorded with exact topology, station, interval, entity,
  relationship-group, and candidate-lifecycle locators
- any unresolved ambiguity makes the plan non-executable
- the executor must refuse a plan that carries unresolved ambiguity records

The planner should distinguish:

- invalid authored input
- missing rails or anchors
- ambiguous branch/split/merge correspondence
- point birth/death ambiguity
- loop or region containment ambiguity
- incompatible lifecycle declarations
- unsupported topology transitions

### 4. Seam, Continuity, And Topology Validity

Surface-body completion requires more than storing patches.

The topology layer needs:

- explicit boundary-use records for every shell
- seam participation validation across every promoted family
- continuity request records beyond the current C0/G0 baseline
- diagnostics for unsupported continuity classes
- watertightness, open-shell, non-manifold, and duplicate-boundary gates
- transform-stable identity and adjacency behavior

Higher continuity support should be promoted deliberately. Unsupported G1/G2 or
curvature-continuity requests must remain structured diagnostics, not silent
downgrades.

### 5. `.impress` Surface-Native Persistence

The `.impress` format is complete only when it can persist the whole authored
surface-body store.

The format must cover:

- multi-body document roots
- units and coordinate policy
- every promoted patch family
- trims, seams, adjacency, shell metadata, and stable identity
- topology rails and authored-lifecycle records when they affect execution
- operation provenance and diagnostic metadata
- refusal of malformed or unsupported payloads without mesh recovery

Mesh payloads may exist only as explicit import/export/cache artifacts. They
are not canonical surface truth.

### 6. Primitive And Feature Integration

Primitive and feature builders should select the appropriate patch family for
the authored geometry they emit.

The completion target includes:

- primitives using analytic families where exact
- loft using B-spline, NURBS, or sweep families when requested and validated
- feature builders producing `SurfaceBody` or `SurfaceConsumerCollection`
- no feature path producing mesh as hidden substitute geometry
- generic external-feature integration for sibling projects

### 7. Verification And Completion Evidence

Every promoted family or operation needs evidence:

- focused unit tests for records and diagnostics
- round-trip `.impress` tests
- tessellation-boundary tests
- no-hidden-mesh-fallback tests
- reference images/STLs for model-outputting capabilities
- negative tests that prove unsupported states refuse with exact diagnostics

Completion claims should cite verified capability matrices, not checklist
completion alone.

## Specification Manifest for Discovery

The following manifest was added after a critical review of the new and updated
architecture documents. The first pass exposed that "complete surface body
support" was too broad to implement as one branch. The second and third review
passes split that umbrella into flat implementation candidates that coordinate,
rather than duplicate, the existing patch-family, CSG, loft, `.impress`, and
mesh-boundary manifests.

All candidates below were rescored after splitting. No candidate remains at
`25+`; candidates in the `16-24` review band include an explicit cohesion
reason.

### Candidate Spec: Surface Body Completion Capability And Evidence Gate

Discovery purpose:
- Define the release-level capability matrix and evidence gate required before
  Impression can claim complete authored surface-body support.

Responsibilities:
- Functions/methods:
  - capability matrix audit command or maintained checker
  - completion gate evaluator
- Data structures/models:
  - completion capability record
  - evidence status record
- Dependencies/services:
  - patch-family manifests
  - CSG manifests
  - loft manifests
  - `.impress` manifests
- Returns/outputs/signals:
  - pass/fail completion report
  - missing evidence diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: architecture work tracker and existing
    capability/spec matrices
  - Additions to existing reusable library/module: release verification helper
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
  - bounded repository scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `project/release-0.1.0a/planning/` and release verification tooling
- Chosen defaults / parameters:
  - incomplete evidence blocks completion claims
- Test strategy:
  - documentation checker plus unit tests for matrix classification
- Data ownership:
  - completion matrix owns release-level support truth
- Routes:
  - architecture work tracker to progression and verification evidence
- Reuse/extraction decision:
  - extend existing planning/checker patterns; no new runtime geometry module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This gate should not mark implementation work done; it only prevents false
  completion claims.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 4 x 1 = 4
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
- Total: 15.5

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
- Small.
- This remains one release-gate spec because it coordinates evidence rather
  than implementing geometry.

### Candidate Spec: Patch Family Promotion Readiness Audit

Discovery purpose:
- Audit every authored patch family against the promotion criteria and produce
  exact missing-work records for storage, evaluation, seams, tessellation,
  `.impress`, CSG, loft, and diagnostics.

Responsibilities:
- Functions/methods:
  - patch-family readiness auditor
  - promotion status updater
  - missing-work reporter
- Data structures/models:
  - family promotion checklist
  - family gap record
  - support status transition record
- Dependencies/services:
  - `surface.py`
  - tessellation adapters
  - `.impress` codec manifests
  - CSG eligibility diagnostics
- Returns/outputs/signals:
  - per-family promotion verdict
  - per-family blocker diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: patch family capability matrix
  - Additions to existing reusable library/module: family readiness checker
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may update support status documentation/spec records
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded static and fixture scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, tessellation, and release verification
- Chosen defaults / parameters:
  - `available` requires implementation plus evidence, not architectural intent
- Test strategy:
  - unit tests for auditor classification and missing-work diagnostics
- Data ownership:
  - patch family capability matrix owns support status truth
- Routes:
  - family records to readiness checker to progression/spec gaps
- Reuse/extraction decision:
  - extend capability matrix rather than creating per-family silo checkers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This is separate from implementing each family; it discovers and gates the
  remaining work.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 4 x 1 = 4
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
- Total: 21.5

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
- Review for split. Cohesion reason: this is one audit/gate over existing
  family-specific implementation specs; splitting by family would duplicate the
  full patch-family manifest.

### Candidate Spec: CSG Completion Support Matrix And Refusal Records

Discovery purpose:
- Define the authoritative CSG family-pair support matrix and refusal record
  schema for unsupported or staged boolean combinations.

Responsibilities:
- Functions/methods:
  - CSG support matrix resolver
  - refusal record factory
  - family-pair diagnostic formatter
- Data structures/models:
  - CSG family-pair support record
  - CSG refusal record
  - operation support phase record
- Dependencies/services:
  - SurfaceBody CSG architecture
  - patch family capability matrix
  - no-hidden-mesh-fallback enforcement
- Returns/outputs/signals:
  - support verdict
  - structured refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG unsupported result posture
  - Additions to existing reusable library/module: CSG capability/refusal table
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes unsupported CSG reporting posture
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - constant-time matrix lookup
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported family pairs refuse explicitly and never route to mesh
- Test strategy:
  - family-pair support/refusal matrix tests
- Data ownership:
  - CSG owns operation support truth; patch families own family availability
- Routes:
  - boolean API to support matrix to executor or refusal result
- Reuse/extraction decision:
  - extend CSG result diagnostics rather than adding caller-specific checks
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This matrix must be cited by higher-order CSG specs before they add solver
  support.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Review for split. Cohesion reason: matrix and refusal records are one
  boundary contract used by all CSG solvers.

### Candidate Spec: Analytic CSG Expansion For Primitive Surface Families

Discovery purpose:
- Expand executable surface CSG coverage for planar, ruled, and revolution
  primitive-derived families without broadening into higher-order spline or
  sampled solvers.

Responsibilities:
- Functions/methods:
  - planar/ruled/revolution intersection dispatch
  - primitive analytic pair handlers
  - tolerance-aware curve classification
- Data structures/models:
  - analytic intersection curve record
  - primitive pair solver record
  - tolerance policy record
- Dependencies/services:
  - CSG support matrix
  - primitive surface families
  - seam/adjacency rebuild
- Returns/outputs/signals:
  - analytic intersection records
  - unsupported analytic pair diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current bounded CSG result types
  - Additions to existing reusable library/module: analytic CSG solver library
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean execution coverage
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded pair solver tolerances and iteration limits
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py` and analytic intersection helpers
- Chosen defaults / parameters:
  - analytic primitive family pairs execute when support matrix marks them
    supported
- Test strategy:
  - primitive pair boolean fixtures and tolerance-bound negative tests
- Data ownership:
  - analytic solver owns intersection records; CSG owns result topology
- Routes:
  - support matrix to analytic solver to trim/result reconstruction
- Reuse/extraction decision:
  - keep solver helpers reusable under CSG; do not place pair logic in
    primitives
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Torus and cone cases may require declared-tolerance support before exact
  closed-form coverage.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Review for split. Cohesion reason: this is intentionally limited to analytic
  primitive families; higher-order and sampled policies are separate candidates.

### Candidate Spec: Higher-Order And Sampled CSG Policy Boundary

Discovery purpose:
- Define whether B-spline, NURBS, subdivision, implicit, heightmap, and
  displacement booleans are executable kernel operations, declared-tolerance
  adapters, or explicit non-CSG refusals.

Responsibilities:
- Functions/methods:
  - higher-order support classifier
  - sampled/implicit operation policy resolver
- Data structures/models:
  - higher-order CSG policy record
  - sampled operation boundary record
- Dependencies/services:
  - CSG support matrix
  - patch family promotion audit
  - tessellation-boundary policy
- Returns/outputs/signals:
  - operation policy verdict
  - refusal or adapter diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: family capability matrix
  - Additions to existing reusable library/module: CSG policy table
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes support/refusal policy
- Security/privacy-sensitive behavior:
  - implicit field policy must preserve declarative safety boundaries
- Performance-sensitive behavior:
  - bounded adapter and refusal decisions; no unbounded numerical solving
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py` and family capability policy
- Chosen defaults / parameters:
  - higher-order operations refuse until explicitly classified as exact,
    declared-tolerance, or bounded adapter support
- Test strategy:
  - policy matrix tests for each higher-order/sampled family
- Data ownership:
  - CSG owns operation policy; family modules own family validity
- Routes:
  - boolean API to CSG support matrix to higher-order policy resolver
- Reuse/extraction decision:
  - share refusal diagnostics with CSG matrix candidate
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This split prevents sampled-family policy from quietly becoming hidden
  tessellation CSG.

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
- Review for split. Cohesion reason: this is one policy boundary for
  non-analytic CSG; solver implementation remains in separate future specs.

### Candidate Spec: Loft Ambiguity Accumulation And Execution Refusal Gate

Discovery purpose:
- Ensure authored loft planning accumulates all ambiguity and invalid-input
  records, while execution refuses any plan that still carries unresolved
  ambiguity.

Responsibilities:
- Functions/methods:
  - ambiguity accumulation pass
  - plan executability gate
  - executor refusal check
- Data structures/models:
  - unresolved ambiguity record
  - plan executability status
  - invalid-input aggregate
- Dependencies/services:
  - loft planner
  - loft plan object
  - topology correspondence architecture
- Returns/outputs/signals:
  - non-executable plan result
  - aggregate ambiguity diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current loft plan and diagnostics records
  - Additions to existing reusable library/module: executability gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes loft execution refusal behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - planner continues after errors but remains bounded by topology size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - unresolved ambiguity always blocks execution; planning still reports all
    detectable problems
- Test strategy:
  - authored ambiguous loft fixtures with multiple simultaneous ambiguity
    records and executor refusal assertions
- Data ownership:
  - `LoftPlan` owns executability status; planner owns ambiguity records
- Routes:
  - authored topology to planner to plan status to surface executor
- Reuse/extraction decision:
  - extend existing diagnostics rather than creating a separate validator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This is not an automatic ambiguity solver; deterministic inference policy is
  intentionally separate.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Review for split. Cohesion reason: accumulation and execution refusal are one
  plan lifecycle invariant and should be tested together.

### Candidate Spec: Loft Ambiguity Locator Diagnostics

Discovery purpose:
- Make every unresolved loft ambiguity report exact topology, station,
  interval, entity, relationship group, candidate lifecycle, and suggested rail
  locators.

Responsibilities:
- Functions/methods:
  - ambiguity locator builder
  - suggested rail formatter
- Data structures/models:
  - ambiguity locator payload
  - suggested authored rail record
  - relationship group reference
- Dependencies/services:
  - topology path records
  - lifecycle records
  - loft diagnostics
- Returns/outputs/signals:
  - exact ambiguity diagnostic
  - missing-locator assertion failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: topology path and lifecycle identity records
  - Additions to existing reusable library/module: locator payload helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes diagnostic payload shape
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - locator creation bounded by ambiguity count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - ambiguity records without exact locators are invalid
- Test strategy:
  - fixtures for split/merge, point birth/death, containment, and missing rail
    ambiguity diagnostics
- Data ownership:
  - topology owns entity identities; loft diagnostics own ambiguity payloads
- Routes:
  - planner ambiguity detection to locator builder to diagnostic aggregate
- Reuse/extraction decision:
  - use shared topology identity records rather than local string paths
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Suggested rails must be advice, not automatic mutation of authored topology.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
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
- Total: 18.5

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
- Review for split. Cohesion reason: all locator fields serve the same
  diagnostic contract and must be complete together.

### Candidate Spec: Seam Continuity Promotion And Diagnostics

Discovery purpose:
- Promote seam and continuity handling beyond C0/G0 storage by defining
  request records, supported continuity classes, validation, and diagnostics.

Responsibilities:
- Functions/methods:
  - continuity request validator
  - seam participation validator
  - unsupported continuity diagnostic builder
- Data structures/models:
  - continuity request record
  - continuity support record
  - seam participation validation result
- Dependencies/services:
  - seam/adjacency architecture
  - patch family promotion audit
  - tessellation watertightness checks
- Returns/outputs/signals:
  - continuity support verdict
  - unsupported continuity diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam and adjacency records
  - Additions to existing reusable library/module: continuity validation helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes validation and diagnostics for seam requests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - validation bounded by seam and boundary-use count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py` and seam/adjacency helpers
- Chosen defaults / parameters:
  - unsupported G1/G2 or curvature requests refuse with diagnostics rather
    than silently downgrading
- Test strategy:
  - seam validation, continuity request, watertightness, and unsupported-class
    tests across promoted families
- Data ownership:
  - seam records own boundary identity; continuity records own requested class
- Routes:
  - patch boundary records to seam validator to tessellation/CSG consumers
- Reuse/extraction decision:
  - extend seam/adjacency module; do not duplicate continuity rules in CSG or
    loft
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Exact G1/G2 enforcement may remain family-specific; the shared layer owns
  request and diagnostic consistency.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Review for split. Cohesion reason: request, validation, and diagnostics are
  one continuity contract; family-specific math remains in family specs.

### Candidate Spec: `.impress` Whole-Store Fixture Coverage Gate

Discovery purpose:
- Prove the `.impress` whole-store fixture covers all promoted families,
  topology rails, lifecycle records, trims, seams, identities, metadata, and
  operation provenance.

Responsibilities:
- Functions/methods:
  - whole-store fixture builder
  - all-family coverage assertion
- Data structures/models:
  - whole-store fixture record
  - topology rail payload
  - operation provenance payload
- Dependencies/services:
  - `.impress` root/store codecs
  - patch payload codecs
  - topology path records
- Returns/outputs/signals:
  - fixture coverage report
  - missing payload diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `.impress` codec specs and surface records
  - Additions to existing reusable library/module: whole-store fixture suite
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes temporary `.impress` fixture files during tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - fixture size and deterministic traversal bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `.impress` serialization modules and test fixtures
- Chosen defaults / parameters:
  - missing promoted-family payloads fail fixture coverage
- Test strategy:
  - all-family round-trip and identity/provenance preservation tests
- Data ownership:
  - `.impress` owns persistence; surface store owns runtime object identity
- Routes:
  - SurfaceBodyStore to writer to reader to validated SurfaceBodyStore
- Reuse/extraction decision:
  - extend existing `.impress` codec tests with whole-store fixture coverage
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This candidate intentionally excludes malformed/unsafe refusal behavior, which
  belongs in the paired refusal gate below.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
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
- Total: 18.5

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
- Review for split. Cohesion reason: this is one fixture coverage gate after
  separating refusal/security behavior.

### Candidate Spec: `.impress` Unsafe Payload Refusal And Determinism Gate

Discovery purpose:
- Prove malformed, unsupported, or unsafe `.impress` payloads refuse with
  deterministic diagnostics and never recover by inventing mesh or executing
  unsafe implicit payloads.

Responsibilities:
- Functions/methods:
  - malformed payload refusal assertion
  - deterministic error reporter
- Data structures/models:
  - diagnostic metadata payload
  - invalid payload fixture record
- Dependencies/services:
  - `.impress` reader
  - implicit payload safety policy
  - deterministic writer
- Returns/outputs/signals:
  - invalid payload refusal diagnostic
  - deterministic error report
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reader/load result contract
  - Additions to existing reusable library/module: unsafe payload fixture set
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes temporary invalid fixture files during tests
- Security/privacy-sensitive behavior:
  - rejects unsafe implicit and malformed payloads without recovery execution
- Performance-sensitive behavior:
  - deterministic error path bounded by payload size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `.impress` reader/load result modules and test fixtures
- Chosen defaults / parameters:
  - invalid payloads refuse; no mesh recovery or best-effort mutation
- Test strategy:
  - malformed payload, unsafe implicit payload, unsupported family payload, and
    deterministic error snapshot tests
- Data ownership:
  - `.impress` reader owns refusal diagnostics
- Routes:
  - reader validation to load result to deterministic diagnostic
- Reuse/extraction decision:
  - extend reader/load-result contract tests
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This split keeps security-sensitive payload refusal visible instead of hiding
  it inside general round-trip testing.

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
- Review for split. Cohesion reason: unsafe payload refusal and deterministic
  diagnostics are one reader/load-result gate.

### Candidate Spec: Primitive Patch Producer Selection

Discovery purpose:
- Make primitives choose exact surface patch families and refuse unsupported
  primitive producer requests without hidden mesh substitution.

Responsibilities:
- Functions/methods:
  - primitive family selector
  - unsupported primitive diagnostic
- Data structures/models:
  - primitive producer selection record
  - unsupported primitive producer result
- Dependencies/services:
  - primitive surface constructors
  - no-hidden-mesh-fallback gate
- Returns/outputs/signals:
  - selected patch family
  - surface truth result
  - explicit unsupported diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: primitive modules
  - Additions to existing reusable library/module: primitive producer selection
    helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes primitive authored output routing
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - selection should be deterministic and constant-time per primitive
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - primitive constructors
- Chosen defaults / parameters:
  - exact analytic families are preferred when available; mesh is never a
    hidden substitute
- Test strategy:
  - producer selection and no-hidden-mesh-fallback tests for primitives
- Data ownership:
  - primitives own authored family choice; surface body owns emitted geometry
- Routes:
  - public primitive API to producer selector to SurfaceBody output
- Reuse/extraction decision:
  - share primitive selection helper across primitive constructors
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Feature-builder handoff is split out because it depends on different output
  contracts.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
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
- Review for split. Cohesion reason: this is one primitive-family routing
  contract after feature-builder handoff was split out.

### Candidate Spec: Feature Builder Patch Producer Handoff

Discovery purpose:
- Make Impression-owned feature builders hand off `SurfaceBody` or
  `SurfaceConsumerCollection` truth with explicit unsupported diagnostics and
  no hidden mesh substitution.

Responsibilities:
- Functions/methods:
  - feature handoff validator
  - feature producer diagnostic
- Data structures/models:
  - feature surface output contract
  - feature unsupported producer result
- Dependencies/services:
  - loft surface producer
  - hinge feature builders
  - no-hidden-mesh-fallback gate
- Returns/outputs/signals:
  - surface truth result
  - consumer collection result
  - explicit unsupported diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft and feature builder modules
  - Additions to existing reusable library/module: feature handoff helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes authored feature output routing
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - handoff validation should be deterministic and bounded by feature output
    size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - loft surface producer and Impression-owned feature builders
- Chosen defaults / parameters:
  - feature builders return surface truth or explicit unsupported diagnostics;
    mesh is never a hidden substitute
- Test strategy:
  - no-hidden-mesh-fallback tests for loft and feature builders
- Data ownership:
  - feature builders own authored output contract; surface body owns emitted
    geometry
- Routes:
  - feature API to handoff validator to SurfaceBody/consumer output
- Reuse/extraction decision:
  - use one feature handoff helper rather than per-feature ad hoc checks
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Sibling projects may consume these helpers but are not Impression-owned
  implementation scope.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
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
- Total: 18.5

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
- Review for split. Cohesion reason: this is one feature-builder handoff
  contract after primitive routing was split out.

### Candidate Spec: Surface Body Completion Reference Evidence Matrix

Discovery purpose:
- Define the required reference images, STL/tessellation artifacts, round-trip
  fixtures, and negative diagnostics needed for every promoted model-outputting
  capability.

Responsibilities:
- Functions/methods:
  - evidence matrix checker
  - reference artifact completeness assertion
- Data structures/models:
  - completion evidence record
  - reference fixture requirement record
- Dependencies/services:
  - reference artifact lifecycle
  - no-hidden-mesh-fallback tests
  - `.impress` round-trip fixtures
  - CSG/loft/family fixtures
- Returns/outputs/signals:
  - missing evidence report
  - promotion-blocking failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reference artifact lifecycle tooling
  - Additions to existing reusable library/module: completion evidence matrix
    checker
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may write dirty reference artifacts during verification runs
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - fixture generation bounded by named matrix
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - reference artifact tests and release verification tooling
- Chosen defaults / parameters:
  - promoted model-outputting capabilities require positive and negative
    evidence
- Test strategy:
  - matrix completeness tests and promotion-gate failure tests
- Data ownership:
  - evidence matrix owns completion evidence requirements
- Routes:
  - capability matrix to required fixtures to reference/test gates
- Reuse/extraction decision:
  - extend existing reference artifact lifecycle instead of adding a separate
    artifact system
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Dirty reference artifacts should remain test output until explicitly
  promoted by the reference lifecycle.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 4 x 1 = 4
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
- Total: 18.5

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
- Review for split. Cohesion reason: this is one evidence matrix and promotion
  gate; individual fixtures remain owned by feature specs.

### Manifest Critical Review Summary

- First review: the umbrella completion candidate was too broad and hid CSG,
  loft, persistence, producer-routing, and evidence work.
- First split: separated patch-family promotion, CSG completion, loft
  ambiguity, seam continuity, `.impress`, producer routing, and evidence.
- Second review: `.impress` whole-store completion still scored `25.5`, because
  it bundled fixture coverage with unsafe payload refusal and deterministic
  error behavior.
- Second split: separated `.impress` whole-store fixture coverage from unsafe
  payload refusal and determinism.
- Third review: primitive and feature producer selection was still underdefined
  despite scoring below `25`, so primitive routing was split from feature-builder
  handoff.
- Fourth review: all remaining implementation candidates scored below `25`;
  `16-24` candidates retain explicit cohesion explanations and no readiness
  blockers.

## Work Register

The architecture work tracker owns the durable to-do list for this completion
program:

- [Architecture Work Tracker](architecture-work-tracker.md)

No downstream specification manifest should be considered complete until the
tracker entries for this document are either specified, implemented, or
explicitly retired by a later architecture decision.

## Change History

- 2026-05-27: Added the specification manifest for discovery, critically
  reviewed the broad completion work, split oversized `.impress` persistence
  work, split primitive producer routing from feature-builder handoff, and
  rescored all final candidates below the split-required threshold.
- 2026-05-27: Created the surface-body completion umbrella architecture after
  the release progression was fully checked but the capability audit still
  found planned patch families, bounded CSG support, loft ambiguity execution
  boundaries, seam continuity limits, and `.impress` promotion work.
