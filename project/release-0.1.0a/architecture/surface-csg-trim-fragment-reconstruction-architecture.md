# Surface CSG Trim Fragment Reconstruction Architecture

This document defines the shared downstream CSG architecture required after
surface-surface intersections have been computed.

Intersection solvers are necessary but not sufficient. A CSG row is not
implemented until its intersection curves can be mapped into patch-local trim
space, used to split surface fragments, classified against the operation, and
reassembled into a valid `SurfaceBody` with seams, provenance, and persistence
intact.

## Relationship To Other Architecture

This document refines:

- [Surface CSG Executable Completion Architecture](surface-csg-executable-completion-architecture.md)
- [Higher-Order Surface CSG Solver Architecture](higher-order-surface-csg-solver-architecture.md)
- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)

It is consumed by:

- [Higher-Order Parametric CSG Routes Architecture](higher-order-parametric-csg-routes-architecture.md)
- [Sampled and Implicit Surface CSG Support Architecture](sampled-implicit-surface-csg-support-architecture.md)
- [Exact Surface Intersection Kernel Architecture](exact-surface-intersection-kernel-architecture.md)

Solver architecture owns how intersections are found. This document owns what
happens after intersections exist.

## Completion Definition

A surface CSG execution path is complete only when it can:

- map every intersection curve into the local domains of all affected patches
- represent coincident regions as first-class split inputs
- build a trim arrangement graph per patch
- split patches into candidate fragments without mesh CSG
- classify fragments for union, intersection, and difference
- generate required caps or refuse with exact unsupported-cap diagnostics
- assemble a watertight or explicitly open result shell according to operation
- rebuild seams, adjacency, and provenance
- serialize the result through `.impress`
- tessellate the result only after the surface body exists

## Components

### Patch-Local Curve And Region Mapper

The mapper receives intersection records from solver routes and converts them
into patch-local trim entities.

Inputs:

- patch ids
- family ids
- world-space curve records
- patch-domain curve records where already available
- coincident-region records
- declared tolerance and residual metadata

Outputs:

- patch-local trim curves
- trim-loop candidates
- coincident-region loops
- mapping diagnostics

The mapper must preserve both sides of every split. A curve that maps on one
patch but not the other is a non-executable diagnostic.

### Trim Arrangement Graph

The trim arrangement graph owns the per-patch subdivision induced by:

- existing patch boundaries
- existing trims
- new CSG intersection curves
- coincident-region boundaries
- generated cap boundaries

The graph is topological, not mesh-based. It stores vertices, oriented edges,
loops, face candidates, tolerances, and source provenance.

### Fragment Builder

The fragment builder converts arrangement faces into surface fragments:

- original patch fragment
- trimmed patch fragment
- coincident shared fragment
- generated cap fragment

Fragments carry their source patch family unless a cap policy creates a new
patch family explicitly.

### Fragment Classifier

The classifier determines whether each fragment is inside, outside, on-boundary,
or coincident relative to the opposite body.

Classification uses:

- surface normal and orientation records
- patch-local point sampling inside arrangement faces
- exact or declared-tolerance point-in-body tests
- coincident-region ownership rules
- operation-specific selection rules

Classification diagnostics must identify ambiguous or numerically unstable
fragments by source patch, local region, operation, and pair route.

### Operation Selector

The operation selector applies boolean semantics:

- union keeps exterior fragments and resolves coincident shared surfaces
- intersection keeps interior/shared fragments
- difference keeps left exterior fragments and reversed right interior/cap
  fragments

The selector owns orientation changes, but not geometry mutation.

### Cap Policy

CSG can produce open boundaries that need caps. The cap policy determines:

- whether the boundary is cap-eligible
- which patch family can represent the cap
- whether the cap is exact, declared-tolerance, or unsupported
- what provenance and `.impress` payload the cap receives

Unsupported cap cases refuse execution. They must not fall back to mesh caps.

### Shell Assembler

The shell assembler creates the result `SurfaceBody`:

- fragments become patches
- trim loops become patch boundaries
- matching boundaries become seams
- open boundaries are retained only for operations that explicitly allow them
- provenance maps result patches to source patches or generated caps

The assembler validates adjacency before returning the body.

### Seam And Adjacency Rebuilder

The rebuilder owns:

- seam identity
- edge orientation
- tolerance records
- continuity class records
- source and generated adjacency provenance

It integrates with higher-order continuity architecture. C0/G0 seams can be
created immediately; G1/G2/C1/C2 enforcement requires the continuity
implementation path and cannot be silently claimed.

### Result Validity Gate

The validity gate checks:

- no dangling trim edges
- no unclassified fragments
- no missing cap policies
- no non-executable route diagnostics
- no mesh-backed fragments
- result body can serialize through `.impress`
- tessellation succeeds as a boundary operation

Failure returns structured diagnostics and no result body.

## Data Flow

1. Intersection routes emit curves, regions, residuals, and degeneracy records.
2. The patch-local mapper translates route output into per-patch trim space.
3. The trim arrangement graph integrates new curves with existing boundaries
   and trims.
4. The fragment builder creates candidate surface fragments.
5. The classifier labels fragments against the opposite body.
6. The operation selector chooses fragments according to boolean semantics.
7. The cap policy creates supported caps or emits unsupported-cap diagnostics.
8. The shell assembler builds a `SurfaceBody`.
9. The seam rebuilder records adjacency and continuity data.
10. The validity gate approves the result for persistence and later
    tessellation.

## Cross-Domain Decisions

### Reconstruction Is Shared Infrastructure

Higher-order, sampled, implicit, and analytic CSG routes all feed the same
reconstruction pipeline. Pair-specific solver code must not embed one-off shell
assembly logic.

### Tessellation Is After SurfaceBody Creation

Tessellation may verify, preview, or export the final result. It cannot provide
fragment splitting, inside/outside classification, caps, or shell assembly.

### Coincidence Is A First-Class Geometry State

Coincident and overlapping regions are not errors by default. They are explicit
inputs to arrangement and selection. They become diagnostics only when ownership
or orientation is ambiguous.

### Caps Are Patch Families, Not Mesh Patches

Generated caps must be represented as supported patch families with
provenance. If no supported cap family can represent the required geometry, the
operation refuses.

### Validity Must Be Reportable

Every refusal must identify the source body, patch, route, fragment, or trim
entity responsible. Planning can accumulate all diagnostics, but execution
cannot produce partial ambiguous results.

## Specification Manifest for Discovery

### Five-Round Critical Review Log

Round 1:
- Found two oversized candidates that bundled runtime validation with
  persistence/reference evidence and shell assembly with seam/provenance work.
- Split all entries at or above 25 before implementation promotion.

Round 2:
- Found patch-local mapping, arrangement, classification, selection, and cap
  policy were near the split threshold but cohesive if kept to one processing
  stage each.
- Kept them under explicit split-review status with readiness blockers carried.

Round 3:
- Found generated caps needed a stronger no-mesh boundary and unsupported-cap
  diagnostic contract.
- Added cap-specific readiness and fixture requirements to the operation/cap
  candidate.

Round 4:
- Found result validity was mixing runtime checks, `.impress` round trip, and
  reference artifact promotion.
- Split runtime validity from persistence/tessellation evidence.

Round 5:
- Rechecked for deferred, unsupported, intentionally-not-implemented, mesh
  fallback, dirty reference evidence, and source geometry mutation.
- Final review found no remaining implementation candidate over 25; remaining
  16-24.5 candidates have explicit cohesion explanations.

### Candidate Spec: CSG Patch-Local Curve And Region Mapping Completion

Discovery purpose:
- Convert solver intersection output into patch-local trim entities for every
  affected surface family.

Responsibilities:
- Functions/methods:
  - patch-local curve mapper
  - coincident-region mapper
  - mapping diagnostic builder
- Data structures/models:
  - patch-local trim curve
  - coincident-region loop
  - mapping failure diagnostic
- Dependencies/services:
  - intersection route results
  - patch evaluators
  - tolerance records
- Returns/outputs/signals:
  - patch-local trim entities
  - mapping diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: patch identity and trim records
  - Additions to existing reusable library/module: CSG trim mapping helpers
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
  - bounded curve projection and validation budgets
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - both affected patches must receive valid local mappings
- Test strategy:
  - crossing, tangent, boundary, coincident, and mapping-failure fixtures
- Data ownership:
  - intersection kernel owns raw curves; CSG owns trim mapping
- Routes:
  - solver result to mapper to trim arrangement graph
- Open questions / nuance discovered:
  - subdivision and sweep local domains need explicit adapter records
- Readiness blockers:
  - family evaluator adapters

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 0 x 2 = 0
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only mapping into
  trim space, not arrangement or reconstruction.

### Candidate Spec: CSG Trim Arrangement Graph Completion

Discovery purpose:
- Build the per-patch trim arrangement graph used to split surfaces after CSG
  intersections.

Responsibilities:
- Functions/methods:
  - arrangement graph builder
  - curve intersection/node insertion
  - loop extraction
- Data structures/models:
  - arrangement vertex
  - arrangement edge
  - arrangement face candidate
- Dependencies/services:
  - patch-local trim curves
  - existing patch trims
  - tolerance policy
- Returns/outputs/signals:
  - arrangement graph
  - invalid arrangement diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current trim payloads
  - Additions to existing reusable library/module: arrangement graph builder
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG execution behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded graph construction for dense trim curves
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - invalid graph topology refuses execution
- Test strategy:
  - simple split, multiple curves, coincident boundaries, nested loops, and
    invalid loop fixtures
- Data ownership:
  - CSG owns arrangement graphs
- Routes:
  - patch-local trim entities to arrangement graph to fragment builder
- Open questions / nuance discovered:
  - graph tolerance must align with intersection residuals
- Readiness blockers:
  - patch-local curve mapping

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
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: graph building is a reusable boundary but
  does not include classification or shell assembly.

### Candidate Spec: CSG Fragment Builder And Classifier Completion

Discovery purpose:
- Split arrangement faces into fragments and classify each fragment for boolean
  operation selection.

Responsibilities:
- Functions/methods:
  - fragment builder
  - inside/outside classifier
  - coincident ownership resolver
- Data structures/models:
  - surface fragment
  - classification record
  - coincident ownership diagnostic
- Dependencies/services:
  - trim arrangement graph
  - point-in-body tests
  - operation planner
- Returns/outputs/signals:
  - classified fragments
  - classification diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: surface body topology records
  - Additions to existing reusable library/module: fragment classification
    helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG execution behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded classification probes per fragment
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - ambiguous classification refuses execution after all diagnostics are
    collected
- Test strategy:
  - union/intersection/difference fragment selection, coincident ownership,
    tangent contact, and ambiguous classification fixtures
- Data ownership:
  - CSG owns fragment classification
- Routes:
  - arrangement graph to fragment builder to operation selector
- Open questions / nuance discovered:
  - sampled families may require separate point-in-body policy
- Readiness blockers:
  - arrangement graph

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
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: builder and classifier are tightly linked
  by arrangement-face ownership but remain separate from shell assembly.

### Candidate Spec: CSG Operation Selection And Cap Policy Completion

Discovery purpose:
- Apply boolean operation semantics to classified fragments and generate or
  refuse required caps.

Responsibilities:
- Functions/methods:
  - operation fragment selector
  - cap eligibility classifier
  - cap patch generator/refusal builder
- Data structures/models:
  - selected fragment record
  - cap boundary record
  - unsupported cap diagnostic
- Dependencies/services:
  - fragment classifier
  - patch family capability matrix
  - generated patch builders
- Returns/outputs/signals:
  - selected fragments
  - generated caps or cap diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: patch family builders
  - Additions to existing reusable library/module: CSG cap policy helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG result construction
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - cap fit validation must be bounded
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported caps refuse; mesh caps are forbidden
- Test strategy:
  - union, intersection, difference, planar cap, spline cap, non-representable
    cap, and cap refusal fixtures
- Data ownership:
  - CSG owns operation selection and generated cap provenance
- Routes:
  - classified fragments to selector to shell assembler
- Open questions / nuance discovered:
  - non-planar cap promotion may need B-spline or NURBS cap builders
- Readiness blockers:
  - patch family cap builders

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
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: operation selection and cap policy are the
  boolean semantic layer between classification and assembly.

### Split Discovery Finding: CSG Shell Assembly And Seam Rebuild Completion

Discovery purpose:
- Assemble selected fragments and generated caps into a valid `SurfaceBody`
  with rebuilt seams, adjacency, orientation, and provenance.

Responsibilities:
- Functions/methods:
  - result shell assembler
  - seam adjacency rebuilder
  - orientation/provenance mapper
- Data structures/models:
  - result shell record
  - rebuilt seam record
  - result provenance map
- Dependencies/services:
  - operation selector
  - surface body store
  - seam adjacency infrastructure
- Returns/outputs/signals:
  - result `SurfaceBody`
  - shell validity diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: `SurfaceBody` and seam records
  - Additions to existing reusable library/module: CSG shell assembly helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - returns new body without mutating operands
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded adjacency rebuild across fragment count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - operands are immutable; result provenance points back to source patches
- Test strategy:
  - closed shell, open refusal, orientation reversal, seam rebuild, provenance,
    and `.impress` round-trip fixtures
- Data ownership:
  - CSG owns result construction; surface body store owns persistence shape
- Routes:
  - selected fragments to shell assembler to validity gate
- Open questions / nuance discovered:
  - higher-order continuity enforcement may remain operation-owned, not source
    mutation
- Readiness blockers:
  - operation selection and cap policy

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 26

Split decision:
- Split completed in this manifest. This broad discovery finding is superseded
  by the shell assembly and seam/provenance rebuild candidates below.

### Candidate Spec: CSG Result Shell Assembly Completion

Discovery purpose:
- Assemble selected fragments and generated caps into a new immutable
  `SurfaceBody` result without mutating operands.

Responsibilities:
- Functions/methods:
  - result shell assembler
  - fragment orientation applicator
  - open-boundary diagnostic builder
- Data structures/models:
  - result shell record
  - oriented fragment record
  - open-boundary diagnostic
- Dependencies/services:
  - operation selector
  - surface body store
  - cap policy
- Returns/outputs/signals:
  - candidate result `SurfaceBody`
  - shell assembly diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: `SurfaceBody` patch and trim records
  - Additions to existing reusable library/module: CSG shell assembly helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - returns a new body and never mutates operands
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded assembly across selected fragment count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - missing caps or invalid open boundaries refuse execution
- Test strategy:
  - closed shell, open-boundary refusal, difference orientation, cap inclusion,
    and operand immutability fixtures
- Data ownership:
  - CSG owns result construction; source operands remain immutable
- Routes:
  - selected fragments to shell assembler to seam/provenance rebuild
- Open questions / nuance discovered:
  - shell assembly must preserve patch-family payloads without promoting to
    mesh
- Readiness blockers:
  - operation selection and cap policy

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: this candidate owns assembly of selected
  fragments only; seam/provenance rebuild is split out.

### Candidate Spec: CSG Seam Adjacency And Provenance Rebuild Completion

Discovery purpose:
- Rebuild result seams, adjacency, orientation metadata, and provenance after
  CSG shell assembly.

Responsibilities:
- Functions/methods:
  - seam adjacency rebuilder
  - orientation/provenance mapper
  - continuity handoff recorder
- Data structures/models:
  - rebuilt seam record
  - result provenance map
  - continuity handoff record
- Dependencies/services:
  - result shell assembler
  - seam adjacency infrastructure
  - higher-order continuity architecture
- Returns/outputs/signals:
  - rebuilt adjacency graph
  - seam/provenance diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: seam and adjacency records
  - Additions to existing reusable library/module: CSG seam/provenance rebuild
    helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - annotates the new result body only
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded adjacency matching across fragment edges
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - C0/G0 adjacency can be recorded immediately; G1/G2/C1/C2 enforcement is
    handed to continuity implementation and cannot be silently claimed
- Test strategy:
  - seam matching, reversed orientation, generated cap provenance,
    continuity-handoff, and missing adjacency fixtures
- Data ownership:
  - CSG owns result provenance; seam system owns adjacency representation
- Routes:
  - shell assembler to seam rebuild to result validity gate
- Open questions / nuance discovered:
  - continuity enforcement must be operation-owned, not source mutation
- Readiness blockers:
  - result shell assembly

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: this candidate owns seam, adjacency, and
  provenance rebuild as one post-assembly boundary.

### Split Discovery Finding: CSG Result Validity And Persistence Gate

Discovery purpose:
- Validate that CSG results are surface-native, persistable, and tessellated
  only after result-body creation.

Responsibilities:
- Functions/methods:
  - CSG result validity checker
  - no-mesh-fragment assertion
  - `.impress` round-trip gate
- Data structures/models:
  - result validity report
  - persistence evidence record
  - tessellation-boundary assertion
- Dependencies/services:
  - shell assembler
  - `.impress` codec
  - tessellation boundary
- Returns/outputs/signals:
  - pass/fail validity report
  - exact invalid-result diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: `.impress` codec and tessellation helpers
  - Additions to existing reusable library/module: CSG result validity gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes reference artifacts through existing lifecycle when requested
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - round-trip and tessellation checks must remain CI-bounded
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `tests/test_impress_io.py`
- Chosen defaults / parameters:
  - dirty references and mesh-backed fragments fail completion evidence
- Test strategy:
  - valid body, invalid dangling trim, missing seam, mesh-backed fragment,
    `.impress` round-trip, and tessellation-boundary fixtures
- Data ownership:
  - CSG owns validity; `.impress` owns persistence encoding
- Routes:
  - shell assembler to validity gate to persistence/tessellation evidence
- Open questions / nuance discovered:
  - reference promotion may need separate smoke and acceptance tiers
- Readiness blockers:
  - shell assembly

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 25

Split decision:
- Split completed in this manifest. This broad discovery finding is superseded
  by runtime validity and persistence/evidence candidates below.

### Candidate Spec: CSG Runtime Result Validity Gate

Discovery purpose:
- Validate that CSG runtime results are surface-native, topologically valid, and
  free of unresolved execution diagnostics before return.

Responsibilities:
- Functions/methods:
  - CSG result validity checker
  - dangling-trim detector
  - no-mesh-fragment assertion
- Data structures/models:
  - result validity report
  - invalid topology diagnostic
  - tessellation-boundary assertion
- Dependencies/services:
  - shell assembler
  - seam adjacency rebuild
  - tessellation boundary
- Returns/outputs/signals:
  - pass/fail validity report
  - exact invalid-result diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: surface body and tessellation helpers
  - Additions to existing reusable library/module: CSG runtime validity gate
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
  - validity checks must be bounded by patch/edge count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - mesh-backed fragments, dangling trims, and unresolved diagnostics fail
    runtime validity
- Test strategy:
  - valid body, dangling trim, missing seam, mesh-backed fragment, unresolved
    diagnostic, and tessellation-boundary fixtures
- Data ownership:
  - CSG owns runtime validity
- Routes:
  - seam rebuild to runtime validity gate to returned `SurfaceBody`
- Open questions / nuance discovered:
  - validity errors must point to result patch and source provenance
- Readiness blockers:
  - seam adjacency and provenance rebuild

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
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 0 x 2 = 0
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns runtime validity only;
  persistence and reference evidence are split out.

### Candidate Spec: CSG Persistence Tessellation And Reference Evidence Gate

Discovery purpose:
- Prove CSG results serialize through `.impress`, tessellate only at the
  boundary, and promote clean reference evidence.

Responsibilities:
- Functions/methods:
  - `.impress` round-trip gate
  - tessellation-boundary verifier
  - reference evidence promoter
- Data structures/models:
  - persistence evidence record
  - tessellation evidence record
  - reference promotion report
- Dependencies/services:
  - `.impress` codec
  - tessellation boundary
  - reference artifact lifecycle
- Returns/outputs/signals:
  - persistence pass/fail
  - promoted or dirty reference evidence
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: `.impress` codec and reference artifact helpers
  - Additions to existing reusable library/module: CSG evidence gate helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes reference artifacts through the existing lifecycle
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - round-trip and tessellation checks must remain CI-bounded
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/test_impress_io.py`
  - `tests/test_surface_kernel.py`
  - `project/release-0.1.0a/reference-artifacts/`
- Chosen defaults / parameters:
  - dirty artifacts do not count as completion evidence
- Test strategy:
  - `.impress` round trip, tessellation boundary, clean reference promotion,
    dirty reference rejection, and no-mesh-execution fixtures
- Data ownership:
  - `.impress` owns persistence; tests own evidence promotion
- Routes:
  - runtime-valid CSG result to persistence gate to reference evidence report
- Open questions / nuance discovered:
  - large CSG fixture sets may need smoke and acceptance tiers
- Readiness blockers:
  - runtime result validity gate

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 24

Split decision:
- Review for split. Cohesion reason: this candidate owns persistence,
  tessellation-boundary, and evidence promotion as one completion gate.

## Change History

- 2026-05-30: Critically reviewed the specification manifest through five
  review/rescore/split loops. Reason: result shell assembly and validity
  candidates still bundled hidden downstream work; the manifest now splits
  shell assembly, seam/provenance rebuild, runtime validity, and
  persistence/reference evidence.
- 2026-05-28: Added shared CSG trim, fragment, shell reconstruction, seam
  rebuild, and validity architecture. Reason: higher-order and sampled CSG rows
  require common surface-native reconstruction work after solver routes exist;
  intersections alone do not make rows executable.
