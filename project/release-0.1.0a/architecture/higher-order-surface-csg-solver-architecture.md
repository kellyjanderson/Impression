# Higher-Order Surface CSG Solver Architecture

## Overview

This document defines the architecture for expanding SurfaceBody CSG from
bounded analytic cases into a full higher-order boolean solver across every
promoted patch-family pair.

The target system is not "try mesh if the surface solver cannot answer." The
target system is:

- choose a surface-native solver route for every patch-family pair
- execute exact or declared-tolerance intersections when supported
- refuse unsupported pairs with structured, family-specific diagnostics
- keep all result truth in surfaces, trims, seams, shells, and provenance
- use tessellation only after the result is already a `SurfaceBody`

## Related Architecture

This document extends:

- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [Surface Body Completion Architecture](surface-body-completion-architecture.md)
- [Exact Surface Intersection Kernel Architecture](exact-surface-intersection-kernel-architecture.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)

## Patch-Family Pair Coverage

The solver matrix must include every promoted family as both left and right
operand:

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

Each ordered pair is classified into one of these support states:

- `exact`: analytic or symbolic intersection and trimming are supported
- `declared-tolerance`: numeric solver returns bounded-error surface-native
  curves, trims, and diagnostics
- `sampled-boundary`: accepted only if the family is fundamentally sampled and
  the result remains a sampled surface record, not mesh truth
- `unsupported`: solver refuses before execution with a diagnostic

No pair is allowed to be "unknown." Unknown means the family matrix is
incomplete.

## Components

### Family Pair Solver Registry

The registry owns the durable solver map:

- operand family pair
- operation set covered
- solver class
- expected intersection curve family
- result trim representation
- tolerance policy
- refusal policy

The registry is the only place where pair coverage is declared. Boolean
entrypoints, primitive helpers, and feature builders must not carry local
hard-coded pair decisions.

### Operation Planner

The planner converts operands and operation intent into a bounded solver plan.

It owns:

- operand eligibility
- operation-specific pair ordering
- shell and patch traversal order
- preclassification of no-cut, containment, overlap, disjoint, tangent, and
  coincident cases
- solver dispatch records

The planner may produce a plan containing unsupported pair diagnostics. Such a
plan is reportable but not executable.

### Patch Pair Intersection Executor

The executor consumes one pair plan at a time and emits surface-native
intersection records.

It owns:

- calling the exact or numeric intersection kernel
- capturing residuals and tolerance metadata
- mapping 3D curves to both patch-local parameter spaces
- detecting coincident overlap regions versus simple crossing curves

The executor never emits triangles as boolean truth.

### Fragment Graph Builder

The graph builder consumes intersection records and operand trims.

It owns:

- splitting trim loops and patch domains
- labeling fragments by source patch and operation
- preserving source provenance
- grouping fragments into candidate result shells

The graph is transient execution state, but every emitted fragment must have a
deterministic provenance path.

### Result Reconstructor

The reconstructor turns classified fragments into a `SurfaceBody`.

It owns:

- patch reuse versus trimmed patch creation
- cap and cut-boundary construction
- seam rebuild requests
- shell assembly order
- empty, multi-shell, and nested-shell outcomes

It must not replace higher-order fragments with planar or mesh approximations
unless the result family itself is explicitly sampled and the operation policy
allows that sampled-family output.

### Validity And Refusal Gate

The final gate owns:

- closed/open shell validity
- trim loop validity
- seam pairing validity
- non-manifold detection
- bounded cleanup policy
- refusal records for unresolved solver conditions

Failure is a first-class outcome. It is better to return a complete diagnostic
than to invent geometry that looks plausible.

## Data Flow

```text
SurfaceBody operands
-> family-pair solver registry
-> operation planner
-> patch-pair intersection executor
-> fragment graph builder
-> result reconstructor
-> seam/validity gate
-> SurfaceBody boolean result or structured refusal
```

## Cross-Domain Decisions

### Pair Coverage Is A Matrix, Not A Set Of If Statements

The family matrix is part of the kernel contract. The code should support
auditing it directly.

### Exact And Declared-Tolerance Solvers Share The Same Result Shape

Exact and numeric solvers differ in how they compute curves. They should not
produce different downstream object shapes. Both feed the same fragment graph,
trim, seam, and provenance records.

### Sampled Families Are Still Surface Families

Heightmap and displacement operations may need sampled-surface policies. Their
presence does not authorize mesh fallback. A sampled result is still a
surface-family record with sampling metadata, bounds, and `.impress` payloads.

### Unsupported Is Better Than Hidden Fallback

If a pair cannot produce surface truth, the solver refuses before execution.
The diagnostic must identify:

- operation
- left family
- right family
- solver stage
- missing capability
- recommended authored workaround when one exists

## Specification Manifest for Discovery

### Candidate Spec: Surface CSG Family Pair Solver Registry

Discovery purpose:
- Define the auditable registry that classifies every promoted family pair for
  every boolean operation.

Responsibilities:
- Functions/methods:
  - family-pair registry builder
  - coverage assertion
  - support-state lookup
- Data structures/models:
  - solver registry record
  - family-pair support record
  - unsupported-pair diagnostic
- Dependencies/services:
  - patch family capability matrix
  - CSG operation set
- Returns/outputs/signals:
  - support classification
  - missing-pair coverage failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG family support records
  - Additions to existing reusable library/module: CSG solver registry helpers
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
  - constant-time pair lookup
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unknown pairs fail coverage; unsupported pairs must be explicit
- Test strategy:
  - matrix completeness tests for all promoted family pairs
- Data ownership:
  - CSG owns solver support truth
- Routes:
  - boolean entrypoints to registry to planner
- Open questions / nuance discovered:
  - sampled-family pairs need support-state names that do not imply mesh truth
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Split decision:
- Review for split. Cohesion reason: registry, lookup, and coverage assertion
  are one durable matrix contract.

### Candidate Spec: Surface CSG Operation Planning And Executability

Discovery purpose:
- Produce boolean plans that accumulate unsupported pair and invalid topology
  diagnostics before execution.

Responsibilities:
- Functions/methods:
  - operation planner
  - executability gate
  - diagnostic accumulator
- Data structures/models:
  - CSG operation plan
  - pair dispatch record
  - plan diagnostic record
- Dependencies/services:
  - solver registry
  - operand preparation
  - seam validation
- Returns/outputs/signals:
  - executable plan
  - non-executable diagnostic bundle
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: boolean operand records
  - Additions to existing reusable library/module: planner records and gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean execution refusal behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by patch-pair count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - plans with unresolved unsupported diagnostics cannot execute
- Test strategy:
  - multi-diagnostic plan tests and executor refusal tests
- Data ownership:
  - operation plan owns executability state
- Routes:
  - public boolean API to planner to executor
- Open questions / nuance discovered:
  - coincident cases may need separate exact-overlap records
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Split decision:
- Review for split. Cohesion reason: planning and executability are one
  lifecycle invariant; intersection execution is split separately.

### Candidate Spec: Surface CSG Fragment Graph And Result Reconstruction

Discovery purpose:
- Convert classified patch fragments into durable result shells without mesh
  substitution.

Responsibilities:
- Functions/methods:
  - fragment graph builder
  - operation fragment selector
  - result reconstructor
- Data structures/models:
  - fragment graph
  - result shell assembly record
  - cap/cut-boundary record
- Dependencies/services:
  - intersection records
  - seam rebuild
  - trim loops
- Returns/outputs/signals:
  - reconstructed SurfaceBody
  - invalid reconstruction diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current fragment/provenance records
  - Additions to existing reusable library/module: graph assembly helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean result construction
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded graph traversal by fragment count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - invalid graph reconstruction refuses; no geometric invention
- Test strategy:
  - trim, cap, multi-shell, empty-result, and provenance tests
- Data ownership:
  - fragment graph owns transient execution truth; result owns durable truth
- Routes:
  - intersection records to graph to reconstructed SurfaceBody
- Open questions / nuance discovered:
  - cap family selection must coordinate with patch-family registry
- Readiness blockers:
  - exact intersection record contract from the intersection architecture

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: graph building and reconstruction form the
  single handoff from transient boolean truth to durable body truth. Cap family
  details may split later if this candidate grows.

## Change History

- 2026-05-27: Added higher-order CSG solver architecture and manifest.
  Context: the surface-body system needed architecture for broad family-pair
  CSG coverage beyond the current bounded support matrix.
