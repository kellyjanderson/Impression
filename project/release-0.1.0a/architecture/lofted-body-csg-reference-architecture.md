# Lofted Body CSG Reference Architecture

## Overview

This document defines the architecture required before `RT-LOFT-CSG-001`
through `RT-LOFT-CSG-014` can become honest reference fixtures.

Current runtime probes show representative loft CSG operations refusing because
the loft result is not yet eligible as a connected single closed-valid loft
shell, or because branching loft topology is not accepted. That means the
missing work is not fixture plumbing; it is loft-result eligibility, solver
routing, topology policy, and reference evidence.

## Related Architecture

This document extends:

- [Reference CSG Gap Closure Architecture](reference-csg-gap-closure-architecture.md)
- [Loft Planner / Executor Architecture](loft-planner-executor-architecture.md)
- [Loft Plan Object Architecture](loft-plan-object-architecture.md)
- [Loft N->M / M->N Decomposition Architecture](loft-nm-mn-decomposition-architecture.md)
- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [Higher-Order Surface CSG Solver Architecture](higher-order-surface-csg-solver-architecture.md)
- [Reference Artifact Promotion Architecture](reference-artifact-promotion-architecture.md)

## Reference Items Owned

This architecture owns:

- `RT-LOFT-CSG-001` lofted cylinder difference box slot
- `RT-LOFT-CSG-002` lofted cylinder difference cross-drilled cylinder
- `RT-LOFT-CSG-003` lofted vessel difference sphere scoop
- `RT-LOFT-CSG-004` hourglass loft intersection box
- `RT-LOFT-CSG-005` square-correspondence loft union post
- `RT-LOFT-CSG-006` phase-shifted loft difference vertical slot
- `RT-LOFT-CSG-007` branching manifold difference sphere at branch joint
- `RT-LOFT-CSG-008` branching manifold intersection cutter window
- `RT-LOFT-CSG-009` loft union loft with overlapping ruled patches
- `RT-LOFT-CSG-010` loft intersection loft with crossing axes
- `RT-LOFT-CSG-011` loft difference loft where cutter shares station plane
- `RT-LOFT-CSG-012` lofted body CSG with authored colors preserved
- `RT-LOFT-CSG-013` lofted body CSG with section expected/actual/diff evidence
- `RT-LOFT-CSG-014` lofted body CSG refusal where topology is underconstrained

## Components

### Loft CSG Eligibility Gate

The eligibility gate decides whether a loft result may enter CSG execution.

It owns:

- connected shell validation
- closed shell validation
- cap validity checks
- seam and adjacency completeness
- loft provenance requirements
- branching topology acceptance or refusal policy
- authored color and metadata preservation requirements

### Loft Patch Family Route Mapper

The mapper translates loft-generated patches into CSG solver families:

- ruled side walls
- planar caps
- revolution-compatible lofts when applicable
- higher-order fitted or smooth loft patches
- branch connector patches

The mapper must not erase loft provenance merely to fit existing primitive CSG
shortcuts.

### Loft CSG Operation Planner

The planner owns operation-specific behavior for loft operands:

- loft difference primitive cutter
- loft intersection primitive cutter
- loft union primitive post
- loft/loft union
- loft/loft intersection
- loft/loft difference
- underconstrained topology refusal

### Branching Loft CSG Policy

Branching lofts need their own policy because a branch joint is not the same as
a single ruled shell. The policy decides which branch structures can execute,
which require decomposition into sub-bodies, and which must refuse.

### Reference Evidence Producer

The producer owns the reference artifacts for successful and refused loft CSG
cases.

Success fixtures produce dirty STLs. Refusal fixtures produce structured
diagnostic evidence. Section expected/actual/diff cases produce comparison
evidence alongside or instead of a single STL, depending on the final fixture
schema.

## Data Flow

```text
loft inputs
-> loft planner/executor
-> SurfaceBody with loft provenance
-> loft CSG eligibility gate
-> patch family route mapper
-> CSG operation planner
-> solver route / branching policy / refusal
-> SurfaceBody result or diagnostic evidence
-> reference fixture registry
```

## Cross-Domain Decisions

### Loft Eligibility Must Be Explicit

The CSG entrypoint should not guess whether a loft is valid enough to cut. The
loft executor must provide connectedness, closedness, cap, seam, and provenance
evidence that CSG can consume.

### Branching Loft CSG Is A Separate Policy

Branching manifolds should not be forced through the same path as a single
lofted cylinder. They may require decomposition or deliberate refusal.

### Metadata And Color Are Kernel Inputs To The Reference Contract

`RT-LOFT-CSG-012` requires authored colors to survive CSG. The CSG result
provenance map must therefore preserve color/material ownership rules in a
deterministic way.

### Section Evidence Is Not Just A Render

`RT-LOFT-CSG-013` requires expected/actual/diff section evidence. That is a
cross-section verification product, not only an STL export.

## Specification Manifest for Discovery

### Candidate Spec: Loft CSG Eligibility Gate

Discovery purpose:
- Define the loft-result validity and provenance contract required before any
  lofted `SurfaceBody` may enter CSG execution.

Responsibilities:
- Functions/methods:
  - loft CSG eligibility checker
  - loft shell validity summarizer
  - eligibility diagnostic builder
- Data structures/models:
  - loft CSG eligibility record
  - loft shell validity record
  - eligibility diagnostic
- Dependencies/services:
  - loft executor output
  - SurfaceBody seam/adjacency records
  - CSG operand preparation
- Returns/outputs/signals:
  - eligible loft operand signal
  - structured refusal diagnostic
  - provenance evidence payload
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft executor and SurfaceBody validity records
  - Additions to existing reusable library/module: loft/CSG eligibility helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG acceptance/refusal behavior for loft operands
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by loft patch, seam, and cap count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - disconnected, open, invalid-cap, or underconstrained branch lofts refuse
- Test strategy:
  - eligibility unit tests plus reference probes for single-shell and branching
    lofts
- Data ownership:
  - loft owns provenance/validity evidence; CSG owns acceptance decision
- Routes:
  - loft executor to eligibility checker to CSG operand preparation
- Reuse/extraction decision:
  - add to existing loft and CSG modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Existing loft outputs may need stronger closed-shell evidence before this
  gate can distinguish implementation gaps from invalid input.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Readiness blockers:
- [ ] Loft executor must expose reliable connected, closed, cap, seam, and
  provenance evidence.

Split decision:
- Review for split. Cohesion reason: eligibility is one boundary between loft
  output and CSG operand preparation.

### Candidate Spec: Loft Primitive Difference And Intersection Routes

Discovery purpose:
- Implement single-shell loft difference and intersection routes for primitive
  cutters such as boxes, cylinders, and spheres.

Responsibilities:
- Functions/methods:
  - loft primitive difference route
  - loft primitive intersection route
- Data structures/models:
  - loft CSG operation plan
  - loft patch-family route record
- Dependencies/services:
  - loft CSG eligibility gate
  - higher-order CSG solver registry
- Returns/outputs/signals:
  - CSG SurfaceBody result
  - operation refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: higher-order solver registry
  - Additions to existing reusable library/module: loft-aware CSG routes
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG execution for eligible loft operands
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by loft and primitive patch pair counts
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - only eligible single-shell lofts execute; others refuse
- Test strategy:
  - unit tests plus fixture probes for `RT-LOFT-CSG-001` through `004` and
    `006`
- Data ownership:
  - CSG owns operation result
- Routes:
  - eligibility gate to solver route to result assembler
- Reuse/extraction decision:
  - add to existing CSG module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Cylinder and sphere cutters may require different loft patch mapping evidence
  from box cutters.

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
- Readiness blockers: 2 x 2 = 4
- Total: 24.5

Readiness blockers:
- [ ] Loft CSG eligibility gate must exist.
- [ ] Higher-order CSG solver routes must support the mapped loft patch
  families.

Split decision:
- Review for split. Cohesion reason: difference and intersection share cutter
  classification and patch mapping behavior for primitive cutters.

### Candidate Spec: Loft Primitive Union Route

Discovery purpose:
- Implement single-shell loft union with primitive posts while preserving loft
  topology, seams, and provenance.

Responsibilities:
- Functions/methods:
  - loft primitive union route
  - union result provenance mapper
- Data structures/models:
  - loft union operation plan
  - loft CSG result provenance record
- Dependencies/services:
  - loft CSG eligibility gate
  - result shell assembler
- Returns/outputs/signals:
  - CSG SurfaceBody result
  - union refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: result shell assembler
  - Additions to existing reusable library/module: loft-aware union route
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG execution for eligible loft operands
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by loft and primitive patch pair counts
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - only eligible single-shell lofts execute; others refuse
- Test strategy:
  - unit tests plus fixture probe for `RT-LOFT-CSG-005`
- Data ownership:
  - CSG owns operation result and provenance
- Routes:
  - eligibility gate to union route to result assembler
- Reuse/extraction decision:
  - add to existing CSG module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Union should avoid dropping internal source provenance when the post is
  absorbed into the loft body.

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
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Readiness blockers:
- [ ] Loft CSG eligibility gate must exist.

Split decision:
- Review for split. Cohesion reason: primitive union has one operation shape
  and one provenance rule set.

### Candidate Spec: Loft Loft Pair Operation Routes

Discovery purpose:
- Implement union, intersection, and difference routes where both operands are
  eligible single-shell loft bodies.

Responsibilities:
- Functions/methods:
  - loft/loft union route
  - loft/loft intersection route
  - loft/loft difference route
- Data structures/models:
  - paired loft operation plan
  - paired loft parameterization record
- Dependencies/services:
  - loft CSG eligibility gate
  - higher-order CSG solver registry
- Returns/outputs/signals:
  - CSG SurfaceBody result
  - paired-loft refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: higher-order solver registry
  - Additions to existing reusable library/module: paired loft CSG routes
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG execution for eligible loft operands
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by loft patch-pair count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unrelated station parameterizations are accepted only through patch-family
    route mapping, not station-index assumptions
- Test strategy:
  - unit tests plus fixture probes for `RT-LOFT-CSG-009` through `011`
- Data ownership:
  - CSG owns operation result; loft owns source provenance
- Routes:
  - eligibility gate to paired loft planner to solver routes
- Reuse/extraction decision:
  - add to existing CSG module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Paired loft routes must not assume matching station counts or path axes.

Score:
- Functions/methods: 3 x 2 = 6
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
- Readiness blockers: 2 x 2 = 4
- Total: 24.5

Readiness blockers:
- [ ] Loft CSG eligibility gate must exist.
- [ ] Higher-order CSG solver routes must support the mapped loft patch
  families.

Split decision:
- Review for split. Cohesion reason: the three paired-loft operations share
  the same operand eligibility and parameterization boundary.

### Candidate Spec: Single-Shell Loft CSG Reference Fixtures

Discovery purpose:
- Add review fixtures for successful single-shell loft CSG routes after the
  route-level specs execute.

Responsibilities:
- Functions/methods:
  - single-shell loft CSG fixture builders
  - dirty STL generation cases
- Data structures/models:
  - fixture context record
  - loft CSG artifact record
- Dependencies/services:
  - loft CSG operation routes
  - reference fixture registry
- Returns/outputs/signals:
  - dirty STL artifact
  - fixture registry row
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - purpose
  - methodology
  - render description
- Reusable code plan:
  - Existing code reused as-is: reference fixture source contract
  - Additions to existing reusable library/module: loft CSG fixture builders
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty STL artifacts in tests when update flag is set
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded reference tessellation settings
- Cross-screen reusable behavior:
  - review fixture context reused by review app

Project readiness fields:
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - dirty fixtures are unreviewed until human promotion
- Test strategy:
  - reference generation and review registry tests
- Data ownership:
  - fixture registry owns review context; builders own deterministic models
- Routes:
  - source builder to CSG API to STL writer to fixture registry
- Reuse/extraction decision:
  - add to existing reference fixture module
- UI field/control inventory:
  - purpose, methodology, render description

Open questions / nuance discovered:
- Fixture creation is blocked until at least one route-level loft CSG spec
  succeeds.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Readiness blockers:
- [ ] Route-level loft CSG specs must produce at least one successful result.

Split decision:
- Review for split. Cohesion reason: this candidate is only the reference
  fixture layer and has already been separated from route implementation.

### Candidate Spec: Branching Loft CSG Execution Policy

Discovery purpose:
- Decide which branching loft structures can execute, decompose into sub-body
  CSG operations, or refuse before fixture generation.

Responsibilities:
- Functions/methods:
  - branching loft CSG classifier
  - branch decomposition planner
- Data structures/models:
  - branching CSG policy record
  - branch decomposition record
- Dependencies/services:
  - N-to-M loft decomposition plan
  - loft CSG eligibility gate
- Returns/outputs/signals:
  - executable branch plan
  - decomposition-required signal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: N-to-M loft decomposition records
  - Additions to existing reusable library/module: branching loft CSG policy
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG acceptance/refusal behavior for branching lofts
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - branch decomposition should be bounded by branch graph size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - underconstrained branch topology refuses; decomposable branch topology
    produces explicit sub-body plans
- Test strategy:
  - branch execution/decomposition policy unit tests
- Data ownership:
  - loft decomposition owns branch graph; CSG owns boolean operation policy
- Routes:
  - loft decomposition to branching policy to CSG planner
- Reuse/extraction decision:
  - add to existing loft and CSG modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Some branching cases may need validated recomposition after sub-body CSG.

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
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Readiness blockers:
- [ ] Branching loft topology must expose enough decomposition and validity
  evidence for CSG policy.

Split decision:
- Review for split. Cohesion reason: execution/decomposition policy is one
  branching-loft routing boundary.

### Candidate Spec: Underconstrained Branching Loft CSG Refusal Fixture

Discovery purpose:
- Represent `RT-LOFT-CSG-014` as explicit diagnostic reference evidence when
  branching topology is underconstrained.

Responsibilities:
- Functions/methods:
  - underconstrained branch refusal builder
  - refusal fixture registry writer
- Data structures/models:
  - underconstrained topology diagnostic
  - diagnostic fixture context record
- Dependencies/services:
  - branching loft CSG execution policy
  - reference fixture registry
- Returns/outputs/signals:
  - refusal diagnostic
  - fixture registry row
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - purpose
  - methodology
  - diagnostic description
- Reusable code plan:
  - Existing code reused as-is: reference fixture context fields
  - Additions to existing reusable library/module: branching CSG refusal
    fixture helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may write diagnostic artifact
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to one refusal fixture
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - refusal fixtures do not write success STLs
- Test strategy:
  - `RT-LOFT-CSG-014` diagnostic fixture and registry tests
- Data ownership:
  - fixture registry owns review evidence
- Routes:
  - branching policy diagnostic to reference evidence producer
- Reuse/extraction decision:
  - add to existing reference fixture module
- UI field/control inventory:
  - purpose, methodology, diagnostic description

Open questions / nuance discovered:
- This may need the same diagnostic artifact policy as other refusal fixtures.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Readiness blockers:
- [ ] Reference fixture schema needs a diagnostic artifact policy.

Split decision:
- Review for split. Cohesion reason: this is one refusal fixture bound to
  underconstrained branching topology.

### Candidate Spec: Loft CSG Metadata Color Propagation

Discovery purpose:
- Preserve authored colors and metadata through lofted-body CSG results using
  source-fragment provenance.

Responsibilities:
- Functions/methods:
  - loft CSG metadata propagation mapper
  - color/material ownership resolver
- Data structures/models:
  - metadata propagation record
  - color provenance record
- Dependencies/services:
  - CSG result provenance
  - loft source metadata
- Returns/outputs/signals:
  - color/material propagation evidence
  - metadata diagnostic signal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: CSG result provenance records
  - Additions to existing reusable library/module: loft CSG metadata mapper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes result metadata behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by source fragment count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - color/material propagation follows source-fragment provenance
- Test strategy:
  - `RT-LOFT-CSG-012` unit and reference fixture assertions
- Data ownership:
  - CSG provenance owns metadata lineage
- Routes:
  - CSG result provenance to metadata propagation map
- Reuse/extraction decision:
  - add to existing CSG helper modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Generated cap and cut-boundary colors need deterministic fallback rules.

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
- None.

Split decision:
- Review for split. Cohesion reason: metadata propagation is one CSG result
  provenance contract and has no fixture schema responsibilities.

### Candidate Spec: Loft CSG Section Evidence Artifacts

Discovery purpose:
- Produce expected/actual/diff section evidence for lofted-body CSG reference
  cases without bundling metadata propagation.

Responsibilities:
- Functions/methods:
  - section evidence generator
  - section diff diagnostic builder
- Data structures/models:
  - section expected/actual/diff record
  - section diagnostic payload
- Dependencies/services:
  - sectioning utilities
  - reference fixture registry
- Returns/outputs/signals:
  - section comparison artifact
  - fixture diagnostic context
- UI surfaces/components:
  - reference review app
- UI fields/elements:
  - purpose
  - methodology
  - section evidence description
- Reusable code plan:
  - Existing code reused as-is: section utilities and fixture context fields
  - Additions to existing reusable library/module: loft CSG section evidence
    helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes section evidence artifacts
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded section sample count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - section evidence is deterministic and tied to declared section planes
- Test strategy:
  - `RT-LOFT-CSG-013` fixture tests plus section comparison unit tests
- Data ownership:
  - reference fixture owns evidence; CSG result owns source geometry
- Routes:
  - CSG result to section evidence producer to fixture registry
- Reuse/extraction decision:
  - add to existing reference helper modules
- UI field/control inventory:
  - purpose, methodology, section evidence description

Open questions / nuance discovered:
- The fixture schema may need multi-artifact support for section expected,
  actual, and diff evidence.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Readiness blockers:
- [ ] Fixture schema needs multi-artifact evidence policy for section
  expected/actual/diff outputs.

Split decision:
- Review for split. Cohesion reason: section evidence is one reference
  artifact contract and has already been split from metadata propagation.

## Change History

- 2026-07-11: Completed five manifest review/update/rescore rounds. Context:
  oversized single-shell, branching, metadata, and section-evidence candidates
  were split into route-sized and artifact-sized candidates.
- 2026-07-11: Added lofted-body CSG reference architecture. Context:
  representative loft CSG probes still refuse, so the unchecked
  `RT-LOFT-CSG-*` items need eligibility, route, branching, metadata, and
  evidence architecture before fixture generation.
