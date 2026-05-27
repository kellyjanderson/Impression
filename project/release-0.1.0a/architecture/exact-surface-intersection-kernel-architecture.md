# Exact Surface Intersection Kernel Architecture

## Overview

This document defines the shared intersection kernel needed for richer exact
and declared-tolerance intersections across NURBS, B-spline, subdivision,
implicit, and sampled/deformed surface families.

The intersection kernel is not owned only by CSG. CSG is the largest consumer,
but the same intersection records should support:

- CSG cut curves
- seam construction and validation
- trims and boundary reconstruction
- loft/sweep diagnostics where generated surfaces meet authored geometry
- reference verification of exact versus approximate outcomes

The goal is one shared intersection vocabulary rather than one-off algorithms
inside CSG, loft, or tessellation.

## Related Architecture

This document extends:

- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [Higher-Order Surface CSG Solver Architecture](higher-order-surface-csg-solver-architecture.md)
- [Higher-Order Seam Continuity Architecture](higher-order-seam-continuity-architecture.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)

## Intersection Families

The kernel must handle these major pair classes:

- analytic / analytic
- analytic / spline
- spline / spline
- spline / subdivision
- analytic / implicit
- spline / implicit
- subdivision / implicit
- sampled/deformed / analytic
- sampled/deformed / spline
- sampled/deformed / sampled/deformed

The solver may choose different algorithms for different pairs, but the result
record shape should be common.

## Components

### Intersection Request

The request owns:

- left patch reference
- right patch reference
- operation consumer
- tolerance policy
- domain/trim restriction
- desired result curve class

Consumers should not call family-specific solvers directly. They build
requests and receive typed results or refusal diagnostics.

### Pair Algorithm Registry

The registry maps family pairs to algorithm classes:

- closed-form analytic
- marching / subdivision with residual bounds
- Newton refinement from seeded curves
- implicit contouring with surface projection
- sampled heightfield/deformation intersection under declared sampling policy

The registry exposes support state and expected result quality.

### Intersection Curve Record

The curve record owns the durable result:

- 3D curve representation
- left patch-local parameter curve
- right patch-local parameter curve
- quality class: exact, declared-tolerance, sampled
- residual metrics
- endpoints, loops, and degeneracy classification
- provenance back to request and solver

The curve record is the handoff object for CSG trims and seam rebuild.

### Degeneracy Classifier

The classifier identifies:

- transverse intersections
- tangent touches
- coincident overlap curves
- coincident overlap regions
- isolated points
- empty intersections
- singular or pole-contact cases

Degenerate outcomes must be explicit. They are often the difference between a
valid boolean and a refusal.

### Patch-Local Curve Mapper

The mapper converts 3D intersection truth into each patch's parameter space.

It owns:

- UV curve creation
- residual validation against 3D curve
- trim-loop compatibility
- parameter-domain clipping
- orientation relative to each patch

### Numerical Safety And Determinism

Declared-tolerance solvers must record enough metadata to make failures and
successes deterministic:

- seed strategy
- iteration caps
- residual threshold
- subdivision depth
- sampled-cell limits
- failure reason

The kernel should prefer deterministic seeding from topology and patch identity
over randomized or machine-order-dependent traversal.

## Data Flow

```text
Intersection request
-> pair algorithm registry
-> solver execution
-> degeneracy classification
-> patch-local curve mapping
-> residual and tolerance validation
-> intersection curve record or refusal diagnostic
```

## Cross-Domain Decisions

### One Result Shape, Many Solvers

Closed-form, numeric, implicit, subdivision, and sampled solvers should all
produce a shared record shape. That prevents downstream code from growing
family-specific result handling.

### Exactness Is Recorded, Not Assumed

Each result carries quality metadata. A curve can be exact, declared-tolerance,
or sampled, and consumers decide whether that quality is acceptable.

### Degeneracy Is A First-Class Result

Tangency, overlap, pole contact, singularity, and empty intersection are not
exceptions by default. They are classified outcomes that may be executable or
refusable depending on the consuming operation.

### Sampled And Implicit Solvers Need Safety Budgets

Implicit, subdivision, heightmap, and displacement intersections can grow
expensive. Their requests must carry bounded budgets, and exceeding a budget is
a deterministic refusal.

## Specification Manifest for Discovery

### Candidate Spec: Intersection Request And Solver Registry

Discovery purpose:
- Define the shared request shape and pair algorithm registry for all surface
  intersection consumers.

Responsibilities:
- Functions/methods:
  - intersection request builder
  - pair algorithm lookup
  - registry coverage assertion
- Data structures/models:
  - intersection request
  - solver registry record
  - support diagnostic
- Dependencies/services:
  - patch family matrix
  - tolerance policy
- Returns/outputs/signals:
  - solver dispatch record
  - unsupported pair diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: CSG curve records and patch family records
  - Additions to existing reusable library/module: shared intersection registry
  - New reusable library/module to create: `surface_intersections.py`
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - constant-time registry lookup
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - unsupported pair returns diagnostic; unknown registry entry fails coverage
- Test strategy:
  - registry coverage tests and consumer lookup tests
- Data ownership:
  - intersection module owns pair solver dispatch truth
- Routes:
  - CSG/seam/loft consumers to shared request builder
- Open questions / nuance discovered:
  - module extraction is justified because CSG and seams are both consumers
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 1 x 3 = 3
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: request and registry are one reusable
  dispatch boundary.

### Candidate Spec: Intersection Curve Result And Degeneracy Records

Discovery purpose:
- Define a common intersection result record that can express curves, points,
  overlap regions, quality, residuals, and degeneracy.

Responsibilities:
- Functions/methods:
  - result normalizer
  - degeneracy classifier
- Data structures/models:
  - intersection curve record
  - overlap region record
  - degeneracy record
- Dependencies/services:
  - tolerance policy
  - patch-local curve mapper
- Returns/outputs/signals:
  - normalized intersection result
  - degeneracy diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG curve primitive records
  - Additions to existing reusable library/module: shared result records
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
  - normalization bounded by curve count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - tangent and overlap cases are explicit result classes
- Test strategy:
  - result normalization and degeneracy classification tests
- Data ownership:
  - intersection module owns transient intersection truth
- Routes:
  - solver output to CSG trim/seam consumers
- Open questions / nuance discovered:
  - overlap region representation must support both trim loops and shell splits
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
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
- Total: 14.5

Split decision:
- No split needed. The candidate is a cohesive result-record contract.

### Candidate Spec: Analytic And Spline Surface Intersection Solvers

Discovery purpose:
- Implement exact or declared-tolerance intersections for analytic/spline and
  spline/spline pairs.

Responsibilities:
- Functions/methods:
  - analytic-spline solver
  - spline-spline solver
  - Newton/refinement routine
- Data structures/models:
  - solver iteration record
  - residual report
- Dependencies/services:
  - B-spline/NURBS evaluators
  - registry and result records
- Returns/outputs/signals:
  - intersection curve records
  - non-convergence diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: spline evaluation helpers
  - Additions to existing reusable library/module: solver routines
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
  - iteration and subdivision budgets
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - deterministic seeds; bounded iteration; residual metadata required
- Test strategy:
  - planar/spline, revolution/spline, spline/spline positive and refusal tests
- Data ownership:
  - solver owns numeric process; result record owns reusable curve truth
- Routes:
  - registry dispatch to solver to result record
- Open questions / nuance discovered:
  - exact closed-form is not expected for all spline pairs; declared-tolerance
    is acceptable when residuals are explicit
- Readiness blockers:
  - spline derivative/evaluation coverage must be verified

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
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
- Readiness blockers: 1 x 2 = 2
- Total: 17.5

Split decision:
- Review for split. Cohesion reason: analytic/spline and spline/spline share
  evaluator, seeding, refinement, and residual infrastructure.

### Candidate Spec: Subdivision And Implicit Intersection Boundary

Discovery purpose:
- Define bounded declared-tolerance intersection support for subdivision and
  implicit pairs without unsafe execution or mesh fallback.

Responsibilities:
- Functions/methods:
  - subdivision intersection adapter
  - implicit intersection adapter
  - budget/refusal checker
- Data structures/models:
  - budget record
  - sampled contour record
  - refusal diagnostic
- Dependencies/services:
  - implicit field safety policy
  - subdivision evaluator
  - result records
- Returns/outputs/signals:
  - declared-tolerance result
  - budget refusal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: implicit safety and subdivision refinement
  - Additions to existing reusable library/module: bounded intersection
    adapters
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - unsafe implicit fields must refuse before solver execution
- Performance-sensitive behavior:
  - strict budgets for cells, depth, and iterations
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - budget exhaustion is deterministic refusal
- Test strategy:
  - safe implicit positive fixtures, unsafe refusal, budget refusal
- Data ownership:
  - implicit/subdivision families own evaluation; intersection owns result
- Routes:
  - registry dispatch to bounded adapter
- Open questions / nuance discovered:
  - sampled contour record must remain surface-native, not mesh truth
- Readiness blockers:
  - implicit and subdivision derivative/evaluation contracts

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: subdivision and implicit share the bounded
  declared-tolerance/safety-budget boundary; concrete family sub-solvers may
  split after this boundary is implemented.

## Change History

- 2026-05-27: Added exact surface intersection kernel architecture and
  manifest. Context: higher-order CSG and seam continuity require one shared
  intersection result vocabulary and solver registry.
