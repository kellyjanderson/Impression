# Surface CSG Executable Completion Architecture

## Overview

This document defines the work required to move SurfaceBody CSG from
classification/refusal coverage to executable support across the promoted
surface patch-family matrix.

The current availability gates prove that every patch family can be authored,
stored, tessellated at the consumer boundary, diagnosed, and carried as
surface-body truth. They do not prove that every boolean family pair can
execute.

As of the current CSG matrix, each boolean operation has:

- 9 exact executable analytic rows
- 40 higher-order parametric rows marked `declared-tolerance`
- 51 sampled or implicit rows marked `unsupported`

Across `union`, `difference`, and `intersection`, that means the remaining
non-executable set is the 153 sampled/implicit operation/family rows now owned
by the row-level implementation architecture.

The target state is:

- no `not-yet-implemented` CSG rows for promoted families
- no `unsupported` CSG rows used as a substitute for missing design
- every promoted pair routes to an exact solver, declared-tolerance solver,
  supported sampled/implicit operation policy, or a deliberately non-CSG
  operation class with a replacement authored workflow
- no mesh fallback before the tessellation boundary

## Related Architecture

This document extends and sharpens:

- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [Higher-Order Surface CSG Solver Architecture](higher-order-surface-csg-solver-architecture.md)
- [Exact Surface Intersection Kernel Architecture](exact-surface-intersection-kernel-architecture.md)
- [Surface Body Completion Architecture](surface-body-completion-architecture.md)
- [Advanced Family Availability Producer Architecture](advanced-family-availability-producer-architecture.md)
- [Sampled and Implicit Surface CSG Support Architecture](sampled-implicit-surface-csg-support-architecture.md)
- [Sampled and Implicit CSG Unsupported Row Implementation Architecture](sampled-implicit-csg-unsupported-row-implementation-architecture.md)

## Current Non-Executable Row Classes

### Resolved Higher-Order Parametric Rows

Rows:

- analytic with B-spline, NURBS, sweep, or subdivision
- B-spline/NURBS/sweep/subdivision with each other

Current state:

- `declared-tolerance`
- phase: `intersection-kernel`

Why they are no longer part of the unsupported-row body of work:

- the higher-order route registry now declares executable routes for these
  family pairs
- analytic/B-spline, analytic/NURBS, spline/NURBS, sweep, and subdivision route
  records have been implemented as declared-tolerance support
- the higher-order fixture matrix verifies executable coverage without mesh
  fallback evidence

What keeps them supported:

- pair-specific exact or declared-tolerance intersection dispatch
- bounded numeric residual and convergence records
- patch-local intersection curves for both operands
- trim arrangement and fragment classification for all generated curves
- result shell reconstruction with seams and validity gates
- reference fixtures for success, refusal, tangent, coincident, and degenerate
  cases

### Sampled And Implicit Rows

Rows:

- any pair involving implicit, heightmap, or displacement

Current state:

- `unsupported`
- phase: `operand-family-eligibility`

Why they are unsupported:

- implicit fields are volumetric predicates, while the current CSG executor is
  patch-boundary and trim-graph based
- heightmap and displacement are sampled 2.5D surface families; many boolean
  results cannot be represented as a single height field or source-surface
  displacement
- no native operation policy exists for when sampled results should remain
  sampled, promote to implicit, promote to subdivision, or refuse
- no inside/outside classifier exists for hybrid sampled/parametric fragments
- no `.impress` result contract exists for sampled CSG provenance and lossiness
  beyond individual family payloads

What makes them supported:

- an explicit sampled/implicit operation policy by pair and operation
- representability checks for heightmap/displacement outputs
- implicit field-composition routes for supported implicit operations
- sampled-contour extraction only when the result remains surface-native and
  lossiness is recorded
- promotion rules to a richer result family when the original sampled family
  cannot represent the boolean result
- diagnostics that distinguish impossible representation from missing solver
  implementation

## Components

### Executable CSG Coverage Matrix

The executable matrix is separate from the family availability matrix.

It owns:

- operation
- left family
- right family
- executable support state
- solver route
- result family policy
- refusal/non-CSG policy
- required future capability when not executable

The matrix must fail completion when any promoted pair remains
`not-yet-implemented` or `unsupported` without an accepted non-CSG replacement
workflow.

### Solver Route Registry

The solver registry maps executable rows to implementation routes:

- analytic/analytic solver
- analytic/spline solver
- spline/spline solver
- sweep solver
- subdivision adapter
- implicit field-composition route
- sampled operation route
- hybrid promotion route

The registry is the only place that declares an operation executable.

### Intersection And Classification Kernel

Executable rows require a common result shape:

- 3D intersection curves or overlap regions
- left patch-local curve mappings
- right patch-local curve mappings
- tolerance and residual metadata
- degeneracy diagnostics
- fragment classification predicates

Exact and declared-tolerance solvers feed the same downstream graph.

### Result Family Policy

CSG results do not always have to preserve the source family. The result family
must be explicit.

Allowed result-family policies:

- preserve source patch family where trims are sufficient
- create generated cap patches
- promote to NURBS/B-spline when exact rational or spline representation is
  possible
- promote to subdivision for bounded approximated surface-native outputs
- promote to implicit for volumetric field-composition results
- refuse when no surface-native result family can represent the operation

Promotion is not mesh fallback. It produces a native surface patch family with
provenance, lossiness metadata when relevant, and `.impress` persistence.

### Completion Gate

The completion gate must report:

- executable row count
- non-executable row count
- `not-yet-implemented` rows
- `unsupported` rows
- accepted non-CSG rows, if any
- missing solver routes
- missing reference fixtures

The gate must not pass merely because a row has a diagnostic.

## Data Flow

```text
SurfaceBody operands
-> executable CSG coverage matrix
-> solver route registry
-> operation planner
-> pair solver / sampled policy / implicit policy
-> intersection and classification records
-> trim and fragment graph
-> result family policy
-> shell/seam/provenance reconstruction
-> validity gate
-> SurfaceBody boolean result or deliberate non-CSG refusal
```

## Cross-Domain Decisions

### Unsupported Is Not A Final Completion State

`unsupported` is useful during discovery because it blocks mesh fallback and
names the missing capability. It is not a final completion state for promoted
CSG unless the row is deliberately reclassified as non-CSG with a supported
replacement workflow.

### Sampled Families Need Representation Policy Before Solver Code

Heightmap and displacement cannot represent arbitrary boolean results. The
architecture must decide when to:

- preserve the sampled family
- promote to a richer surface family
- compose an implicit field
- refuse with a representation diagnostic

### Exactness And Lossiness Must Be Explicit

Declared-tolerance and sampled outputs are acceptable only when the output
records expose:

- tolerance policy
- residuals
- sampling resolution
- lossiness
- source provenance
- reproducibility inputs

### CSG Completeness Is Stronger Than Family Availability

Patch family availability proves a family can participate in the surface-body
model. CSG completeness proves every requested boolean operation has an
executable route or an intentionally non-CSG replacement contract.

## Specification Manifest for Discovery

### Candidate Spec: Executable CSG Coverage Gate

Discovery purpose:
- Replace diagnostic-only CSG completion with an executable coverage gate that
  fails on `not-yet-implemented` and unresolved `unsupported` rows.

Responsibilities:
- Functions/methods:
  - executable coverage matrix builder
  - non-executable row collector
  - completion gate assertion
- Data structures/models:
  - executable coverage row
  - non-executable row diagnostic
  - CSG executable completion report
- Dependencies/services:
  - existing CSG support matrix
  - patch family capability matrix
  - solver route registry
- Returns/outputs/signals:
  - executable completion pass/fail
  - exact missing row list
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: `SurfaceBooleanFamilyPairSupport`
  - Additions to existing reusable library/module: CSG executable coverage
    report helpers
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
  - bounded matrix scan across operations and family pairs
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - `not-yet-implemented` and unresolved `unsupported` rows fail completion
- Test strategy:
  - matrix report tests covering exact, not-yet-implemented, unsupported, and
    accepted non-CSG rows
- Data ownership:
  - CSG owns executable operation truth
- Routes:
  - capability matrix to CSG support matrix to executable coverage report
- Open questions / nuance discovered:
  - accepted non-CSG rows require an authored replacement workflow contract
- Readiness blockers:
  - sampled/implicit support policy

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
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only the executable
  coverage gate and row diagnostics, not solver implementation.

### Candidate Spec: Higher-Order Parametric CSG Solver Routes

Discovery purpose:
- Implement executable routes for the 40 per-operation higher-order parametric
  rows involving B-spline, NURBS, sweep, and subdivision families.

Responsibilities:
- Functions/methods:
  - analytic/spline pair solver dispatch
  - spline/spline pair solver dispatch
  - sweep/subdivision adapter dispatch
- Data structures/models:
  - higher-order solver route record
  - declared-tolerance intersection record
  - convergence diagnostic
- Dependencies/services:
  - exact surface intersection kernel
  - CSG operation planner
  - patch-local curve mapping
- Returns/outputs/signals:
  - intersection records
  - supported route rows
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current intersection and CSG records
  - Additions to existing reusable library/module: higher-order CSG solver
    dispatch
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
  - bounded numeric iteration budgets
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - declared-tolerance routes require residuals and convergence metadata
- Test strategy:
  - analytic/spline, spline/spline, sweep, subdivision, tangent, coincident,
    and non-convergence cases
- Data ownership:
  - intersection kernel owns curves; CSG owns operation semantics
- Routes:
  - operation planner to solver route registry to intersection kernel
- Open questions / nuance discovered:
  - sweep and subdivision may route through adapter-specific intermediate
    representations
- Readiness blockers:
  - executable coverage gate
  - patch-local curve mapping for higher-order curves

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
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 2 x 2 = 4
- Total: 28.5

Split decision:
- Split required. This candidate must split into analytic/spline,
  spline/spline, sweep route, subdivision route, and convergence diagnostics
  before implementation.

### Candidate Spec: Surface CSG Fragment Classification And Reconstruction Completion

Discovery purpose:
- Complete the downstream CSG execution path that turns intersection records
  into durable `SurfaceBody` boolean results.

Responsibilities:
- Functions/methods:
  - trim arrangement builder
  - fragment classifier
  - result shell reconstructor
  - seam/provenance rebuild
- Data structures/models:
  - trim arrangement graph
  - classified fragment record
  - result shell assembly record
  - CSG provenance map
- Dependencies/services:
  - intersection records
  - seam/adjacency subsystem
  - validity gate
- Returns/outputs/signals:
  - reconstructed `SurfaceBody`
  - validity/refusal diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: trim, seam, and CSG provenance records
  - Additions to existing reusable library/module: reconstruction helpers
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
  - bounded graph traversal and classification
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - invalid reconstruction refuses; no mesh-derived reconstruction substitute
- Test strategy:
  - fragment graph, inside/outside classification, caps, multi-shell outputs,
    seam rebuild, and provenance fixtures
- Data ownership:
  - CSG owns transient graph; `SurfaceBody` owns durable result topology
- Routes:
  - intersection records to fragment graph to shell assembly to validity gate
- Open questions / nuance discovered:
  - generated non-planar caps may require producer-policy integration
- Readiness blockers:
  - higher-order and sampled/implicit intersection/result routes

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 4 x 1 = 4
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
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 5 x 1 = 5
- Readiness blockers: 1 x 2 = 2
- Total: 29.5

Split decision:
- Split required. This must split into trim arrangement, classification,
  cap/result-family policy, shell assembly, seam/provenance rebuild, and
  validity fixtures.

## Change History

- 2026-05-30: Updated current matrix language after higher-order CSG rows were
  promoted to declared-tolerance support and linked the sampled/implicit
  unsupported-row implementation architecture for the remaining 153 rows.
- 2026-05-28: Created executable CSG completion architecture after the
  availability gates exposed that higher-order and sampled CSG rows were still
  diagnostic-only rather than executable.
