# Higher-Order Parametric CSG Routes Architecture

This document defines the architecture required to move the higher-order
parametric CSG rows from planned or `not-yet-implemented` into executable,
surface-native support.

It covers the rows involving:

- B-spline
- NURBS
- sweep
- subdivision

The goal is not to make the capability matrix look complete. The goal is that
each supported family-pair row has an actual solver route that can produce
surface-native intersection records, declared residuals, diagnostics, and
downstream trim inputs without falling back to mesh execution.

## Relationship To Other Architecture

This document refines:

- [Surface CSG Executable Completion Architecture](surface-csg-executable-completion-architecture.md)
- [Higher-Order Surface CSG Solver Architecture](higher-order-surface-csg-solver-architecture.md)
- [Exact Surface Intersection Kernel Architecture](exact-surface-intersection-kernel-architecture.md)

It depends on:

- [Surface CSG Trim Fragment Reconstruction Architecture](surface-csg-trim-fragment-reconstruction-architecture.md)
- [Patch Family Integration Architecture](patch-family-integration-architecture.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)

The route architecture owns pair classification and intersection execution. It
does not own boolean operation selection, shell assembly, or final validity;
those are owned by the trim/fragment/reconstruction architecture.

## Completion Definition

A higher-order parametric CSG row is implemented only when all of the following
are true:

- the row has a registered solver route for union, intersection, and difference
- the solver route produces patch-local intersection curves or explicit
  coincident-region records
- all numeric routes emit declared tolerance, residual, iteration, and
  degeneracy diagnostics
- non-convergence is a precise refusal with location and route metadata
- downstream trim reconstruction receives surface-native data
- the result remains a `SurfaceBody`
- no route invokes mesh CSG or treats tessellation as execution
- the row is represented in reference fixtures and capability evidence

## Row Taxonomy

The 40 per-operation higher-order rows are grouped by pair class, not by
operation. Each pair class must be implemented once and exercised across the
three boolean operations.

### Analytic To Spline/NURBS

Pairs:

- planar to B-spline
- planar to NURBS
- ruled to B-spline
- ruled to NURBS
- revolution to B-spline
- revolution to NURBS

Expected route:

- analytic surface evaluator provides exact or closed-form constraints where
  available
- spline or NURBS evaluator provides basis, derivatives, and patch-domain
  bounds
- intersection kernel solves for curve sets in the parametric domain of the
  higher-order patch
- analytic patch receives mapped local curves or generated trim curves

### Analytic To Sweep

Pairs:

- planar to sweep
- ruled to sweep
- revolution to sweep

Expected route:

- sweep is reduced to path/profile evaluator with frame transport
- intersection solves against the authored sweep parameter domain
- singular frame, profile cusp, and path endpoint events become diagnostics

### Analytic To Subdivision

Pairs:

- planar to subdivision
- ruled to subdivision
- revolution to subdivision

Expected route:

- subdivision adapter exposes bounded refined surface patches
- intersection is solved on refined subdivision charts with residual metadata
- result records preserve subdivision identity and refinement level

### Spline/NURBS To Spline/NURBS

Pairs:

- B-spline to B-spline
- B-spline to NURBS
- NURBS to NURBS

Expected route:

- shared basis and rational evaluator infrastructure is reused
- curve marching and subdivision of parameter domains provide robust roots
- overlap and tangency events are represented explicitly

### Spline/NURBS To Sweep

Pairs:

- B-spline to sweep
- NURBS to sweep

Expected route:

- sweep evaluator participates as a parametric surface with derivative access
- profile/path event points are injected as solver seeds
- resulting curves are recorded in both patch-local domains

### Sweep To Sweep

Pairs:

- sweep to sweep

Expected route:

- both sweeps expose path/profile evaluators
- seed generation includes path intersections, profile intersections, endpoint
  events, and frame singularities
- repeated path/profile symmetries produce ambiguity diagnostics instead of
  arbitrary correspondence

### Subdivision With Higher-Order Families

Pairs:

- B-spline to subdivision
- NURBS to subdivision
- sweep to subdivision
- subdivision to subdivision

Expected route:

- subdivision surfaces expose bounded refined charts
- non-subdivision patch families provide evaluator and derivative access
- exactness is declared-tolerance unless the pair has a closed-form route
- refinement budget exhaustion is a refusal, not a mesh fallback

## Components

### Higher-Order CSG Route Registry

The route registry maps:

- operation
- left family
- right family
- pair class
- solver exactness class
- tolerance policy
- supported degeneracy classes

The registry is the source of truth for whether a row is executable. A row must
not be labeled supported merely because the input families are available.

### Parametric Surface Evaluator Adapter

Each participating family exposes a common evaluator shape:

- point evaluation
- first derivative evaluation
- optional second derivative evaluation
- domain bounds
- singularity/event hints
- trim-boundary evaluators
- patch identity and provenance

Adapters are surface-native. They may use approximation internally only when the
family itself is defined as declared-tolerance, and must report that
approximation in the route result.

### Seed And Event Collector

The seed collector provides deterministic starting data for numeric routes:

- boundary intersections
- knot-line events
- rational weight discontinuity guards
- sweep path/profile events
- subdivision extraordinary vertices
- tangency candidates
- coincident boundary candidates

The collector improves robustness but does not invent user intent. Ambiguous
authored inputs produce accumulated diagnostics.

### Intersection Solver Routes

Routes are selected by pair class:

- analytic-to-parametric constrained solving
- spline/NURBS curve marching with domain subdivision
- sweep parameter reduction
- subdivision refined-chart solving
- overlap/coincident-region classification

Each route returns the same intersection result model so downstream CSG does
not need family-specific branching.

### Residual And Convergence Diagnostics

Every declared-tolerance route reports:

- route id
- seed source
- iteration count
- residual distance
- parametric residual
- tolerance used
- convergence state
- failed interval or event location when available

Diagnostics are part of execution correctness. A route that cannot produce
these diagnostics is not complete.

### Pair Fixture Matrix

Reference fixtures prove that rows are executable across:

- crossing intersections
- tangent intersections
- boundary-only contact
- coincident or overlapping regions
- trim-boundary intersections
- endpoint and singular events
- non-convergence refusals

Fixtures are recorded by pair class and operation.

## Data Flow

1. CSG receives two `SurfaceBody` operands and an operation.
2. The operation planner enumerates patch-family pair rows.
3. The higher-order route registry selects an executable solver route for each
   pair.
4. Family adapters expose evaluator, derivative, domain, and event data.
5. The seed collector emits deterministic initial candidates.
6. The selected route computes intersection curves or coincident regions.
7. The route emits residuals, degeneracy records, and patch-local mappings.
8. The trim/fragment/reconstruction layer consumes the route results.
9. The executable coverage gate records the row as supported only when the
   route and downstream reconstruction both pass.

## Cross-Domain Decisions

### Mesh Is Not A Solver Boundary

Mesh may exist only at tessellation, preview, export, and artifact comparison
boundaries. Higher-order CSG routes cannot call mesh CSG, compare triangle
fragments as boolean execution, or return mesh-backed bodies.

### Declared-Tolerance Is Acceptable, Silent Approximation Is Not

NURBS, B-spline, sweep, and subdivision combinations may require numeric
solving. Numeric solving is acceptable when the route declares tolerance,
residuals, convergence state, and affected patch ids.

### Authored Ambiguity Is Reported, Not Solved Arbitrarily

When input topology, sweep frames, coincident regions, or repeated features
create multiple valid interpretations, the route accumulates exact ambiguity
diagnostics. A plan containing unresolved ambiguity cannot execute.

### Subdivision Stays Surface-Native

Subdivision routes may use refined charts, but the result remains a
surface-body result with subdivision provenance. Refinement budget failure is a
diagnostic refusal.

### Sweep Uses Authored Path/Profile Semantics

Sweep routes must respect path start anchors, profile anchors, frame policy, and
named topology entities. They must not reinterpret the sweep as a generic mesh
or unstructured sampled shape.

## Specification Manifest for Discovery

### Five-Round Critical Review Log

Round 1:
- Found four oversized route candidates that bundled adapter readiness,
  numeric solving, patch-local mapping, diagnostics, and fixture evidence.
- Split candidates over 25 before implementation promotion.

Round 2:
- Found analytic-to-spline and analytic-to-NURBS were different enough to split
  because rational weights and exact conic cases change solver readiness.
- Added explicit B-spline and NURBS route candidates.

Round 3:
- Found spline-pair intersection mixed ordinary curve solving with
  coincident-region ownership.
- Split crossing/tangent curve routes from overlap/coincident-region routes.

Round 4:
- Found sweep and subdivision routes were hiding evaluator-adapter work.
- Split evaluator/chart extraction work from actual pair-route execution.

Round 5:
- Rechecked for deferred, unsupported, intentionally-not-implemented, mesh
  fallback, missing persistence, missing diagnostics, and hidden fixture work.
- Final review found no remaining implementation candidate over 25; remaining
  16-24.5 candidates have explicit cohesion explanations.

### Candidate Spec: Higher-Order CSG Row Taxonomy And Route Registry

Discovery purpose:
- Create the route registry that turns higher-order CSG row labels into
  executable pair-class routes.

Responsibilities:
- Functions/methods:
  - higher-order row classifier
  - route lookup
  - executable row report
- Data structures/models:
  - route registry row
  - pair-class enum
  - route support diagnostic
- Dependencies/services:
  - CSG capability matrix
  - patch-family registry
  - executable coverage gate
- Returns/outputs/signals:
  - executable route id
  - missing route diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current CSG family support rows
  - Additions to existing reusable library/module: route registry fields
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
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unregistered rows fail executable completion
- Test strategy:
  - route lookup tests for every higher-order pair class and operation
- Data ownership:
  - CSG owns route executability truth
- Routes:
  - capability matrix to route registry to executable coverage report
- Open questions / nuance discovered:
  - route exactness must be visible separately from family availability
- Readiness blockers:
  - none

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
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only classification
  and registry evidence, not solver implementation.

### Split Discovery Finding: Analytic To Spline And NURBS CSG Intersections

Discovery purpose:
- Implement executable CSG intersection routes between analytic patches and
  B-spline or NURBS patches.

Responsibilities:
- Functions/methods:
  - analytic constraint builder
  - spline/NURBS domain root solver
  - patch-local curve emitter
- Data structures/models:
  - analytic-parametric route request
  - declared-tolerance intersection curve
  - tangency/coincidence diagnostic
- Dependencies/services:
  - exact surface intersection kernel
  - spline basis evaluator
  - NURBS rational evaluator
- Returns/outputs/signals:
  - patch-local intersection curves
  - overlap diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: spline and NURBS evaluators
  - Additions to existing reusable library/module: analytic-parametric route
    solver
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
  - bounded numeric solve iterations and subdivision depth
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - declared-tolerance routes require residual and convergence metadata
- Test strategy:
  - planar/spline, ruled/NURBS, revolution/NURBS, tangent, boundary, and
    coincident fixtures
- Data ownership:
  - intersection kernel owns curve records; CSG owns operation use
- Routes:
  - CSG planner to route registry to analytic-parametric solver
- Open questions / nuance discovered:
  - exact analytic constraints may become declared-tolerance when trimmed
- Readiness blockers:
  - patch-local curve mapping

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
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 27

Split decision:
- Split completed in this manifest. This broad discovery finding is superseded
  by the B-spline and NURBS candidates below and must not be promoted directly.

### Candidate Spec: Analytic To B-Spline CSG Intersections

Discovery purpose:
- Implement executable CSG intersection routes between analytic patches and
  B-spline patches.

Responsibilities:
- Functions/methods:
  - analytic constraint builder
  - B-spline domain root solver
  - B-spline-local curve emitter
- Data structures/models:
  - analytic-B-spline route request
  - B-spline-local intersection curve
  - analytic/spline residual diagnostic
- Dependencies/services:
  - exact surface intersection kernel
  - B-spline basis evaluator
  - route registry
- Returns/outputs/signals:
  - patch-local intersection curves
  - convergence diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: B-spline patch storage and basis evaluator
  - Additions to existing reusable library/module: analytic-to-B-spline CSG
    route
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
  - bounded root iterations and domain subdivision
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - declared-tolerance routes require residual, seed, and convergence metadata
- Test strategy:
  - planar/B-spline, ruled/B-spline, revolution/B-spline, tangent, boundary,
    and non-convergence fixtures
- Data ownership:
  - intersection kernel owns curve records; CSG owns operation semantics
- Routes:
  - planner to route registry to analytic-to-B-spline solver to trim mapper
- Open questions / nuance discovered:
  - trimmed analytic patches require both analytic and generated local curves
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
- Review for split. Cohesion reason: this candidate owns one pair-class route
  and depends on shared mapping/reconstruction rather than implementing it.

### Candidate Spec: Analytic To NURBS CSG Intersections

Discovery purpose:
- Implement executable CSG intersection routes between analytic patches and
  NURBS patches, including rational-weight validation and exact conic cases.

Responsibilities:
- Functions/methods:
  - analytic/rational constraint builder
  - NURBS domain root solver
  - rational residual validator
- Data structures/models:
  - analytic-NURBS route request
  - rational intersection curve
  - weight/domain diagnostic
- Dependencies/services:
  - exact surface intersection kernel
  - NURBS rational evaluator
  - route registry
- Returns/outputs/signals:
  - patch-local NURBS intersection curves
  - rational residual diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: NURBS patch storage and rational evaluator
  - Additions to existing reusable library/module: analytic-to-NURBS CSG route
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
  - bounded root iterations and weight validation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
  - `src/impression/modeling/surface_spline.py`
- Chosen defaults / parameters:
  - exact conic-compatible cases may report exact; all others report
    declared-tolerance residuals
- Test strategy:
  - planar/NURBS, ruled/NURBS, revolution/NURBS, exact conic, tangent, and
    invalid-weight diagnostics
- Data ownership:
  - NURBS evaluator owns rational basis math; intersection kernel owns route
    output
- Routes:
  - planner to route registry to analytic-to-NURBS solver to trim mapper
- Open questions / nuance discovered:
  - rational singularities need exact patch-local diagnostic locations
- Readiness blockers:
  - rational derivative evaluator

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
- Review for split. Cohesion reason: this candidate owns the NURBS-specific
  analytic route and keeps rational validation out of the B-spline route.

### Split Discovery Finding: Spline And NURBS Pair CSG Intersections

Discovery purpose:
- Implement executable B-spline/B-spline, B-spline/NURBS, and NURBS/NURBS CSG
  intersection routes.

Responsibilities:
- Functions/methods:
  - pair-domain subdivision
  - intersection curve marching
  - overlap region classifier
- Data structures/models:
  - spline-pair route request
  - marched curve segment
  - coincident patch-region record
- Dependencies/services:
  - spline basis infrastructure
  - rational NURBS evaluator
  - exact surface intersection kernel
- Returns/outputs/signals:
  - intersection curve chains
  - overlap/coincidence records
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: spline/NURBS patch storage and evaluation
  - Additions to existing reusable library/module: spline-pair CSG route
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
  - bounded subdivision, root finding, and curve marching budgets
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
  - `src/impression/modeling/surface_spline.py`
- Chosen defaults / parameters:
  - all numeric roots report residuals and domain intervals
- Test strategy:
  - crossing, tangent, coincident, trimmed, and non-convergent spline-pair
    fixtures
- Data ownership:
  - spline/NURBS evaluators own basis math; intersection kernel owns results
- Routes:
  - route registry to spline-pair solver to trim reconstruction
- Open questions / nuance discovered:
  - coincident-region records must be consumed by downstream fragment logic
- Readiness blockers:
  - overlap handling
  - trim arrangement graph

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
- Performance-sensitive behavior: 3 x 1 = 3
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 2 x 2 = 4
- Total: 29.5

Split decision:
- Split completed in this manifest. This broad discovery finding is superseded
  by the curve-intersection and coincident-region candidates below.

### Candidate Spec: Spline And NURBS Pair Curve Intersections

Discovery purpose:
- Implement crossing and tangent curve intersection routes for B-spline and
  NURBS patch pairs.

Responsibilities:
- Functions/methods:
  - pair-domain subdivision
  - curve marching solver
  - tangent root classifier
- Data structures/models:
  - spline-pair route request
  - marched curve segment
  - tangent event record
- Dependencies/services:
  - spline basis infrastructure
  - NURBS rational evaluator
  - exact surface intersection kernel
- Returns/outputs/signals:
  - intersection curve chains
  - tangent diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: spline/NURBS evaluation
  - Additions to existing reusable library/module: spline-pair curve route
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
  - bounded domain subdivision and curve marching
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
  - `src/impression/modeling/surface_spline.py`
- Chosen defaults / parameters:
  - tangent cases must report residual and tangent classification
- Test strategy:
  - B-spline/B-spline, B-spline/NURBS, NURBS/NURBS, crossing, tangent, trimmed,
    and non-convergence fixtures
- Data ownership:
  - intersection kernel owns curve records
- Routes:
  - route registry to spline-pair curve solver to trim mapper
- Open questions / nuance discovered:
  - coincident regions are intentionally excluded and handled separately
- Readiness blockers:
  - tangent residual diagnostics

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
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: this candidate owns curve intersections
  only; overlap regions are split out.

### Candidate Spec: Spline And NURBS Coincident Region CSG Intersections

Discovery purpose:
- Represent and classify coincident or overlapping B-spline/NURBS patch regions
  as first-class CSG intersection output.

Responsibilities:
- Functions/methods:
  - coincident-region detector
  - overlap boundary extractor
  - ownership diagnostic builder
- Data structures/models:
  - coincident patch-region record
  - overlap boundary loop
  - ownership ambiguity diagnostic
- Dependencies/services:
  - spline/NURBS evaluators
  - tolerance records
  - trim arrangement graph
- Returns/outputs/signals:
  - coincident-region records
  - ambiguous-overlap diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current diagnostic and patch identity records
  - Additions to existing reusable library/module: overlap-region route
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
  - bounded overlap sampling and boundary extraction
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - ambiguous ownership blocks execution after all diagnostics are collected
- Test strategy:
  - identical patches, partial overlap, reversed orientation, near-coincident
    refusal, and overlap-boundary fixtures
- Data ownership:
  - intersection kernel owns overlap records; CSG owns ownership semantics
- Routes:
  - route registry to overlap detector to trim arrangement graph
- Open questions / nuance discovered:
  - overlap boundaries must be consumed by fragment classification
- Readiness blockers:
  - trim arrangement graph

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
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns coincident-region
  detection and output records only.

### Split Discovery Finding: Sweep CSG Intersection Routes

Discovery purpose:
- Implement executable CSG routes for sweep participation against analytic,
  spline, NURBS, sweep, and subdivision families.

Responsibilities:
- Functions/methods:
  - sweep evaluator adapter
  - path/profile event seeding
  - sweep pair solver dispatch
- Data structures/models:
  - sweep route request
  - frame-event diagnostic
  - sweep-local intersection curve
- Dependencies/services:
  - sweep path/frame policy
  - exact surface intersection kernel
  - route registry
- Returns/outputs/signals:
  - sweep-local curves
  - authored ambiguity diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: sweep patch payload and frame transport policy
  - Additions to existing reusable library/module: sweep intersection adapter
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
  - bounded path/profile sampling and solve budgets
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - authored path/profile anchors define correspondence; ambiguity refuses
    execution
- Test strategy:
  - analytic/sweep, spline/sweep, sweep/sweep, endpoint, cusp, frame singular,
    and repeated-profile fixtures
- Data ownership:
  - sweep owns evaluator semantics; intersection kernel owns route result
- Routes:
  - route registry to sweep adapter to trim reconstruction
- Open questions / nuance discovered:
  - sweep/sweep repeated features can produce multiple valid roots
- Readiness blockers:
  - sweep local-domain curve mapping

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
- Readiness blockers: 1 x 2 = 2
- Total: 26.5

Split decision:
- Split completed in this manifest. This broad discovery finding is superseded
  by the sweep adapter and sweep pair-route candidates below.

### Candidate Spec: Sweep CSG Evaluator And Event Adapter

Discovery purpose:
- Expose sweep path/profile/frame data as a CSG-ready parametric evaluator with
  deterministic event seeds.

Responsibilities:
- Functions/methods:
  - sweep evaluator adapter
  - path/profile event collector
  - frame singularity diagnostic builder
- Data structures/models:
  - sweep evaluation request
  - sweep event seed
  - frame-event diagnostic
- Dependencies/services:
  - sweep path/frame policy
  - authored anchor records
  - route registry
- Returns/outputs/signals:
  - evaluator adapter
  - event seed list
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: sweep patch payload and frame transport policy
  - Additions to existing reusable library/module: CSG-ready sweep adapter
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
  - bounded event enumeration
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - authored path/profile anchors are the correspondence source of truth
- Test strategy:
  - path endpoints, profile cusps, frame singularities, repeated features, and
    valid seed ordering fixtures
- Data ownership:
  - sweep owns evaluator semantics; intersection kernel owns route use
- Routes:
  - sweep patch to evaluator adapter to route registry
- Open questions / nuance discovered:
  - repeated features can be valid but ambiguous for execution
- Readiness blockers:
  - none

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
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: this candidate owns adapter readiness and
  does not implement boolean pair routes.

### Candidate Spec: Sweep Pair CSG Intersections

Discovery purpose:
- Implement executable sweep participation against analytic, spline, NURBS,
  sweep, and subdivision route partners.

Responsibilities:
- Functions/methods:
  - sweep pair solver dispatch
  - sweep-local curve mapper
  - authored ambiguity detector
- Data structures/models:
  - sweep pair route request
  - sweep-local intersection curve
  - sweep ambiguity diagnostic
- Dependencies/services:
  - sweep evaluator adapter
  - exact surface intersection kernel
  - route registry
- Returns/outputs/signals:
  - sweep-local curves
  - non-executable ambiguity diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: sweep adapter and route diagnostics
  - Additions to existing reusable library/module: sweep pair route solvers
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
  - bounded path/profile solve budgets
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unresolved authored ambiguity blocks execution
- Test strategy:
  - analytic/sweep, spline/sweep, NURBS/sweep, sweep/sweep, subdivision/sweep,
    and ambiguity fixtures
- Data ownership:
  - intersection kernel owns curves; CSG owns execution gating
- Routes:
  - route registry to sweep pair solver to trim mapper
- Open questions / nuance discovered:
  - sweep/sweep route may need specialized seed ordering
- Readiness blockers:
  - sweep evaluator adapter

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
- Review for split. Cohesion reason: this candidate owns sweep pair execution
  while using shared sweep adapter and shared reconstruction.

### Split Discovery Finding: Subdivision CSG Refined-Chart Routes

Discovery purpose:
- Implement executable CSG routes for subdivision participation against
  analytic, spline, NURBS, sweep, and subdivision families.

Responsibilities:
- Functions/methods:
  - subdivision chart extraction
  - refined-chart intersection solve
  - refinement budget refusal
- Data structures/models:
  - subdivision route request
  - refined chart record
  - refinement residual diagnostic
- Dependencies/services:
  - subdivision evaluator
  - exact surface intersection kernel
  - route registry
- Returns/outputs/signals:
  - subdivision-local curves
  - bounded-refinement diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: subdivision cage/evaluation infrastructure
  - Additions to existing reusable library/module: refined-chart CSG adapter
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
  - bounded refinement, chart count, and solve budget
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - refinement is declared-tolerance and never mesh CSG
- Test strategy:
  - analytic/subdivision, spline/subdivision, sweep/subdivision,
    subdivision/subdivision, extraordinary vertex, and budget refusal fixtures
- Data ownership:
  - subdivision evaluator owns chart data; intersection kernel owns results
- Routes:
  - route registry to subdivision adapter to trim reconstruction
- Open questions / nuance discovered:
  - extraordinary vertices need explicit location diagnostics
- Readiness blockers:
  - subdivision local-domain curve mapping

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
- Performance-sensitive behavior: 3 x 1 = 3
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 27.5

Split decision:
- Split completed in this manifest. This broad discovery finding is superseded
  by the subdivision chart adapter and subdivision pair-route candidates below.

### Candidate Spec: Subdivision CSG Refined Chart Adapter

Discovery purpose:
- Expose subdivision patches as bounded refined charts that CSG routes can solve
  against without mesh execution.

Responsibilities:
- Functions/methods:
  - subdivision chart extraction
  - extraordinary vertex locator
  - refinement budget diagnostic builder
- Data structures/models:
  - refined chart record
  - extraordinary vertex event
  - refinement budget diagnostic
- Dependencies/services:
  - subdivision evaluator
  - patch family registry
  - route registry
- Returns/outputs/signals:
  - bounded refined chart set
  - refinement diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: subdivision cage/evaluation infrastructure
  - Additions to existing reusable library/module: CSG refined-chart adapter
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
  - bounded chart count and refinement depth
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - chart refinement is declared-tolerance and records source subdivision
    provenance
- Test strategy:
  - regular chart, extraordinary vertex, boundary chart, budget refusal, and
    provenance fixtures
- Data ownership:
  - subdivision evaluator owns chart extraction
- Routes:
  - subdivision patch to refined chart adapter to route registry
- Open questions / nuance discovered:
  - extraordinary vertices need body-level and chart-local diagnostics
- Readiness blockers:
  - none

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
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns chart adapter
  readiness only.

### Candidate Spec: Subdivision Pair CSG Intersections

Discovery purpose:
- Implement executable subdivision participation against analytic, spline,
  NURBS, sweep, and subdivision route partners.

Responsibilities:
- Functions/methods:
  - refined-chart pair solver
  - subdivision-local curve mapper
  - refinement convergence validator
- Data structures/models:
  - subdivision pair route request
  - subdivision-local curve
  - refinement residual diagnostic
- Dependencies/services:
  - subdivision refined chart adapter
  - exact surface intersection kernel
  - route registry
- Returns/outputs/signals:
  - subdivision-local curves
  - bounded-refinement diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: refined chart adapter and route diagnostics
  - Additions to existing reusable library/module: subdivision pair route
    solvers
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
  - bounded chart-pair solving and refinement budget
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface_intersections.py`
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - budget exhaustion refuses execution; it never falls back to mesh CSG
- Test strategy:
  - analytic/subdivision, spline/subdivision, NURBS/subdivision,
    sweep/subdivision, subdivision/subdivision, and budget refusal fixtures
- Data ownership:
  - intersection kernel owns curves; CSG owns route executability
- Routes:
  - route registry to subdivision pair solver to trim mapper
- Open questions / nuance discovered:
  - subdivision/subdivision route may produce many chart-pair candidates
- Readiness blockers:
  - subdivision refined chart adapter

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
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: this candidate owns subdivision pair
  route execution while using shared adapter and reconstruction.

### Candidate Spec: Higher-Order CSG Residual And Degeneracy Diagnostics

Discovery purpose:
- Standardize residual, convergence, ambiguity, and degeneracy diagnostics for
  all higher-order parametric CSG routes.

Responsibilities:
- Functions/methods:
  - residual collector
  - degeneracy classifier
  - route diagnostic formatter
- Data structures/models:
  - residual record
  - degeneracy record
  - ambiguity diagnostic
- Dependencies/services:
  - intersection kernel
  - CSG operation planner
  - route registry
- Returns/outputs/signals:
  - structured route diagnostics
  - non-executable plan diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current diagnostic records
  - Additions to existing reusable library/module: higher-order route
    diagnostic helpers
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
  - diagnostics must not require re-solving the pair
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - unresolved ambiguity blocks execution but planning continues to collect all
    diagnostics
- Test strategy:
  - non-convergence, ambiguous sweep, overlap, singularity, and budget refusal
    diagnostics
- Data ownership:
  - route owns local diagnostics; planner owns aggregate diagnostics
- Routes:
  - solver route to planner diagnostics to user-facing error payload
- Open questions / nuance discovered:
  - diagnostic locations need both patch-local and body-level identifiers
- Readiness blockers:
  - none

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
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 0 x 2 = 0
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns shared diagnostics
  only and is intentionally independent of solver route implementation.

### Candidate Spec: Higher-Order CSG Pair Fixture Matrix

Discovery purpose:
- Add fixture coverage proving that higher-order CSG rows are executable across
  operation, pair class, and degeneracy category.

Responsibilities:
- Functions/methods:
  - fixture matrix enumerator
  - expected residual validator
  - no-mesh-execution assertion
- Data structures/models:
  - pair fixture row
  - expected diagnostic record
  - fixture evidence report
- Dependencies/services:
  - CSG executable coverage gate
  - reference artifact lifecycle
  - route registry
- Returns/outputs/signals:
  - fixture coverage report
  - missing evidence diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: reference artifact helpers
  - Additions to existing reusable library/module: higher-order CSG fixture
    matrix helpers
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
  - fixture generation must remain bounded for CI
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/test_surface_kernel.py`
  - `project/release-0.1.0a/reference-artifacts/`
- Chosen defaults / parameters:
  - dirty reference artifacts do not count as completion evidence
- Test strategy:
  - crossing, tangent, coincident, boundary, singular, and refusal fixtures for
    every higher-order pair class
- Data ownership:
  - tests own fixture evidence; CSG owns executable behavior
- Routes:
  - fixture builder to CSG route execution to reference artifact promotion
- Open questions / nuance discovered:
  - large fixture matrix may need smoke and acceptance tiers
- Readiness blockers:
  - route registry

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
- Review for split. Cohesion reason: this candidate owns evidence coverage and
  reference promotion only; it does not implement solver routes.

## Change History

- 2026-05-30: Critically reviewed the specification manifest through five
  review/rescore/split loops. Reason: broad higher-order route candidates still
  hid implementation work and scored above the split threshold; the manifest now
  contains final split candidates for B-spline, NURBS, sweep, subdivision,
  coincident regions, diagnostics, and fixtures.
- 2026-05-28: Added architecture for executable higher-order parametric CSG
  routes. Reason: the CSG matrix exposed `not-yet-implemented` rows for
  B-spline, NURBS, sweep, and subdivision combinations that require explicit
  solver architecture before they can be promoted to supported.
