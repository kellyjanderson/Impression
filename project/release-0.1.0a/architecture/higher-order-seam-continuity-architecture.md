# Higher-Order Seam Continuity Architecture

## Overview

This document defines the architecture for true G1/G2/C1/C2 seam enforcement
in the surface-body kernel.

The current system can record requested continuity and refuse unsupported
classes. That is the correct safety posture, but it is not the final geometry
target. The final target is to validate and, where explicitly allowed, construct
seams whose participating patch-boundary uses satisfy the requested continuity
class.

Continuity is kernel truth. It is not a tessellation smoothing hint.

## Related Architecture

This document extends:

- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [Surface Body Completion Architecture](surface-body-completion-architecture.md)
- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)

## Continuity Classes

The kernel distinguishes:

- `C0`: positional continuity with identical evaluated boundary points
- `G0`: geometric positional continuity where orientation/parameterization may
  differ but the boundary curve coincides
- `C1`: first derivative continuity in a compatible parameterization
- `G1`: tangent-plane or tangent-direction continuity independent of parameter
  speed
- `C2`: second derivative continuity in a compatible parameterization
- `G2`: curvature continuity independent of parameter speed

C-class continuity is parameterization-sensitive. G-class continuity is
geometric and must account for reparameterization.

## Components

### Boundary Use Evaluator

The boundary use evaluator samples and evaluates patch-local boundary uses.

It owns:

- boundary curve evaluation in 3D
- first and second boundary derivatives
- patch normal evaluation along the seam
- parameter mapping between seam coordinate and patch-local UV
- tolerance and residual summaries

The evaluator must use patch-family evaluation APIs. It must not infer
continuity by comparing tessellated mesh normals.

### Continuity Constraint Record

The constraint record is the durable expression of authored continuity intent.

It owns:

- requested class
- participating boundary-use identities
- tolerance policy
- whether construction/enforcement is required or validation-only
- source of the request

### Continuity Validator

The validator compares boundary uses against a requested class.

It owns:

- C0/G0 point coincidence checks
- C1/G1 tangent checks
- C2/G2 curvature checks
- residual reporting
- localizing violations along the seam

The validator returns structured reports that identify where continuity fails,
not just that it fails.

### Continuity Enforcer

The enforcer is optional and bounded. It may adjust or construct geometry only
when the owning operation explicitly asks for it.

Allowed enforcement examples:

- loft constructing a sweep or spline surface with matched tangents
- blend/fillet operations creating a new transition patch
- surface extension or trim adjustment within declared tolerances

Forbidden enforcement examples:

- silently moving authored source geometry
- replacing surfaces with mesh smoothing
- downgrading G2 to G1 or C1 without a refusal diagnostic

### Tessellation Consumer

Tessellation consumes seam continuity as metadata for sampling and welding
quality, but it does not define continuity truth.

## Data Flow

```text
Authored continuity request
-> continuity constraint record
-> boundary use evaluator
-> continuity validator
-> pass report or localized violation report
-> optional continuity enforcer when operation owns construction
-> updated SurfaceBody seam truth
```

## Cross-Domain Decisions

### Validation And Enforcement Are Separate

Validation asks whether a seam satisfies a class. Enforcement changes or
constructs geometry to satisfy a class. They must not be the same function.

### Reparameterization Is First-Class

G-continuity cannot be implemented as naive derivative equality. The system
needs a seam parameter map that can compare tangent direction and curvature
independent of parameter speed.

### Higher Continuity Requires Patch Derivatives

Every family promoted to C1/C2/G1/G2 participation must expose derivative
evaluation or an explicit not-supported diagnostic.

### Refusal Remains Valid

If a patch family cannot provide derivative information, the seam validator
must refuse the requested class with exact family and boundary-use locators.

## Specification Manifest for Discovery

### Candidate Spec: Seam Continuity Constraint Records

Discovery purpose:
- Add durable records for authored C1/C2/G1/G2 continuity requests and their
  participating boundary uses.

Responsibilities:
- Functions/methods:
  - constraint normalizer
  - request validator
- Data structures/models:
  - continuity constraint record
  - tolerance policy record
  - boundary-use reference
- Dependencies/services:
  - seam records
  - boundary-use records
- Returns/outputs/signals:
  - normalized constraint
  - invalid request diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current continuity request/support records
  - Additions to existing reusable library/module: higher-order constraint
    records
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
  - constant-time validation per request
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - C0/G0 remain default; higher classes require explicit request
- Test strategy:
  - record normalization and invalid request tests
- Data ownership:
  - seam topology owns continuity intent
- Routes:
  - authored seam to constraint record to validator
- Open questions / nuance discovered:
  - tolerance policy should align with loft and CSG tolerances
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
- No split needed. The candidate is a cohesive record and validation contract.

### Candidate Spec: Boundary Derivative Evaluation For Continuity

Discovery purpose:
- Provide derivative and normal evaluation along patch boundaries for promoted
  families.

Responsibilities:
- Functions/methods:
  - boundary evaluator
  - first derivative evaluator
  - second derivative evaluator
- Data structures/models:
  - boundary derivative sample
  - residual summary
- Dependencies/services:
  - patch family evaluators
  - seam boundary-use records
- Returns/outputs/signals:
  - derivative samples
  - unsupported-family diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: patch evaluation APIs
  - Additions to existing reusable library/module: boundary derivative helpers
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
  - bounded by seam sample count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - exact derivative preferred; numeric derivative allowed only with residual
    metadata
- Test strategy:
  - analytic planar/revolution/ruled derivative tests plus unsupported-family
    diagnostics
- Data ownership:
  - patch families own evaluation; seam layer owns boundary sampling
- Routes:
  - seam boundary use to patch evaluator to continuity validator
- Open questions / nuance discovered:
  - subdivision and implicit derivative support may need declared-tolerance mode
- Readiness blockers:
  - promoted family derivative API coverage

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
- Review for split. Cohesion reason: derivative evaluation is one shared seam
  service; family-specific evaluator work may split after this contract lands.

### Candidate Spec: Higher-Order Continuity Residual Validation

Discovery purpose:
- Validate C1/C2/G1/G2 requests by computing residuals from boundary
  derivative samples without downgrading requested continuity.

Responsibilities:
- Functions/methods:
  - continuity validator
  - residual classifier
- Data structures/models:
  - continuity validation report
  - residual metrics
  - observed continuity class record
- Dependencies/services:
  - constraint records
  - boundary derivative evaluator
- Returns/outputs/signals:
  - pass/fail report
  - residual summary
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam validation result records
  - Additions to existing reusable library/module: higher-order report helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes higher-continuity request behavior from blanket refusal to
    validation where supported
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by seam sample count and derivative cost
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - failing validation never downgrades requested continuity
- Test strategy:
  - positive C1/G1 fixtures and negative residual threshold fixtures
- Data ownership:
  - validator owns observed continuity; seam owns requested continuity
- Routes:
  - constraint and derivative samples to validation report
- Open questions / nuance discovered:
  - G2 residual thresholds need family-specific numerical stability notes
- Readiness blockers:
  - boundary derivative evaluator must exist

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
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
- Total: 19.5

Split decision:
- Review for split. Cohesion reason: residual validation is now separated from
  user-facing violation locator diagnostics.

### Candidate Spec: Higher-Order Continuity Violation Locators

Discovery purpose:
- Convert failed higher-order continuity validation into exact seam, boundary
  use, parameter, and residual diagnostics.

Responsibilities:
- Functions/methods:
  - violation locator builder
  - residual hot-spot selector
  - diagnostic formatter
- Data structures/models:
  - violation record
  - seam parameter locator
  - boundary-use diagnostic
- Dependencies/services:
  - continuity validation report
  - seam boundary-use records
- Returns/outputs/signals:
  - localized violation diagnostics
  - suggested authored fix hints
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam validation result records
  - Additions to existing reusable library/module: higher-order locator helpers
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
  - bounded by failed residual sample count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - diagnostics name the requested class and observed residual failure
- Test strategy:
  - exact locator tests for tangent and curvature failures
- Data ownership:
  - validator owns observed residuals; locator owns user-facing diagnostic path
- Routes:
  - validation report to localized diagnostics
- Open questions / nuance discovered:
  - fix hints should remain advice and never mutate source geometry
- Readiness blockers:
  - residual validation report must exist

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
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: locator diagnostics are one user-facing
  report contract and are now separated from residual math.

### Candidate Spec: Bounded Continuity Enforcement Boundary

Discovery purpose:
- Define when operations may construct or adjust geometry to satisfy requested
  continuity.

Responsibilities:
- Functions/methods:
  - enforcement eligibility checker
  - enforcement result validator
- Data structures/models:
  - enforcement request
  - enforcement result
  - refusal diagnostic
- Dependencies/services:
  - continuity validator
  - loft/sweep/blend producers
- Returns/outputs/signals:
  - accepted enforcement result
  - explicit refusal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: feature handoff gates
  - Additions to existing reusable library/module: enforcement boundary helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may alter generated operation output; must not alter source geometry
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by owning operation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py` plus operation-owned producers
- Chosen defaults / parameters:
  - validation-only unless an operation explicitly owns construction
- Test strategy:
  - refusal tests for source mutation and positive operation-owned enforcement
- Data ownership:
  - operation owns generated geometry; authored source owns source geometry
- Routes:
  - operation output to enforcement boundary to validation
- Open questions / nuance discovered:
  - blend/fillet producers may need their own architecture before enforcement
- Readiness blockers:
  - continuity validator must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
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
- Total: 19.5

Split decision:
- Review for split. Cohesion reason: this spec owns the enforcement boundary
  only; individual operation-specific enforcement can split later.

## Change History

- 2026-05-27: Ran two additional critical manifest cycles; no additional seam
  continuity split was needed after residual validation, locator diagnostics,
  and enforcement were reviewed again.
- 2026-05-27: Critically reviewed, rescored, and split the specification
  manifest. Context: continuity residual validation and user-facing violation
  locators needed separate candidate specs.
- 2026-05-27: Added higher-order seam continuity architecture and manifest.
  Context: current C0/G0 support plus G1/G2 diagnostics needed a path toward
  true higher-order continuity enforcement.
