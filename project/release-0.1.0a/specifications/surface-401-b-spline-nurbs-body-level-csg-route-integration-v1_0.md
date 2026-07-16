# Surface Spec 401: B-Spline/NURBS Body-Level CSG Route Integration (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: this leaf owns one body-level integration route after the patch-level evidence contract is complete: B-spline/NURBS patch evidence produced by Surface Specs 402A, 402B1, 402B2, 402C, 402D, and 402E must become body-level `SurfaceBooleanResult` success or deterministic refusal through the existing CSG planner, fragment graph, shell assembly, provenance, and validity gates.

## Overview

Implement the body-level CSG integration route for `SurfaceBody` operands that contain B-spline and/or NURBS patches.

The architecture for patch-level B-spline/NURBS CSG intersections already exists in the higher-order parametric CSG route documents and final specs. The missing work is the executable bridge from those patch-level routes into the public body-level boolean flow used by `boolean_union`, `boolean_difference`, `boolean_intersection`, and `surface_boolean_result`.

This spec exists because reference fixture Spec 384 cannot honestly produce success STLs until a validated body-level route returns a `SurfaceBody` result before tessellation.

## Backlinks

- [Architecture: Surface CSG Executable Completion Architecture](../architecture/surface-csg-executable-completion-architecture.md)
- [Architecture: Higher-Order Parametric CSG Routes Architecture](../architecture/higher-order-parametric-csg-routes-architecture.md)
- [Architecture: Patch-Family Reference CSG Completion Architecture](../architecture/patch-family-reference-csg-completion-architecture.md)
- [Specification: Surface Spec 341: Analytic To B-Spline CSG Intersections](surface-341-analytic-to-b-spline-csg-intersections-v1_0.md)
- [Specification: Surface Spec 342: Analytic To NURBS CSG Intersections](surface-342-analytic-to-nurbs-csg-intersections-v1_0.md)
- [Specification: Surface Spec 343: Spline And NURBS Pair Curve Intersections](surface-343-spline-and-nurbs-pair-curve-intersections-v1_0.md)
- [Specification: Surface Spec 344: Spline And NURBS Coincident Region CSG Intersections](surface-344-spline-and-nurbs-coincident-region-csg-intersections-v1_0.md)
- [Specification: Surface Spec 402: B-Spline/NURBS Patch Evidence Completion For Body CSG](surface-402-b-spline-nurbs-patch-evidence-completion-for-body-csg-v1_0.md)
- [Specification: Surface Spec 402A: B-Spline/NURBS Body CSG Evidence Contract And Prerequisite Audit](surface-402a-b-spline-nurbs-body-csg-evidence-contract-and-prerequisite-audit-v1_0.md)
- [Specification: Surface Spec 402B: Analytic To B-Spline/NURBS Body CSG Evidence Completion](surface-402b-analytic-to-b-spline-nurbs-body-csg-evidence-completion-v1_0.md)
- [Specification: Surface Spec 402B1: Analytic To B-Spline Body CSG Evidence Completion](surface-402b1-analytic-to-b-spline-body-csg-evidence-completion-v1_0.md)
- [Specification: Surface Spec 402B2: Analytic To NURBS Body CSG Evidence Completion](surface-402b2-analytic-to-nurbs-body-csg-evidence-completion-v1_0.md)
- [Specification: Surface Spec 402C: Spline/NURBS Pair Curve Body CSG Evidence Completion](surface-402c-spline-nurbs-pair-curve-body-csg-evidence-completion-v1_0.md)
- [Specification: Surface Spec 402D: Spline/NURBS Coincident Region Body CSG Evidence Completion](surface-402d-spline-nurbs-coincident-region-body-csg-evidence-completion-v1_0.md)
- [Specification: Surface Spec 402E: B-Spline/NURBS Patch Evidence Collector And Body-Route Readiness Gate](surface-402e-b-spline-nurbs-patch-evidence-collector-and-body-route-readiness-gate-v1_0.md)
- [Specification: Surface Spec 384: B-Spline And NURBS CSG Success Fixtures](surface-384-b-spline-and-nurbs-csg-success-fixtures-v1_0.md)

## Scope

This specification covers:

- routing prepared `SurfaceBody` operands with B-spline/NURBS patch families through body-level CSG execution
- consuming patch-level intersection curves and coincident-region records from existing higher-order routes
- mapping those records into trim arrangement, fragment classification, operation selection, cap policy, shell assembly, seam/provenance rebuild, and runtime validity gates
- returning a `SurfaceBooleanResult` with a valid `SurfaceBody` only when the complete body-level route succeeds
- returning deterministic diagnostics when patch-level evidence exists but body-level reconstruction cannot safely complete
- removing the concrete blocker currently preventing Spec 384 from generating
  B-spline/NURBS success fixtures: the public CSG route must no longer stop at
  patch-level records or generic unsupported diagnostics for at least the
  declared B-spline/NURBS success routes

This specification does not cover:

- implementing new B-spline/NURBS intersection solvers already owned by Specs 341-344
- creating reference STL fixtures; that remains owned by Spec 384 after this route is implemented
- using tessellated mesh fragments, mesh boolean operations, or triangle fallback as CSG truth

## Blocker Removal Contract

This is the blocker-removal specification for B-spline/NURBS reference CSG.
Spec 384 is blocked until this spec is implemented and verified.

The implementation must make the blocked state executable by adding the missing
body-level bridge, not by weakening fixture requirements. In practical terms,
the work must provide:

- a higher-order body dispatch stage that detects B-spline/NURBS operand
  participation after operand canonicalization and before generic unsupported
  fallback
- a patch-pair evidence collector that invokes the existing analytic/B-spline,
  analytic/NURBS, spline/NURBS curve, and spline/NURBS coincident-region routes
  for the participating operand patches
- a normalized evidence payload that carries patch-local curves, coincident
  region loops, residual/convergence metadata, tolerance metadata, route ids,
  source patch ids, and `no_mesh_fallback=true`
- an adapter from that evidence payload into the existing trim arrangement,
  fragment graph, operation selection, cap policy, shell assembly, seam rebuild,
  provenance map, and runtime validity gate records
- a success path that returns a valid `SurfaceBody` through `surface_boolean_result`
  or the public boolean helpers before tessellation
- a refusal path that names the precise incomplete stage when evidence is
  missing, non-convergent, ambiguous, unmappable, uncappable, invalid, or
  cannot be assembled into a valid shell

Surface Specs 402A, 402B1, 402B2, 402C, 402D, and 402E own the complete
patch-level evidence audit, repair, and collector work for Specs 341-344. This
spec may not begin implementation until those leaves are complete, because the
body-level bridge must consume normalized B-spline/NURBS evidence instead of
rediscovering or repairing patch-level route gaps.

## Implementation Boundary

Owner modules:

- `src/impression/modeling/csg.py`
- `src/impression/modeling/surface_intersections.py` only when adapter output shape must be normalized for the CSG body route
- focused tests in `tests/test_surface_csg.py`

The implementation must use the existing public CSG flow:

```text
prepare_surface_boolean_operands / prepare_surface_boolean_difference_operands
-> surface_boolean_result
-> plan_prepared_surface_csg_operation
-> surface_boolean_intersection_stage
-> patch-level B-spline/NURBS route records
-> trim arrangement and fragment graph
-> operation selection and cap policy
-> result shell assembly
-> seam/provenance rebuild
-> runtime validity gate
-> SurfaceBooleanResult
```

No successful route may bypass the body-level result gates by directly exporting a tessellated artifact.

## Data And Defaults

Required records or payloads:

- body-level route diagnostic that names operation, operand ids, patch family pair, and failed stage
- patch-level evidence reference linking intersection curves or overlap regions to body-level fragments
- result provenance metadata that records B-spline/NURBS source patch ids, route id, support state, residual/tolerance metadata, and `no_mesh_fallback=true`
- refusal diagnostic when patch-level routes succeed but body-level shell assembly, cap policy, continuity, or validity cannot complete
- route summary record for each public CSG call that states whether the
  B-spline/NURBS body path succeeded, refused before trim arrangement, refused
  during fragment/cap/shell assembly, or refused at runtime validity

Defaults:

- declared-tolerance B-spline/NURBS routes remain supported only when residual and convergence metadata are present
- coincident-region ambiguity blocks execution unless ownership and trim boundaries are resolved
- generated caps must remain surface-native; unsupported cap families refuse instead of falling back to mesh
- result family policy must preserve B-spline/NURBS payloads where trims are sufficient and may only promote when an existing architecture/spec explicitly permits the target family

## Behavior

The implementation must:

- detect B-spline/NURBS participation from prepared `SurfaceBody` operands
- dispatch analytic/B-spline, analytic/NURBS, B-spline/B-spline, B-spline/NURBS, and NURBS/NURBS patch pairs through the existing route registry
- require patch-local curves or coincident-region records before fragment graph construction
- feed all generated patch-local curves into the same trim arrangement and fragment classification path used by other surfaced CSG routes
- preserve the operation semantics for union, difference, and intersection; a
  route that only proves no-cut containment or disjoint cases is insufficient
  unless at least one non-trivial B-spline/NURBS trim/reconstruction case also
  succeeds or produces the stage-specific refusal required above
- keep B-spline/NURBS success fixtures from collapsing to primitive-equivalent
  planar or revolution cases; success evidence must show authored B-spline or
  NURBS patch participation in the input route and result provenance
- preserve source patch identities, authored metadata, and route provenance in the assembled result
- fail deterministically when any stage is incomplete, non-convergent, ambiguous, invalid, or missing required evidence
- keep `SurfaceBooleanResult.status == "succeeded"` reserved for results with a valid `SurfaceBody` or a deliberate empty result

## Verification

Focused tests must cover:

- analytic/B-spline body-level success route returning `SurfaceBooleanResult.status == "succeeded"` with a `SurfaceBody`
- analytic/NURBS body-level success route returning `SurfaceBooleanResult.status == "succeeded"` with a `SurfaceBody`
- B-spline/NURBS pair route either succeeding with body-level reconstruction or refusing with a diagnostic that names the missing body-level stage
- coincident or overlap-region ambiguity refusing before tessellation
- no-hidden-mesh-fallback evidence on every B-spline/NURBS body-level success and refusal path
- `.impress` or metadata payload checks when the result body preserves B-spline/NURBS provenance
- at least one public-route test whose input contains authored B-spline or NURBS
  patches and whose result is not merely a primitive-equivalent planar or
  revolution shortcut
- regression coverage for stale checked prerequisites: the tests must fail if
  Specs 341-344 provide only isolated patch-level records that cannot be
  consumed by the public body-level CSG route

The implementation is not complete if only patch-level intersection tests pass. At least one public body-level boolean call must validate the route through `surface_boolean_result` or the public boolean helpers.

## Readiness And Sequencing

This spec is a prerequisite for:

- [Surface Spec 384: B-Spline And NURBS CSG Success Fixtures](surface-384-b-spline-and-nurbs-csg-success-fixtures-v1_0.md)

This spec is blocked by:

- [Surface Spec 402A: B-Spline/NURBS Body CSG Evidence Contract And Prerequisite Audit](surface-402a-b-spline-nurbs-body-csg-evidence-contract-and-prerequisite-audit-v1_0.md)
- [Surface Spec 402B1: Analytic To B-Spline Body CSG Evidence Completion](surface-402b1-analytic-to-b-spline-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402B2: Analytic To NURBS Body CSG Evidence Completion](surface-402b2-analytic-to-nurbs-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402C: Spline/NURBS Pair Curve Body CSG Evidence Completion](surface-402c-spline-nurbs-pair-curve-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402D: Spline/NURBS Coincident Region Body CSG Evidence Completion](surface-402d-spline-nurbs-coincident-region-body-csg-evidence-completion-v1_0.md)
- [Surface Spec 402E: B-Spline/NURBS Patch Evidence Collector And Body-Route Readiness Gate](surface-402e-b-spline-nurbs-patch-evidence-collector-and-body-route-readiness-gate-v1_0.md)

Spec 384 must remain unchecked until this spec is complete and the fixture generation route proves that dirty STLs are exported only after a successful body-level `SurfaceBody` result exists.

Implementation order:

1. Complete Surface Specs 402A, 402B1, 402B2, 402C, 402D, and 402E so the
   patch-level evidence contract is ready for body-level consumption.
2. Implement this body-level bridge and verify it through the public boolean
   route.
3. Only then resume Spec 384 fixture generation.

## Refinement Status

Final implementation leaf.

## Five-Pass Review History

- Pass 1 - Scope Completeness: Confirmed 401 owns only the body-level bridge after patch evidence is available.
- Pass 2 - Dependency Check: Found 401 blocked by the patch-evidence work now split into 402A, 402B1, 402B2, 402C, 402D, and 402E.
- Pass 3 - Rescore: Count remains 1 IWU because 401 has one post-evidence integration route and one public-route verification surface.
- Pass 4 - Split: No split required after moving patch evidence repair into 402A, 402B1, 402B2, 402C, 402D, and 402E.
- Pass 5 - Final Review: 401 remains a final leaf, blocked until all 402 child leaves are complete.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- B-spline/NURBS `SurfaceBody` operands can reach a body-level CSG success route through the public CSG flow
- successful results return `SurfaceBooleanResult` with a valid `SurfaceBody` before tessellation
- incomplete or unsafe body-level stages refuse with deterministic diagnostics
- no implementation path performs mesh CSG or uses tessellation as a substitute for surface-body execution
- focused body-level tests and no-hidden-mesh-fallback assertions pass
