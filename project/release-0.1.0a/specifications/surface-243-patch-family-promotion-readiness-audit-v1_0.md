# Surface Spec 243: Patch Family Promotion Readiness Audit (v1.0)

## Overview

Audit every authored patch family against promotion criteria and produce exact
missing-work records for storage, evaluation, seams, tessellation, `.impress`,
CSG, loft, and diagnostics.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Patch Family Promotion
Readiness Audit` into a final implementation leaf.

## Responsibilities

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

## Implementation Boundary

Owner/module:

- `src/impression/modeling/surface.py`, tessellation, and release verification

Routes:

- family records to readiness checker to progression/spec gaps

Reuse/extraction decision:

- extend the capability matrix rather than creating per-family silo checkers

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- `available` requires implementation plus evidence, not architectural intent

Data ownership:

- patch family capability matrix owns support status truth

## Behavior

The implementation must:

- audit planar, ruled, revolution, B-spline, NURBS, sweep, subdivision,
  implicit, heightmap, and displacement families
- report missing record, evaluator, derivative, seam, tessellation, `.impress`,
  CSG, loft, and diagnostic coverage separately
- distinguish unsupported operation diagnostics from family availability
- block promotion when round-trip or reference evidence is missing

## Verification

Test strategy:

- unit tests for auditor classification
- missing-work diagnostics tests
- fixture tests for at least one available, planned, unsupported-operation, and
  partially implemented family

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 4 x 1 = 4
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
- Total: 21.5

Split decision:

- Review for split. Cohesion reason: this is one audit/gate over existing
  family-specific implementation specs; splitting by family would duplicate the
  full patch-family manifest.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- every authored patch family has a promotion verdict
- every missing promotion criterion is named explicitly
- no family is promoted without implementation and evidence
