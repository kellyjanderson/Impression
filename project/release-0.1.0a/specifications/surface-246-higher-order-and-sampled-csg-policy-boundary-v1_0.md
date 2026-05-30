# Surface Spec 246: Higher-Order And Sampled CSG Policy Boundary (v1.0)

## Overview

Define whether B-spline, NURBS, subdivision, implicit, heightmap, and
displacement booleans are executable kernel operations, declared-tolerance
adapters, or explicit non-CSG refusals.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Higher-Order And Sampled
CSG Policy Boundary` into a final implementation leaf.

## Responsibilities

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

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py` and family capability policy

Routes:

- boolean API to CSG support matrix to higher-order policy resolver

Reuse/extraction decision:

- share refusal diagnostics with the CSG matrix spec

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- higher-order operations refuse until explicitly classified as exact,
  declared-tolerance, or bounded adapter support

Data ownership:

- CSG owns operation policy
- family modules own family validity

## Behavior

The implementation must:

- classify every higher-order and sampled family as exact, declared-tolerance,
  bounded adapter, unsupported, or not-yet implemented for each boolean
  operation
- refuse implicit and sampled operations that would require hidden
  tessellation-based CSG
- preserve declarative safety rules for implicit fields
- report refusal diagnostics that identify the family, operation, and required
  future solver boundary

## Verification

Test strategy:

- policy matrix tests for each higher-order and sampled family
- refusal tests proving no hidden tessellation CSG
- implicit safety policy tests for CSG entrypoints

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Split decision:

- Review for split. Cohesion reason: this is one policy boundary for
  non-analytic CSG; solver implementation remains in separate future specs.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- higher-order and sampled CSG support policy is explicit
- hidden tessellation CSG is refused
- future solver gaps are named rather than left ambiguous
