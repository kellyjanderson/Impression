# Surface Spec 247: Loft Ambiguity Accumulation And Execution Refusal Gate (v1.0)

## Overview

Ensure authored loft planning accumulates all ambiguity and invalid-input
records, while execution refuses any plan that still carries unresolved
ambiguity.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Loft Ambiguity Accumulation
And Execution Refusal Gate` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - ambiguity accumulation pass
  - plan executability gate
  - executor refusal check
- Data structures/models:
  - unresolved ambiguity record
  - plan executability status
  - invalid-input aggregate
- Dependencies/services:
  - loft planner
  - loft plan object
  - topology correspondence architecture
- Returns/outputs/signals:
  - non-executable plan result
  - aggregate ambiguity diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current loft plan and diagnostics records
  - Additions to existing reusable library/module: executability gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes loft execution refusal behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - planner continues after errors but remains bounded by topology size
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/loft.py`

Routes:

- authored topology to planner to plan status to surface executor

Reuse/extraction decision:

- extend existing diagnostics rather than creating a separate validator

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- unresolved ambiguity always blocks execution
- planning still reports all detectable problems

Data ownership:

- `LoftPlan` owns executability status
- planner owns ambiguity records

## Behavior

The implementation must:

- continue planning after detecting an ambiguity when doing so is bounded and
  safe
- accumulate all ambiguity and invalid-input records in the returned plan
- mark any plan with unresolved ambiguity as non-executable
- require every executor to refuse non-executable plans before emitting surface
  output
- avoid automatic ambiguity resolution unless a separate high-confidence policy
  authorizes it

## Verification

Test strategy:

- authored ambiguous loft fixtures with multiple simultaneous ambiguity records
- executor refusal assertions
- tests proving all reportable ambiguity records are preserved
- tests proving valid unambiguous plans still execute

## Manifest Assessment

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

- Review for split. Cohesion reason: accumulation and execution refusal are one
  plan lifecycle invariant and should be tested together.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- ambiguous plans report all detectable ambiguity records
- any unresolved ambiguity prevents execution
- executor refusal is deterministic and diagnostic-rich
