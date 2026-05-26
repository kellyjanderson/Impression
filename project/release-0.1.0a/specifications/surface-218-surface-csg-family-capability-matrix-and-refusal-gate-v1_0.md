# Surface Spec 218: Surface CSG Family Capability Matrix And Refusal Gate (v1.0)

## Overview

Define the authoritative family-pair and operation matrix for CSG support,
including explicit refusal behavior for unsupported pairs.

## Backlink

- [Architecture: SurfaceBody CSG Architecture](../architecture/surfacebody-csg-architecture.md)

## Scope

This specification promotes the manifest candidate `Surface CSG Family
Capability Matrix And Refusal Gate` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - CSG capability matrix lookup
  - operation/family eligibility gate
- Data structures/models:
  - family-pair support record
  - unsupported CSG diagnostic payload
- Dependencies/services:
  - `src/impression/modeling/csg.py`
  - patch family capability matrix
- Returns/outputs/signals:
  - executable support decision
  - explicit refusal result
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG diagnostics and patch family matrix
  - Additions to existing reusable library/module: CSG family support table
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - updates CSG behavior and tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded matrix lookup on CSG entry
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `src/impression/modeling/csg.py`

Routes:

- public boolean API to CSG support gate

Reuse/extraction decision:

- extend existing CSG diagnostics

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- unsupported pairs refuse before any mesh path is reachable

Data ownership:

- CSG owns CSG operation support; patch families own patch capabilities

## Behavior

The implementation must:

- satisfy every function, data-structure, dependency, and output responsibility
  listed above
- preserve the architecture boundary named in the backlink
- reject unsupported or ambiguous states with explicit diagnostics rather than
  silent fallback behavior
- keep mesh data outside canonical authored-surface state unless this spec
  explicitly names a tessellation, compatibility, or mesh-utility boundary
- expose only the public API surface needed by downstream specs and tests

## Constraints

- The implementation must remain deterministic for equivalent inputs.
- The implementation must not introduce hidden mesh execution in authored
  modeling paths.
- Refusal is a valid surface CSG result until exact execution supports the
  family pair.

## Verification

Test strategy:

- matrix coverage for supported and unsupported pairs

Automated or review verification must prove support decisions are deterministic
and unsupported pairs never invoke mesh boolean behavior.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
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
- Total: 17

Split decision:

- Review for split. Cohesion reason: the matrix and refusal gate are one
  decision boundary.

Open questions / nuance resolved for implementation:

- Unsupported pairs must fail before any compatibility mesh path can be chosen.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when the CSG matrix gates every public boolean
entrypoint and unsupported family pairs return explicit diagnostics without
mesh fallback.
