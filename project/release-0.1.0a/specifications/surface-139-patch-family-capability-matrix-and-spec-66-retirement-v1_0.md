# Surface Spec 139: Patch Family Capability Matrix And Spec 66 Retirement (v1.0)

## Overview

Replace deferred-family/exclusion posture with a first-class capability matrix that tracks staged support without declaring families out of scope.

## Backlink

- [Architecture: Full Surface Patch Family Architecture](../architecture/full-surface-patch-family-architecture.md)

## Scope

This specification promotes the manifest candidate `Patch Family Capability Matrix And Spec 66 Retirement` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - capability matrix generator or maintained table
  - Spec 66 retirement note
- Data structures/models:
  - patch family capability record
  - operation support phase record
- Dependencies/services:
  - existing Surface Specs 65-67
  - full patch family architecture
- Returns/outputs/signals:
  - capability matrix
  - retired/replaced deferred-family spec status
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current patch family docs/specs
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - revises/retire existing spec posture
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - none
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `project/release-0.1.0a/specifications/`

Routes:

- architecture to replacement specs

Reuse/extraction decision:

- revise existing specs; no code module

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- no patch family is architecturally deferred; unsupported operations report
    capability-aware diagnostics

Data ownership:

- capability matrix owns support status truth

## Behavior

The implementation must:

- satisfy every function, data-structure, dependency, and output responsibility listed above
- preserve the architecture boundary named in the backlink
- reject unsupported or ambiguous states with explicit diagnostics rather than silent fallback behavior
- keep mesh data outside canonical authored-surface state unless this spec explicitly names a tessellation, compatibility, or mesh-utility boundary
- expose only the public API surface needed by downstream specs and tests

## Constraints

- The implementation must remain deterministic for equivalent inputs.
- The implementation must keep metadata and stable identity behavior explicit when the leaf touches persisted or reusable surface state.
- The implementation must not introduce hidden mesh execution in authored modeling paths.
- The implementation must not broaden industry interchange, patch-family, or mesh compatibility scope beyond what this leaf names.

## Verification

Test strategy:

- documentation/spec review plus tests in family leaf specs

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 13.5

Split decision:

- Small.
- This can remain one replacement spec because it is a policy/matrix artifact.

Open questions / nuance resolved for implementation:

- Matrix must distinguish "family exists" from "operation supports this family."

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- all manifest responsibilities are implemented or explicitly refused by the leaf
- owner/module, routes, data ownership, reuse, UI inventory, defaults, and test strategy are represented in code or verification artifacts
- related progression items can be checked without relying on unstated architecture assumptions
- downstream specs can cite this leaf instead of re-reading the manifest candidate
