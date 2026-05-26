# Impress Spec 10: .impress Reader And Load Result Contract (v1.0)

## Overview

Define public load behavior, load result shape, and schema/payload error reporting.

## Backlink

- [Architecture: Impress Surface Native File Format Architecture](../architecture/impress-surface-native-file-format-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Reader And Load Result Contract` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - `load_impress`
  - deterministic JSON reader
- Data structures/models:
  - load result
  - schema error
  - payload error
- Dependencies/services:
  - `.impress` codec
  - JSON decoder
  - filesystem paths
- Returns/outputs/signals:
  - loaded body store
  - IO/schema errors
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: standard JSON and pathlib
  - Additions to existing reusable library/module: `.impress` persistence module
  - New reusable library/module to create: none beyond persistence module
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads user-selected files
- Security/privacy-sensitive behavior:
  - refuses unsafe schema and does not execute payloads
- Performance-sensitive behavior:
  - load validation linear in entity count
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- future `.impress` persistence module

Routes:

- public load API to reader to codec

Reuse/extraction decision:

- add to `.impress` persistence module

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- load returns `SurfaceBodyStore` or raises structured diagnostics according to final error policy

Data ownership:

- reader owns file boundary; codec owns payload validation

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

- tests for load round trip, malformed JSON, unsupported schema, and invalid payload

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
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
- Total: 21.5

Split decision:

- Review for split. Cohesion reason: reading, load result, and load diagnostics are one input boundary.

Open questions / nuance resolved for implementation:

- Need final error style before implementation spec promotion: exceptions versus result object.

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
