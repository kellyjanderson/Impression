# Impress Spec 06: .impress Trim Payload Codec (v1.0)

## Overview

Define trim-loop payload encoding/decoding independent of patch-family geometry.

## Backlink

- [Architecture: Impress Surface Native File Format Architecture](../architecture/impress-surface-native-file-format-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Trim Payload Codec` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - encode trim
  - decode trim
  - trim constructor validation bridge
- Data structures/models:
  - trim payload
  - trim orientation/category record
- Dependencies/services:
  - surface model
  - `.impress` root
- Returns/outputs/signals:
  - encoded trim payload
  - decoded trim object
  - trim diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: trim loop constructors
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes or changes only the scoped artifact described by this spec
- Security/privacy-sensitive behavior:
  - none unless explicitly named in this candidate
- Performance-sensitive behavior:
  - bounded to the scoped artifact described by this spec
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- future `.impress` codec module

Routes:

- save/load to trim codec after patch decode

Reuse/extraction decision:

- add to `.impress` codec module

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- trims validate orientation/category without evaluating dense geometry

Data ownership:

- codec owns serialized form; trim constructors own invariants

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

- trim round-trip, invalid orientation, and missing-loop tests

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
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
- Total: 22.5

Split decision:

- Review for split. Cohesion reason: trim payload is one non-family-specific boundary contract.

Open questions / nuance resolved for implementation:

- Trim codec is separated because trim validity can fail even when patch payloads are valid.

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
