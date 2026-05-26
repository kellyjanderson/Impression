# Impress Spec 09: .impress Deterministic JSON Writer (v1.0)

## Overview

Define byte-stable deterministic JSON writing for `.impress` payloads.

## Backlink

- [Architecture: Impress Surface Native File Format Architecture](../architecture/impress-surface-native-file-format-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Deterministic JSON Writer` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - `save_impress` writer path
  - deterministic JSON writer
- Data structures/models:
  - save options
  - serialized document payload
- Dependencies/services:
  - `.impress` codec
  - JSON encoder
  - filesystem paths
- Returns/outputs/signals:
  - deterministic persisted file
  - write error
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
  - writes user-selected files
- Security/privacy-sensitive behavior:
  - does not execute payloads; writes validated data only
- Performance-sensitive behavior:
  - V1 may serialize in memory; output order must be deterministic
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- future `.impress` persistence module

Routes:

- public save API to codec to deterministic writer

Reuse/extraction decision:

- add to `.impress` persistence module

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- V1 writes sorted-key deterministic JSON with explicit schema version

Data ownership:

- writer owns file boundary; codec owns payload validation

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

- tests for byte-stable output and write errors

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

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

- Review for split. Cohesion reason: deterministic writing is one file-output behavior.

Open questions / nuance resolved for implementation:

- Atomic replacement is handled by the separate atomic write/error spec.

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
