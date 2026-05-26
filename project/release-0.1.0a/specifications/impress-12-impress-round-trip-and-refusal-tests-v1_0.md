# Impress Spec 12: .impress Round Trip And Refusal Tests (v1.0)

## Overview

Define acceptance tests proving `.impress` preserves identity/metadata and refuses invalid files.

## Backlink

- [Architecture: Impress Surface Native File Format Architecture](../architecture/impress-surface-native-file-format-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Round Trip And Refusal Tests` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - round-trip fixture builder
  - invalid-file fixture builder
  - assertion helpers
- Data structures/models:
  - round-trip fixture
  - invalid-file case
  - metadata preservation assertion
- Dependencies/services:
  - `.impress` IO API
  - surface model fixtures
- Returns/outputs/signals:
  - passing round-trip test
  - explicit refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing surface fixtures
  - Additions to existing reusable library/module: test fixture helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes temporary test files
- Security/privacy-sensitive behavior:
  - invalid-file tests include unsafe implicit/code payload refusal
- Performance-sensitive behavior:
  - fixture sizes are bounded
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `tests/`

Routes:

- tests to public `.impress` save/load API

Reuse/extraction decision:

- add reusable test helpers

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- tests compare semantic payload and deterministic serialized output where
    appropriate

Data ownership:

- tests own fixtures; IO API owns behavior under test

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

- this is the test specification branch for `.impress`

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:

- Review for split.
- Cohesion reason: round-trip and refusal tests are the paired acceptance gate
  for the same file format boundary.

Open questions / nuance resolved for implementation:

- Should be paired with implementation specs rather than waiting until the end.

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
