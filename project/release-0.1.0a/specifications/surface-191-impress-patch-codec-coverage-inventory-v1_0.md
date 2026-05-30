# Surface Spec 191: .impress Patch Codec Coverage Inventory (v1.0)

## Overview

Establish the authoritative `.impress` family codec coverage matrix before adding missing codecs.

## Backlink

- [Architecture: Patch Family Integration Architecture](../architecture/patch-family-integration-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Patch Codec Coverage Inventory` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Responsibilities

- Functions/methods:
  - codec inventory scanner
  - family map assertion
- Data structures/models:
  - codec coverage record
  - patch family list
- Dependencies/services:
  - `src/impression/io/impress.py`
  - `src/impression/modeling/surface.py`
- Returns/outputs/signals:
  - missing codec diagnostic
  - coverage matrix test result
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `.impress` codec helpers
  - Additions to existing reusable library/module: test helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - test/spec only unless inventory reveals stale declarations
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded static inspection of codec maps
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

  - `tests/` plus `src/impression/io/impress.py` if stale map entries are found

Routes:

  - test helper reads public constants and codec module maps

Reuse/extraction decision:

  - add test helper; do not move codec tables unless duplication appears

UI field/control inventory:

  - not applicable

## Data And Defaults

Chosen defaults / parameters:

  - the codec matrix must include every family marked `available`

Data ownership:

  - `.impress` owns persistence codec truth; `surface.py` owns family list truth

## Behavior

The implementation must:

- satisfy every function, data-structure, dependency, and output responsibility listed above
- preserve the architecture boundary named in the backlink
- reject unsupported or ambiguous states with explicit diagnostics rather than silent fallback behavior
- keep mesh data outside canonical authored-surface state unless this spec explicitly names a tessellation, compatibility, or mesh-utility boundary
- preserve family-native payloads, stable identity, and capability metadata when the leaf touches surface storage, traversal, persistence, or tessellation
- expose only the public API surface needed by downstream specs and tests

## Constraints

- The implementation must remain deterministic for equivalent inputs.
- The implementation must keep metadata and stable identity behavior explicit when the leaf touches persisted or reusable surface state.
- The implementation must not introduce hidden mesh execution in authored modeling paths.
- The implementation must not broaden industry interchange, patch-family, or mesh compatibility scope beyond what this leaf names.
- Bounded fixture, sampling, and performance limits named in the manifest must be represented in code or tests before this leaf is marked complete.

## Verification

Test strategy:

  - static test comparing codec map against patch-family capability matrix

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks. Where this spec touches tessellation, verification must prove mesh output is created only at the explicit tessellation boundary and records lossiness or approximation metadata when applicable.

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Split decision:

- Review for split.
- Keep together because this is a bounded inventory and assertion spec, not the
  implementation of every missing codec.

Open questions / nuance resolved for implementation:

- The codec inventory should not require codecs for `planned` families unless
  the architecture decides `.impress` leads availability.

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
