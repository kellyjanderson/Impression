# Surface Spec 353: Implicit Field Safety Validation (v1.0)

## Overview

Implement `Implicit Field Safety Validation` as a final sampled/implicit SurfaceBody CSG completion leaf.

This specification is part of the effort to promote the 153 sampled/implicit CSG rows out of raw `unsupported` state without using mesh as source truth.

## Backlink

- [Architecture: Sampled and Implicit CSG Unsupported Row Implementation Architecture](../architecture/sampled-implicit-csg-unsupported-row-implementation-architecture.md)

## Scope

This specification promotes the manifest candidate `Implicit Field Safety Validation` into a final implementation leaf.

This specification covers:

- Validate implicit field payloads before sampled/implicit CSG composition so unsafe or unbounded fields refuse deterministically.
- the records, helpers, diagnostics, route rows, payloads, and evidence named by the owning manifest candidate
- deterministic refusal behavior for unsafe, unrepresentable, missing-route, dirty-evidence, or non-applicable sampled/implicit CSG states

This specification does not permit mesh-backed CSG execution or tessellation as a substitute for surface-body execution.

## Manifest Candidate

Implicit Field Safety Validation

Discovery purpose:
- Validate implicit field payloads before sampled/implicit CSG composition so unsafe or unbounded fields refuse deterministically.

Responsibilities:
- Functions/methods: safety validator; bounded-domain checker; evaluation-budget checker
- Data structures/models: field safety report; unsafe-field diagnostic
- Dependencies/services: implicit payload builder; route planner
- Returns/outputs/signals: accepted safety report; unsafe refusal diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: reuse existing implicit safety records and add CSG route integration
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/surface.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters: unbounded or external executable fields refuse before composition
- Test strategy: safe field, missing bounds, unsafe external payload, budget overflow
- Data ownership: implicit payload owns safety facts; CSG consumes them before execution
- Routes: field graph to safety validator to composition route
- Open questions / nuance discovered: none
- Readiness blockers: none

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 1
- Total: 19.5

Split decision:
- Keep cohesive. This is the shared security/safety gate.

## Implementation Boundary

The implementation must stay inside the owner modules named above unless a small reusable helper is required to keep CSG route behavior shared.

Required boundaries:

- preserve `SurfaceBody` as the runtime result or supported refusal carrier
- preserve authored patch-family payloads, source identities, and operation provenance
- use sampling only for native sampled payloads, bounded diagnostics, declared promotion records, or explicit tessellation-boundary consumers
- never create a mesh and then treat it as the boolean source of truth
- update the CSG support, route, policy, or evidence matrix when this leaf changes row status

## Data And Defaults

The chosen defaults, routes, data ownership, open questions, and readiness blockers are the project-readiness fields in the manifest candidate above.

Additional defaults:

- supported means executable native behavior, supported promotion, or supported representation refusal, not merely a diagnostic string
- raw `unsupported` is not an acceptable final row state for the rows covered by this architecture branch
- promotion must record result family, source families, operation, tolerance, sampling or reconstruction lossiness, and provenance
- representation refusal must prove that the result is mathematically unrepresentable or deliberately non-CSG, not merely missing code

## Behavior

The implementation must:

- satisfy every responsibility listed in the manifest candidate with explicit records, helpers, diagnostics, route rows, payloads, or evidence gates
- report all unsupported-row state transitions through durable route or policy records
- keep unsafe, unavailable, unrepresentable, under-evidenced, dirty, or malformed states deterministic and inspectable
- distinguish missing solver code from impossible representation
- preserve no-hidden-mesh-fallback guarantees for every route touched by this leaf

## Verification

The test strategy is the strategy named by the manifest candidate.

Additional verification requirements:

- add focused unit coverage for every new record, helper, diagnostic, route row, payload, or evidence gate introduced by this leaf
- add negative coverage for malformed, unsafe, unrepresentable, missing-route, dirty-evidence, or non-applicable states named by this leaf
- include no-hidden-mesh-fallback assertions for every native, promoted, or refusal route touched by this leaf
- include `.impress` round-trip checks when this leaf changes persisted surface-body shape or route metadata
- update reference or diagnostic fixtures when this leaf changes visible model output or durable refusal behavior

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- `Implicit Field Safety Validation` is implemented in the owner/module named by the manifest candidate
- all manifest responsibilities are represented by explicit records, helpers, diagnostics, route records, payloads, or evidence gates
- affected sampled/implicit CSG rows no longer depend on raw `unsupported` as their final behavior
- unsupported, unsafe, unavailable, unrepresentable, missing-route, dirty-evidence, malformed, or non-applicable cases fail with deterministic diagnostics rather than hidden fallback behavior
- no implementation path performs mesh CSG or uses tessellation as a substitute for surface-body execution
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
