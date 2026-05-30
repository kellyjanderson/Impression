# Surface Spec 355: Implicit CSG Impress Payload Persistence (v1.0)

## Overview

Implement `Implicit CSG Impress Payload Persistence` as a final sampled/implicit SurfaceBody CSG completion leaf.

This specification is part of the effort to promote the 153 sampled/implicit CSG rows out of raw `unsupported` state without using mesh as source truth.

## Backlink

- [Architecture: Sampled and Implicit CSG Unsupported Row Implementation Architecture](../architecture/sampled-implicit-csg-unsupported-row-implementation-architecture.md)

## Scope

This specification promotes the manifest candidate `Implicit CSG Impress Payload Persistence` into a final implementation leaf.

This specification covers:

- Persist composed implicit CSG payloads and round-trip them through `.impress` without converting to mesh truth.
- the records, helpers, diagnostics, route rows, payloads, and evidence named by the owning manifest candidate
- deterministic refusal behavior for unsafe, unrepresentable, missing-route, dirty-evidence, or non-applicable sampled/implicit CSG states

This specification does not permit mesh-backed CSG execution or tessellation as a substitute for surface-body execution.

## Manifest Candidate

Implicit CSG Impress Payload Persistence

Discovery purpose:
- Persist composed implicit CSG payloads and round-trip them through `.impress` without converting to mesh truth.

Responsibilities:
- Functions/methods: implicit CSG payload encoder; decoder; round-trip verifier
- Data structures/models: implicit composition payload record; operation provenance payload
- Dependencies/services: implicit field expression graph; `.impress` root codec
- Returns/outputs/signals: serialized payload; round-trip diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing `.impress` implicit codec dispatch
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/io/impress.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters: payload includes operation provenance and bounded domain
- Test strategy: round-trip composed fields, malformed payload refusal, version compatibility
- Data ownership: `.impress` owns serialization; CSG owns semantic payload
- Routes: CSG result to codec to restored SurfaceBody
- Open questions / nuance discovered: none
- Readiness blockers: composition payload record

Score:
- Functions/methods: 3
- Data structures/models: 2
- Dependencies/services: 2
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 2
- Readiness blockers: 0
- Total: 18.5

Split decision:
- Keep cohesive. This leaf only covers implicit persistence.

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

- `Implicit CSG Impress Payload Persistence` is implemented in the owner/module named by the manifest candidate
- all manifest responsibilities are represented by explicit records, helpers, diagnostics, route records, payloads, or evidence gates
- affected sampled/implicit CSG rows no longer depend on raw `unsupported` as their final behavior
- unsupported, unsafe, unavailable, unrepresentable, missing-route, dirty-evidence, malformed, or non-applicable cases fail with deterministic diagnostics rather than hidden fallback behavior
- no implementation path performs mesh CSG or uses tessellation as a substitute for surface-body execution
- verification requirements are covered by implementation tests and any paired test specification created from this leaf
