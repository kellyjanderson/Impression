# Surface Spec 240: .impress Implicit Patch Payload Codec (v1.0)

## Overview

Define safe `.impress` encoding and decoding for declarative implicit patch
payloads, excluding hostile payload refusal fixtures.

## Backlink

- [Architecture: Full Surface Patch Family Architecture](../architecture/full-surface-patch-family-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Implicit Patch
Payload Codec` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - implicit payload encoder
  - implicit payload decoder
- Data structures/models:
  - implicit payload
  - allow-listed field node
- Dependencies/services:
  - `.impress` codec
  - implicit field validator
- Returns/outputs/signals:
  - encoded implicit payload
  - decoded implicit patch
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: implicit field validator once implemented
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads/writes `.impress` files
- Security/privacy-sensitive behavior:
  - declarative allow-listing is required; hostile refusal fixtures are split
    out
- Performance-sensitive behavior:
  - validation bounds field tree size
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- future `.impress` codec module

Routes:

- file codec to implicit field validator

Reuse/extraction decision:

- add to `.impress` codec module and reuse the implicit field validator

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- implicit payloads are declarative allow-listed data only

Data ownership:

- `.impress` payload owns serialized field data
- evaluator owns runtime interpretation

## Behavior

The implementation must:

- encode declarative implicit field nodes without executable code
- decode only allow-listed field node payloads
- preserve field parameters, patch metadata, identity, and parameter-domain
  references
- enforce field tree size limits before returning a decoded implicit patch
- delegate hostile payload refusal cases to the security refusal fixture spec

## Verification

Test strategy:

- safe implicit payload round-trip tests
- field-node allow-list tests
- metadata and identity preservation tests
- field tree size boundary tests for valid payloads

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
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
- Total: 19.5

Split decision:

- Review for split. Cohesion reason: this is one safe declarative payload codec
  after hostile refusal fixtures were split out.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- safe implicit payloads round-trip deterministically
- decoded implicit payloads remain declarative
- field tree bounds are enforced before runtime interpretation
