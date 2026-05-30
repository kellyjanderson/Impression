# Impress Spec 14: .impress Base Patch Payload Codec (v1.0)

## Overview

Define encoding and decoding for base patch fields shared by every `.impress`
surface patch family, excluding family dispatch and constructor validation.

## Backlink

- [Architecture: .impress Surface-Native File Format Architecture](../architecture/impress-surface-native-file-format-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Base Patch Payload
Codec` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - encode base patch fields
  - decode base patch fields
- Data structures/models:
  - base patch payload
  - patch identity payload
- Dependencies/services:
  - surface model
  - `.impress` root
- Returns/outputs/signals:
  - encoded base patch payload
  - decoded base patch field bundle
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface patch base fields
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes `.impress` patch payloads
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to patch field size
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- future `.impress` codec module

Routes:

- save/load to base patch codec

Reuse/extraction decision:

- add shared base patch helpers to the `.impress` codec module

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- all family codecs consume the same base patch field bundle

Data ownership:

- codec owns serialized base fields
- patch constructors own runtime invariants

## Behavior

The implementation must:

- encode identity, family tag, domain, transform, metadata, and base patch
  fields that are shared across families
- decode those shared fields into a typed intermediate bundle before family
  dispatch
- reject missing or malformed base fields with deterministic diagnostics
- avoid embedding family-specific payload interpretation in the base codec
- preserve stable identity and metadata exactly enough for later family codecs
  and whole-store round-trip tests

## Verification

Test strategy:

- base patch round-trip tests
- missing-field and malformed-field refusal tests
- identity and metadata preservation tests
- tests proving family-specific payload validation is delegated to family
  dispatch

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
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Split decision:

- Review for split. Cohesion reason: this is one shared base payload codec
  after family dispatch and constructor validation were split out.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- shared patch fields round-trip through `.impress`
- malformed base fields refuse with deterministic diagnostics
- family dispatch remains outside the base codec
- progression can mark this leaf complete without relying on unstated
  architecture assumptions
