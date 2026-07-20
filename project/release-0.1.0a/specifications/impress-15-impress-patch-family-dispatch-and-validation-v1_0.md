# Impress Spec 15: .impress Patch Family Dispatch And Validation (v1.0)

## Overview

Define patch-family dispatch, decoded payload routing, and constructor
validation diagnostics for `.impress` patch loading.

## Backlink

- [Architecture: .impress Surface-Native File Format Architecture](../architecture/impress-surface-native-file-format-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Patch Family
Dispatch And Validation` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - patch family dispatcher
  - patch constructor validation bridge
  - invalid family diagnostic builder
- Data structures/models:
  - family dispatch record
  - patch diagnostic
- Dependencies/services:
  - patch family modules
  - base patch payload codec
  - `.impress` reader
- Returns/outputs/signals:
  - decoded patch object
  - invalid family diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface patch constructors
  - Additions to existing reusable library/module: `.impress` codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - refuses unknown or unsafe family payloads
- Performance-sensitive behavior:
  - dispatch bounded by allow-listed family registry
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- future `.impress` codec module

Routes:

- reader to family dispatcher to patch constructor

Reuse/extraction decision:

- add a dispatch registry to the `.impress` codec module

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- decode validates through public patch constructors and refuses unknown
  families

Data ownership:

- family dispatcher owns serialized-to-runtime routing
- patch constructors own runtime invariants

## Behavior

The implementation must:

- dispatch only to allow-listed patch families
- pass decoded base fields and family payloads through the appropriate
  constructor or validator
- produce deterministic diagnostics for unknown families, wrong payload shapes,
  constructor validation failures, and unsupported family payload versions
- preserve the separation between base patch payloads and family-specific
  payloads
- reuse security-sensitive validators for implicit or otherwise constrained
  family payloads

## Verification

Test strategy:

- invalid family tests
- invalid constructor payload tests
- family dispatch tests for every promoted patch family
- refusal tests for unsafe or unsupported payload families

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
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

- Review for split. Cohesion reason: dispatch and constructor validation are
  one load-time routing/refusal boundary.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- all promoted families route through explicit dispatch
- unknown and invalid family payloads refuse deterministically
- constructors remain the source of runtime invariants
- no family payload is silently downgraded to mesh or generic patch data
