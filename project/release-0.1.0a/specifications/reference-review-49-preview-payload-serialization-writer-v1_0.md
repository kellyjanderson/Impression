# Reference Review Spec 49: Preview Payload Serialization Writer (v1.0)

## Overview

Convert loaded preview datasets into immutable or file-backed payloads that the UI-owned preview widget can consume.

## Backlink

- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Payload Serialization Writer`.
- Manifest score: 22

## Scope

This specification covers:

- Convert loaded preview datasets into immutable or file-backed payloads that the UI-owned preview widget can consume.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - payload serializer
  - temporary payload file writer
- Data structures/models:
  - `PreviewPayload`
  - payload file metadata
- Dependencies/services:
  - preview payload records
- Returns/outputs/signals:
  - serialized payload
  - serialization diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - none
  - Additions to existing reusable library/module:
    - preview payload module
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - writer runs outside UI thread
- Destructive/write behavior:
  - writes temporary payload files only
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - avoids passing live VTK or Qt objects across boundaries
- Cross-screen reusable behavior:
  - serialized payload feeds the embedded preview widget

## Implementation Boundary

Owner/module:

- future preview payload module

Routes:

- loaded preview datasets to serialized payload result

Reuse/extraction decision:

- Existing code reused as-is:
  - none
- Additions to existing reusable library/module:
  - preview payload module
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- file-backed payloads are used when immutable in-memory payloads would be too
  large

Data ownership:

- writer owns payload files until controller cleanup takes over

Open questions and resolved assumptions:

- exact payload file format remains a specification default

Implementation prerequisites:

- preview payload records

## Behavior

The implementation must:

- Convert loaded preview datasets into immutable or file-backed payloads that the UI-owned preview widget can consume.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- serialization, temp file, and invalid payload tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 0 x 0.5 = 0
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22

Split decision:

- No split needed. Cohesive serialization/file-backed payload leaf.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Payload Serialization Writer boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
