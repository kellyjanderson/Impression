# Reference Review Spec 48: Preview Source Load And Tessellation Adapter (v1.0)

## Overview

Load the selected fixture source and tessellate it for preview payload creation outside the UI thread.

## Backlink

- [Architecture: Reference Review Preview Payload Boundary Architecture](../architecture/reference-review-preview-payload-boundary-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Preview Source Load And Tessellation Adapter`.
- Manifest score: 21.5

## Scope

This specification covers:

- Load the selected fixture source and tessellate it for preview payload creation outside the UI thread.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - source load adapter
  - tessellation adapter
- Data structures/models:
  - loaded preview dataset
- Dependencies/services:
  - fixture source contract
  - Impression source loading and tessellation
- Returns/outputs/signals:
  - loaded preview datasets
  - source/tessellation diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - fixture source records and Impression loading
      code
  - Additions to existing reusable library/module:
    - none
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - runs outside the Qt UI thread
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - diagnostics redact unsafe environment details
- Performance-sensitive behavior:
  - no source loading or tessellation on the UI thread
- Cross-screen reusable behavior:
  - loaded preview datasets feed payload serialization

## Implementation Boundary

Owner/module:

- future preview payload builder module

Routes:

- payload request to source/tessellation adapter

Reuse/extraction decision:

- Existing code reused as-is:
  - fixture source records and Impression loading
    code
- Additions to existing reusable library/module:
  - none
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- adapter imports no workbench UI shell or PySide widget modules

Data ownership:

- adapter owns loaded preview datasets until serialization

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- payload request records

## Behavior

The implementation must:

- Load the selected fixture source and tessellate it for preview payload creation outside the UI thread.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- import-boundary and fixture load/tessellation tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:

- No split needed. Cohesive non-UI source-load/tessellation leaf.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Preview Source Load And Tessellation Adapter boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
