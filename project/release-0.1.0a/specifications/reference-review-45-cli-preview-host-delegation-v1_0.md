# Reference Review Spec 45: CLI Preview Host Delegation (v1.0)

## Overview

Rewire the existing CLI preview host to use the shared preview controller without changing CLI launch behavior.

## Backlink

- [Architecture: Reference Review Preview Engine Sharing Architecture](../architecture/reference-review-preview-engine-sharing-architecture.md)

## Source Manifest

- This leaf is derived from the manifest candidate `CLI Preview Host Delegation`.
- Manifest score: 20.5

## Scope

This specification covers:

- Rewire the existing CLI preview host to use the shared preview controller without changing CLI launch behavior.
- the data structures, helpers, diagnostics, and verification gates named by the manifest
- the exact implementation boundary needed by the owning architecture document

## Responsibilities

- Functions/methods:
  - CLI host delegation call
  - compatibility adapter for existing CLI options
- Data structures/models:
  - CLI-to-controller option mapping
- Dependencies/services:
  - `impression.preview`
  - shared preview controller
- Returns/outputs/signals:
  - CLI preview scene result
- UI surfaces/components:
  - CLI preview window
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is:
    - CLI application lifecycle
  - Additions to existing reusable library/module:
    - CLI controller delegation
  - New reusable library/module to create:
    - none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - unchanged from CLI preview host
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - no additional renderer creation during delegation
- Cross-screen reusable behavior:
  - keeps CLI and workbench preview behavior aligned

## Implementation Boundary

Owner/module:

- `src/impression/preview.py`

Routes:

- CLI options to controller options to scene application

Reuse/extraction decision:

- Existing code reused as-is:
  - CLI application lifecycle
- Additions to existing reusable library/module:
  - CLI controller delegation
- New reusable library/module to create:
  - none

UI field/control inventory:

- none

## Data And Defaults

Chosen defaults / parameters:

- CLI defaults are preserved unless the shared controller already exposes the
  same behavior

Data ownership:

- CLI host owns launch and renderer lifecycle; controller owns scene behavior

Open questions and resolved assumptions:

- none

Implementation prerequisites:

- shared scene application and camera policy

## Behavior

The implementation must:

- Rewire the existing CLI preview host to use the shared preview controller without changing CLI launch behavior.
- preserve the shared-preview boundary so the workbench does not grow a separate renderer path
- preserve renderer, payload, and UI-thread ownership described by the architecture
- emit deterministic diagnostics for invalid, unsupported, unsafe, or stale preview states

## Verification

Test strategy:

- CLI delegation smoke and option mapping tests

Additional verification requirements:

- add focused unit coverage for the records, helpers, or routes introduced by this specification
- add regression coverage for stale-result, cancellation, diagnostic, or renderer-lifetime behavior when this leaf touches those paths
- ensure real `.impress` fixtures remain the preferred manual smoke path for interactive preview behavior

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 1 x 3 = 3
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:

- No split needed. Cohesive host-migration leaf; controller extraction remains
  separate.

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the CLI Preview Host Delegation boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
- listed implementation prerequisites are satisfied or carried as explicit blocking failures before execution
