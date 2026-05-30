# Surface Spec 253: Feature Builder Patch Producer Handoff (v1.0)

## Overview

Make Impression-owned feature builders hand off `SurfaceBody` or
`SurfaceConsumerCollection` truth with explicit unsupported diagnostics and no
hidden mesh substitution.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `Feature Builder Patch
Producer Handoff` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - feature handoff validator
  - feature producer diagnostic
- Data structures/models:
  - feature surface output contract
  - feature unsupported producer result
- Dependencies/services:
  - loft surface producer
  - hinge feature builders
  - no-hidden-mesh-fallback gate
- Returns/outputs/signals:
  - surface truth result
  - consumer collection result
  - explicit unsupported diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: loft and feature builder modules
  - Additions to existing reusable library/module: feature handoff helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes authored feature output routing
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - handoff validation should be deterministic and bounded by feature output
    size
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- loft surface producer and Impression-owned feature builders

Routes:

- feature API to handoff validator to SurfaceBody/consumer output

Reuse/extraction decision:

- use one feature handoff helper rather than per-feature ad hoc checks

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- feature builders return surface truth or explicit unsupported diagnostics
- mesh is never a hidden substitute

Data ownership:

- feature builders own authored output contract
- surface body owns emitted geometry

## Behavior

The implementation must:

- validate feature-builder outputs before they leave authored modeling APIs
- allow `SurfaceBody` and `SurfaceConsumerCollection` as surface truth handoff
  forms
- report unsupported feature producer requests explicitly
- keep sibling-project consumers outside Impression-owned implementation scope
- preserve explicit mesh compatibility only when the API name declares mesh
  compatibility

## Verification

Test strategy:

- no-hidden-mesh-fallback tests for loft and feature builders
- feature handoff validator tests
- unsupported feature diagnostic tests

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
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
- Total: 18.5

Split decision:

- Review for split. Cohesion reason: this is one feature-builder handoff
  contract after primitive routing was split out.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- feature builders hand off surface truth or explicit unsupported diagnostics
- hidden mesh substitute outputs are impossible in authored feature routes
- sibling project integration remains consumer-side
