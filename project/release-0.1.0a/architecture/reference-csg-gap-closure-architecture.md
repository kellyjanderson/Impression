# Reference CSG Gap Closure Architecture

## Overview

This document maps the unchecked items in
[Reference Test Expansion Plan](../planning/reference-test-expansion-plan.md)
to the architecture work required before more reference STL fixtures can be
created honestly.

The reference suite may only add a dirty STL fixture after the model generator
executes the relevant surface-native behavior and produces a `SurfaceBody`
without hidden mesh fallback. Unsupported routes should remain visible as
architecture, specification, or implementation gaps.

## Related Architecture

This document coordinates the supplemental reference-gap architecture branch:

- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [Surface CSG Executable Completion Architecture](surface-csg-executable-completion-architecture.md)
- [Higher-Order Surface CSG Solver Architecture](higher-order-surface-csg-solver-architecture.md)
- [Sampled and Implicit Surface CSG Support Architecture](sampled-implicit-surface-csg-support-architecture.md)
- [CSG Coincident Contact Architecture](csg-coincident-contact-architecture.md)
- [Patch-Family Reference CSG Completion Architecture](patch-family-reference-csg-completion-architecture.md)
- [Loft Self-Intersection Reference Architecture](loft-self-intersection-reference-architecture.md)
- [Lofted Body CSG Reference Architecture](lofted-body-csg-reference-architecture.md)

## Current Unchecked Reference Groups

### Primitive CSG Contact Gap

- `RT-CSG-009` coincident-face box union and difference

The difference half is represented by the existing coincident-face difference
fixture. The missing behavior is the face-touch union case, where two solids
share a boundary face without overlapping volume.

### Patch-Family CSG Gaps

- `RT-PATCH-CSG-001` planar patch CSG against box and sphere cutters
- `RT-PATCH-CSG-003` revolution patch CSG from cylinder/cone/sphere/torus-like
  bodies
- `RT-PATCH-CSG-004` B-spline patch CSG
- `RT-PATCH-CSG-005` NURBS patch CSG
- `RT-PATCH-CSG-006` sweep patch CSG
- `RT-PATCH-CSG-007` subdivision patch CSG
- `RT-PATCH-CSG-011` mixed planar/ruled/revolution matrix
- `RT-PATCH-CSG-012` sampled/implicit mixed-family promotion fixtures
- `RT-PATCH-CSG-013` unsupported-family route fixtures
- `RT-PATCH-CSG-014` no-hidden-mesh-fallback evidence fixtures

These are not just fixture additions. They require a reference-facing policy for
which solver rows count as exact, declared-tolerance, promoted-family, explicit
refusal, or non-CSG replacement behavior.

### Loft Reference Gap

- `RT-LOFT-037` self-intersection detection case

This is a loft validity and diagnostic reference, not a CSG solver feature.

### Lofted Body CSG Gaps

- `RT-LOFT-CSG-001` through `RT-LOFT-CSG-014`

These require loft result eligibility, connected closed-valid shell evidence,
ruled/revolution patch provenance, branch/topology refusal policy, color and
metadata propagation, and section evidence.

## Components

### Progression Gap Mapper

The mapper owns the relationship between unchecked reference-test progression
items and architecture documents. It prevents a fixture task from guessing which
implementation program owns a missing capability.

### Runtime Support Gate

The runtime support gate owns the rule that reference fixtures can only be
created from routes that currently return successful `SurfaceBody` results.

It must report:

- operation
- source reference item
- builder function
- expected fixture id
- result status
- failure reason when unsupported
- no-hidden-mesh-fallback evidence

### Reference Fixture Completion Gate

The completion gate owns the final readiness check for each reference item:

- deterministic model generator
- dirty STL path
- review fixture record
- purpose, methodology, and render description
- source entrypoint
- non-empty STL signal
- review-app loadability

### Gap Backlog

The gap backlog records reference items that cannot produce honest STL fixtures
yet. It links each item to architecture and, later, final implementation specs.

## Data Flow

```text
reference-test progression item
-> runtime support probe
-> architecture owner lookup
-> implementation/spec readiness decision
-> fixture generation only after successful SurfaceBody route
-> dirty STL and review fixture record
```

## Cross-Domain Decisions

### Dirty STL Is Not Proof Of Feature Completion

Dirty reference artifacts prove that a generator produced output. They do not
prove that the output is geometrically correct or reviewed. Promotion remains a
human review step.

### Unsupported Routes Stay Architectural Until Executable

An unsupported route should not get a placeholder STL. It should get one of:

- architecture coverage
- final implementation spec coverage
- explicit refusal fixture coverage when the intended behavior is refusal

### Reference Coverage Tracks Kernel Truth

The reference fixture system follows the modeling kernel. It must not introduce
fixture-specific geometry shortcuts to make a planned visual case appear ready.

## Specification Manifest for Discovery

### Candidate Spec: Reference CSG Gap Audit And Fixture Gate

Discovery purpose:
- Define the reusable audit that maps unchecked reference-test items to runtime
  support, architecture ownership, and fixture-generation readiness.

Responsibilities:
- Functions/methods:
  - reference CSG gap collector
  - runtime support probe runner
  - fixture readiness reporter
- Data structures/models:
  - reference gap record
  - runtime support result
  - fixture readiness record
- Dependencies/services:
  - reference-test expansion plan
  - CSG/loft public modeling API
  - review fixture registry
- Returns/outputs/signals:
  - ready-for-fixture list
  - unsupported implementation gap list
  - stale or missing artifact diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reference fixture registry and STL signal helpers
  - Additions to existing reusable library/module: test-side reference audit helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may generate dirty reference artifacts when explicitly requested
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded to selected reference cases and avoids broad artifact regeneration
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/test_reference_stl_expansion.py`
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - unsupported routes are reported and do not create STL artifacts
- Test strategy:
  - unit test for gap classification; focused CSG reference fixture smoke
- Data ownership:
  - progression owns desired cases; runtime probe owns support truth; fixture
    JSON owns review records
- Routes:
  - progression item to runtime probe to fixture registry
- Reuse/extraction decision:
  - add to existing test/reference helper modules
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- The audit should preserve human-readable reference IDs while also recording
  concrete fixture IDs and source entrypoints.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
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
- Total: 21.5

Readiness blockers:
- None.

Split decision:
- Review for split. Cohesion reason: the audit/gate is one reusable reference
  workflow that binds progression, runtime support, and fixture readiness.

## Change History

- 2026-07-11: Completed five manifest review/update/rescore rounds. Context:
  the single gap-audit candidate stayed below the split threshold and retained
  its existing cohesive scope.
- 2026-07-11: Added gap-closure architecture for unchecked reference-test
  progression items. Context: the reference suite had new CSG artifacts but
  still had unchecked primitive-contact, patch-family, loft, and lofted-CSG
  groups.
