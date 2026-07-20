# CSG Coincident Contact Architecture

## Overview

This document defines the missing architecture for `RT-CSG-009`: coincident or
face-touching primitive CSG.

The reference plan asks for "coincident-face box union and difference." The
difference case is already represented by a difference fixture where the cutter
shares a boundary face with the base and the result remains the base. The
missing successful fixture is face-touch union: two closed solids share a
boundary face but have no overlapping volume.

The architecture goal is to support degenerate-contact CSG without slivers,
duplicate coincident faces, invalid seams, or hidden mesh fallback.

## Related Architecture

This document extends:

- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [Surface CSG Executable Completion Architecture](surface-csg-executable-completion-architecture.md)
- [Surface CSG Trim Fragment Reconstruction Architecture](surface-csg-trim-fragment-reconstruction-architecture.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [Reference CSG Gap Closure Architecture](reference-csg-gap-closure-architecture.md)

## Components

### Contact Classifier

The contact classifier distinguishes:

- disjoint with positive gap
- tangent point contact
- tangent curve contact
- coincident face contact
- positive-volume overlap
- containment

This classification must occur before the general fragment solver treats the
case as an ordinary crossing intersection.

### Coincident Region Resolver

The resolver owns surface regions that overlap exactly or within the declared
coincidence tolerance.

It decides whether the coincident region should:

- merge as an internal face removed from the union result
- remain as an exposed face in a difference result
- become a diagnostic refusal for ambiguous or non-manifold contact

### Union Seam Merger

The union merger owns topology reconstruction for face-touching solids.

It must:

- remove duplicate internal coincident faces
- create one closed outer shell
- rebuild seams around the merged boundary
- preserve deterministic patch ordering
- reject non-manifold edge-only or vertex-only unions unless explicitly
  supported as multi-shell contact output

### Diagnostic Refusal Gate

The gate owns cases that look coincident but cannot produce valid manifold
output:

- partial coplanar overlap without closed region resolution
- edge-only contact when the result would be non-manifold
- tolerance-ambiguous near contact
- conflicting face normals or inverted source shells

## Data Flow

```text
SurfaceBody operands
-> contact classifier
-> coincident region resolver
-> operation-specific topology policy
-> seam merger / exposed-boundary policy
-> validity gate
-> SurfaceBody result or structured contact diagnostic
```

## Cross-Domain Decisions

### Coincident Union Is Not Disjoint Union

Face-touch union should not return two independent shells. It represents a
single connected volume when the shared face covers a valid closed interface.

### Contact Tolerance Must Be Explicit

Near-touching operands must not randomly choose union, disjoint, or overlap
behavior. The tolerance policy belongs to the CSG kernel and must be recorded in
diagnostics.

### Non-Manifold Contact Should Refuse Clearly

Point-only and edge-only contact can produce non-manifold results. Unless a
future architecture explicitly supports them, these routes should refuse with a
contact diagnostic rather than producing invalid geometry.

## Specification Manifest for Discovery

### Candidate Spec: Coincident Contact Classifier

Discovery purpose:
- Classify primitive CSG contacts before general fragment reconstruction so
  face-touch, tangent, near-touch, overlap, disjoint, and containment cases do
  not share one brittle path.

Responsibilities:
- Functions/methods:
  - contact classifier
  - coincidence tolerance evaluator
  - contact diagnostic builder
- Data structures/models:
  - contact classification record
  - coincidence tolerance record
  - contact diagnostic
- Dependencies/services:
  - operand bounds
  - patch-local face data
  - shell validity records
- Returns/outputs/signals:
  - contact classification
  - tolerance diagnostic
  - execution eligibility signal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: surface CSG operand preparation
  - Additions to existing reusable library/module: `src/impression/modeling/csg.py`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes boolean execution classification
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded patch-pair contact checks
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - use existing CSG tolerance policy; near-contact ambiguity refuses
- Test strategy:
  - point, edge, face, near-touch, disjoint, overlap, and containment tests
- Data ownership:
  - CSG owns contact classification records and diagnostics
- Routes:
  - operand preparation to contact classifier to operation planner
- Reuse/extraction decision:
  - add to existing CSG module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Edge-only and point-only union may be intentionally non-manifold; current
  architecture treats them as refusal unless later product intent changes.

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
- Review for split. Cohesion reason: all listed responsibilities define the
  single contact-classification boundary needed before operation-specific
  topology handling.

### Candidate Spec: Face-Touch Union Shell Merger

Discovery purpose:
- Define the operation-specific reconstruction path for union of solids that
  share a full coincident face.

Responsibilities:
- Functions/methods:
  - coincident face union resolver
  - duplicate internal face remover
  - merged seam rebuilder
- Data structures/models:
  - coincident region record
  - removed-face provenance record
  - merged-shell diagnostic
- Dependencies/services:
  - contact classification record
  - result shell assembler
  - seam adjacency builder
- Returns/outputs/signals:
  - merged SurfaceBody
  - internal-face removal provenance
  - non-manifold refusal diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: result shell assembly and seam records
  - Additions to existing reusable library/module: CSG shell reconstruction
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes union results for face-touching operands
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - linear in coincident face boundary count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - full-face contact merges into one shell; non-manifold partial contact refuses
- Test strategy:
  - `RT-CSG-009` reference test plus unit tests for adjacent, identical,
    near-touching, and partial-overlap boxes
- Data ownership:
  - result body owns final shell truth; provenance map owns removed-face trace
- Routes:
  - contact classifier to union resolver to result validity gate
- Reuse/extraction decision:
  - add to existing CSG shell reconstruction path
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Identical-box union and contained-box union already have shortcut behavior;
  this spec must avoid regressing those cases.

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
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Readiness blockers:
- [ ] Result shell assembly must expose enough seam-rebuild hooks for full-face
  internal face removal.

Split decision:
- Review for split. Cohesion reason: this remains one operation-specific
  topology behavior for the missing `RT-CSG-009` union fixture.

## Change History

- 2026-07-11: Completed five manifest review/update/rescore rounds. Context:
  coincident contact candidates remained below the split threshold after review.
- 2026-07-11: Added coincident-contact architecture for `RT-CSG-009`.
  Context: coincident-face difference is covered, but face-touch union still
  needs contact classification and shell merger behavior.
