# Surface Body Completion Architecture

## Overview

This document defines the remaining architecture needed for Impression to make
surface bodies feel complete as an authored modeling kernel.

The target is not merely "no mesh fallback." The target is:

- all surface patch families are first-class authored data
- surface operations either execute natively or return precise diagnostics
- authored topology rails drive correspondence
- ambiguous authored lofts report every ambiguity and refuse execution until
  resolved
- `.impress` persists complete surface truth
- mesh appears only at the tessellation, preview, export, analysis, or explicit
  compatibility boundary

## Related Architecture

This document coordinates and supersedes the completion posture in:

- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [Patch Family Integration Architecture](patch-family-integration-architecture.md)
- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)
- [Loft Evolution System Architecture](loft-evolution-system.md)
- [Loft Ambiguity and Diagnostics Architecture](loft-ambiguity-and-diagnostics.md)
- [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)

## Ownership Boundaries

### Impression Owns

- surface-body storage and topology
- patch family records, evaluation, tessellation adapters, and `.impress`
  payloads
- surface-native CSG over `SurfaceBody`
- loft planning, diagnostics, and surface execution
- primitive and feature-builder integration boundaries that produce
  `SurfaceBody` or `SurfaceConsumerCollection`
- no-hidden-mesh-fallback enforcement

## Completion Streams

### 1. Patch Family Promotion

The current capability matrix distinguishes `available` and `planned` families.
Completion requires every family intended for authored modeling to pass a
family-specific promotion gate.

Each family needs:

- canonical in-memory patch record
- parameter domain and boundary model
- evaluation and derivative contract where meaningful
- trim and seam participation
- tessellation adapter with lossiness metadata when approximate
- `.impress` codec and round-trip identity tests
- diagnostic behavior for unsupported operations
- CSG and loft eligibility classification

The promotion target includes:

- planar
- ruled
- revolution
- B-spline
- NURBS
- sweep
- subdivision
- implicit
- heightmap
- displacement

### 2. Surface CSG Completion

Surface CSG must move from a bounded planar/box slice to a broad surface-body
boolean system.

The completed system needs:

- exact or declared-tolerance intersection records for supported family pairs
- operation-specific trim graph construction
- fragment classification that works on trimmed and curved surfaces
- shell reconstruction for multi-shell and nested-loop results
- cap construction for cut regions
- seam and adjacency rebuild for all result shells
- validity, healing, provenance, and diagnostics at the result boundary
- explicit refusal records for family pairs that genuinely remain outside the
  kernel scope

The priority order should be:

1. analytic planar, ruled, and revolution pair coverage
2. box/cylinder/sphere/cone/torus primitive boolean coverage
3. trimmed-surface fragment graph coverage
4. B-spline/NURBS curve-surface and surface-surface intersection boundary
5. subdivision, implicit, heightmap, and displacement operation policy

### 3. Loft Completion

Impression targets authored topologies.

That means the planner should rely on user-authored rails, named entities,
anchors, path order, lifecycle records, and topology paths before attempting
automatic ambiguity resolution.

Automatic resolution is allowed only when it is deterministic, high confidence,
and produces a better user experience without hiding ambiguity. It is not a
requirement for this version.

The core execution rule is:

- planning continues after ambiguity is discovered so all problems can be
  reported together
- each ambiguity is recorded with exact topology, station, interval, entity,
  relationship-group, and candidate-lifecycle locators
- any unresolved ambiguity makes the plan non-executable
- the executor must refuse a plan that carries unresolved ambiguity records

The planner should distinguish:

- invalid authored input
- missing rails or anchors
- ambiguous branch/split/merge correspondence
- point birth/death ambiguity
- loop or region containment ambiguity
- incompatible lifecycle declarations
- unsupported topology transitions

### 4. Seam, Continuity, And Topology Validity

Surface-body completion requires more than storing patches.

The topology layer needs:

- explicit boundary-use records for every shell
- seam participation validation across every promoted family
- continuity request records beyond the current C0/G0 baseline
- diagnostics for unsupported continuity classes
- watertightness, open-shell, non-manifold, and duplicate-boundary gates
- transform-stable identity and adjacency behavior

Higher continuity support should be promoted deliberately. Unsupported G1/G2 or
curvature-continuity requests must remain structured diagnostics, not silent
downgrades.

### 5. `.impress` Surface-Native Persistence

The `.impress` format is complete only when it can persist the whole authored
surface-body store.

The format must cover:

- multi-body document roots
- units and coordinate policy
- every promoted patch family
- trims, seams, adjacency, shell metadata, and stable identity
- topology rails and authored-lifecycle records when they affect execution
- operation provenance and diagnostic metadata
- refusal of malformed or unsupported payloads without mesh recovery

Mesh payloads may exist only as explicit import/export/cache artifacts. They
are not canonical surface truth.

### 6. Primitive And Feature Integration

Primitive and feature builders should select the appropriate patch family for
the authored geometry they emit.

The completion target includes:

- primitives using analytic families where exact
- loft using B-spline, NURBS, or sweep families when requested and validated
- feature builders producing `SurfaceBody` or `SurfaceConsumerCollection`
- no feature path producing mesh as hidden substitute geometry
- generic external-feature integration for sibling projects

### 7. Verification And Completion Evidence

Every promoted family or operation needs evidence:

- focused unit tests for records and diagnostics
- round-trip `.impress` tests
- tessellation-boundary tests
- no-hidden-mesh-fallback tests
- reference images/STLs for model-outputting capabilities
- negative tests that prove unsupported states refuse with exact diagnostics

Completion claims should cite verified capability matrices, not checklist
completion alone.

## Work Register

The architecture work tracker owns the durable to-do list for this completion
program:

- [Architecture Work Tracker](architecture-work-tracker.md)

No downstream specification manifest should be considered complete until the
tracker entries for this document are either specified, implemented, or
explicitly retired by a later architecture decision.

## Change History

- 2026-05-27: Created the surface-body completion umbrella architecture after
  the release progression was fully checked but the capability audit still
  found planned patch families, bounded CSG support, loft ambiguity execution
  boundaries, seam continuity limits, and `.impress` promotion work.
