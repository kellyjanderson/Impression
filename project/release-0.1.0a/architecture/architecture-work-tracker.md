# Architecture Work Tracker

## Overview

This document tracks architecture-defined work that is not yet complete.

It exists so architecture can stay useful after documents are written. When an
architecture document creates new implementation or specification obligations,
the obligation should appear here until it is either:

- converted into final specifications and paired test specifications
- implemented and verified
- retired by a later architecture decision

This tracker is intentionally architecture-facing. It is not a replacement for
release progression, implementation checklists, or GitHub issues.

## Audit Date

2026-05-27

## Audit Scope

Audited architecture documents under:

- `project/release-0.1.0a/architecture/`

Cross-checked downstream specification coverage against:

- `project/release-0.1.0a/specifications/`

The audit looked for:

- specification manifests that do not yet have matching final specs
- newly-added architecture that supersedes older specs
- explicit deferred, unsupported, or not-yet-implemented architecture
- migration plans that still need specification or implementation follow-through
- architecture that still allows mesh execution outside the tessellation
  boundary

## Status Legend

- `Manifest Assessed`: architecture has a template-assessed specification
  manifest in its owning architecture document; final specs may still need
  promotion.
- `Needs Spec Revision`: specs exist, but the architecture changed enough that
  one or more specs must be revised, retired, or replaced.
- `Specified, Needs Implementation`: final specs appear to exist; remaining work
  is implementation and verification.
- `Monitor`: architecture has downstream specs or process coverage; keep visible
  but it is not a current manifest blocker.

## Manifest Coverage Review

- [x] Added the missing manifest to
  [Surface Body Completion Architecture](surface-body-completion-architecture.md),
  the new umbrella document that owns the newly discovered completion work.
- [x] Reviewed existing manifests in
  [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md),
  [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md),
  [Loft Topology Point Correspondence Architecture](loft-topology-point-correspondence-architecture.md),
  [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md),
  and [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md).
- [x] Confirmed the reviewed manifests have no remaining candidate scored
  `25+` after the current split pass.
- [x] Left index, tracker, and already-covered sibling architecture documents
  without duplicate manifests where their new work is owned by the umbrella or
  component manifest listed above.

## Priority 0: Surface Body Completion Architecture Needed

These items capture the new surface-body completion standard. They supersede
the assumption that a checked progression means the surface-body kernel is
complete.

### Surface Body Completion Program

- Status: `Manifest Assessed`
- Architecture: [Surface Body Completion Architecture](surface-body-completion-architecture.md)
- [x] Specification manifest added, critically reviewed, rescored, and split.

Architecture-defined work:

- [ ] promote every authored patch family from `planned` to verified `available`
  where it belongs in the surface-body kernel
- [ ] make B-spline, NURBS, sweep, subdivision, implicit, heightmap, and
  displacement first-class surface-body citizens across storage, evaluation,
  seams, tessellation, `.impress`, and diagnostics
- [ ] expand surface CSG from bounded planar/box execution to broad analytic and
  higher-order surface-body boolean coverage
- [ ] build exact or declared-tolerance CSG intersection, trim graph, fragment
  classification, shell assembly, cap construction, seam rebuild, validity, and
  provenance for supported family pairs
- [ ] define remaining CSG solver boundaries as explicit refusal records, not as
  mesh fallback or vague "not implemented" strings
- [ ] make authored topology rails the primary loft correspondence mechanism
- [ ] continue loft planning after ambiguities are found so all ambiguity and
  invalid-input records are reported together
- [ ] refuse loft execution for any plan carrying unresolved ambiguity records
- [ ] provide exact ambiguity locators: topology, station, interval, entity,
  relationship group, candidate lifecycle, and suggested authored rail
- [ ] promote seam and continuity support beyond the current C0/G0 baseline with
  structured diagnostics for unsupported continuity classes
- [ ] complete `.impress` persistence for the full authored surface-body store,
  including all promoted patch families, trims, seams, adjacency, metadata,
  identities, topology rails, and operation provenance
- [ ] ensure primitives and Impression-owned feature builders select appropriate
  patch families and return surface truth
- [ ] keep mesh output only at tessellation, preview, export, analysis, or explicit
  compatibility boundaries
- [ ] add reference artifact and round-trip evidence for every promoted
  model-outputting capability

Open architecture decisions:

- [ ] what exact threshold promotes a family from `planned` to `available`
- [ ] which CSG family pairs must be exact for the completion claim and which may
  be declared-tolerance or explicitly unsupported
- [ ] whether implicit, heightmap, and displacement booleans are kernel operations,
  bounded adapters, or explicit non-CSG families
- [ ] how `.impress` versions family payloads independently without fragmenting the
  document root
- [ ] how much automatic loft ambiguity resolution is permitted when authored rails
  are missing but the deterministic answer is obvious

Manifest notes:

- [x] Manifest added to the owning architecture document.
- [x] Broad completion work split into patch-family promotion, CSG completion,
  loft ambiguity gates, seam/continuity promotion, `.impress` whole-store
  gates, primitive/feature integration, and verification evidence.
- [x] Manifest reviewed through repeated rescore/split passes with no remaining
  candidate at `25+`.

### Authored Loft Ambiguity Contract Refresh

- Status: `Needs Spec Revision`
- Architecture:
  - [Surface Body Completion Architecture](surface-body-completion-architecture.md)
  - [Loft Ambiguity and Diagnostics Architecture](loft-ambiguity-and-diagnostics.md)
  - [Loft Plan Object Architecture](loft-plan-object-architecture.md)
  - [Loft Topology Point Correspondence Architecture](loft-topology-point-correspondence-architecture.md)

Issue:

- The intended product contract is authored topology first.
- Automatic ambiguity resolution is useful but not required for this version.
- A plan should continue collecting all ambiguity records, but unresolved
  ambiguity must make execution impossible.

Required work:

- [ ] ensure loft architecture consistently states authored rails and named
  topology entities are primary
- [ ] ensure ambiguity records include exact "what and where" locators
- [ ] ensure planners accumulate all ambiguities before returning
- [ ] ensure executors refuse unresolved ambiguity records
- [ ] remove any implication that automatic ambiguity resolution is required for
  completion

## Priority 1: Manifest Promoted To Final Specs

These items have assessed manifests in their owning architecture documents and
now have one final specification document for each manifest candidate.

### .impress Surface-Native File Format

- Status: `Manifest Promoted`
- [x] Final specifications created for every manifest candidate.
- Architecture: [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)
- Current downstream specs found: none dedicated to `.impress`

Architecture-defined work:

- [ ] `.impress` document root and `SurfaceBodyStore` record
- [ ] surface payload encoder for bodies, shells, patches, trims, seams, and
  adjacency
- [ ] surface payload decoder and constructor validation
- [ ] deterministic JSON writer and reader API
- [ ] round-trip identity and metadata preservation tests
- [ ] invalid-file refusal diagnostics
- [ ] industry interchange adapter boundary that explicitly excludes STEP/IGES from
  V1 unless separately planned

Open architecture decisions that must be resolved before final specs:

- [ ] single-body versus multi-body default file semantics
- [ ] whether stored stable identities are required or optional
- [ ] document units policy
- [ ] whether topology rails belong in V1 `.impress` or in a later modeling document
  layer

Manifest notes:

- This should become a coherent `.impress` spec branch, not a single oversized
  spec.
- The file format must inherit the mesh-boundary rule: tessellated mesh may be a
  cache or imported mesh object, not canonical surface truth.

### Full Surface Patch Family Program

- Status: `Manifest Promoted`
- [x] Final specifications created for every manifest candidate.
- Architecture: [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- Current downstream specs found: older surface patch specs exist, but they
  include [Surface Spec 66: Deferred Patch Families and Explicit Exclusions](../specifications/surface-66-deferred-patch-families-exclusions-v1_0.md)
  which the new architecture supersedes.

Architecture-defined work:

- [ ] full patch-family scope and implementation matrix
- [ ] B-spline surface patch record and evaluation
- [ ] NURBS surface patch record and rational evaluation
- [ ] sweep surface patch record and frame transport
- [ ] subdivision surface patch record and Catmull-Clark evaluation
- [ ] implicit surface patch record and declarative field model
- [ ] cross-family tessellation adapters
- [ ] cross-family seam and boundary participation
- [ ] `.impress` payload support for all patch families
- [ ] family-aware boolean eligibility and refusal diagnostics
- [ ] migration spec replacing deferred-family constants with a capability matrix

Open architecture decisions that must be resolved before final specs:

- [ ] whether B-spline and NURBS are separate classes or one shared rational-capable
  implementation over common basis infrastructure
- [ ] sweep payload shape: 2D topology, 3D curves, or both
- [ ] subdivision V1 limit evaluation versus finite-level deterministic evaluation
- [ ] allowed declarative implicit field nodes
- [ ] exact boolean support threshold for declaring a family complete
- [ ] whether `.impress` versions all family payloads together or independently

Manifest notes:

- This is the most important replacement for the old deferred-family posture.
- Spec 66 should be retired, replaced, or explicitly downgraded to historical
  context.
- The manifest should avoid bundling all families into one implementation spec.

### Mesh Execution To Tessellation Boundary

- Status: `Manifest Promoted`
- [x] Final specifications created for every manifest candidate.
- Architecture: [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)
- Current downstream specs found: older mesh decommission and surface-return
  specs exist, but they do not fully cover the new boundary audit or primitive
  excision details.

Architecture-defined work:

- [ ] mesh execution inventory and classification
- [ ] loft mesh executor boundary migration
- [ ] primitive surface defaults and primitive mesh path excision
- [ ] text and drafting surface defaults
- [ ] heightmap native surface representation
- [ ] hinge surface assemblies as canonical modeled output
- [ ] surface transform and composition layer replacing `MeshGroup` as authored
  composition
- [ ] mesh utility quarantine
- [ ] no-hidden-mesh-fallback enforcement

Primitive-specific work called out by architecture:

- [ ] `make_box` must default to `make_surface_box`
- [ ] `make_cylinder` must default to `make_surface_cylinder`
- [ ] `make_ngon` must default to `make_surface_ngon`
- [ ] `make_polyhedron` must default to `make_surface_polyhedron`
- [ ] `make_nhedron` must remain a surface-safe compatibility alias
- [ ] `make_sphere` must default to `make_surface_sphere`
- [ ] `make_torus` must default to `make_surface_torus`
- [ ] `make_cone` must default to `make_surface_cone`
- [ ] `make_prism` must default to `make_surface_prism`

Private primitive mesh helpers requiring deletion, quarantine, or tessellation
relocation:

- [ ] `_orient_mesh(...)`
- [ ] `_box_mesh(...)`
- [ ] `_sphere_mesh(...)`
- [ ] `_torus_mesh(...)`
- [ ] `_circular_frustum_mesh(...)`
- [ ] `_rectangular_frustum_mesh(...)`

Manifest notes:

- This branch should explicitly reconcile older specs such as Surface Specs
  44, 45, 50, 51, 57, 99-105, and Loft Spec 60.
- Loft Spec 60 is now suspect as canonical architecture because mesh executor
  correspondence consumption should move to legacy/debug/tessellation-boundary
  language.
- The first spec should probably be the inventory/classification spec so later
  leaf specs can cite exact symbols and owners.

## Priority 2: Spec Revision Or Reconciliation Promoted

These areas had specs, but the architecture changed enough that at least one
spec required revision, retirement, or replacement. The required reconciliation
work now has final specification coverage.

### Loft Legacy Mesh Debug Correspondence Consumption

- Status: `Spec Revision Promoted`
- [x] Replacement/reconciliation specification created.
- Architecture:
  - [Loft Topology Point Correspondence Architecture](loft-topology-point-correspondence-architecture.md)
  - [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)
- Current downstream spec:
  - [Loft Spec 60: Legacy Mesh Debug Correspondence Consumption](../specifications/loft-60-mesh-executor-correspondence-consumption-v1_0.md)

Issue:

- The point-correspondence manifest created a mesh executor consumption spec.
- The newer mesh-boundary architecture says mesh execution is not canonical and
  must move behind tessellation or into explicit legacy/debug tooling.

Required work:

- [ ] revise or replace Loft Spec 60
- [ ] make `LoftPlan -> SurfaceBody` the canonical executor contract
- [ ] ensure mesh face emission is tessellation/debug/compatibility only
- [ ] add tests proving surface loft does not fall back to mesh

### Deferred Patch Family Specs

- Status: `Spec Revision Promoted`
- [x] Replacement/reconciliation specifications created.
- Architecture: [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- Current downstream specs:
  - [Surface Spec 65: Required V1 Patch Families](../specifications/surface-65-required-v1-patch-families-v1_0.md)
  - [Surface Spec 66: Deferred Patch Families and Explicit Exclusions](../specifications/surface-66-deferred-patch-families-exclusions-v1_0.md)
  - [Surface Spec 67: Patch-Family to Feature Coverage Matrix](../specifications/surface-67-patch-family-feature-coverage-v1_0.md)

Issue:

- The new architecture rejects deferred patch families as the target posture.

Required work:

- [ ] replace deferred-family language with a capability and implementation matrix
- [ ] define every patch family as first-class, even when staged
- [ ] require explicit unsupported-operation diagnostics without treating the family
  as excluded

### Surface API Mesh Defaults

- Status: `Spec Revision Promoted`
- [x] Replacement/reconciliation specifications created.
- Architecture:
  - [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)
  - [Surface-Native Capability Replacement Architecture](surface-native-capability-replacement-architecture.md)
- Current downstream specs:
  - [Surface Spec 44: Primitive API Surface Return-Type Migration](../specifications/surface-44-primitive-api-surface-return-migration-v1_0.md)
  - [Surface Spec 45: Modeling Operation Surface Return-Type Migration](../specifications/surface-45-modeling-op-surface-return-migration-v1_0.md)
  - [Surface Spec 99: Surface-Native Replacement Program](../specifications/surface-99-surface-native-replacement-program-v1_0.md)
  - [Surface Specs 100-105](../specifications/README.md)

Issue:

- Older specs may describe migration, but the new architecture is stricter:
  new authored APIs should not use `backend="mesh"` defaults, and hidden mesh
  fallback is forbidden.

Required work:

- [ ] audit older specs for compatibility language that still allows mesh-primary
  authored modeling
- [ ] add explicit no-hidden-fallback acceptance criteria
- [ ] ensure primitive-by-primitive excision is represented in final specs

## Priority 3: Specified, Needs Implementation

These architecture branches already have downstream specs in the release tree.
They still represent uncompleted work unless implementation and verification
have landed.

### SurfaceBody CSG

- Status: `Specified, Needs Implementation`
- Architecture: [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- Current downstream specs:
  - Surface Specs 108-110
  - Surface Specs 117-120
  - Surface Specs 126-138

Remaining architecture work to implement:

- [ ] canonical surface boolean operand preparation
- [ ] intersection and classification
- [ ] operand fragment graph
- [ ] result topology reconstruction
- [ ] validity and bounded healing gate
- [ ] metadata and provenance propagation
- [ ] explicit unsupported result behavior without mesh fallback

### Surface-Native Capability Replacements

- Status: `Specified, Needs Implementation`
- Architecture: [Surface-Native Capability Replacement Architecture](surface-native-capability-replacement-architecture.md)
- Current downstream specs:
  - Surface Specs 99-105

Remaining architecture work to implement:

- [ ] surface-native drafting
- [ ] surface-native text
- [ ] surface-body booleans
- [ ] surface-native hinges
- [ ] surface-native heightfields and displacement

This area overlaps with the mesh-boundary branch and should be reconciled before
implementation sequencing is refreshed.

### Mesh Analysis And Repair Toolchain

- Status: `Specified, Needs Implementation`
- Architecture: [Mesh Analysis and Repair Architecture](surface-mesh-decommission-architecture.md)
- Current downstream specs:
  - Surface Specs 121-125

Remaining architecture work to implement:

- [ ] retained mesh capability matrix
- [ ] mesh analysis contract
- [ ] mesh repair contract
- [ ] standalone mesh utility contract
- [ ] clear separation between retained mesh tools and deleted mesh modeling paths

### SurfaceBody Seam And Adjacency

- Status: `Specified, Needs Implementation`
- Architecture: [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- Current downstream specs:
  - Surface Specs 26-28
  - Surface Specs 77-85

Remaining architecture work to implement:

- [ ] explicit seam identity
- [ ] patch-boundary use records
- [ ] shell adjacency view
- [ ] seam-first tessellation for watertight output
- [ ] transform and cache identity interactions

### Testing, Reference Artifacts, And Computer Vision Verification

- Status: `Specified, Needs Implementation`
- Architecture:
  - [Testing Architecture](testing-architecture.md)
  - [Model Output Reference Verification](model-output-reference-verification.md)
  - [Computer Vision Verification Architecture](computer-vision-verification-architecture.md)
- Current downstream specs:
  - Testing Specs 01-23
  - Surface Specs 106-107

Remaining architecture work to implement:

- [ ] reference artifact baseline lifecycle
- [ ] grouped model output completeness rules
- [ ] deterministic render products
- [ ] CV text, slice, object-view, handedness, and diagnostic lanes

### Next-Generation Loft Architecture

- Status: `Specified, Needs Implementation`
- Architecture:
  - [Loft Evolution System Architecture](loft-evolution-system.md)
  - [Loft Planner / Executor Architecture](loft-planner-executor-architecture.md)
  - [Loft Plan Object Architecture](loft-plan-object-architecture.md)
  - [Loft Ambiguity and Diagnostics Architecture](loft-ambiguity-and-diagnostics.md)
  - [Loft Tolerance and Degeneracy Architecture](loft-tolerance-and-degeneracy-architecture.md)
  - [Loft N->M / M->N Decomposition Architecture](loft-nm-mn-decomposition-architecture.md)
  - [Loft Topology Point Correspondence Architecture](loft-topology-point-correspondence-architecture.md)
- Current downstream specs:
  - Loft Specs 21-64
  - Surface Specs 95-98

Remaining architecture work to implement:

- [ ] placed topology state and directional correspondence
- [ ] evolution plan and transition operator family
- [ ] ambiguity and constraint request records
- [ ] tolerance and degeneracy policy
- [ ] many-to-many decomposition
- [ ] topology path, segment, landmark, and lifecycle records
- [ ] correspondence-preserving resampling
- [ ] generated shape default rails
- [ ] surface executor consumption

Reconciliation required:

- [ ] mesh executor consumption must be revised as noted in Priority 2.

## Monitor

These documents are important but did not produce new manifest or spec-promotion
blockers during this audit.

### Surface-First Internal Model

- Status: `Monitor`
- Architecture: [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- Current downstream specs:
  - Surface Specs 01-19

Notes:

- This is the parent architecture for much of the current release.
- No new manifest branch is needed unless later architecture changes the core
  `SurfaceBody` model again.

### Architecture Index

- Status: `Monitor`
- Architecture: [Architecture Index](README.md)

Notes:

- Keep links updated as new architecture work documents are added.

## Completed Spec Promotion Order

1. [x] Mesh Execution Inventory And Classification
2. [x] Mesh Execution To Tessellation Boundary leaf specs
3. [x] Full Surface Patch Family manifest and Spec 66 replacement
4. [x] `.impress` file format manifest

Reasoning:

- The mesh inventory gives immediate clarity about what current code paths must
  move or be quarantined.
- Patch-family completeness informs `.impress` payload shape.
- `.impress` should not be finalized before the patch-family payload policy is
  settled, but its document-root/store work can be drafted in parallel once the
  payload boundary is clear.

Spec-promotion note:

- 2026-05-26: Completed final specification promotion for the Priority 1
  manifests and Priority 2 reconciliation items. Remaining unchecked release
  work belongs in progression as implementation work, not in this architecture
  promotion tracker.

## Maintenance Rule

When a new architecture document is added or materially updated:

1. Add or update its entry here.
2. Mark whether it needs manifest work, spec revision, implementation, or only
   monitoring.
3. Link the relevant architecture and specs.
4. Move completed entries to a release-complete archive section when the release
   closes.

## Change History

- 2026-05-27: Marked the surface-body completion program as manifest assessed
  after adding and splitting the owning specification manifest.
- 2026-05-26: Moved assessed specification manifests out of this tracker and
  into the owning architecture documents. The tracker remains an index and
  status rollup.
- 2026-05-26: Added tracker and initial audit of release architecture documents
  to identify manifest/spec-promotion blockers, spec revisions, and
  implementation follow-up work before the next specification pass.
