# Loft Primitive Cut-Shell Geometric Kernel Umbrella ACD

Date: 2026-07-16
Status: Manifesting
Canonical architecture targets:

- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/surface-csg-trim-fragment-reconstruction-architecture.md`
- `project/release-0.1.0a/architecture/surfacebody-csg-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Parent ACD:
  `project/release-0.1.0a/architecture/acd-loft-primitive-trim-fragment-execution-adapter.md`
- Triggering spec:
  `project/release-0.1.0a/specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`

## Change Intent

Coordinate the architectural split required before cut-producing
loft/primitive CSG can be implemented honestly.

This ACD originally tried to describe the whole missing kernel in one document.
The deep review found that shape was too broad: source normalization,
patch-local loop construction, generated primitive caps, topology selection,
seam/shell assembly, runtime validity, public route wiring, and reference
handoff are distinct architecture concerns. They should be owned by separate
ACDs so implementation specs do not become catch-all discovery documents.

## Current Architecture

The current implementation has:

- exact no-cut/containment loft/primitive execution
- loft/primitive trim-adapter and fragment classification records
- branch graph, policy, decomposition, provenance, color, and fixture evidence
  records
- adapter-only refusal for intersecting loft/primitive cut cases

The current architecture does not yet define a complete cut-shell geometric
kernel. Surface Spec 422 therefore remains blocked.

## Target Architecture

The target architecture is split across child ACDs:

- [ACD: Loft Primitive Intersection And Cut-Loop Kernel](acd-loft-primitive-intersection-and-cut-loop-kernel.md)
  - Owns primitive/loft source normalization, patch-local curve inversion,
    clipping, arrangement, loop closure, and loop diagnostics.
- [ACD: Loft Primitive Generated Cap And Topology Policy](acd-loft-primitive-generated-cap-and-topology-policy.md)
  - Owns generated primitive caps, supported/unsupported cap policy,
    operation-specific fragment selection, cavity/exterior/multi-shell topology,
    and orientation policy.
- [ACD: Loft Primitive Seam Shell Validity Execution](acd-loft-primitive-seam-shell-validity-execution.md)
  - Owns seam/use pairing, shell assembly, runtime validity, persistence and
    no-hidden-mesh gates, and public `surface_boolean_result` wiring.
- [ACD: Loft CSG Reference Geometry Handoff](acd-loft-csg-reference-geometry-handoff.md)
  - Owns the workflow handoff from accepted public CSG result geometry to
    reference STL proof and section evidence readiness.

Implementation specs should be derived from the child ACDs, not from this
umbrella document.

## Non-Goals

- This umbrella does not define implementation specs directly.
- This umbrella does not close Surface Spec 422 by itself.
- This umbrella does not replace canonical architecture until child ACDs
  conform and close.

## Canonical Document Impact

- Architecture docs to update on closure:
  - `lofted-body-csg-reference-architecture.md` - replace the broad
    "loft primitive route" gap with the conformed staged cut-shell pipeline.
  - `surface-csg-trim-fragment-reconstruction-architecture.md` - include
    loft-specific source, loop, cap, topology, seam, and validity boundaries.
  - `surfacebody-csg-architecture.md` - document cut-producing loft/primitive
    execution as a surface-native public Boolean route.
- Specs or plans affected:
  - Surface Spec 422 - must be superseded by or 100% covered by child specs.
  - Surface Spec 407 - waits for accepted public result geometry.
  - Surface Spec 418 - waits for result geometry handoff and section readiness.

## Readiness Blocker Resolution

- Blocker being resolved:
  - Surface Spec 422 named cut-shell assembly but lacked architectural
    decomposition for the required geometric kernel.
- Source artifact:
  - `project/release-0.1.0a/specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md`
- Resolution provided by this ACD:
  - Split the broad blocker into child ACDs with explicit architecture
    ownership.
- Follow-on artifact:
  - Final specs derived from the child ACD manifests.
- Resolution status:
  - resolved at umbrella routing level; child ACDs must be reviewed and
    promoted before implementation.

## Compatibility And Migration Strategy

- Surface Specs 420 and 421 remain valid.
- Intersecting loft/primitive cases keep returning adapter-only structured
  refusal until child ACD-derived specs implement the full kernel.
- Existing primitive, B-spline/NURBS, sweep/subdivision, ruled-affine, and
  sampled/implicit CSG routes retain dispatch precedence.
- No child ACD may introduce tessellation, rasterization, or mesh execution as
  a hidden fallback.

## Application Integration Contract

- App type: library-only with downstream workflow consumers
- User/caller surface: public Boolean API consumers and reference generation
  workflows
- Invocation route: `surface_boolean_result` to loft route selection to the
  child cut-shell pipeline, then reference workflow handoff after accepted
  result geometry exists
- Wiring owner/module: `src/impression/modeling/csg.py`,
  `tests/reference_review_fixtures/stl_review_sources.py`,
  `tests/reference_images.py`
- Observable result: accepted cut cases return valid `SurfaceBody` results;
  unsupported cases return structured diagnostics; reference workflows consume
  only accepted result geometry
- Integration validation: public Boolean route tests plus reference workflow
  proof tests from child specs

## Specification Manifest for Discovery

No umbrella implementation manifest is promoted from this document.

Child ACDs own the manifest candidates that can become final implementation
specs. This keeps architecture boundaries explicit and prevents a single giant
specification manifest from hiding missing prerequisites.

## Specification Conformance

- Parent specs created or affected:
  - `project/release-0.1.0a/specifications/surface-422-loft-primitive-cut-shell-assembly-and-validity-v1_0.md` - blocked parent spec; must be superseded or fully covered by child specs.
- Canonical child specs:
  - pending child ACD promotion.
- Paired test specs:
  - pending child ACD promotion.

## Conformance Checklist

- [ ] Child ACDs exist for source/loop, cap/topology, seam/shell/validity, and
  reference handoff architecture.
- [ ] Child ACD manifests are reviewed, rescored, and split until no unresolved
  readiness blockers remain.
- [ ] Surface Spec 422 is superseded or fully covered by child specs.
- [ ] Progression links the final child specs in prerequisite order.
- [ ] Public API tests prove accepted cut results and structured refusals.
- [ ] Surface Specs 407 and 418 consume accepted public result geometry only.
- [ ] Canonical architecture documents are updated after implementation
  conforms.

## Closure Criteria

- All child ACDs are either closed or replaced by canonical architecture.
- Surface Spec 422 no longer blocks progression as a broad under-specified
  parent.
- Cut-producing loft/primitive CSG has route-level public API proof.
- Reference geometry proof and section evidence generation no longer rely on
  adapter-only payloads.

## Closure Notes

- Canonical architecture updated:
  - pending
- Archived or removed scaffolding:
  - pending
- Follow-up ACDs:
  - child ACDs listed above

## Change History

- 2026-07-16 - Initial draft. Reason: Surface Spec 422 could not truthfully
  produce cut-shell results from Surface Spec 421 adapter records alone.
- 2026-07-16 - Split into umbrella plus child ACDs. Reason: deep review found
  the original ACD was too broad and was hiding multiple architecture
  responsibilities inside one giant manifest.
