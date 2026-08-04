# Loft Identity And Junction Correctness Architectural Change Document

Date: 2026-08-04
Status: Drafting Specs
Canonical architecture targets:

- `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md`

Related:

- Release / plan / issue: `project/release-1.0.0a4/README.md`; GitHub #244, #245, #246
- Parent ACD, if any: none

## Change Intent

Extend authored identity-first loft planning from stable region sets to named
holes, count-changing region sets, and synthetic planning stations, then execute
hole split/merge transitions as surface junctions rather than interior caps.

## Current Architecture

Named `TopologyPath` values survive in section metadata, but planner-native
`PlannedLoopRef` values carry only actual/synthetic kind and index. Equal-count
holes therefore use geometric assignment. `_expand_split_merge_stations(...)`
reconstructs anonymous sections and calls transition pairing without caller
configuration, losing identities and resetting `ambiguity_max_branches` to 64.
Hole births/deaths become shrunken synthetic loops plus planar closure caps,
which adds a third cap and invalid orientation in the published split/merge
example.

## Target Architecture

- Normalized region and loop records retain stable topology path identity.
- Identity resolution runs before geometric assignment at both region and hole
  levels; geometric search sees only unnamed residue.
- Synthetic station records retain the source/target identity lineage and every
  planner configuration value supplied by the caller.
- Birth/death planning emits an explicit junction operator record. The surface
  executor builds a continuous branch/junction patch and does not cap an
  interior hole transition.
- `cap_ends=True` creates exactly the requested terminal caps. Cap validity
  distinguishes terminal caps from junction surfaces.
- Duplicate and contradictory IDs fail before ambiguity enumeration with stable
  diagnostics.

## Non-Goals

- Probabilistic identity inference, arbitrary self-intersecting topology, or
  redesign of point-level correspondence tracks already owned by canonical
  architecture.

## Canonical Document Impact

- Architecture docs to update on closure:
  - loft topology point correspondence architecture - add loop identity,
    synthetic lineage, and junction execution ownership.
- Specs or plans affected:
  - Fixes 03 through 06 and their paired tests.

## Readiness Blocker Resolution

- Blocker being resolved: issues did not define whether names belonged only to
  metadata, how synthetic stations inherit them, or who replaces closure caps.
- Source artifact: GitHub #244-#246 and the audio-cube reproductions.
- Resolution provided by this ACD: planner-owned loop identity, synthetic
  lineage/configuration, and executor-owned junction patches.
- Follow-on artifact: Fixes 03-06.
- Resolution status: resolved.

## Compatibility And Migration Strategy

Unnamed models retain deterministic geometric matching. Named models become
more authoritative: matching path IDs override proximity. Existing public
planner options retain names and defaults; propagation changes only eliminate
internal resets.

## Application Integration Contract

- App type: library-only.
- User/caller surface: `Loft(...)`, `loft_plan_sections(...)`, and
  `loft_plan_ambiguities(...)`.
- Invocation route: authored sections/stations -> identity normalization ->
  expanded plan -> junction-aware surface executor.
- Wiring owner/module: `src/impression/modeling/loft.py`.
- Observable result: deterministic plan metadata and closed `SurfaceBody`
  output with exact terminal cap count.
- Integration validation: public planner/executor tests plus the published
  split/merge example and audio-cube rail-pair reproduction.

## Specification Sources

- Fix 03: named hole identity preservation and assignment.
- Fix 04: hole split/merge junction planning and surface execution.
- Fix 05: count-changing region identity lineage through synthetic stations.
- Fix 06: propagation of all caller planner configuration through expansion.
- Reuse existing `Station`, `TopologyPath`, planned refs, ambiguity diagnostics,
  surface patch builders, seam graph, and closure evidence.

## Specification Conformance

- Parent specs created or affected:
  - [Fix 03 draft](../specifications/fix-03-named-hole-identity-pairing-v1_0.md) - named-hole identity pairing.
  - [Fix 04 draft](../specifications/fix-04-hole-split-merge-junction-surfaces-v1_0.md) - junction surface execution.
  - [Fix 05 draft](../specifications/fix-05-count-changing-region-identity-preservation-v1_0.md) - synthetic lineage.
  - [Fix 06 draft](../specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md) - planner configuration propagation.
- Canonical child specs: none yet; independent review is pending.
- Paired test specs:
  - [Fix 03 test draft](../test-specifications/fix-03-named-hole-identity-pairing-v1_0.md)
  - [Fix 04 test draft](../test-specifications/fix-04-hole-split-merge-junction-surfaces-v1_0.md)
  - [Fix 05 test draft](../test-specifications/fix-05-count-changing-region-identity-preservation-v1_0.md)
  - [Fix 06 test draft](../test-specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md)

## Conformance Checklist

- [ ] Implementation conforms to the target architecture.
- [ ] Draft leaves are independently reviewed and canonicalized.
- [ ] Paired test specs point to canonical leaves.
- [ ] Final progression preserves prerequisite order.
- [ ] Canonical loft architecture is reconciled after implementation.

## Closure Criteria

Close after named hole swaps, count-changing named regions, non-default branch
limits, both hole-transition directions, and project-scale rail geometry pass
through public routes with closed-valid surface output.

## Closure Notes

- Canonical architecture updated: none yet.
- Archived or removed scaffolding: none.
- Follow-up ACDs: none.

## Change History

- 2026-08-04 - Linked the full-template Fix 03-06 and paired test drafts. Reason: complete the `do specs` creation handoff.
- 2026-08-04 - Initial draft. Reason: plan issues #244-#246 for `v1.0.0a4`.
