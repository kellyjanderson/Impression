# Loft Identity And Junction Correctness Architectural Change Document

Date: 2026-08-04
Status: In Progress
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

## Count-Changing Exact Region Pairing Boundary

For every identity-bearing interval, the planner builds unique source and
target identity indexes in linear time and emits source-ordered exact pairs
before any geometric search. Only regions that are anonymous on both sides may
enter the bounded geometric assignment. Named source residue becomes explicit
death records and named target residue becomes explicit birth records; geometry
cannot repurpose either one as a continuation.

The public plan preserves one canonical transition-resolution payload per
identity-bearing interval: exact pairs, anonymous geometric pairs, unnamed
candidate indexes, births, and deaths. This allows a net expanding interval to
contain both named deaths and births without losing authored identity. Fix 05B
owns propagation of those identities through later synthetic stations.

## Synthetic Station Identity Lineage Boundary

Split/merge expansion consumes the exact transition resolution before it emits
staged geometry. Every inserted station carries an immutable lineage record for
each region and loop: a direction-independent identity, predecessor and
successor references, directional region IDs, loop endpoint IDs, and a complete
set of synthetic `TopologyPath` records. Authored identities remain authoritative;
anonymous geometry receives deterministic planner-local lineage without being
promoted into authored exact correspondence.

The staged `Station` records expose authored directional IDs on both sides of
each inserted station, so the first, intermediate, and last expanded intervals
all resolve without anonymous rebuilding. Reversing the transition swaps source
and target refs while preserving region and loop identity. The plan and surfaced
`Loft(...)` result publish both immutable records and canonical diagnostic
payloads, and incomplete or duplicate derived lineage fails before execution.

## Specification Sources

- Fix 03: named hole identity preservation and assignment.
- Fix 04A and Fix 04B: hole split/merge junction planning and surface execution.
- Fix 05A and Fix 05B: exact count-changing pairing and synthetic identity lineage.
- Fix 06: propagation of all caller planner configuration through expansion.
- Reuse existing `Station`, `TopologyPath`, planned refs, ambiguity diagnostics,
  surface patch builders, seam graph, and closure evidence.

## Specification Conformance

- Archived split parents:
  - [Fix 04](../specifications/fix-04-hole-split-merge-junction-surfaces-v1_0.md)
  - [Fix 05](../specifications/fix-05-count-changing-region-identity-preservation-v1_0.md)
- Canonical specs:
  - [Fix 03](../specifications/fix-03-named-hole-identity-pairing-v1_0.md)
  - [Fix 04A](../specifications/fix-04a-hole-junction-plan-records-v1_0.md)
  - [Fix 04B](../specifications/fix-04b-hole-junction-surface-execution-v1_0.md)
  - [Fix 05A](../specifications/fix-05a-count-changing-exact-region-pairing-v1_0.md)
  - [Fix 05B](../specifications/fix-05b-synthetic-station-identity-lineage-v1_0.md)
  - [Fix 06](../specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md)
- Paired canonical test specs use the matching filenames under
  [test specifications](../test-specifications/README.md).
- Progression: [v1.0.0a4 corrective release progression](../planning/progression.md).

## Conformance Checklist

- [ ] Implementation conforms to the target architecture.
- [x] Fix 03 named-hole identity resolution conforms through public planning and execution routes.
- [x] Fix 05A exact region identities resolve before anonymous residue and publish explicit count-changing births/deaths.
- [x] Fix 05B synthetic stations preserve deterministic region/loop lineage through public planning and execution.
- [x] Fix 04A count-changing holes publish validated interior-junction direction, lineage, and boundary inputs through planning and execution.
- [x] Fix 06 immutable planner configuration propagates through direct, expanded, and nested pairing routes.
- [x] Final leaves are independently reviewed and canonicalized.
- [x] Paired test specs point to canonical leaves.
- [x] Final progression preserves prerequisite order.
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

- 2026-08-04 - Completed Fix 04A. Reason: hole births and deaths now publish
  immutable identity-bearing interior-junction events, deterministic boundary
  rings, and exact pre-execution lineage validation through the surface executor.
- 2026-08-04 - Completed Fix 05B. Reason: identity-aware expansion now carries
  deterministic region and loop lineage through every synthetic station,
  preserves topology paths in both directions, and reaches the public surface
  executor without rebuilding an anonymous target.
- 2026-08-04 - Completed Fix 05A. Reason: count-changing public plans now
  preserve exact region IDs, restrict geometric assignment to anonymous
  residue, and record named births/deaths explicitly.
- 2026-08-04 - Completed Fix 06 and reconciled canonical loft correspondence
  architecture. Reason: public planner settings now become one immutable options
  value consumed by direct, expanded, and synthetic transition pairing; branch
  limit failures identify the effective cap and planning location.
- 2026-08-04 - Completed Fix 03 and reconciled canonical loft correspondence
  architecture. Reason: named holes now resolve before anonymous geometric
  residue and execution consumes the resolved plan.
- 2026-08-04 - Linked the final dependency-ordered progression. Reason: preserve identity, lineage, junction-plan, and execution prerequisites.
- 2026-08-04 - Recorded the six canonical loft leaves and archived split parents after fixed-point review.
- 2026-08-04 - Linked the full-template Fix 03-06 and paired test drafts. Reason: complete the `do specs` creation handoff.
- 2026-08-04 - Initial draft. Reason: plan issues #244-#246 for `v1.0.0a4`.
