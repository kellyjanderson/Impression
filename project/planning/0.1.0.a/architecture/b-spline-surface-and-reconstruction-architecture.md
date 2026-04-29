# B-Spline Surface And Reconstruction Architecture

## Status

`0.1.0.a` feature-path architecture branch.

Parent architecture:

- [B-Spline Implementation Architecture](b-spline-implementation-architecture.md)

Related research:

- [External Spanwise Loft Consolidation](../../../research/2026-04-23-external-spanwise-loft-consolidation.md)
- [External Model-Assisted Mesh Repair](../../../research/2026-04-23-external-model-assisted-mesh-repair.md)
- [External Patchwise Loft Repair](../../../research/2026-04-23-external-patchwise-loft-repair.md)

## Purpose

This branch defines the later, heavier usage areas for B-spline beyond path and
control-station work.

It covers:

- future surfaced B-spline patch families
- reconstruction and repair tooling
- approximate refit or consolidation work

## Why This Branch Exists

Once B-spline exists as a first-class curve primitive, the project will be
tempted to use it everywhere.

This branch exists to keep that expansion disciplined.

The key architectural point is:

- B-spline curves are likely an early need
- B-spline surfaces are a later and heavier need

So the system should separate:

- immediate B-spline curve adoption
- later surfaced/reconstruction adoption

## Main Usage Areas

### 1. Surface Patch Families

Future surfaced patch work may benefit from a true B-spline surface patch family
when the project needs:

- higher-order consolidated spans
- explicit smooth patch refits
- surfaced reconstruction that is richer than ruled or revolution patches

This is not a first milestone.

It should be treated as a later surfaced-family branch once the curve layer is
stable and the exact use cases are clearer.

### 2. Spanwise Consolidation / Refit

If spanwise loft consolidation progresses from exact merge toward approximate
refit, B-spline surfaces become a plausible target representation for:

- grouped interval runs
- larger-span smooth approximations
- exact-vs-approximate reporting

This branch should remain downstream of the exact-only early consolidation lane.

### 3. Reconstruction And Repair

Mesh repair and patchwise reconstruction are also natural consumers of
B-spline-based reconstruction once:

- section extraction is stable
- structural recovery is good enough
- surfaced repair wants a compact smooth patch instead of only triangle repair

This is especially plausible for:

- section-derived surface recovery
- local patch repair from recovered boundaries
- approximate smooth reconstruction from damaged spans

## Scope Boundary

This branch should not force `0.1.0.a` to implement surfaced B-spline patches.

Instead, it should define:

- where those later uses belong
- what preconditions they require
- why they should not be collapsed into the first B-spline milestone

## Preconditions

Before this branch becomes active implementation work, the project should have:

1. stable B-spline curve objects
2. deterministic evaluation/sampling
3. clear path/trajectory adoption
4. better evidence that a consumer truly needs a surfaced B-spline target

For reconstruction-specific use:

5. section extraction and structural recovery that are already trustworthy

## Architectural Rule

The project should prefer:

- exact surfaced-native structures first
- B-spline surface refit only where a consumer explicitly needs smooth compact
  approximation

That keeps B-spline from becoming a vague “smooth fallback” with unclear
guarantees.

## Open Questions

- When does a consumer truly require a B-spline surface rather than a B-spline
  curve plus existing surfaced patches?
- Should surfaced B-spline patch families be native kernel families or imported
  approximation targets?
- How should exact versus approximate surfaced B-spline results be reported?
- Which reconstruction branches should be allowed to depend on surfaced
  B-spline support first:
  - spanwise consolidation
  - band repair
  - patchwise repair

## Architectural Conclusion

B-spline should have future use in surfaced reconstruction and refit work, but
that usage should stay as a later branch. `0.1.0.a` should define the branch and
its boundaries without making surfaced B-spline support the first required
milestone.
