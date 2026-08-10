# Impression v1.0.0a5 Corrective Release Definition

Date: 2026-08-09
Status: Qualified Candidate
Issue set: [#267](https://github.com/kellyjanderson/Impression/issues/267) and [#268](https://github.com/kellyjanderson/Impression/issues/268)
Base release: `v1.0.0a4`

## Intent

`v1.0.0a5` is a focused corrective alpha release for the remaining public
surface-CSG failures exposed by the diagonal audio-cube model. It completes
attached polygon-loft union and preserves declarative polygon-loft field
provenance across repeated public difference calls.

## User-Visible Outcomes

- A shell plus attached snap-tab polygon lofts fuses into one closed surfaced
  result through `boolean_union`.
- A shell plus the two attached microphone-rail polygon lofts fuses through the
  same route.
- Six copied snap-groove cutters can be applied sequentially, with every
  successful result immediately accepted as the next difference base.
- Equivalent union operand permutations produce stable result identity.
- Modeling composition remains surface-native; mesh conversion occurs only at
  preview/export boundaries.

## Canonical Scope

- [Surface Spec 432](../release-0.1.0a/specifications/surface-432-attached-polygon-loft-surface-union-completion-v1_0.md)
- [Surface Spec 432 Test](../release-0.1.0a/test-specifications/surface-432-attached-polygon-loft-surface-union-completion-v1_0.md)
- [Surface Spec 433](../release-0.1.0a/specifications/surface-433-repeated-snap-groove-surface-difference-provenance-preservation-v1_0.md)
- [Surface Spec 433 Test](../release-0.1.0a/test-specifications/surface-433-repeated-snap-groove-surface-difference-provenance-preservation-v1_0.md)
- [Closed architecture transition](../release-0.1.0a/architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md)

## Release Gates

The release tag may be pushed only after:

1. Both public sibling-project issue reproductions succeed.
2. The self-contained focused CSG tests and complete configured suite pass.
3. Dirty image and STL references are generated and inspected without being
   promoted to clean evidence.
4. Package/runtime versions are exactly `1.0.0a5` and build metadata agrees.
5. Wheel, source distribution, and docs archive pass candidate qualification.
6. The release PR is stable in GitHub CI and merged to `main`.
7. The tag-triggered release workflow publishes and verifies every asset.
8. Live assets pass independent hash and fresh-install smoke verification.

## Exclusions

- General implicit or polygon-loft intersection completion.
- Mesh-backed modeling fallback.
- General disconnected multi-component union represented as one shell.
- Changes to the sibling audio-cube model's authored dimensions or transforms.

## Candidate Evidence

- Feature branch began exactly at current `origin/main` commit
  `91ca617ba0d6c56be70192a9778e3122f9ca2776`.
- Both issue failures reproduced on that baseline.
- After implementation, both attached-feature unions, multi-cutter difference,
  first/second sequential cuts, and all six sequential sibling cuts succeed.
- Focused self-contained regression: 3 passed with refreshed XML/HTML coverage.
- Complete configured coverage: 1,785 passed in 440.69 seconds with 82.9%
  branch-aware coverage; canonical XML and HTML reports refreshed.
- Release metadata/workflow regression: 15 passed.
- Candidate wheel, sdist, and docs archive built and independently qualified in
  fresh environments; immutable artifact manifest verification passed.
- GitHub PR, merge, and publication evidence remain pending.

## Timing History

Durations and future wait recommendations are maintained in
[Release Timing History](planning/release-timing-history.md).
