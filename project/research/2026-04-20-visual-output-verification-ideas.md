# Visual Output Verification Ideas Research

## Topic

Durable verification ideas for rendered-image, STL, and loft-specific visual
output checks.

Related documents:

- [Reference Artifacts](../agents/reference-images.md)
- [Surface Spec 106: Reference Artifact Regression Suite (v1.0)](../specifications/surface-106-reference-artifact-regression-suite-v1_0.md)
- [Loft Spec 50: Simple Correspondence Regression Fixtures (v1.0)](../specifications/loft-50-simple-correspondence-regression-fixtures-v1_0.md)
- [Legacy Mesh Audit](2026-04-18-legacy-mesh-audit.md)
- [Legacy Loft / Extrude Tool Extraction](2026-04-19-legacy-loft-extrude-tool-extraction.md)

## Findings

### 1. The current reference-artifact harness is intentionally limited

The current reference-image and reference-STL harness is useful because it
proves that:

- output was actually produced
- the artifact contains lightweight model-related signal
- the artifact changed or did not change relative to a stored reference

It does **not** prove that the output is aesthetically correct, geometrically
correct, or even the best result. That distinction should remain explicit in
future docs and tests. Reference artifacts are existence-and-change checks
first, not beauty or geometric-truth checks.

### 2. The durable general-harness ideas are already visible in code and policy

The current harness direction is:

- keep dirty and clean references separate
- treat dirty references as unreviewed change detectors
- prefer clean references when available
- never silently promote dirty references to clean
- keep reference rendering deterministic enough to compare repeatedly

The current test helpers already support several durable ideas worth keeping:

- deterministic camera/framing for reference renders via a bounds-derived camera
- lightweight image signal checks such as occupancy and luma variation
- lightweight STL signal checks such as file size, facet count, and vertex count
- stable reference selection with clean-first and dirty-fallback behavior

Preview and reference verification should remain distinct lanes. Preview is for
interactive inspection and iteration. Reference-artifact verification is for
repeatable regression detection under a fixed render/export harness.

### 3. The current loft-specific diagnostic direction is station-slice comparison

The strongest currently established loft lane is reconstruction-style station
comparison:

- author a loft from known sections/stations
- reconstruct a planar slice at a known station position
- compare the recovered section loops to the authored section loops

The existing harness already records this as three reference artifacts per
fixture:

- expected section bitmap
- actual recovered section bitmap
- expected/actual diff bitmap

This direction is strong because it checks the specific place where twist,
orientation drift, contour drift, or region permutation becomes visible without
pretending the whole rendered body image is enough on its own.

This is also the best current answer to "how do we inspect loft correctness
visually?" because it keeps the comparison local, inspectable, and tied to the
authoring input.

### 4. Metric-based checks should complement, not replace, reference artifacts

The loft verification discussion points toward a layered approach:

- keep reference images and STL files as coarse change detectors
- add station/section reconstruction checks for localized diagnosis
- add lightweight metrics that detect twist or deformation drift without
  relying on exact mesh identity

The metric ideas that appear durable are:

- loop count / region count agreement
- outer-versus-hole classification agreement
- normalized area and perimeter comparisons
- centroid drift
- winding/orientation agreement
- contour similarity after alignment

Those metrics fit the intent of [Loft Spec 50](../specifications/loft-50-simple-correspondence-regression-fixtures-v1_0.md):
simple fixtures should reveal twist or deformation both visually and
numerically.

### 5. Fixture breadth matters more than one showcase render

The current loft fixture family is already broader than a single hero render
and that breadth should continue:

- simple rectangular correspondence fixtures
- simple cylindrical correspondence fixtures
- anchor-shift / start-index-sensitive fixtures
- phase-shift cylindrical fixtures
- dual-region fixtures
- perforated / hole-sensitive fixtures

That breadth matters because different regressions show up in different ways:

- twist can hide in symmetric circular fixtures
- region permutation needs multi-region fixtures
- hole correspondence needs perforated fixtures
- anchor/start-index mistakes need orientation-sensitive fixtures

### 6. Orientation-sensitive fixtures should become more intentionally asymmetric

A useful preserved idea is to use notched or otherwise asymmetric fixtures when
the test is supposed to detect orientation-sensitive errors.

Plain circles and highly symmetric profiles can mask rotation or phase mistakes.
Notched asymmetric fixtures are better at exposing:

- unintended rotation
- phase drift
- station-to-station orientation mistakes
- swapped or mirrored reconstruction

The discussed follow-on work was to switch more orientation-sensitive station
tests toward asymmetric notched fixtures and then refresh their references. That
work was paused, so it should be treated as planned follow-up rather than
current completed state.

### 7. Twist/normal visualizers were discussed, but are not the established lane

Twist diagnostics and normal-oriented debugging ideas were part of the broader
discussion, but there is not yet an established implemented verification lane
in the repo built around dedicated twist or normal visualizers.

The currently agreed diagnostic direction is narrower and more concrete:

- station-slice comparison
- orientation-sensitive fixtures
- lightweight comparison metrics

### 8. A practical CV lane is silhouette classification, not beauty-image judging

A promising computer-vision-style lane is to compare canonical expected and
actual slice silhouettes after deterministic normalization rather than trying to
judge shaded beauty renders directly.

The useful high-level result classes are:

- same shape, same orientation
- same shape, rotated
- different shape

That lane is valuable because it can:

- ignore translation differences
- ignore size differences when the fixture only cares about contour identity
- distinguish orientation mismatch from genuine contour mismatch

This is especially attractive for orientation-sensitive fixtures with an
asymmetric witness such as an outward notch.

The project should treat this as a contour/silhouette classification tool, not
as a general-purpose aesthetic image judge.

If twist or normal visualizers are added later, they should be framed as
diagnostic helpers, not as the primary proof lane, unless the project later
decides otherwise.

## Implications

- Future verification docs should keep saying that reference artifacts prove
  output existence and change, not aesthetic or geometric correctness.
- Loft verification should keep investing in section/station reconstruction,
  because it is the clearest currently grounded way to inspect correctness near
  authored topology.
- Orientation-sensitive tests should prefer asymmetric fixtures when the goal is
  to catch rotation, phase, or twist regressions.
- Canonical slice verification can grow into a CV-style silhouette
  classification lane if it stays focused on expected-vs-actual contour
  classes rather than on subjective render judgment.
- Any future reference refresh for orientation-sensitive loft tests should be
  paired with an explicit note that the fixture family changed, not just the
  rendered outputs.
- Twist/normal visualizers should remain an optional future diagnostic lane
  unless they become real implemented harness behavior.

## References

- `project/agents/reference-images.md`
- `project/specifications/surface-106-reference-artifact-regression-suite-v1_0.md`
- `project/specifications/loft-50-simple-correspondence-regression-fixtures-v1_0.md`
- `project/research/2026-04-18-legacy-mesh-audit.md`
- `project/research/2026-04-19-legacy-loft-extrude-tool-extraction.md`
- `tests/reference_images.py`
- `tests/test_reference_images.py`
- `tests/loft_showcases.py`
- `docs/modeling/loft.md`
