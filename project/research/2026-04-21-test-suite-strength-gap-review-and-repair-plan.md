# Test Suite Strength / Gap Review and Repair Plan

## Topic

Review of the current Impression test suite, with a repair plan focused on the
largest remaining surfaced CSG verification gap: `intersection_box_sphere`.

This note is a synthesis of one local review pass plus three isolated read-only
sub-agent tracks:

- surfaced CSG execution and reference coverage
- reference-image / STL / section-verification harness behavior
- broader suite balance outside the narrow CSG lane

## Current Strengths

### 1. The strongest parts of the suite are strong for the right reasons

The loft and surface branches are not just smoke-tested.
They cover:

- deterministic planning and execution
- closure and seam behavior
- split / merge cases
- planner / executor equivalence
- structural metadata and public API contracts

Representative files:

- [tests/test_loft.py](../../tests/test_loft.py)
- [tests/test_loft_correspondence.py](../../tests/test_loft_correspondence.py)
- [tests/test_surface.py](../../tests/test_surface.py)
- [tests/test_surface_kernel.py](../../tests/test_surface_kernel.py)

### 2. The surfaced CSG lane is honest about bounded scope

The surfaced boolean tests do a good job of separating:

- supported no-cut and exact-reuse cases
- supported bounded box/box overlap cases
- explicit unsupported broader cases
- explicit invalid-result posture when cleanup cannot rescue reconstruction

That honesty is valuable.
The suite does not currently pretend that general surfaced CSG is finished.

Representative files:

- [tests/test_surface_csg.py](../../tests/test_surface_csg.py)
- [docs/modeling/csg.md](../../docs/modeling/csg.md)

### 3. The repo has real migration and documentation guardrails

The test suite protects:

- no stale deprecation wiring in finished surface-first modules
- explicit deprecation posture where mesh-primary behavior still belongs
- documentation promises that are supposed to remain durable

Representative files:

- [tests/test_modern_geometry_no_deprecations.py](../../tests/test_modern_geometry_no_deprecations.py)
- [tests/test_mesh_deprecations.py](../../tests/test_mesh_deprecations.py)
- [tests/test_documentation_rules.py](../../tests/test_documentation_rules.py)

### 4. The model-output verification harness is real infrastructure

The reference-artifact lane now has durable policy and real enforcement for:

- dirty bootstrap
- clean-over-dirty preference
- paired image / STL verification
- contract invalidation
- section-image expected / actual / diff artifacts
- silhouette relationship classes for slice comparison

Representative files:

- [project/architecture/model-output-reference-verification.md](../architecture/model-output-reference-verification.md)
- [tests/reference_images.py](../../tests/reference_images.py)
- [tests/test_reference_images.py](../../tests/test_reference_images.py)

## Current Weaknesses

### 1. The suite is imbalanced

Test density is heavily concentrated in loft and surface:

- `test_loft.py`: `114`
- loft family total: `139`
- surface family total: `119`
- `test_surface_csg.py`: `29`
- `test_reference_images.py`: `17`

Several other modeling lanes are still thin and mostly smoke-level:

- [tests/test_drafting.py](../../tests/test_drafting.py)
- [tests/test_heightmap.py](../../tests/test_heightmap.py)
- [tests/test_text.py](../../tests/test_text.py)
- [tests/test_extrude.py](../../tests/test_extrude.py)
- [tests/test_path3d.py](../../tests/test_path3d.py)

These files are useful sanity checks, but they are weaker regression detectors
than the loft and surface branches.

### 2. Many reference tests still prove change detection more than geometry truth

The reference harness is valuable, but much of it still primarily proves:

- a render happened
- an STL was produced
- the output changed or did not change relative to a stored baseline

That is exactly what the policy says it should do.
The weakness is not the harness itself.
The weakness is that many fixtures do not yet pair those artifacts with enough
geometry-specific truth.

This is especially visible outside loft section reconstruction and the newer
CSG slice lane.

### 3. Triptych rendering is visually helpful but not geometrically honest yet

The current triptych helper renders each panel independently and then stitches
the images together.

That means it does **not** preserve:

- true relative scale
- true relative spatial framing
- trustworthy visual comparison of operand-vs-result size

So it is useful for human readability, but it is weaker evidence than a shared
camera or shared-scene triptych would be.

Relevant file:

- [tests/reference_images.py](../../tests/reference_images.py)

### 4. Section verification is stronger than beauty renders, but still incomplete

The slice-based silhouette lane is the most promising current “truth” layer for
visual verification, but it still has important limitations:

- it normalizes translation and size away
- it currently only distinguishes rotational classes at `90/180/270` degrees
- it has no explicit mirror class
- it does not yet enforce section artifacts as a grouped completeness contract

That means it is good at classifying contour identity, but weaker than the
project discussion intended for exact fixture-local placement.

### 5. Some documentation promises are ahead of executable truth

The biggest current example is surfaced CSG reference readiness.
Docs and paired specs still name `surfacebody/csg_intersection_box_sphere` as a
required active reference fixture, but the current executable reference lane
only supports:

- `surfacebody/csg_union_box_post`
- `surfacebody/csg_difference_slot`

Relevant files:

- [docs/modeling/csg.md](../../docs/modeling/csg.md)
- [project/specifications/surface-137-surface-boolean-initial-reference-fixture-matrix-and-promotion-gates-v1_0.md](../specifications/surface-137-surface-boolean-initial-reference-fixture-matrix-and-promotion-gates-v1_0.md)
- [tests/test_reference_images.py](../../tests/test_reference_images.py)

## Focused Review: `intersection_box_sphere`

### What is strong today

The unit suite does cover box/sphere surfaced boolean posture, but only in the
exact no-cut lane:

- box contains sphere
- sphere contains box
- partial overlap remains explicitly unsupported

That is good bounded-scope honesty.
It keeps callers from mistaking containment reuse for true overlap execution.

Relevant file:

- [tests/test_surface_csg.py](../../tests/test_surface_csg.py)

### What is weak today

`intersection_box_sphere` is still the largest surfaced CSG gap because the
named fixture is not currently honest enough to be active evidence.

The weak historical version was a containment case.
That meant the intersection result was just the sphere.
It proved exact reuse, not boolean cut truth.

The current stronger problem is deeper than fixture naming:

- partial-overlap box/sphere intersection is not implemented in the surfaced
  execution slice
- the reference harness has no honest active intersection builder for it
- the docs and specs still name it as a required fixture

So the gap is now three-layered:

1. execution gap
2. reference-fixture gap
3. doc/spec alignment gap

### Why it is blocked

The current bounded surfaced overlap lane supports box-style planar operands.
A sphere introduces a new surface family and new work around:

- cut discovery between planar and spherical patches
- patch classification across different surface families
- trim reconstruction on spherical patches
- result-shell reconstruction for the mixed-family intersection

There is also a fixture-design issue:
if orientation-sensitive slice truth is required, the asymmetry cannot come
from the sphere itself.
It must come from the box-side witness and survive the recovered section in a
way that still makes the intended result obvious.

## Repair Plan

The repair plan should be split into isolated tracks so work can proceed
without multiple agents fighting over the same files or decisions.

### Track A: `intersection_box_sphere` Surfaced Execution

Goal:
implement an honest surfaced partial-overlap `intersection` between one box and
one sphere.

Scope:

- [src/impression/modeling/csg.py](../../src/impression/modeling/csg.py)
- [tests/test_surface_csg.py](../../tests/test_surface_csg.py)

Needed outcomes:

- partial-overlap box/sphere intersection no longer returns `unsupported`
- containment reuse remains separate and explicitly tested
- the supported result is a real mixed-family surfaced body, not a stand-in

This is the critical path for restoring `intersection_box_sphere` as a real
fixture rather than a documentation promise.

### Track B: Mixed-Family Cut / Trim Reconstruction

Goal:
add the geometry sub-problems that Track A depends on.

Scope:

- [src/impression/modeling/csg.py](../../src/impression/modeling/csg.py)
- surfaced CSG architecture / specification notes as needed

Needed outcomes:

- box/sphere cut discovery
- mixed-family patch classification
- planar and spherical trim reconstruction
- bounded shell / seam reconstruction for the supported intersection case

This track is the true geometry blocker behind Track A.

### Track C: `intersection_box_sphere` Fixture Design

Goal:
define a fixture that is worth trusting once execution exists.

Scope:

- [tests/csg_reference_fixtures.py](../../tests/csg_reference_fixtures.py)
- [tests/test_reference_images.py](../../tests/test_reference_images.py)

Needed outcomes:

- a non-containment overlap case
- triptych presentation
- canonical slice definition
- explicit asymmetric witness on the box side if orientation is required
- PNG + STL + section-artifact contract

The fixture should not be activated until Track A produces a real surfaced
result.

### Track D: Reference-Harness Hardening

Goal:
make the visual verification harness closer to the project’s intended truth
contract.

Scope:

- [tests/reference_images.py](../../tests/reference_images.py)
- [tests/test_reference_images.py](../../tests/test_reference_images.py)
- [scripts/dev/update_dirty_reference_images.py](../../scripts/dev/update_dirty_reference_images.py)

Needed outcomes:

- grouped completeness checks for section artifact sets
- more honest triptych framing
- stronger orientation / mirror classification
- fix latent image-to-mask polarity bug

This track improves the verification machinery for both CSG and loft without
depending on Track A.

### Track E: CSG Spec / Doc Reconciliation

Goal:
bring the durable requirements back into line with executable truth.

Scope:

- [docs/modeling/csg.md](../../docs/modeling/csg.md)
- [project/specifications/surface-137-surface-boolean-initial-reference-fixture-matrix-and-promotion-gates-v1_0.md](../specifications/surface-137-surface-boolean-initial-reference-fixture-matrix-and-promotion-gates-v1_0.md)
- [project/specifications/surface-138-surface-boolean-reference-fixture-composition-and-slice-verification-v1_0.md](../specifications/surface-138-surface-boolean-reference-fixture-composition-and-slice-verification-v1_0.md)
- paired test specifications
- [project/planning/progression.md](../planning/progression.md)

Needed outcomes:

- either keep `intersection_box_sphere` as required and leave it explicitly open
- or temporarily narrow the requirement until execution is real

This track must not paper over the geometry gap.
It should reflect reality, not excuse it.

### Track F: Non-CSG High-Value Follow-On Tracks

These can proceed independently while the surfaced CSG lane remains active:

- surface threading reference-fixture track
- thin-geometry invariants track for drafting / text / heightmap
- loft orientation-hardening track with asymmetric witnesses
- documentation-contract hardening where docs currently outpace executable
  cross-checks

Representative files:

- [tests/test_surface_threading.py](../../tests/test_surface_threading.py)
- [tests/test_surface_threading_docs.py](../../tests/test_surface_threading_docs.py)
- [tests/test_drafting.py](../../tests/test_drafting.py)
- [tests/test_heightmap.py](../../tests/test_heightmap.py)
- [tests/test_text.py](../../tests/test_text.py)
- [tests/test_loft_correspondence.py](../../tests/test_loft_correspondence.py)

## Recommended Order

1. Keep `intersection_box_sphere` remediation as the dedicated active CSG lane.
2. Split the CSG work into Track A, Track B, Track C, and Track E.
3. Run Track D in parallel when it can stay isolated from execution changes.
4. Use Track F for all non-CSG unattended work so the repo keeps moving without
   creating collisions in the surfaced CSG slice.

## Conclusion

The current suite is stronger than a normal regression suite in the parts of
the repo that matter most today.
Its biggest weakness is not lack of testing effort.
It is the remaining mismatch between:

- bounded surfaced CSG execution truth
- reference-fixture evidence
- durable documentation promises

`intersection_box_sphere` is the clearest example of that mismatch.
It should remain the lead repair target until execution, fixtures, and docs all
describe the same honest state again.
