# Reference Artifacts

## Purpose

Reference-artifact tests are used to prove three things:

1. the renderer actually produced an image
2. the image contains visible model-driven content
3. the rendered or exported output has or has not changed relative to a stored reference

Reference artifacts do **not** automatically prove that an image or STL is aesthetically or geometrically correct.

## Completeness Requirement

Any capability that outputs a model must have a reference-artifact test before
that capability is considered complete.

For this project, that means model-outputting work is expected to carry, by
default:

- a rendered reference image
- an exported reference STL

If a capability cannot legitimately produce one of those artifacts, the
exception must be documented explicitly in the relevant specification and test
specification.

## Storage

Reference images live under:

```text
project/reference-images/
  dirty/
  clean/
```

Suggested grouping:

```text
project/reference-images/
  dirty/
    surfacebody/
    loft/
  clean/
    surfacebody/
    loft/
```

Reference STL files live under:

```text
project/reference-stl/
  dirty/
  clean/
```

Suggested grouping:

```text
project/reference-stl/
  dirty/
    surfacebody/
    loft/
  clean/
    surfacebody/
    loft/
```

Planar section comparison artifacts may also live under `project/reference-images/`
when the feature is fundamentally a 2D reconstruction or sectioning regression.
Those artifacts may include:

- expected station bitmap
- recovered section bitmap
- visual diff bitmap

For planar section comparisons, the expected and recovered images should stay
reducible to two-color bitmaps so contour drift remains easy to inspect.

## Dirty References

Dirty references are unreviewed snapshots.

They may be used to detect output changes, but they must **not** be treated as proof that the rendered result is correct or desirable.

Dirty references are appropriate when:

- a render pipeline has just been created
- a new showcase or regression target has just been added
- an image has not yet received human visual review

When a dirty reference changes, tests may report that the render changed. They must not imply the new output is bad; only that it is different.

## First Run

The first run of a reference test for a named fixture should bootstrap the
fixture by creating:

- a dirty reference image
- a dirty reference STL

Bootstrap applies only when that fixture has no existing dirty or clean
references yet.

Bootstrap creates the initial change-detection baseline.
It does not imply acceptance, and it must not silently promote anything to
clean.

## Contract Changes

When the reference-test contract for an existing fixture changes, the previous
dirty and clean reference sets for that fixture are no longer valid.

Contract changes include cases such as:

- different fixture geometry
- different camera or composition rules
- adding or removing required artifact types
- changing a single-view render into a triptych or other composed view
- adding or changing canonical slice expectations
- changing the meaning of pass/fail for the fixture

In those cases, agents must invalidate the existing reference set before
trusting another test run.

Invalidation means removing the old dirty and clean references for that fixture
so the next intentional run bootstraps a new dirty baseline that matches the new
test contract.

Agents must not treat an old reference file as reusable evidence when the test
itself now means something different.

## Clean References

Clean references are visually reviewed and explicitly accepted.

A clean reference may be used as a stronger signal that:

- the render path is stable
- the composition is acceptable
- the rendered output is consistent with the intended model and camera

Promotion from dirty to clean requires explicit human review.

## Promotion Rules

Dirty -> clean promotion requires:

1. human visual inspection
2. confirmation that the model shown is the intended model
3. confirmation that camera, framing, and visible topology are acceptable
4. explicit replacement or copy from `dirty/` into `clean/`

Agents must not silently promote dirty references to clean.

## Test Rules

Reference-image and reference-STL tests must:

1. render a fresh image during the test
2. export a fresh STL during the test when STL references are part of the feature
3. prove the artifact is non-empty and model-related
4. bootstrap dirty references on the first run when the fixture does not exist yet
5. compare against the selected reference artifact on subsequent runs

If a clean reference exists, tests should prefer it.
If no clean reference exists, tests should fall back to the dirty reference.

If a fixture already exists and only part of the required artifact set is
missing, the test should fail clearly rather than silently treating that as a
fresh bootstrap event.

If the rendered or exported output differs from the selected reference in a way
the comparison contract does not consider clean, the test must fail.

## Regeneration

Dirty references may be regenerated intentionally with:

```bash
./.venv/bin/python scripts/dev/update_dirty_reference_images.py
```

or:

```bash
./scripts/dev/run_reference_image_suite.sh --update-dirty-reference-images
```

STL references should be regenerated alongside the image references by the same
project helper script when the feature's reference set includes STL outputs.

Agents should note when regeneration occurred and whether the outputs remain dirty or were visually promoted.

## Agent Behavior

Agents should:

- treat reference-artifact coverage as part of completion for model-outputting work
- treat clean references as stronger than dirty references
- describe dirty references as change detectors, not quality truth
- avoid rewriting clean references without explicit reason
- invalidate existing dirty and clean references when a fixture's reference-test
  contract changes
- note whether a fixture run bootstrapped new dirty references or compared
  against an existing baseline
- mention when a rendered or exported test only proves output changed, not whether it improved or regressed visually
