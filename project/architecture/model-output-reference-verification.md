# Model Output Reference Verification

## Overview

This document defines the project-wide verification contract for any capability
that outputs a model.

This architecture is a child branch of the top-level
[Testing Architecture](testing-architecture.md).
It describes reusable testing infrastructure rather than feature-owned
structure.

In Impression, a capability is treated as model-outputting when it produces a
result that a caller can preview, render, tessellate, export, analyze, or hand
off as geometry.

Representative examples include:

- surface-body generators
- loft outputs
- modeling operations that return geometry
- geometry collections or handoff objects that still represent model output

## Core Requirement

Any capability that outputs a model must have a durable reference-artifact test
before that capability is considered complete.

That requirement is not optional polish.
It is part of the definition of done for model-outputting work in this project.

By default, the required reference set is:

- a rendered reference image
- an exported reference STL

If a capability cannot legitimately produce one of those artifacts, that
exception must be made explicit in the relevant specification and test
specification rather than being omitted implicitly.

## Fixture Identity

Each model-outputting capability must have at least one stable named reference
fixture under `project/reference-images/` and `project/reference-stl/`.

Fixture names should be:

- durable
- capability-specific
- stable enough to survive implementation churn

The reference name is part of the verification contract and should not drift
casually.

## First-Run Bootstrap

When a reference test for a named fixture runs for the first time and neither a
dirty nor clean reference exists yet for that fixture, the test must bootstrap
the fixture by writing:

- a dirty reference image
- a dirty reference STL

That first run creates the initial change-detection baseline.

Bootstrap does not imply correctness review.
It only establishes the first dirty baseline for later comparison.

Bootstrap must not silently promote anything to clean.

## Subsequent-Run Comparison

After bootstrap, reference tests must compare fresh outputs against the selected
reference set.

Reference selection rules are:

1. prefer the clean reference if one exists
2. otherwise compare against the dirty reference

The test must fail when the fresh output differs from the selected reference in
a way the comparison contract does not consider clean.

That means:

- a meaningful rendered-image diff fails the test
- a meaningful STL diff fails the test

This project treats dirty references as legitimate comparison baselines for
change detection even though they are not proof of visual or geometric quality.

## Reference Invalidation

If the verification contract for a fixture changes, the existing dirty and clean
references for that fixture must be invalidated before the new test contract is
trusted.

Examples of contract changes include:

- changing the modeled geometry or operand composition
- changing the camera, framing, or rendered layout
- changing which artifact types are required
- adding or changing canonical slice expectations
- changing orientation policy or other pass/fail meaning

Invalidation means deleting the existing dirty and clean references for that
fixture so the next intentional run bootstraps a fresh dirty baseline under the
new rules.

The project must not compare a new contract against an old baseline and pretend
that is still durable evidence.

## Partial-Missing Reference State

Bootstrap applies only when a fixture has no existing reference set yet.

If a fixture already exists and only part of the required set is missing, that
is an incomplete fixture state and should be treated as a failure, not as a new
bootstrap event.

In practice, once a fixture exists, the project expects the required artifact
pair to remain present together.

## Failure Meaning

Reference-artifact failures mean one of the following:

- the model output changed
- the rendering or export path changed
- the reference set is incomplete or missing

Reference-artifact failures do not automatically mean the new model is worse.
They mean the durable baseline no longer matches and human review is required.

## Canonical Slice Verification

For capabilities where a full rendered image is too weak to describe geometric
correctness, the project may add canonical slice verification as a stronger
evidence lane.

This is especially appropriate for:

- loft section recovery
- orientation-sensitive boolean results
- fixtures where local contour truth matters more than a full shaded render

Canonical slice verification should compare an expected and actual silhouette in
the same logical fixture frame rather than relying on raw world-space mesh
identity.

The expected and actual slice do not need to match by raw pixel position or raw
pixel scale.
Instead, they should be compared after deterministic normalization that removes
translation and size differences while preserving contour identity.

## Silhouette Relationship Classes

When a fixture uses slice verification, the preferred high-level result classes
are:

- `same_shape_same_orientation`
- `same_shape_rotated`
- `different_shape`

This classification allows the test to separate:

- correct shape and orientation
- correct shape with an orientation mismatch
- genuinely different contour outcome

The test fixture then decides whether an orientation difference is acceptable or
is itself a failure.

## Orientation Policy

Slice-based fixtures should declare whether orientation is:

- required
- irrelevant

If orientation is required, the fixture should include an asymmetric cue so the
silhouette has a meaningful directional witness.

Good cues include:

- an outward notch
- an offset tab
- an asymmetric hole
- a clearly one-sided protrusion

Without an asymmetric cue, some silhouettes are too symmetric for orientation
classification to be meaningful.

## Expected Slice Definition

Expected slices should be defined in fixture-local terms rather than inferred
from arbitrary world-space framing.

That means a slice-based fixture should make explicit:

- the fixture-local frame
- the slice location and normal
- the expected silhouette source
- whether orientation mismatch is a failure

Expected silhouettes may come from:

- authored section input
- hand-defined 2D expected contours for simple fixtures
- a separately trusted oracle lane when explicitly allowed by specification

## Delivery Implication

When a capability claims slice-based correctness evidence, the test is not
complete unless it defines:

- what the expected slice is
- how the actual slice is normalized into the same comparison frame
- what silhouette relationship classes are accepted
- whether orientation mismatch is acceptable for that fixture

## Clean Promotion

Dirty references remain dirty until explicitly reviewed and promoted.

Promotion to clean requires:

1. human review
2. confirmation that the model shown or exported is the intended one
3. confirmation that framing and visible topology are acceptable for the image
4. explicit promotion of the dirty artifacts into the clean location

## Delivery Rule

A model-outputting capability is not complete when any of the following is true:

- it has no named reference fixture
- it has no reference test
- it has no dirty bootstrap artifacts after the first reference run
- it has no comparison behavior for subsequent runs
- it is still using a reference set created for an older fixture contract
- it relies on manual memory instead of durable reference artifacts

This requirement applies equally to new capabilities and to finished surfaced
replacements that are being promoted as the canonical lane.
