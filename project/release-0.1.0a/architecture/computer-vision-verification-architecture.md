# Computer Vision Verification Architecture

## Overview

This document defines how computer-vision-style verification fits into
Impression's model-output test system.

This architecture is a child branch of the top-level
[Testing Architecture](testing-architecture.md).
It describes one testing subsystem.
Its tooling specifications should refine the testing architecture/spec branch
rather than treating CV as a feature-owned program.

The goal is not to replace the existing reference-image and reference-STL
contract.
The goal is to add a stronger interpretation layer for cases where "an image
changed" or "an STL exists" is not enough to describe whether the output is
correct.

The primary target cases discussed so far are:

- canonical slice silhouette verification for loft and surfaced CSG
- orientation-sensitive difference/section fixtures using asymmetric notches
- text-output verification that reads the rendered glyphs rather than trusting
  font-specific geometry assumptions
- STL/viewer-space orientation verification through canonical rendered views
- handedness and mirror detection across modeling space, export space, and
  viewer space
- camera-position and framing agreement under a declared fixture view contract

## Relationship To Existing Verification

This architecture extends, but does not replace:

- [Model Output Reference Verification](model-output-reference-verification.md)
- [Reference Artifact rules](../agents/reference-images.md)
- existing reference image and STL dirty/clean lifecycle policy

Reference artifacts remain the project's baseline change detectors.
CV verification is an additional truth and interpretation layer.

The durable split is:

- reference artifacts answer "did the produced output change relative to the
  stored baseline?"
- CV verification answers "does the produced output still mean the same thing
  under a declared fixture contract?"

## Core Design Principles

### 1. CV must operate on deterministic harness outputs

The project should not ask CV tools to guess arbitrary camera pose,
orientation, or expected meaning after the fact.

Instead, CV verification must operate on deterministic harness products such
as:

- canonical slice bitmaps
- canonical orthographic text renders
- canonical front / side / top / isometric object renders
- depth, silhouette, normal, or ID-mask images derived from a fixed view spec

The harness defines the frame.
CV interprets the result inside that frame.

### 2. CV should classify meaning, not judge beauty

The project should not treat CV as a subjective image critic.

The preferred role is:

- classify contour relationships
- recognize visible glyphs
- detect rotation or mirroring
- detect orientation witnesses such as outward notches
- identify handedness or axis-swap failures
- compare silhouettes, edge maps, depth maps, and normal-map structure where
  those products are more stable than shaded renders

Shaded beauty-image similarity is too weak and too ambiguous to be the primary
proof lane.

### 3. CV should be strongest where the fixture has an explicit witness

CV is most trustworthy when the fixture deliberately includes a visible cue
that makes the intended answer obvious.

Examples:

- an outward notch for orientation-sensitive section comparison
- rendered text for OCR-driven glyph verification
- a left/right asymmetric calibration body for handedness checks
- a declared axis witness for camera and view agreement

Symmetric fixtures should not pretend to support orientation-sensitive CV
classification unless the fixture explicitly declares orientation irrelevant.

### 4. ML-assisted interpretation is secondary to deterministic gates

Traditional image-processing and contour-comparison tools should be the default
gate when a deterministic formulation exists.

Learned vision tools may be used as:

- a secondary diagnostic lane
- a confidence signal
- a review helper for ambiguous outputs

They should not be the primary pass/fail gate when a deterministic silhouette,
OCR, or view-spec comparison is available.

## Verification Lanes

### 1. Canonical Slice Silhouette Lane

This lane verifies a 3D result by slicing it in a declared fixture-local frame
and comparing the resulting 2D silhouette to an expected silhouette.

Primary use cases:

- loft station recovery
- orientation-sensitive surfaced boolean results
- local contour checks where a full beauty render is too weak

Required inputs:

- fixture-local frame
- slice plane origin and normal
- expected silhouette source
- orientation policy
- contour comparison tolerances

Preferred result classes:

- `same_shape_same_orientation`
- `same_shape_rotated`
- `different_shape`

Future extension:

- `same_shape_mirrored`

This lane is currently the best grounded CV-style verification direction for
loft and surfaced CSG.

### 2. Orientation-Witness Notch Lane

This is a specialization of the slice silhouette lane for fixtures where
orientation itself is part of correctness.

The preferred witness is an outward asymmetric notch or other one-sided cue.

Why this matters:

- correct alignment causes the expected and actual silhouettes to cancel
  cleanly
- wrong orientation leaves an obvious residual notch pattern in the diff
- the failure remains machine-classifiable and human-readable

Preferred target cases:

- surfaced `difference` fixtures
- loft section-orientation fixtures
- any slice-based test where rotation or mirroring must fail

This lane should not be used with symmetric contours unless the fixture marks
orientation as irrelevant.

### 3. Text OCR / Glyph Verification Lane

This lane verifies rendered text by asking a vision system to identify the
visible text content and its orientation, instead of assuming that a chosen
font fixture produced the intended glyph geometry.

Primary use cases:

- confirming that the rendered result contains the intended characters
- detecting `.notdef` box output or other fallback glyph failures
- detecting rotated, mirrored, or otherwise misoriented text
- reducing dependency on font-specific contour assumptions in tests

Expected outputs:

- recognized text
- per-glyph or whole-string confidence
- orientation classification
- unreadable / low-confidence status when recognition is not trustworthy

Preferred high-level result classes:

- `same_text_same_orientation`
- `same_text_rotated`
- `same_text_mirrored`
- `different_text`
- `unreadable`

This lane is the strongest currently discussed answer for text verification
because it tests the user-visible meaning of the output rather than just its
bounds or extrusion depth.

### 4. Canonical View Object Verification Lane

This lane verifies an STL or other model output by rendering it from a declared
set of canonical views and then evaluating the resulting views.

Primary use cases:

- STL export verification
- viewer-space orientation checks
- modeling-space vs output-space vs viewer-space consistency
- shared camera-position agreement across fixtures

Preferred canonical view set:

- front
- side
- top
- isometric

Preferred derived render products:

- silhouette images
- depth maps
- normal maps
- optional shaded beauty render for human review

The key rule is that the view specification must be declared by the fixture.
CV should evaluate a known camera contract, not infer one.

### 5. Handedness And Mirror Detection Lane

This lane checks whether the object has crossed a left/right mirror boundary or
otherwise changed handedness between modeling, tessellation, export, and
viewing.

Primary use cases:

- export pipeline validation
- STL import/export regression checks
- viewer-adapter verification
- axis-flip and mirror regression detection

This lane depends on a deliberately asymmetric witness fixture.

Examples of useful witnesses:

- a left-handed or right-handed calibration body
- directional embossed text
- one-sided tabs, holes, or arrows
- a labeled axis marker body

Expected result classes:

- `same_handedness`
- `mirrored`
- `orientation_unknown`

This lane should remain explicit about uncertainty.
If the witness is too symmetric, the test should report the ambiguity rather
than pretending handedness was verified.

### 6. Camera And Framing Agreement Lane

This lane verifies that the harness camera/view contract itself is stable and
that the rendered result agrees with the declared fixture framing.

Primary use cases:

- ensuring reference renders are generated from the intended camera pose
- detecting drift between modeling space and rendered view framing
- preserving comparability across repeated runs and across artifact types

Verification targets:

- declared camera position
- declared target/look-at point
- declared up vector
- declared projection mode
- expected visible extents for the fixture

This lane is important because many higher-level CV checks become unreliable if
camera and framing are allowed to drift silently.

### 7. Diagnostic Multi-View / Triptych Lane

Triptych or multi-panel presentation remains useful for human review, but it is
not by itself strong geometric proof.

This lane should therefore be treated as diagnostic-first.

Recommended uses:

- operand A / result / operand B human-readable CSG review
- visual explanation of why a failure occurred
- paired presentation of expected / actual / diff section artifacts

Triptych presentation becomes stronger when built from a shared scene or shared
camera contract, but even then it should remain a supporting lane unless the
project later defines a stricter shared-view proof contract.

## Supporting CV Analyses

The major lanes above may rely on smaller CV analyses that should be treated as
shared building blocks rather than separate architectural systems.

### 1. Silhouette And Mask Comparison

Useful for:

- foreground/background occupancy checks
- canonical slice comparison
- canonical STL/object view comparison

Typical outputs:

- mask IoU
- area delta
- centroid delta
- residual diff mask

### 2. Contour And Shape Metrics

Useful for:

- comparing expected and actual slice boundaries
- determining whether two silhouettes are the same shape after normalization
- distinguishing contour drift from orientation drift

Representative metrics and tools:

- contour extraction
- perimeter and area comparison
- Hu-moment-style shape summaries
- Hausdorff-style contour distance
- chamfer-style boundary distance

These metrics should complement, not replace, the lane-level result classes.

### 3. Edge / Depth / Normal Comparison

Useful for:

- object-view verification where shaded renders are lighting-sensitive
- view-space orientation checks
- highlighting geometric discontinuities and silhouette truth more clearly than
  beauty images

Preferred artifacts:

- edge maps
- depth maps
- normal maps
- ID masks when panel or object isolation matters

### 4. Connected-Component And Hole Counting

Useful for:

- counting visible islands or bodies in a rendered mask
- checking hole presence in a canonical section
- coarse CSG or text sanity checks before deeper comparison

This is a lightweight support check, not a full proof lane.

### 5. Panel / Region Segmentation

Useful for:

- triptych or multi-panel fixtures
- separating operand A, result, and operand B for downstream comparison
- OCR isolation for text-only subregions

This may be done with deterministic cropping when layout is fixed, or with a
learned/heuristic segmenter when the harness later needs more flexibility.

## Shared Fixture Contract

Every CV-backed fixture should declare enough information for deterministic
evaluation.

The minimum fixture contract should identify:

- fixture name
- capability under test
- canonical fixture frame
- rendered artifact set
- CV verification lane or lanes used
- expected interpretation contract
- pass/fail mapping for each result class

Depending on lane, the fixture may also need:

- slice plane definitions
- expected silhouette source
- orientation-required vs orientation-irrelevant policy
- text string and readable orientation policy
- canonical camera set
- handedness witness definition
- allowed tolerances and uncertainty policy

## Shared Harness Components

The architecture assumes a reusable harness with distinct layers.

### 1. Fixture Builder

Builds the model, slice inputs, witnesses, and canonical metadata.

### 2. Canonical Renderer

Produces deterministic reference artifacts and CV input artifacts such as:

- section bitmaps
- orthographic text renders
- canonical object views
- silhouettes
- depth maps
- normal maps

### 3. Normalization Layer

Normalizes the produced artifacts into the comparison frame defined by the
fixture.

Examples:

- translation / scale normalization for silhouette comparison
- OCR crop and rotation normalization for text
- canonical view naming and ordering for STL review

### 4. CV Interpreter

Runs the appropriate lane-specific analysis.

Examples:

- contour/silhouette classification
- notch-orientation classification
- OCR / glyph recognition
- mirror / handedness classification
- camera/framing compliance checks

### 5. Test Decision Layer

Maps the interpreter output into pass/fail according to the fixture contract.

This is where the test decides, for example, whether:

- `same_shape_rotated` is a failure
- mirrored text is a failure
- low-confidence OCR should fail or request review

### 6. Review Artifact Publisher

Writes human-readable artifacts that help explain failure.

Examples:

- expected / actual / diff bitmaps
- multi-view comparison panels
- OCR overlays
- handedness witness annotations

## Tooling Posture

### Deterministic First

Preferred first-line tools:

- OpenCV-style contour and image operations
- binary-mask and silhouette comparison
- shape/contour metrics
- OCR where text meaning is the target
- deterministic cropping or panel extraction for fixed-layout triptychs
- connected-component and hole counting for quick structural sanity checks

### ML-Assisted Second

Possible later tools:

- embedding-based similarity for review ranking
- anomaly detection on accepted reference render families
- learned segmentation to isolate result panels or text areas
- semantic classification of mirrored/rotated text or object views when the
  deterministic lane is inconclusive
- confidence estimation for ambiguous OCR or silhouette results

These tools may help triage or diagnose, but they should remain secondary until
the project has a strong deterministic reason to trust them as gates.

## Result Taxonomy

Different lanes need different result classes, but they should all follow the
same pattern:

- a positive class for semantic agreement
- explicit orientation/mirror variants when relevant
- a clear "different" class
- an explicit uncertainty or unreadable class when the harness cannot honestly
  decide

Representative examples:

- silhouette lane:
  - `same_shape_same_orientation`
  - `same_shape_rotated`
  - `different_shape`
- text lane:
  - `same_text_same_orientation`
  - `same_text_rotated`
  - `same_text_mirrored`
  - `different_text`
  - `unreadable`
- handedness lane:
  - `same_handedness`
  - `mirrored`
  - `orientation_unknown`

The project should prefer an honest unknown/uncertain result over a false
assertion of correctness.

## Initial Application Priorities

The discussed rollout order should remain conservative.

### Priority 1: Text Verification

Text is currently a strong candidate because CV can check the user-visible
meaning of the render even when font geometry details vary.

Immediate goals:

- confirm intended visible text
- catch `.notdef` box fallback
- classify orientation and mirror state

### Priority 2: Canonical Slice Verification

This extends the current slice-silhouette direction with stronger orientation
and classification language.

Immediate goals:

- loft station recovery verification
- surfaced boolean section verification
- notch-based orientation failures for `difference`-style fixtures

### Priority 3: STL/Viewer-Space Orientation Verification

This is valuable, but it depends on a stable canonical view harness and
asymmetric witness fixtures.

Immediate goals:

- front/side/top/isometric canonical render set
- handedness witness fixture
- axis and mirror drift detection

### Priority 4: ML-Assisted Diagnostics

This should remain a later lane after deterministic fixtures and framing are
stable.

## Non-Goals

This architecture does not claim that CV should:

- replace geometric unit tests
- replace reference images or STL comparisons
- infer fixture intent from arbitrary beauty renders
- silently correct framing, pose, or handedness after the fact
- be trusted as a subjective quality judge

## Implications For Specifications And Tests

If a specification claims CV-backed visual truth, it must also define:

- which lane is authoritative
- the fixture witnesses that make that lane meaningful
- what result classes are acceptable
- what uncertainty policy applies
- which artifacts must be emitted for human review

The project should not add vague "use CV here" language to specs without
declaring the fixture contract and decision policy that make the lane durable.

## References

- [Model Output Reference Verification](model-output-reference-verification.md)
- [Visual Output Verification Ideas Research](../research/2026-04-20-visual-output-verification-ideas.md)
- [Test Suite Strength / Gap Review and Repair Plan](../research/2026-04-21-test-suite-strength-gap-review-and-repair-plan.md)
- [Reference Artifact rules](../agents/reference-images.md)
