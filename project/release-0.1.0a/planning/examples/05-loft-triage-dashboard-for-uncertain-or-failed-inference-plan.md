# 05 Loft Triage Dashboard For Uncertain Or Failed Inference

## Objective

Create a diagnostic-first example family that proves the new inference stack is
inspectable rather than magical.

This should not be one single hero part.
It should be a compact suite of intentionally chosen loft cases with three
outcomes:

- accepted
- uncertain
- refused

## Release Delta

- `0.0.3a2`
  Strong surfaced lofts and planner artifacts exist, but the new shared
  inference bundle and reporting layers do not.
- `0.1.0.a`
  Accepted, uncertain, and refused inference outcomes become explicit,
  reportable, and inspectable.

## Visual Target

This should eventually look like a serious diagnostic board, not a gallery of
pretty shapes.

Recommended composition:

- one clean “accepted” object
- one visually plausible but less confident “uncertain” object
- one visibly problematic “refused” object
- one right-hand diagnostic panel per case

## Real-World Need Links

- [Loft fails when guide curve is too severe](https://forum.onshape.com/discussion/23829/loft-fails-when-guide-curve-is-too-severe)
- [Loft did not generate properly: Current selections would create a self-intersecting body](https://forum.onshape.com/discussion/29436/loft-did-not-generate-properly-current-selections-would-create-a-self-intersecting-body)
- [Loft failure when I use a third guide](https://forum.onshape.com/discussion/27044/loft-failure-when-i-use-a-third-guide)

## Planned Impression Calls

Geometry and planner calls:

- `Loft(...)`
- `loft_plan_ambiguities(...)`
- `loft_plan_sections(...)`

Inference calls:

- `prepare_dense_loft_fit_descriptors(...)`
- `generate_station_derived_curve_fit_candidates(...)`
- `compare_station_derived_curve_fit_candidates(...)`
- `generate_shared_whole_loft_trajectory_candidates(...)`
- `assess_shared_whole_loft_trajectory_candidates(...)`
- `ReducedProgressionBundle.from_progression(...)`
- `assess_control_station_inference(...)`

Diagnostic calls:

- `SharedInferenceDiagnosticBundle(...)`
- `SharedInferenceDiagnosticBundle.from_station_fit(...)`
- `DeveloperInferenceInspection.from_bundle(...)`
- `DownstreamInferenceReport.from_bundle(...)`

## Case Family

### Case A — Accepted

Use a shoulder bottle or clean fairing family.

Intent:

- enough evidence
- stable fitted candidate
- clear retained structure

Recommended geometric source:

- reuse the bottle family from Example `02`

### Case B — Uncertain

Use a curved fairing or shell with a believable but noisy middle band.

Intent:

- fit candidate exists
- residuals are acceptable but not strong
- guidance may be attached
- report must say `uncertain`

Recommended geometric source:

- reuse the fairing family from Example `03`
- add a slightly underconstrained middle station band

### Case C — Refused

Use a contour-stack simplification case where topology-critical structure would
be lost.

Intent:

- the system should refuse reduction or trajectory promotion
- report must be explicit, not vague

Recommended geometric source:

- reuse the keyed shroud family from Example `04`

## Important Geometry Notes Per Case

Accepted case:

- keep sparse and elegant
- no branch topology
- visual result should be obviously clean

Uncertain case:

- asymmetry must remain visible
- evidence should be close enough that the model does not look obviously broken
- uncertainty should come from inference strength, not from ugly geometry

Refused case:

- keyed or topology-critical feature must be obvious enough that refusal feels
  justified

## Dashboard Artifact Plan

For each case, plan to produce:

- hero render
- station/evidence overlay
- one shared diagnostic bundle
- one developer inspection record
- one downstream report

The dashboard should compare the cases in the same layout so the status change
is the message.

## Modeling Sequence

1. Build the accepted geometry family.
2. Build the uncertain geometry family.
3. Build the refused geometry family.
4. For each family, prepare dense descriptors and candidate fits.
5. For accepted and uncertain cases, assess whole-loft trajectory posture.
6. For the refused case, assess control-station structural preservation or fit
   refusal explicitly.
7. Build a shared inference diagnostic bundle for every case.
8. Build one developer-facing inspection record from each bundle.
9. Build one downstream-facing report from each bundle.
10. Lay out the final board so the geometry and reporting are always adjacent.

## Planned File And Helper Structure For Later Execution

Suggested later implementation layout:

- one top-level `build()` that assembles the dashboard scene
- one helper per case:
  - `build_case_accepted()`
  - `build_case_uncertain()`
  - `build_case_refused()`
- one helper to convert diagnostic records into presentable text/labels
- one helper to place panels in a deterministic grid

## Visual Review Loop For Execution

This example especially needs the visual-review sub-agent.

Execution loop:

1. render the geometry-only board
2. ask whether the three cases are visually distinguishable
3. render the geometry-plus-diagnostic board
4. ask whether the diagnostic outcome matches the geometric intuition
5. if the refused case does not look risky, strengthen the risky geometry
6. if the uncertain case looks obviously broken, soften it until “uncertain”
   feels honest

## Risks To Guard Against

- accepted and uncertain look too similar
- uncertain and refused look equally broken
- the board over-indexes on text and under-indexes on geometry
- the reporting layer becomes dense enough that the viewer stops reading
