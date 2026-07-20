# 04 Contour Stack Simplifier For Reverse-Engineered Parts

## Objective

Model a reverse-engineering cleanup workflow from an intentionally over-sampled
contour stack to a replayable reduced progression bundle with explicit retained
station classes and structural-preservation refusal.

This example should feel like engineering work, not product styling.

## Release Delta

- `0.0.3a2`
  Dense contour evidence can be lofted, but simplification is manual and does
  not produce durable machine-readable reduction artifacts.
- `0.1.0.a`
  The same contour stack can generate descriptor bands, fitted candidates,
  reduced progression bundles, retained station records, and explicit accepted
  or refused outcomes.

## Visual Target

Use a scan-derived intake shroud or cast transition shell.

The shape should include:

- a mounting flange at one end
- a full belly in the middle
- a keyed side flat or notch that appears only in part of the run
- a smaller oval outlet

That gives the example something topology-meaningful to preserve.

## Real-World Need Links

- [Self intersecting loft when using offset contour](https://forums.autodesk.com/t5/fusion-design-validate-document/self-intersecting-loft-when-using-offset-contour/td-p/10766237)
- [Creased edges after lofting](https://forums.autodesk.com/t5/fusion-design-validate-document/creased-edges-after-lofting/td-p/9349250)
- [Loft generates surface but not solid](https://forum.onshape.com/discussion/25049/loft-generates-surface-but-not-solid)

## Planned Impression Calls

Geometry construction calls:

- `Path2D.from_points(...)`
- `PlanarShape2D(...)`
- `Station(...)`
- `Loft(...)`

Inference calls:

- `PathBackedProgression(...)`
- `ProgressionProvenanceRecord(...)`
- `ProgressionStationAttachment.from_station(...)`
- `prepare_dense_loft_fit_descriptors(...)`
- `build_curve_intent_descriptor_families(...)`
- `assemble_span_local_curve_intent_evidence(...)`
- `classify_curve_intent_candidate(...)`
- `generate_station_derived_curve_fit_candidates(...)`
- `compare_station_derived_curve_fit_candidates(...)`
- `ReducedProgressionBundle.from_progression(...)`
- `RetainedStationRecord(...)`
- `assess_control_station_inference(...)`

Diagnostics and reporting calls:

- `SharedInferenceDiagnosticBundle.from_station_fit(...)`
- `DeveloperInferenceInspection.from_bundle(...)`
- `DownstreamInferenceReport.from_bundle(...)`

## Model Decomposition

One outer boundary only.
No through-holes.
No branch topology.

The topology-critical feature should be a side flat or keyed indentation that
changes silhouette and must remain visible in retained station classification.

## Profile Strategy

Use one helper:

`make_shroud_section(width, height, flat_side_depth, notch_depth, corner_softness, color)`

Section intent:

- rounded rectangle to soft oval hybrid
- one side slightly flattened
- notch or key only in a middle band
- outlet sections become more oval and cleaner

## Contour Stack Plan

This example should start from a dense evidence stack, not from the sparse
answer.

Suggested dense count:

- `18` to `24` sections

Suggested semantic bands:

| Band | Approx `t` Range | Intended Meaning |
|---|---|---|
| flange | `0.00` to `0.12` | wide, stiff mounting region |
| belly growth | `0.12` to `0.36` | main body expansion |
| keyed body | `0.36` to `0.62` | flat side and notch visible |
| taper | `0.62` to `0.84` | controlled narrowing |
| outlet | `0.84` to `1.00` | clean oval exit |

Important retained targets:

| Retained ID | Kind | Reason |
|---|---|---|
| `flange_start` | topology | mounting signature begins |
| `flange_end` | topology | flange exits |
| `key_start` | topology | keyed side feature appears |
| `belly_max` | control or topology | largest body read |
| `key_end` | topology | keyed side feature disappears |
| `outlet_root` | topology | outlet transition begins |
| `outlet_tip` | topology | terminal outlet shape |

## Modeling Sequence

1. Author the dense contour stack directly.
2. Build a dense hero loft only for comparison reference.
3. Wrap the dense station path in `PathBackedProgression(...)`.
4. Attach all dense stations to progression.
5. Prepare dense fit descriptors.
6. Build descriptor families.
7. Assemble span-local evidence.
8. Classify curve intent posture.
9. Generate and compare station-derived candidates.
10. Build a reduced progression bundle from the accepted candidate if valid.
11. Create retained station records with explicit `topology` vs hidden-control
    meaning.
12. Assess structural preservation.
13. Generate diagnostic and downstream reports.

## Accepted And Refused Variants

This example should plan for two outcomes:

Accepted case:

- keyed feature survives
- reduced progression remains replayable
- retained station list is understandable

Refused case:

- an over-aggressive reduction tries to bridge across the keyed band
- structural-preservation assessment refuses the simplification

Both outcomes should eventually be shown.

## Demo Composition

The final example should present:

- dense section stack
- accepted reduced progression replay
- refused reduction panel

The most important visual should be a side-by-side silhouette that proves the
accepted replay still preserves the key band while the refused case would not.

## Visual Review Loop For Execution

When we later build this example:

- render contour-stack silhouettes before inference
- render reduced replay silhouettes after inference
- ask the visual-review sub-agent whether the keyed side feature still reads as
  intentional structure
- if the keyed feature disappears visually before the refusal gate catches it,
  strengthen the keyed band geometry or raise the required retained stations

## Risks To Guard Against

- too few dense sections make the reverse-engineering story unconvincing
- too subtle a keyed feature makes refusal visually meaningless
- too dramatic a keyed feature turns the model into an unrealistic toy part
