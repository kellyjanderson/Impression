# 02 Tight-Shoulder Bottle Without Rail Explosion

## Objective

Model a premium bottle or decanter with a difficult shoulder transition that
users would normally solve with too many rails, too many intermediate sections,
or fragile hand-tuned curves.

The example should prove that `0.1.0.a` can:

- reason about shared trajectory intent
- keep path/travel semantics explicit
- expose fit drift and uncertainty
- attach explicit guidance only when needed

## Release Delta

- `0.0.3a2`
  The bottle can be lofted, but shoulder quality is mostly a manual station and
  profile-spacing problem.
- `0.1.0.a`
  The same form can be described with a cleaner shared-trajectory story,
  explicit guidance attachments, and fit/residual reporting.

## Visual Target

The bottle should feel like a cosmetic or high-end spirits package:

- broad stable base
- full cylindrical or slightly oval body
- aggressive but graceful shoulder break
- slender neck
- clean lip

The shoulder must be the star.
It should feel taut, not pinched and not lumpy.

## Real-World Need Links

- [How to achieve this loft on this bottle?](https://forums.autodesk.com/t5/fusion-design-validate-document/how-to-achieve-this-loft-on-this-bottle/td-p/9609170)
- [Tight curve lofting improvment](https://forums.autodesk.com/t5/fusion-360-ideastation-archived/tight-curve-lofting-improvment/idi-p/8647961)
- [Loft issue with Bottle design](https://forums.autodesk.com/t5/autocad-forum/loft-issue-with-bottle-design/td-p/8127558)

## Planned Impression Calls

Primary geometry calls:

- `make_circle(...)`
- `PlanarShape2D(...)`
- `as_section(...)`
- `Loft(...)`

Travel and fit calls:

- `BSpline3D(...)`
- `PathBackedProgression(...)`
- `ProgressionProvenanceRecord(...)`
- `prepare_dense_loft_fit_descriptors(...)`
- `generate_shared_trajectory_curve_fit_candidates(...)`
- `compare_shared_trajectory_curve_fit_candidates(...)`
- `generate_shared_whole_loft_trajectory_candidates(...)`
- `assess_shared_whole_loft_trajectory_candidates(...)`
- `ExplicitSharedGuidanceAttachmentRecord.from_candidate(...)`
- `ExplicitSharedGuidancePlannerConsumption(...)`

Diagnostic calls:

- `SharedInferenceDiagnosticBundle.from_station_fit(...)`
- `DownstreamInferenceReport.from_bundle(...)`

## Model Decomposition

Use a single connected ring profile at every station:

- outer wall
- one inner void hole

This is important because the example is about shoulder quality, not topology
branching.

Non-flat cap behavior should stay on one connected region only.

## Profile Strategy

Each profile should come from one helper:

`make_ring_profile(outer_radius_x, outer_radius_y, wall_thickness, color)`

The profile should stay nearly circular through the base and lower body, then
become slightly oval in the shoulder so the transition has a visible designer
intent.

The neck should return closer to circular.

## Progression And Trajectory Plan

Use one vertical spine with a subtle rear lean:

- `BSpline3D` or `Path3D`
- slight rearward offset by the neck so the bottle does not feel lathe-perfect

That lean should be mild enough to preserve a packaging read.

Suggested progression:

| Station ID | `t` | Role |
|---|---:|---|
| `base_outer` | `0.00` | footing |
| `heel_round` | `0.10` | soft base lift |
| `body_low` | `0.22` | body expansion |
| `body_max` | `0.38` | max body diameter |
| `shoulder_start` | `0.56` | first narrowing |
| `shoulder_break` | `0.68` | strongest shoulder event |
| `neck_base` | `0.80` | neck established |
| `neck_mid` | `0.90` | stable neck |
| `lip` | `1.00` | finish |

## Important Station Plan

Suggested dimensions:

| Station ID | X Dia | Y Dia | Inner Dia Offset | Notes |
|---|---:|---:|---:|---|
| `base_outer` | `56` | `56` | `3.2` wall | thick visual footing |
| `heel_round` | `60` | `60` | `2.8` wall | shoulder not started yet |
| `body_low` | `68` | `68` | `2.4` wall | body fullness |
| `body_max` | `74` | `72` | `2.2` wall | slight oval begins |
| `shoulder_start` | `63` | `61` | `2.1` wall | early pull-in |
| `shoulder_break` | `40` | `42` | `2.0` wall | critical station |
| `neck_base` | `30` | `31` | `1.8` wall | neck established |
| `neck_mid` | `24` | `24` | `1.8` wall | stable neck |
| `lip` | `28` | `28` | `2.0` wall | finish flare |

Dense evidence plan:

- build an oversampled shoulder band from `t=0.48` to `t=0.84`
- use `10` to `14` local shoulder sections even though the hero model will
  visually present only the sparse family
- treat this dense shoulder band as the evidence that the shared-trajectory
  lane is trying to explain

## Modeling Sequence

1. Author sparse ring sections for the hero bottle.
2. Build a surfaced `Loft(...)` hero version from those sparse stations.
3. Create a second dense shoulder stack by interpolating only the shoulder and
   neck band.
4. Wrap the bottle spine in `PathBackedProgression(...)`.
5. Prepare dense descriptors from the dense shoulder stack.
6. Generate and compare shared-trajectory candidates.
7. Lift the winning fit to whole-loft shared trajectory candidates.
8. Assess accepted, uncertain, or refused whole-loft posture.
9. If accepted or uncertain, create an
   `ExplicitSharedGuidanceAttachmentRecord`.
10. Record bounded planner consumption with
    `ExplicitSharedGuidancePlannerConsumption`.
11. Build the downstream report for the example panel.

## Demo Composition

This example should ultimately show three panels:

- released-style dense shoulder section stack
- current-version sparse bottle hero
- guidance and residual panel

The hero image should be the sparse bottle on a clean studio background.

## Visual Review Loop For Execution

When executing this plan:

- render a side silhouette and a reflective three-quarter view
- ask the visual-review sub-agent whether the shoulder reads premium,
  pinched, or plastic
- if it reads pinched, widen `shoulder_start`
- if it reads swollen, narrow `body_max` and move `shoulder_break` slightly
  upward
- do not proceed to the guidance-report panel until the silhouette reads like a
  believable package

## Risks To Guard Against

- too much ovality makes the bottle look warped
- too little ovality makes the trajectory story visually weak
- neck reduction too early makes the bottle feel toy-like
- lip flare too large makes the shoulder look accidental
