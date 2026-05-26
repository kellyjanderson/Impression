# 03 Orientation-Safe Motor Mount Or Fairing Spine

## Objective

Model a lofted fairing that must keep a meaningful orientation along a curved
travel path so the user can immediately see the difference between old scalar
progression and the new path-backed progression semantics.

The object should feel like a real engineering part:

- a compact motor mount fairing
- or an intake / cable-routing fairing for a composite body

## Release Delta

- `0.0.3a2`
  Loft progression mostly reads as ordering and station placement.
- `0.1.0.a`
  Progression becomes an owned semantic carrier for path, provenance,
  attachment, transport policy, and deferred twist/scale slots.

## Visual Target

The part should have an obvious “this could twist wrong” asymmetry:

- one side flatter
- one side fuller
- one bolt-boss shoulder or dorsal ridge

If orientation drifts, the viewer should notice immediately.

## Real-World Need Links

- [Keep profile orientation during a loft](https://forum.onshape.com/discussion/28189/keep-profile-orientation-during-a-loft)
- [Guide curve or more profiles for lofting.](https://forum.onshape.com/discussion/8422/guide-curve-or-more-profiles-for-lofting)
- [Loft Not Following Guide Curves](https://forums.autodesk.com/t5/fusion-design-validate-document/loft-not-following-guide-curves/td-p/9370557)

## Planned Impression Calls

Geometry and path calls:

- `Path2D.from_points(...)`
- `PlanarShape2D(...)`
- `Bezier3D(...)`
- `Path3D(...)`
- `BSpline3D(...)`
- `Station(...)`
- `Loft(...)`

Progression and guidance calls:

- `PathBackedProgression(...)`
- `ProgressionProvenanceRecord(...)`
- `ProgressionStationAttachment.from_station(...)`
- `ProgressionTransportPolicy(...)`
- `ProgressionTwistSemanticSlot(...)`
- `ProgressionScaleSemanticSlot(...)`
- `ExplicitSharedGuidanceAttachmentRecord(...)`
- `ExplicitSharedGuidancePlannerConsumption(...)`

## Model Decomposition

The main loft should use one closed outer profile family with no holes.

Optional supporting hardware should not be part of the first pass.
The fairing itself is enough.

## Profile Strategy

Use one helper:

`make_fairing_profile(width, height, dorsal_bias, flat_side_bias, chin_depth, color)`

The section should be asymmetrical:

- top ridge slightly offset
- lower inside corner flatter
- outside flank fuller and rounder

Good profile family:

- rounded triangular-oval hybrid
- not just a rectangle
- not just a circle

## Spine Plan

Use one authored curved spine with clear yaw and pitch change:

- start near horizontal
- rise and sweep outward
- relax back toward neutral near the end

Suggested control points for a single-curve first pass:

- `p0 = (0, 0, 0)`
- `p1 = (12, -2, 18)`
- `p2 = (28, 8, 42)`
- `p3 = (44, 6, 60)`

This gives enough curvature for orientation drift to matter without becoming
cartoonish.

## Important Station Plan

Suggested progression and station family:

| Station ID | `t` | Width | Height | Dorsal Bias | Chin Depth | Notes |
|---|---:|---:|---:|---:|---:|---|
| `mount_face` | `0.00` | `34` | `24` | `0.18` | `0.12` | flange-facing start |
| `clearance_body` | `0.18` | `31` | `23` | `0.20` | `0.14` | body starts to lift |
| `mid_bend` | `0.42` | `26` | `22` | `0.26` | `0.18` | strongest orientation read |
| `sweep_exit` | `0.70` | `22` | `19` | `0.22` | `0.12` | begins slimming |
| `tip_fairing` | `1.00` | `14` | `12` | `0.16` | `0.08` | clean termination |

Station-authoring posture:

- build explicit `Station(...)` records
- derive `u`, `v`, `n` from the intended orientation story rather than relying
  only on accidental section rotations
- also create `ProgressionStationAttachment` records from those same stations

## Modeling Sequence

1. Author the fairing spine with `Bezier3D(...)` inside `Path3D(...)`.
2. Build the semantic progression with `PathBackedProgression(...)`.
3. Add explicit provenance and `parallel_transport` policy.
4. Add deferred twist and scale semantic slots.
5. Author the asymmetric profile family at the five important stations.
6. Build explicit `Station(...)` frames and attach them to progression.
7. Execute the hero `Loft(...)` from the explicit stations.
8. If needed, create one explicit shared-guidance attachment over the
   middle-to-tip band and record planner consumption separately.

## Demo Composition

This example should eventually show:

- the fairing hero render
- the spine and station-frame overlay
- a compact progression semantic panel listing:
  - provenance
  - transport policy
  - twist slot status
  - scale slot status

## Visual Review Loop For Execution

When executing this example:

- render the fairing with the spine visible
- render a second view with frame arrows at the five authored stations
- ask the visual-review sub-agent whether the object still reads as one
  consistently oriented engineering fairing
- if the ridge appears to roll unintentionally, adjust the station frame family
  before touching any guidance logic

The primary visual gate is orientation trustworthiness.

## Risks To Guard Against

- too much symmetry makes the progression upgrade visually invisible
- too much curvature makes the part feel decorative rather than functional
- too much section shrinkage makes it read as a horn, not a fairing
