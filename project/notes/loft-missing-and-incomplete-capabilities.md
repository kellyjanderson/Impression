---
created: 2026-07-27
status: focused current-state inventory
---

# Loft Missing And Incomplete Capabilities

## Purpose

This note isolates Loft from the broader product reviews in `project/notes`.
It asks a narrower question:

> What can the current Loft system not yet do, or not yet do as a complete
> surface-first user feature?

The answer is not “everything described by an unfinished specification.”
Impression already has substantial Loft planning, topology, ambiguity,
fairness, surface execution, and diagnostic machinery. This inventory
distinguishes:

- **Verified unsupported:** the public documentation or runtime explicitly
  refuses the capability.
- **Partial:** meaningful implementation exists, but the capability is split,
  internal-only, diagnostic-only, or missing an end-to-end contract.
- **Opportunity:** a useful Loft capability for which no current public
  contract or implementation route was located in this review.

This is a focused source-and-document review, not a completed conformance audit
of every Loft specification.

## What Loft Already Implements

The following are important baselines and should not be described as missing:

- canonical `SurfaceBody` output through `Loft(...)`, `loft(...)`, and
  `loft_sections(...)`;
- explicit stations and path-following frames;
- multiple regions and holes;
- deterministic region and hole matching;
- bounded `1->N` and `N->1` region/hole split and merge resolution;
- deterministic, probabilistic, and interactive ambiguity controls;
- local and global fairness controls;
- point birth/death and correspondence-preserving resampling machinery;
- planner/executor separation with inspectable `LoftPlan` values;
- ruled, B-spline, NURBS, and sweep patch production at the planner level;
- planar end closure and shaped `flat`, `taper`, `dome`, and `slope` caps on
  the canonical surface path;
- an experimental mesh-only cap comparison path for `FLAT`, `CHAMFER`,
  `ROUND`, and `COVE`;
- branch-crossing and closure evidence diagnostics;
- bounded surface-native Loft CSG routes.

The practical gaps are therefore mostly about unifying, exposing, extending,
and physically qualifying existing machinery.

## Highest-Value Missing Or Incomplete Work

### 1. Unify Endcaps As One Surface-Native Feature

Status: **Partial.**

Endcaps are implemented, but the feature family is split:

- canonical surface Loft accepts `none`, `flat`, `taper`, `dome`, and `slope`;
- experimental `loft_endcaps(...)` accepts `FLAT`, `CHAMFER`, `ROUND`, and
  `COVE`, but returns `Mesh`;
- non-flat canonical caps explicitly require one connected region per profile;
- choosing a shaped cap at only one end currently turns closure on and makes
  the other unspecified end flat, rather than expressing a truly independent
  open/closed end policy.

The missing product feature is not “add endcaps.” It is:

- one cap vocabulary and parameter model;
- surface-native chamfer, round, and cove caps;
- independent start/end placement, including deliberately open opposite ends;
- multi-region and holed-profile behavior, or precise refusals where the
  topology is underconstrained;
- cap-to-side seam and orientation proof for every supported cap/profile
  combination;
- explicit continuity targets at the cap transition;
- parity fixtures against the useful experimental cap shapes.

Useful later extensions include custom cap profiles, asymmetric cap laws, and
per-region cap policy.

Evidence:

- `src/impression/modeling/loft.py::_validate_caps`
- `src/impression/modeling/loft.py::loft_endcaps`
- `src/impression/modeling/loft.py::_prepare_profile_sections_for_loft`
- `docs/modeling/loft.md#end-caps`

### 2. Make Correspondence Confidence A User Contract

Status: **Partial.**

The kernel has authored correspondence, inferred correspondence, ambiguity
records, protected resampling records, and explicit refusal paths. The public
behavior is still not strong enough to guarantee that sharp features remain
physically aligned in unnamed polygonal sections.

The missing capability is a complete correspondence contract:

- automatic matching only when the winning phase/order is measurably
  unambiguous;
- authored names treated as hard rails;
- mixed named and unnamed points inferred only between protected anchors;
- explicit refusal for equivalent rotations, unstable corner phase, or weak
  anchor evidence;
- user-visible selected rails, alternatives, confidence, and resampling
  diagnostics;
- geometry assertions for corner and landmark preservation, not only
  deterministic payload assertions.

This is the likely owner of the softened-corner rectangle/square concern
already recorded in the critical-review notes.

Evidence:

- `src/impression/modeling/loft.py::accept_or_refuse_inferred_correspondence`
- `src/impression/modeling/loft.py::resample_loop_correspondence`
- `project/notes/impression-critical-review-planning.md#correspondence-policy`

### 3. Add General Geometric Self-Intersection Validation And Prevention

Status: **Partial.**

Current Loft validity checks primarily consume planner/executor
`branch_crossing_count` evidence. That is valuable, but it is not a general
geometric proof that arbitrary loft patches do not intersect themselves.

Missing work:

- broad patch/patch self-intersection detection after execution;
- near-touch and tolerance-aware classification;
- pre-execution warnings for high twist, local inversion, foldover, and
  station collapse;
- optional constrained replanning or fairing to avoid detected intersections;
- export refusal when a claimed solid Loft is self-intersecting;
- section artifacts that locate the offending station span and surface region.

This should remain distinct from the future repair tool: detection and refusal
are required before automatic repair.

Evidence:

- `src/impression/modeling/loft.py::detect_loft_plan_self_intersections`
- `src/impression/modeling/loft.py::check_executed_loft_self_intersection_validity`

### 4. Resolve True Many-To-Many Topology Evolution

Status: **Verified unsupported.**

The current resolve mode supports `1->N` and `N->1` region/hole events. True
`N->M`, where both sides contain multiple competing regions or holes, remains
an explicit refusal.

Possible implementation levels:

1. require authored decomposition for every many-to-many interval;
2. generate deterministic candidate decompositions and ask the user to choose;
3. automatically accept only when one decomposition is uniquely supported;
4. eventually blend branch joints instead of realizing every event as a
   locally sharp synthetic transition.

The first useful feature is probably an authored decomposition API plus clear
visual diagnostics, not unrestricted automatic branching.

Evidence:

- `docs/modeling/loft.md#current-constraints`
- `src/impression/modeling/loft.py::loft_plan_sections`

### 5. Integrate Trajectory Guidance Into Loft Execution

Status: **Partial infrastructure; missing end-to-end Loft consumption.**

The repository has shared-trajectory candidate generation, confidence
assessment, and guidance records. `loft(...)` also accepts one placement path,
and the low-level planner can produce sweep patches from an explicit path.
Those are not yet the richer user feature described by trajectory-guided Loft:

- one explicit trajectory for the whole Loft evolution;
- different trajectories per region;
- different trajectories per correspondence track or landmark;
- inferred trajectories from dense stations;
- deterministic precedence between explicit stations and trajectory guidance;
- direct consumption of accepted trajectory evidence by the Loft planner.

The key distinction is that the current `path=` positions and frames stations.
A trajectory-guided Loft also controls how particular features travel between
stations.

Evidence:

- `project/future-features/trajectory-guided-loft-architecture.md`
- `src/impression/modeling/shared_trajectory.py`
- `src/impression/modeling/shared_guidance.py`
- `src/impression/modeling/loft.py::loft_plan_sections`

### 6. Complete Control-Station Inference As A Loft Tool

Status: **Partial infrastructure; no complete author-facing transformation.**

Control-station records, reduced progression bundles, preservation assessment,
and diagnostics exist. A complete Loft feature would:

- analyze a dense authored station sequence;
- identify topology-critical stations that must remain;
- infer hidden control stations where they improve shape;
- produce an accepted reduced progression;
- rebuild and compare the resulting Loft;
- report deviation, retained structure, and refusal reasons;
- let the author accept, reject, or edit the proposed result.

This reduces authored complexity. It is related to, but not the same as,
spanwise surface consolidation.

Evidence:

- `src/impression/modeling/control_station_inference.py`
- `project/release-0.1.0a/architecture/feature-05-control-station-inference-architecture.md`

### 7. Add Higher-Order Loft Continuity And Transition Control

Status: **Verified unsupported beyond positional continuity.**

The broader surface kernel records `C1`, `G1`, `C2`, and `G2` requests as not
yet implemented. Loft therefore cannot yet promise tangent- or
curvature-continuous joins between spans, at caps, or around branch
transitions.

Missing work:

- tangent-direction and tangent-magnitude constraints;
- curvature continuity constraints;
- measurable residuals and violation locators;
- local fairing with bounded deviation;
- blend patches where direct continuity is impossible;
- author controls for sharp, tangent, and curvature-preserving station joins.

This is one of the most important visual-quality improvements for organic and
industrial-design Loft work.

Evidence:

- `src/impression/modeling/surface.py::surface_continuity_support`
- `project/release-0.1.0a/architecture/higher-order-seam-continuity-architecture.md`

### 8. Consolidate Dense Station Spans Into Better Surface Structure

Status: **No implementation located.**

Current Loft primarily realizes local station-to-station intervals. Dense
authored stations can therefore produce more patches and seams than the shape
semantically needs.

The existing future-feature documents identify three possible products:

- inline multi-station span recognition in the planner;
- post-Loft exact or approximate patch consolidation;
- repair/reconstruction of noisy or damaged Loft-like spans.

The recommended order is:

1. post-Loft analysis and exact-equivalence reporting;
2. post-Loft exact consolidation;
3. bounded refitting with error metrics;
4. only then consider changing the core planner or repairing foreign geometry.

Evidence:

- `project/future-features/spanwise-loft-consolidation-architecture.md`
- `project/future-features/spanwise-loft-postprocessing-optimization-architecture.md`
- `project/future-features/spanwise-loft-repair-tool-architecture.md`

### 9. Expose Patch-Family Intent Through The Main Loft APIs

Status: **Implemented at the planner level, incomplete as a public Loft
feature.**

`loft_plan_sections(...)` accepts smooth intent, rational intent and weights,
a sweep path and frame policy, and an explicit patch-family request. The main
`Loft(...)`, `loft(...)`, and `loft_sections(...)` signatures do not expose
those controls.

Possible completion:

- a small typed `LoftSurfaceOptions` object instead of more top-level
  parameters;
- automatic family selection with a visible reason;
- explicit per-span family overrides;
- public B-spline/NURBS degree, knot, fit/interpolate, and weight policy;
- refusal when adjacent family choices cannot meet seam or continuity
  requirements.

This is both a missing capability surface and an opportunity to reduce the
existing Loft parameter train.

Evidence:

- `src/impression/modeling/loft.py::loft_plan_sections`
- `src/impression/modeling/loft.py::loft`
- `project/notes/code-quality-principal-engineer-review.md#p0-the-loft-interface-is-a-parameter-train-repeated-through-wrapper-layers`

## Additional Loft Opportunities

These were not found as current public contracts and should be treated as
design candidates, not verified roadmap commitments.

### Closed Or Periodic Loft Progression

Join the final station back to the first with explicit phase, frame, seam, and
continuity policy. This would support toroidal and closed-loop Loft structures
without requiring a separate sweep construction.

### Per-Span Evolution Laws

Allow authored interpolation laws for scale, rotation, offset, and profile
morphing between stations: linear, eased, held, overshooting, or curve-driven.
Today authors largely communicate these behaviors by adding stations.

### Explicit Guide Rails

Allow one or more spatial curves to constrain named landmarks or
correspondence tracks. This is the author-controlled counterpart to inferred
trajectory guidance.

### Local Loft Editing

Provide stable selection of stations, regions, loops, tracks, and spans so an
author can insert a station, move a rail, change continuity, or replace one
profile without rebuilding unrelated Loft identity.

### Thickness And Hollow-Loft Construction

Create a shell/thicken workflow that offsets a Loft with explicit behavior at
caps, tight curvature, branches, and self-intersections. This is broader than
ordinary end closure and should reuse general surface offset/sewing truth.

## Recommended Loft-Only Priority

For the most useful near-term Loft improvement sequence:

1. unify and surface-promote endcaps;
2. close the correspondence confidence and landmark-preservation contract;
3. add geometric self-intersection/closure qualification;
4. expose authored decomposition for many-to-many topology;
5. connect trajectory and control-station evidence to actual Loft execution;
6. add higher-order continuity;
7. add spanwise consolidation;
8. then pursue periodic Loft, guide rails, per-span laws, and local editing.

The first three make existing Loft output more trustworthy. The next four make
it more expressive. The final opportunities make it substantially more
authorable.

## Related Notes

- [Critical Review Planning](impression-critical-review-planning.md)
- [Technical And Industry Completeness Review](technical-and-industry-completeness-principal-engineer-review.md)
- [Defined But Unimplemented Functionality](../adhoc/2026-07-23-defined-but-unimplemented-functionality.md)
