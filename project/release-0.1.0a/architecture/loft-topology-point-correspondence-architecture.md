# Loft Topology Point Correspondence Architecture

## Overview

Loft currently manages region and loop correspondence, but it does not fully
manage point correspondence inside each topology loop.

That is a critical flaw.

The current system stores topology primarily as ordered point arrays:

- `Section`
- `Region`
- `Loop`

Loft then canonicalizes, anchors, and resamples those loops during planning.
That gives deterministic geometry, but not owned point identity. It means the
planner can know which region or loop corresponds, while still guessing which
point on that loop corresponds to which point on the next station.

The fix is to make point correspondence a topology-owned concept and make loft
consume it explicitly.

The product target is authored models. That matters: loft should prefer simple
author-provided rails over automatic ambiguity solving. Existing ambiguity
machinery can remain for compatibility, diagnostics, and best-effort recovery,
but new correspondence design should assume the author can provide enough
intent to make the mapping deterministic.

## Decision: Authored Rails First

Impression is targeting authored models for loft correspondence.

Going forward, loft correspondence should be designed around explicit user
intent, not automatic ambiguity resolution as the primary path.

The system should expect and preserve simple authored rails:

- the start of a supplied point array is intentional
- the start of a supplied closed path is the loop anchor unless the user opts
  into automatic anchoring
- the direction of a supplied path or point array is intentional
- named topology entities are authoritative correspondence hints
- explicit point or correspondence ids override geometric heuristics

Existing ambiguity and automatic disambiguation machinery should remain in
place for compatibility, diagnostics, and fallback behavior. It should not be
expanded as the main solution for authored-model correspondence.

That does not mean the user should do every piece of correspondence work
manually. When the system can deliver a large usability win with a
high-confidence automatic resolution, it should. The boundary is confidence and
explainability:

- use authored rails first
- automatically resolve obvious or strongly constrained cases
- record whether the result was rail-driven, inferred, or mixed
- surface diagnostics when inference mattered
- refuse or ask for better rails when the correspondence remains genuinely
  ambiguous

The preferred future behavior is not "never infer." It is "infer when the
system has enough evidence, and do not silently pretend a guess was authored
truth."

## Current Behavior

Current relevant code paths:

- `src/impression/modeling/topology.py`
  - `Loop` stores only `points`.
  - `anchor_loop(...)` chooses a deterministic local loop start.
  - `resample_loop(...)` resamples by arc length.
- `src/impression/modeling/loft.py`
  - `Station` carries `predecessor_ids` and `successor_ids`, but only per
    normalized region.
  - `_canonicalize_section_for_loft(...)` reorders regions and holes and
    anchors each loop independently.
  - `_section_to_region_loops(...)` resamples each loop independently.
  - `PlannedLoopPair` stores a previous loop and current loop with equal sample
    count, but no point-level mapping record.
  - executors connect equal sample indices directly.

This works for simple symmetric or similarly authored profiles. It is not a
complete topology correspondence model.

## Failure Mode

The failure is not merely that a heuristic can pick the wrong start vertex. The
deeper issue is that topology does not own a point correspondence graph.

Consequences:

- a semantic corner, notch, shoulder, seam point, or feature vertex can drift to
  a different sampled point after independent anchoring/resampling
- identical topologies with different authored point order can produce a twist
  or phase shift
- holes and outer loops can be paired correctly while their internal point
  tracks are still wrong
- dense station evidence cannot reliably say "this authored point persisted"
  across stations
- future B-spline/control-station inference can fit correspondence tracks only
  after loft has already guessed them
- diagnostics can report region/loop decisions, but not point-track decisions
- point birth and point death inside an otherwise stable loop cannot be
  represented explicitly

## Required Design Principle

Topology owns point identity.

Authored intent wins over inference.

Loft may infer missing correspondence, but inference is secondary. Inferred
correspondence must become an explicit planner product with diagnostics. It must
not remain implicit in array order.

Useful automation is encouraged when it reduces author burden without hiding
uncertainty.

## Authored Rails

The clean solution is to define a small set of author-facing rails that loft can
trust.

### Array And Path Start Is Intentional

When the user supplies an ordered point array, closed path, or sampled path, the
start of that array/path is meaningful unless the user explicitly opts into
automatic anchoring.

For closed loops:

- point `0` is the authored anchor
- order defines traversal direction
- cyclic rotation is not allowed to silently change semantic correspondence
- reversal is not allowed unless explicitly requested or required by winding
  normalization, and any reversal must be recorded

For open paths:

- point `0` is the start anchor
- the final point is the end anchor
- path direction is meaningful

### Named Topology Entities

Topology should allow user-defined names for:

- regions
- loops
- points
- seams
- feature vertices
- correspondence tracks

Names are rails. When names are present, loft should use them before geometric
matching.

Examples:

- `outer`
- `inner-left`
- `canopy-front`
- `wing-root-leading-edge`
- `station-seam`
- `track:A`

### Explicit Correspondence Names

Users should be able to assign the same correspondence id to related points
across stations.

Example intent:

```text
station 0: point "canopy-front" -> correspondence "canopy-front"
station 1: point "canopy-front" -> correspondence "canopy-front"
station 2: point "canopy-front" -> correspondence "canopy-front"
```

Loft should treat that as stronger than nearest-point, angle, area, symmetry,
or other inferred scoring.

### Author-Controlled Anchor Policy

Loops should carry an anchor policy:

- `authored`: preserve point `0` as the semantic anchor
- `named`: use a named point as the semantic anchor
- `auto`: allow current geometric anchoring heuristics

The default for authored point arrays and paths should be `authored`.

The default for generated shapes may remain `auto` unless the shape constructor
can provide meaningful named anchors.

## Decision: B-Spline2D Helps, But Is Not The Correspondence Layer

`BSpline2D` should be accepted as one useful authored curve input for topology
paths, but it should not be treated as the solution to point correspondence by
itself.

B-spline curves help with:

- compact smooth path authoring
- deterministic curve evaluation
- deterministic parameter-space sampling
- preserving continuous curve truth longer than a sampled polyline
- carrying parameter landmarks such as `u=0.0`, `u=0.25`, or named knots
- fitting or reducing dense evidence later

B-spline curves do not automatically solve topology correspondence because:

- control points usually do not lie on the curve
- equal control-point indices are not equal curve landmarks
- equal sample counts do not imply equal semantic points
- closed or periodic curves can still have arbitrary phase unless anchored
- two different B-spline parameterizations can describe visually similar loops
  with different parameter meaning
- topology still needs names, anchors, and point/parameter tracks to know what
  should correspond

The right model is:

```text
authored curve/path input
-> topology path with named anchors and parameter landmarks
-> topology loop with point/landmark identity
-> loft correspondence map
-> correspondence-preserving samples for execution
```

So `BSpline2D` is a high-quality input carrier, not a replacement for topology
correspondence records.

## Better User-Facing Topology Input

The cleaner user-facing solution is a topology path model that can wrap simple
point arrays, `Path2D`, `BSpline2D`, and generated shapes while preserving
authored rails.

## Interface Direction

The kernel model can be rich without forcing every user to author verbose
records.

The public interface should have layers:

```text
helpers
-> topology path builder
-> topology path / landmark records
-> section / region / loop topology
-> loft planner correspondence maps
```

### Simple Helpers

Common cases should stay lightweight:

```python
outer = TopologyPath.from_points(
    [
        ("bottom-left", (0.0, 0.0)),
        ("bottom-right", (4.0, 0.0)),
        ("top-right", (4.0, 2.0)),
        ("top-left", (0.0, 2.0)),
    ],
    closed=True,
    anchor="bottom-left",
)
```

Unnamed points should still work:

```python
outer = TopologyPath.from_points(
    [(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)],
    closed=True,
)
```

For unnamed authored arrays, point `0` remains the anchor and order remains
intentional.

### Builder Helpers For Richer Cases

Point birth/death and segment rails should be available without making users
construct internal records directly:

```python
outer = TopologyPath.closed(anchor="bottom-left")
outer.point("bottom-left", (0.0, 0.0), correspond="bottom-left")
outer.point("bottom-right", (4.0, 0.0), correspond="bottom-right")
outer.birth_span(
    parent=("bottom-right", "top-right"),
    points=[
        ("cutout-right-bottom", (4.0, 1.0)),
        ("cutout-inner-start", (3.2, 1.0)),
    ],
)
outer.birth_arc(
    name="cutout-rounded-inner-corner",
    parent=("top-right", "top-left"),
    start=(3.2, 1.0),
    end=(3.0, 1.4),
    radius=0.2,
)
outer.point("top-left", (0.0, 2.0), correspond="top-left")
```

The builder should emit topology path records with landmarks and lifecycle
events. It should not bypass the same data model used by lower-level APIs.

### Curve Segment Inputs

Curves should be accepted as topology path segments:

```python
outer.segment(
    name="crown",
    curve=BSpline2D(...),
    landmarks=[
        Landmark("crown-start", parameter=0.0, correspond="crown-start"),
        Landmark("crown-peak", parameter=0.5, correspond="crown-peak"),
        Landmark("crown-end", parameter=1.0, correspond="crown-end"),
    ],
)
```

For curve inputs, correspondence should attach to parameters and landmarks, not
to raw sample indices.

### Generated Shape Helpers

Generated shapes should provide default rails:

```python
outer = TopologyPath.named_rect(
    width=4.0,
    height=2.0,
    anchor="bottom-left",
)
```

The helper can create useful names such as:

- corners
- side midpoints
- side segment names

Users can override or add names when needed.

### Escape Hatch

Advanced users should be able to construct topology path, landmark,
correspondence, and lifecycle records directly.

The helper API must compile to those records so diagnostics and tests can
inspect the same truth regardless of authoring style.

### Topology Path

A topology path should preserve:

- source curve segments
- closure policy
- authored start
- traversal direction
- named landmarks
- point or parameter anchors
- segment ids or names
- sampling policy
- correspondence ids

For polyline input, landmarks are points.

For curve input, landmarks are curve parameters.

For generated shapes, constructors should provide useful default named
landmarks where possible.

### Landmark Records

Topology landmarks should be first-class records.

Required fields:

- landmark id
- optional user-facing name
- source segment id
- source parameter or point ordinal
- role, such as corner, seam, feature, tangent transition, or sample
- correspondence id
- protection policy during resampling

This is the missing bridge between smooth curve input and point-level loft
correspondence.

### Segment-Level Rails

For paths made of multiple segments, segment identity is useful correspondence
evidence.

Examples:

- `forebody-top`
- `canopy-arc`
- `wing-root-shoulder`
- `tailcone-bottom`

If two stations have matching segment names, loft can align corresponding
segment parameter ranges before sampling interior points.

### Generated Shape Rails

Generated shapes should not be anonymous when simple named rails are obvious.

Examples:

- rectangle:
  - `right-mid`
  - `top-mid`
  - `left-mid`
  - `bottom-mid`
  - corners
- circle:
  - authored start
  - quadrant landmarks
  - optional named seam
- rounded rectangle:
  - straight segment names
  - corner arc names
  - tangent transition landmarks

These rails give users helpful defaults without requiring manual point tagging
for every common shape.

### Correspondence From Path Families

The preferred order for solving point correspondence should be:

1. explicit correspondence ids
2. matching landmark names
3. matching segment names plus segment-local parameters
4. authored start and direction
5. generated shape default rails
6. high-confidence geometric inference
7. refusal / request for better rails

This order keeps authored intent primary while still letting the system do
useful work automatically.

## Resolved Defaults For Manifest Blockers

These decisions close the architecture-level questions that would otherwise
force final implementation specs to guess.

### Public API Naming

Topology path public APIs should follow existing modeling style:

- factory methods use `from_*` names:
  - `TopologyPath.from_points`
  - `TopologyPath.from_path2d`
  - `TopologyPath.from_bspline`
- generated shape helpers use descriptive snake-case names:
  - `TopologyPath.named_rect`
  - `TopologyPath.named_rounded_rect`
  - `TopologyPath.named_circle`
- builder entrypoints use nouns and simple verbs:
  - `TopologyPath.closed`
  - `path.point`
  - `path.segment`
  - `path.birth_span`
  - `path.birth_arc`
  - `path.death_span`
- durable records use `correspondence_id`; helper calls may accept
  `correspond=` as the lightweight authoring alias and must normalize it into
  `correspondence_id`.
- public names use `name` for user-facing labels and `id` for stable record
  identity. When only `name` is provided, helpers may derive the initial stable
  id from the name and must preserve provenance.

Do not introduce a second naming dialect during this feature. If final specs
find an existing `Path2D` or `PlanarShape2D` convention that directly conflicts
with these names, they should keep the topology names above and add adapter
aliases only after compatibility tests justify them.

### High-Confidence Inference Acceptance

Explicit authored rails do not require confidence scoring. Confidence scoring
applies only when the planner is about to infer correspondence without explicit
ids or names.

An inferred correspondence may be accepted automatically only when all of these
conditions hold:

- no hard refusal condition is present
- at least two stable protected anchors exist, or both loops have compatible
  authored point counts and preserved authored starts
- protected anchor order does not cross after the proposed match
- reversal, if selected, is allowed by topology semantics and does not conflict
  with authored direction
- the normalized best-candidate cost is at or below `0.20`
- the second-best candidate is separated by at least
  `max(0.10, best_cost * 0.50)`
- prior station interval continuity, when available, does not prefer a
  different candidate

Costs are normalized to the loop's bounding-box diagonal and include point
distance, edge-length ratio, tangent continuity, curvature or turning-angle
similarity, protected-anchor agreement, and previous-interval continuity.

If those gates do not pass, the planner must refuse automatic inference and
report the needed rails. It may still return diagnostics with ranked
candidates, but the executable plan must not treat a medium- or low-confidence
candidate as authored truth.

### Span Collapse Tolerance

Point birth/death span collapse belongs to the existing
`collapse_degeneracy` tolerance family from
[Loft Tolerance and Degeneracy Architecture](loft-tolerance-and-degeneracy-architecture.md).

Point-correspondence specs must not introduce an independent epsilon. They
should route through a loft tolerance policy field equivalent to
`collapse_degeneracy.min_point_correspondence_span`. If that policy field is
not yet present, the implementation spec should add it to the loft tolerance
policy rather than hard-coding a local value.

A birth/death transition is refused when the parent span, projected support
span, or resulting neighboring span would fall below that tolerance. The refusal
diagnostic should identify the failing span and the lifecycle event it blocked.

### Protected Landmark Sample Overflow

Protected landmarks and synthetic birth/death support samples are semantic
requirements, not optional decorations.

Default sample-count behavior should be `sample_count="auto"` for topology
helpers. Auto mode must raise the loop sample count enough to include:

- every protected landmark
- every required synthetic birth/death support sample
- the minimum unprotected samples required by each surviving span

When the user supplies an explicit integer sample count, that value is a hard
cap. If protected landmarks and synthetic supports cannot fit within the cap,
plan validation must refuse with a diagnostic that reports:

- requested sample count
- minimum required sample count
- protected landmarks that forced the minimum
- synthetic lifecycle supports that forced the minimum

The planner must not silently drop protected landmarks, merge protected
landmarks, or demote them into unprotected samples to satisfy a too-low explicit
sample count.

## Data Model

### Topology Point

Each loop point should be representable as a point record, not only as a row in
an array.

Required fields:

- stable point id
- 2D coordinates in loop-local section space
- loop-local ordinal
- optional semantic role
- optional authored name
- optional user-authored correspondence id
- optional generated/inferred correspondence id
- anchor eligibility
- source/provenance metadata

Semantic roles may include:

- corner
- seam
- feature
- tangent handle endpoint
- sampled support point
- inferred support point

### Loop Point Track

A loop point track represents the same intended point across one or more
stations.

Required fields:

- track id
- station membership
- loop membership
- point id per station where present
- lifecycle state per station
- interpolation eligibility
- diagnostic provenance

Track ids are the primary bridge between topology and loft.

Lifecycle states should include:

- `present`
- `birth`
- `death`
- `synthetic_birth_support`
- `synthetic_death_support`
- `inferred`

### Point Birth And Death

Point birth and point death are first-class correspondence events.

They occur when a loop remains the same loop, but its semantic point structure
changes between stations.

Examples:

- a new corner appears on a previously smooth side
- a notch begins
- a seam point is introduced for a later branch
- a shoulder point disappears into a smooth run
- a generated helper point exists only to support a transition

This is different from region or loop birth/death. The region and loop may
remain stable while individual point tracks begin or end.

Required event fields:

- event id
- event type: `point_birth` or `point_death`
- station interval
- loop reference
- point id or generated support point id
- optional correspondence id
- parent span or neighboring point-track ids
- authored/inferred source
- interpolation policy
- diagnostic confidence and refusal reason where relevant

Point birth/death should be visible in planner diagnostics and should not be
hidden as merely "extra samples."

### Loop Correspondence Map

A loop correspondence map records how one loop maps to another.

Required fields:

- source loop reference
- target loop reference
- ordered point-track pairs
- cyclic orientation
- source start point id
- target start point id
- reversal status
- explicit vs inferred status
- confidence / refusal diagnostics for inferred maps
- rail source, such as authored index, named point, explicit correspondence id,
  or inferred heuristic
- point birth/death events

### Resampled Loop Correspondence

Loft often needs equal sample counts for tessellation and ruled patch emission.
Resampling must preserve the point-track map.

Required fields:

- source authored point ids
- target authored point ids
- source sample coordinates
- target sample coordinates
- sample-to-track association
- sample arc-length parameter
- protected sample indices for semantic points
- synthetic samples introduced to carry birth/death transitions

Semantic points should survive resampling as protected samples whenever
possible.

## Planner Changes

### Station Normalization

Station normalization must preserve point correspondence metadata while still
normalizing winding, region order, and hole order.

Normalization should produce:

- normalized section geometry
- mapping from original region/loop/point ids to normalized ids
- explicit record of any reorder, reversal, anchor shift, or inferred start
  decision
- authored anchor policy and named anchors

If a loop has `anchor_policy="authored"`, normalization must not rotate the loop
away from authored point `0`. Winding normalization may reverse order only when
needed for validity, and must preserve a mapping back to the authored start.

### Loop Pair Planning

`PlannedLoopPair` should grow from "two same-length arrays" into "two loops plus
a point correspondence map."

Required additions:

- point correspondence map id
- explicit/inferred map status
- start-point ids
- reversed flag
- sample correspondence records
- diagnostics

Executors should connect by sample correspondence, not by assuming equal array
index means semantic identity.

### Inference Policy

When explicit point ids or track ids exist, loft must use them.

When authored point order exists and no stronger names or track ids exist, loft
should treat index/order as intentional authored correspondence.

When they do not exist, loft may infer point correspondence by:

- preserving authored point order when both loops share compatible point counts
  and no normalization reversal occurred
- matching protected corner/feature points first
- selecting cyclic shift and optional reversal by a cost function
- resampling remaining spans by normalized arc length between matched anchors

Inference must refuse or report low confidence when:

- multiple equally plausible cyclic alignments exist
- feature/corner counts disagree
- inferred reversal conflicts with winding or seam direction
- protected point order would cross
- a loop has no stable anchors and high symmetry makes phase arbitrary
- a point birth or death cannot be localized to a stable parent span

For authored-model workflows, refusal is acceptable. The preferred response to
ambiguous correspondence is to ask for or require rails, not to guess harder.
For high-confidence cases, automatic resolution is also acceptable and
desirable.

## Algorithms

### Explicit Track Resolution

1. Normalize section topology.
2. Carry point ids through normalized loop order.
3. Group matching user-authored correspondence ids across adjacent stations.
4. Validate one-to-one order around each loop.
5. Build loop correspondence maps from the validated track order.

### Authored Order Resolution

When no explicit track ids exist:

1. Check each loop's anchor policy.
2. If both source and target loops are authored and have compatible point
   counts, preserve authored order.
3. If point counts differ, preserve authored anchors and distribute samples by
   authored edge spans.
4. If one side is named and the other is unnamed, match named anchors first and
   preserve authored order between them.
5. Identify unmatched points as possible point births or deaths when their
   neighboring spans remain stable.
6. Record the map as rail-driven, not heuristic-inferred.

### Point Birth Resolution

When a target loop contains an authored or inferred point that has no matching
source point:

1. Locate the source span between two stable neighboring tracks.
2. Project the target birth point into that span's parameter space when
   possible.
3. Add a synthetic source support point at the matching span parameter.
4. Mark the target point track lifecycle as `birth`.
5. Emit a `point_birth` event in the loop correspondence map.

The executor then connects:

```text
source synthetic support point -> target born point
```

instead of smearing the birth across unrelated samples.

### Point Death Resolution

When a source loop contains an authored or inferred point that has no matching
target point:

1. Locate the target span between two stable neighboring tracks.
2. Project the source death point into that span's parameter space when
   possible.
3. Add a synthetic target support point at the matching span parameter.
4. Mark the source point track lifecycle as `death`.
5. Emit a `point_death` event in the loop correspondence map.

The executor then connects:

```text
source dying point -> target synthetic support point
```

instead of dropping the point or shifting every downstream correspondence.

### Birth/Death Refusal Conditions

Point birth/death resolution should refuse or request rails when:

- the neighboring stable tracks cannot be identified
- multiple parent spans are equally plausible
- the birth/death point would invert local order
- the transition would collapse a span below tolerance
- the event would conflict with explicit correspondence ids
- too many unmatched points appear for a stable-loop transition and the change
  should be modeled as a different topology event

### Protected Anchor Matching

1. Detect or accept semantic anchors.
2. Match anchors by explicit id first.
3. Match remaining anchors by role, local angle, curvature class, and distance.
4. Reject matches that imply crossing order.
5. Use anchors to partition the loop into spans.

### Cyclic Shift / Reversal Selection

For loops without explicit point tracks:

1. enumerate valid cyclic shifts
2. optionally enumerate reversal when allowed by topology semantics
3. score each candidate by:
   - point distance
   - edge length ratio
   - tangent continuity
   - curvature/turning-angle similarity
   - protected-anchor agreement
   - previous-interval continuity
4. select the unique best candidate or refuse/report ambiguity

This algorithm is a fallback. It should not override authored starts, named
anchors, or explicit correspondence ids.

### Correspondence-Preserving Resampling

1. Convert each loop span between protected anchors into arc-length parameter
   space.
2. Allocate samples per span by length, with minimum samples for protected
   features.
3. Ensure protected points are exact samples.
4. Insert synthetic support samples for point birth/death events.
5. Interpolate unprotected samples within each span.
6. Emit source and target sample arrays with a parallel sample correspondence
   record.

## Relationship To Existing Loft Architecture

This document extends:

- [Loft Planner / Executor Architecture](loft-planner-executor-architecture.md)
- [Loft Plan Object Architecture](loft-plan-object-architecture.md)
- [Loft Ambiguity and Diagnostics Architecture](loft-ambiguity-and-diagnostics.md)
- [Loft Tolerance and Degeneracy Architecture](loft-tolerance-and-degeneracy-architecture.md)

It also supports future control-station and curve-intent work, because those
features need stable point tracks before they can reason about station
reduction or fitted curves.

## Implementation Sequence

1. Add topology point identity records.
2. Add anchor policy to loops, with authored point arrays defaulting to
   authored start order.
3. Add topology path / landmark records that can preserve point-array,
   `Path2D`, `BSpline2D`, and generated-shape rails.
4. Add optional names, point ids, and correspondence ids to loops without
   breaking simple point-array construction.
5. Preserve ids, names, landmarks, segment ids, and anchor policy through
   section normalization.
6. Add loop correspondence map records.
7. Extend `PlannedLoopPair` with point correspondence metadata.
8. Add point birth/death event records and lifecycle states.
9. Change loop resampling to preserve protected point tracks and curve
   parameter landmarks.
10. Add diagnostics for rail-driven, inferred, ambiguous, birth/death, and refused point
   correspondence.
11. Update mesh and surface executors to consume sample correspondence records.
12. Add regression fixtures where old independent anchoring produces a twist,
   phase shift, semantic point drift, or incorrect point birth/death transition.

## Acceptance Criteria

- Explicit point correspondence ids survive `Section` normalization.
- Authored loop starts are preserved as semantic anchors by default.
- Named topology entities drive correspondence before geometric matching.
- `BSpline2D` topology input preserves authored parameter landmarks instead of
  collapsing directly into anonymous samples.
- Matching segment or landmark names align loop spans before interior samples
  are generated.
- Point birth and point death inside a stable loop are represented explicitly
  with lifecycle events and synthetic support samples.
- High-confidence automatic correspondence remains available for cases where it
  clearly reduces author burden.
- Loft connects explicit point tracks even when authored point order differs
  between stations.
- Equal-index connection is treated as an implementation detail of a resolved
  sample correspondence map, not as topology truth.
- Planner diagnostics expose whether point correspondence was explicit or
  rail-driven, explicit, or inferred.
- Ambiguous symmetric point correspondence can be refused or reported rather
  than silently guessed.
- Existing simple loft inputs still work through generated/inferred point
  correspondence.
- Surface and mesh loft executors produce equivalent results from the same
  resolved correspondence map.

## Specification Candidates

- topology point identity records
- authored anchor policy for loops and paths
- topology path records for point-array, `Path2D`, `BSpline2D`, and generated
  shape inputs
- landmark records for point and curve-parameter anchors
- named topology entities and explicit correspondence ids
- topology loop correspondence map records
- point birth/death lifecycle records
- station normalization id-preservation rules
- authored-order and named-rail resolution rules
- birth/death synthetic support sampling
- protected-anchor and cyclic-shift correspondence inference
- correspondence-preserving loop resampling
- `PlannedLoopPair` point-correspondence payload
- point-correspondence diagnostics and refusal posture
- executor consumption of resolved sample correspondence
- regression fixtures for point-track drift, phase ambiguity, and reordered
  authored points

## Specification Manifest for Discovery

### Manifest Summary

Critical review finding: the initial manifest entries below are discovery
notes, not promotion-ready implementation candidates. They include the required
template sections, but several entries under-count applicable public API
fields, reusable boundary work, executor routes, and performance-sensitive
behavior.

The architecture-level blocker questions have now been resolved in
[Resolved Defaults For Manifest Blockers](#resolved-defaults-for-manifest-blockers).
The scores below reflect those defaults; they no longer include readiness
blocker points for public naming, inference confidence, span collapse
tolerance, protected landmark overflow, or helper naming style.

| Original candidate spec | Post-default points | Corrected split posture | Primary owner |
|---|---:|---|---|
| Topology Path And Landmark Records | 35 | Split required | `src/impression/modeling/topology.py` |
| Authored Rail Resolution Rules | 21.5 | Review for split; blocker resolved | `src/impression/modeling/loft.py` |
| Point Birth And Death Correspondence Events | 25.5 | Split required | `src/impression/modeling/loft.py` |
| Correspondence-Preserving Resampling And Executor Consumption | 23.5 | Review for split; blocker resolved | `src/impression/modeling/loft.py` |
| User-Facing Topology Helpers | 36 | Split required | `src/impression/modeling/topology.py` |

The corrected post-default read is that three of five initial candidates cross
the `25+` mandatory-split threshold. The remaining two are still in the
`16-24` explicit split-review range and should be split if the final spec would
otherwise mix planner policy with executor or diagnostics work.

Required split plan before promotion:

| Revised candidate spec | Parent candidate | Target posture |
|---|---|---|
| Topology Path Core Records | Topology Path And Landmark Records | Small |
| Topology Segment And Landmark Identity Records | Topology Path And Landmark Records | Review for split |
| Path Input Adapters To Topology Paths | Topology Path And Landmark Records | Review for split |
| Authored Rail Priority And Diagnostics | Authored Rail Resolution Rules | Small |
| High-Confidence Inference And Refusal Policy | Authored Rail Resolution Rules | Review for split |
| Point Lifecycle Event Records | Point Birth And Death Correspondence Events | Small |
| Birth/Death Synthetic Support Resolution | Point Birth And Death Correspondence Events | Review for split |
| Correspondence-Preserving Resampling Contract | Correspondence-Preserving Resampling And Executor Consumption | Review for split |
| Mesh Executor Correspondence Consumption | Correspondence-Preserving Resampling And Executor Consumption | Small |
| Surface Executor Correspondence Consumption | Correspondence-Preserving Resampling And Executor Consumption | Small |
| Topology Builder Core API | User-Facing Topology Helpers | Review for split |
| Topology Lifecycle Builder API | User-Facing Topology Helpers | Review for split |
| Generated Shape Default Rails | User-Facing Topology Helpers | Small |

Template compliance notes for the detailed entries below:

- UI surfaces/components and UI fields/elements must count public modeling API
  surfaces and arguments where the spec defines user-authored calls.
- Open questions that force an implementation choice must be counted as
  readiness blockers until a default or refusal policy is chosen. The current
  architecture-level blockers are resolved above.
- "Additions to existing reusable library/module" must be counted whenever the
  work adds shared topology or loft-planner behavior, even if no new Python
  module is created.
- Performance-sensitive behavior applies to deterministic correspondence,
  bounded sampling, executor handoff, and inference search space.
- The old broad entries have been rewritten as the active split candidates
  below; the broad entries are retained afterward only as retired discovery
  notes.

### Active Split Candidate Summary

The active implementation candidates are the split specs below. The broad
candidate notes that follow them are retained only as discovery history and
must not be promoted directly.

| Candidate spec | Points | Split posture | Primary owner |
|---|---:|---|---|
| Topology Path Core Records | 14.5 | Small | `src/impression/modeling/topology.py` |
| Topology Segment And Landmark Identity Records | 21 | Review for split; cohesive identity records | `src/impression/modeling/topology.py` |
| Path Input Adapters To Topology Paths | 22.5 | Review for split; cohesive adapter boundary | `src/impression/modeling/topology.py` |
| Authored Rail Priority And Diagnostics | 13.5 | Small | `src/impression/modeling/loft.py` |
| High-Confidence Inference And Refusal Policy | 16.5 | Review for split; cohesive inference policy | `src/impression/modeling/loft.py` |
| Point Lifecycle Event Records | 10.5 | Small | `src/impression/modeling/loft.py` |
| Birth/Death Synthetic Support Resolution | 19.5 | Review for split; symmetric resolver pair | `src/impression/modeling/loft.py` |
| Correspondence-Preserving Resampling Contract | 18.5 | Review for split; planner sample contract | `src/impression/modeling/loft.py` |
| Mesh Executor Correspondence Consumption | 12.5 | Small | `src/impression/modeling/loft.py` |
| Surface Executor Correspondence Consumption | 12.5 | Small | `src/impression/modeling/loft.py` |
| Topology Builder Core API | 24 | Review for split; cohesive builder core | `src/impression/modeling/topology.py` |
| Topology Lifecycle Builder API | 21 | Review for split; cohesive lifecycle helpers | `src/impression/modeling/topology.py` |
| Generated Shape Default Rails | 20.5 | Review for split; cohesive generated-rail contract | `src/impression/modeling/topology.py` |

All active candidates are now below the `25+` mandatory-split threshold.

### Candidate Spec: Topology Path Core Records

Discovery purpose:
- Define the core topology path container that preserves closure, authored
  start, traversal direction, anchor policy, and path-level sampling policy.

Responsibilities:
- Functions/methods:
  - topology path constructor
  - topology path validation
- Data structures/models:
  - `TopologyPath`
  - `TopologyPathSamplingPolicy`
- Dependencies/services:
  - `Loop`
  - `Section`
- Returns/outputs/signals:
  - validated topology path record
  - validation errors
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `Loop`, `Section`, existing validation helpers
  - Additions to existing reusable library/module: `impression.modeling.topology`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - deterministic validation and bounded path record construction
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/topology.py`
- Chosen defaults / parameters:
  - closed paths preserve authored start and authored direction unless an
    explicit normalization map records otherwise
- Test strategy:
  - unit tests for closure, authored start, direction, sampling policy, and
    validation errors
- Data ownership:
  - topology path owns authored path-level intent before station normalization
- Routes:
  - helper or adapter input -> `TopologyPath` -> `Section`
- Reuse/extraction decision:
  - add to existing topology module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 14.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Small.

### Candidate Spec: Topology Segment And Landmark Identity Records

Discovery purpose:
- Define segment, landmark, and point identity records that carry named
  correspondence rails through topology normalization and loft planning.

Responsibilities:
- Functions/methods:
  - segment identity validation
  - landmark identity validation
- Data structures/models:
  - `TopologySegment`
  - `TopologyLandmark`
  - `TopologyPoint`
- Dependencies/services:
  - `TopologyPath`
  - station normalization
- Returns/outputs/signals:
  - validated identity records
  - identity/provenance diagnostics
- UI surfaces/components:
  - public modeling API records
- UI fields/elements:
  - `id`
  - `name`
  - `correspondence_id`
  - `protection_policy`
- Reusable code plan:
  - Existing code reused as-is: `Loop`, `Section`
  - Additions to existing reusable library/module: `impression.modeling.topology`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - deterministic identity validation independent of sample count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/topology.py`
- Chosen defaults / parameters:
  - `name` is user-facing; `id` is durable; `correspondence_id` is the planner
    rail key
- Test strategy:
  - unit tests for id/name/correspondence preservation, duplicate rejection,
    provenance, and protection policy defaults
- Data ownership:
  - topology identity records own user-authored point and landmark identity
- Routes:
  - topology records -> station normalization -> loop correspondence map
- Reuse/extraction decision:
  - add to existing topology module
- UI field/control inventory:
  - public record fields: `id`, `name`, `role`, `correspondence_id`,
    `protection_policy`

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 4 x 1 = 4
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: segment, landmark, and point identity are one record-family
  contract and should be specified together before adapter and planner work
  consumes them.

### Candidate Spec: Path Input Adapters To Topology Paths

Discovery purpose:
- Define how point arrays, `Path2D`, `BSpline2D`, and generated helper output
  become topology path records without losing authored starts, parameters, or
  landmarks.

Responsibilities:
- Functions/methods:
  - `TopologyPath.from_points`
  - `TopologyPath.from_path2d`
  - `TopologyPath.from_bspline`
- Data structures/models:
  - adapter provenance record
- Dependencies/services:
  - `Path2D`
  - `BSpline2D`
  - generated shape helpers
- Returns/outputs/signals:
  - topology path records
  - adapter validation errors
- UI surfaces/components:
  - public modeling API factories
- UI fields/elements:
  - `closed`
  - `anchor`
  - `direction`
  - `landmarks`
- Reusable code plan:
  - Existing code reused as-is: `Path2D`, `BSpline2D`, generated shape helpers
  - Additions to existing reusable library/module: `impression.modeling.topology`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded adapter validation; no hidden correspondence solving
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/topology.py`
- Chosen defaults / parameters:
  - authored point arrays use point `0` as anchor; curve inputs attach
    correspondence to parameters and named landmarks
- Test strategy:
  - unit tests for each adapter, parameter landmark preservation, failure
    diagnostics, and id/name provenance
- Data ownership:
  - adapters translate source geometry; topology path records own the result
- Routes:
  - point array / `Path2D` / `BSpline2D` / generated helper -> `TopologyPath`
- Reuse/extraction decision:
  - add adapter methods to the existing topology module
- UI field/control inventory:
  - factory arguments: `closed`, `anchor`, `direction`, `landmarks`,
    `sampling_policy`, `correspondence_id`

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 4 x 1 = 4
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: the adapters differ by input type but all terminate at the
  same topology path contract and share validation/provenance behavior.

### Candidate Spec: Authored Rail Priority And Diagnostics

Discovery purpose:
- Define deterministic priority order and diagnostics for authored
  correspondence rails before heuristic inference is considered.

Responsibilities:
- Functions/methods:
  - rail priority resolver
  - rail conflict validator
- Data structures/models:
  - rail resolution result
  - rail source enum
- Dependencies/services:
  - topology path records
  - loop correspondence planner
- Returns/outputs/signals:
  - resolved rail map
  - rail conflict diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing loft ambiguity diagnostics
  - Additions to existing reusable library/module: `impression.modeling.loft`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - deterministic resolver order
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - priority order is explicit ids, landmark names, segment names, authored
    start/direction, generated rails, inference, refusal
- Test strategy:
  - unit tests for each priority tier and conflicting rails
- Data ownership:
  - loft planner owns resolved rail maps derived from topology records
- Routes:
  - `TopologyPath` / `Section` -> `loft_plan_sections`
- Reuse/extraction decision:
  - add to existing loft planner module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 13.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Small.

### Candidate Spec: High-Confidence Inference And Refusal Policy

Discovery purpose:
- Define the inference gates, scoring, and refusal behavior used only after
  authored rails fail to resolve correspondence.

Responsibilities:
- Functions/methods:
  - inference candidate scorer
  - inference acceptance gate
  - inference refusal reporter
- Data structures/models:
  - inference candidate score
  - inference refusal reason
- Dependencies/services:
  - rail priority resolver
  - loop correspondence planner
  - ambiguity diagnostics
- Returns/outputs/signals:
  - accepted inferred map
  - ranked refused candidates
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing ambiguity diagnostics
  - Additions to existing reusable library/module: `impression.modeling.loft`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded candidate enumeration and deterministic tie refusal
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - best cost <= `0.20`; second-best separation >=
    `max(0.10, best_cost * 0.50)`; otherwise refuse
- Test strategy:
  - unit tests for accepted inference, equal-score refusal, reversal refusal,
    and diagnostics
- Data ownership:
  - loft planner owns inferred maps and refusal diagnostics
- Routes:
  - unresolved rail map -> inference policy -> loop correspondence map or
    refusal diagnostic
- Reuse/extraction decision:
  - add to existing loft planner module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: scoring, acceptance, and refusal are one policy boundary;
  splitting them would separate the threshold contract from its diagnostics.

### Candidate Spec: Point Lifecycle Event Records

Discovery purpose:
- Define durable point lifecycle event records independently from the algorithms
  that create synthetic support samples.

Responsibilities:
- Functions/methods:
  - lifecycle event validation
- Data structures/models:
  - point lifecycle state
  - point birth/death event
  - synthetic support reference
- Dependencies/services:
  - loop correspondence map
- Returns/outputs/signals:
  - lifecycle event records
  - validation errors
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current loop pairing diagnostics
  - Additions to existing reusable library/module: `impression.modeling.loft`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - not applicable
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - lifecycle states are `present`, `birth`, `death`,
    `synthetic_birth_support`, `synthetic_death_support`, and `inferred`
- Test strategy:
  - unit tests for event construction, lifecycle validation, provenance, and
    diagnostics serialization
- Data ownership:
  - loop correspondence map owns lifecycle events
- Routes:
  - synthetic support resolver -> lifecycle records -> resampling contract
- Reuse/extraction decision:
  - add to existing loft planner module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 1 x 2 = 2
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 10.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Small.

### Candidate Spec: Birth/Death Synthetic Support Resolution

Discovery purpose:
- Define the algorithms that localize unmatched points to stable parent spans
  and insert synthetic support samples for point birth and death.

Responsibilities:
- Functions/methods:
  - point birth resolver
  - point death resolver
  - parent span locator
  - synthetic support inserter
- Data structures/models:
  - parent span match
  - synthetic support sample
- Dependencies/services:
  - lifecycle event records
  - topology landmarks
  - loft tolerance policy
- Returns/outputs/signals:
  - synthetic support samples
  - lifecycle events
  - refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current loop pairing diagnostics
  - Additions to existing reusable library/module: `impression.modeling.loft`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - local span lookup and bounded projection per unmatched point
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - birth/death requires stable neighboring tracks or explicit parent span and
    refuses spans below `collapse_degeneracy.min_point_correspondence_span`
- Test strategy:
  - rectangle-to-rounded-L fixture, isolated birth fixture, isolated death
    fixture, collapse refusal fixture, conflicting-id refusal fixture
- Data ownership:
  - resolver writes lifecycle events into the loop correspondence map
- Routes:
  - unmatched topology points -> support resolver -> lifecycle events ->
    resampling contract
- Reuse/extraction decision:
  - add to existing loft planner module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: birth and death are symmetric operations over the same
  parent-span and synthetic-support contract.

### Candidate Spec: Correspondence-Preserving Resampling Contract

Discovery purpose:
- Define the planner-side sample correspondence contract before mesh or surface
  executors consume it.

Responsibilities:
- Functions/methods:
  - correspondence-preserving resampler
  - protected sample allocator
  - sample correspondence validator
- Data structures/models:
  - resampled loop correspondence
  - sample correspondence record
- Dependencies/services:
  - loop correspondence map
  - lifecycle event records
  - existing `resample_loop`
- Returns/outputs/signals:
  - source sample array
  - target sample array
  - sample correspondence records
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `resample_loop`, loop pairing records, lifecycle
    event records
  - Additions to existing reusable library/module: `impression.modeling.loft`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - deterministic sample allocation bounded by auto or explicit sample count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - `sample_count="auto"` grows to fit protected landmarks; explicit integer
    sample counts are hard caps and refuse when too low
- Test strategy:
  - unit tests for protected landmark preservation, synthetic sample placement,
    explicit cap refusal, and sample-to-track associations
- Data ownership:
  - loop correspondence map owns semantic truth; resampling contract owns
    executable samples
- Routes:
  - correspondence map -> resampled loop correspondence -> executor input
- Reuse/extraction decision:
  - add to loft planner/executor boundary
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: protected allocation, synthetic supports, and sample
  correspondence validation are one planner contract consumed by both
  executors.

### Candidate Spec: Mesh Executor Correspondence Consumption

Discovery purpose:
- Define the mesh loft executor changes needed to consume sample
  correspondence records instead of assuming equal array index means semantic
  identity.

Responsibilities:
- Functions/methods:
  - mesh executor sample-consumption update
  - mesh correspondence validation
- Data structures/models:
  - mesh sample emission diagnostic
- Dependencies/services:
  - resampled loop correspondence
  - mesh face emission
- Returns/outputs/signals:
  - mesh geometry
  - executor diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: mesh executor face emission
  - Additions to existing reusable library/module: `impression.modeling.loft`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - linear pass over correspondence samples
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - executor trusts validated sample correspondence records and refuses missing
    or mismatched records
- Test strategy:
  - unit tests comparing mesh faces for stable correspondence, birth support,
    death support, and missing-record refusal
- Data ownership:
  - mesh executor owns emitted mesh geometry; planner owns correspondence truth
- Routes:
  - resampled loop correspondence -> `loft_execute_plan`
- Reuse/extraction decision:
  - update existing mesh executor path
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 12.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Small.

### Candidate Spec: Surface Executor Correspondence Consumption

Discovery purpose:
- Define the surface loft executor changes needed to carry sample
  correspondence into surface patch construction.

Responsibilities:
- Functions/methods:
  - surface executor sample-consumption update
  - surface correspondence validation
- Data structures/models:
  - surface sample emission diagnostic
- Dependencies/services:
  - resampled loop correspondence
  - `RuledSurfacePatch`
- Returns/outputs/signals:
  - surface body geometry
  - executor diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `RuledSurfacePatch`
  - Additions to existing reusable library/module: `impression.modeling.loft`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - linear pass over correspondence samples and bounded patch construction
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - surface executor consumes validated sample correspondence and refuses
    missing records instead of falling back to index-only matching
- Test strategy:
  - unit tests for ruled patch boundaries, birth/death support preservation,
    and missing-record refusal
- Data ownership:
  - surface executor owns emitted surface patches; planner owns correspondence
    truth
- Routes:
  - resampled loop correspondence -> `_loft_execute_plan_surface`
- Reuse/extraction decision:
  - update existing surface executor path
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 12.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Small.

### Candidate Spec: Topology Builder Core API

Discovery purpose:
- Define the lightweight user-facing builder methods for ordinary points and
  curve segments, excluding lifecycle-specific birth/death helpers.

Responsibilities:
- Functions/methods:
  - `TopologyPath.closed`
  - `path.point`
  - `path.segment`
  - builder validation
- Data structures/models:
  - topology builder state
- Dependencies/services:
  - topology path core records
  - segment and landmark identity records
- Returns/outputs/signals:
  - topology path records
  - builder validation errors
- UI surfaces/components:
  - public modeling API builder
- UI fields/elements:
  - `name`
  - `id`
  - `correspond`
  - `anchor`
  - `points`
- Reusable code plan:
  - Existing code reused as-is: topology path records, landmark records
  - Additions to existing reusable library/module: `impression.modeling.topology`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - builder validation avoids hidden expensive correspondence solving
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/topology.py`
- Chosen defaults / parameters:
  - `correspond=` is a helper alias for `correspondence_id`; omitted ids derive
    from stable names with provenance
- Test strategy:
  - public API unit tests, docs snippets, point/segment builder tests, and
    validation error tests
- Data ownership:
  - builder creates topology records; planner owns later correspondence
    decisions
- Routes:
  - builder calls -> topology records -> section -> loft planner
- Reuse/extraction decision:
  - add builder helpers to topology module as thin wrappers over records
- UI field/control inventory:
  - builder arguments: `name`, `id`, `correspond`, `anchor`, `points`, `curve`,
    `landmarks`

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 5 x 1 = 5
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 24

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: point and segment builder methods share one public authoring
  state and compile to the same topology path record family.

### Candidate Spec: Topology Lifecycle Builder API

Discovery purpose:
- Define birth/death helper methods that compile to lifecycle-capable topology
  records without exposing internal event records directly.

Responsibilities:
- Functions/methods:
  - `path.birth_span`
  - `path.birth_arc`
  - `path.death_span`
- Data structures/models:
  - lifecycle builder request
- Dependencies/services:
  - topology builder state
  - point lifecycle event records
- Returns/outputs/signals:
  - topology path records
  - lifecycle builder validation errors
- UI surfaces/components:
  - public modeling API builder
- UI fields/elements:
  - `parent`
  - `points`
  - `radius`
  - `name`
- Reusable code plan:
  - Existing code reused as-is: topology builder state, lifecycle record fields
  - Additions to existing reusable library/module: `impression.modeling.topology`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - builder validation avoids parent-span solving beyond structural checks
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/topology.py`
- Chosen defaults / parameters:
  - lifecycle helpers require explicit parent spans and preserve authored order
    inside each birth/death helper call
- Test strategy:
  - public API tests for birth span, birth arc, death span, parent validation,
    and generated lifecycle provenance
- Data ownership:
  - helper creates lifecycle-capable topology records; loft planner owns final
    event resolution
- Routes:
  - lifecycle builder call -> topology path records -> loft lifecycle resolver
- Reuse/extraction decision:
  - add lifecycle builder helpers to topology module as thin wrappers
- UI field/control inventory:
  - helper arguments: `parent`, `points`, `name`, `radius`, `start`, `end`

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 4 x 1 = 4
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: lifecycle helper methods share one authored parent-span
  contract and compile to the same lifecycle-capable topology records.

### Candidate Spec: Generated Shape Default Rails

Discovery purpose:
- Define generated rails for common shape helpers so rectangles, circles, and
  rounded rectangles are not anonymous topology inputs.

Responsibilities:
- Functions/methods:
  - `TopologyPath.named_rect`
  - `TopologyPath.named_rounded_rect`
  - `TopologyPath.named_circle`
- Data structures/models:
  - generated rail provenance record
- Dependencies/services:
  - generated shape helpers
  - topology identity records
- Returns/outputs/signals:
  - topology path records with generated landmarks
  - generated rail diagnostics
- UI surfaces/components:
  - public modeling API factories
- UI fields/elements:
  - `width`
  - `height`
  - `radius`
  - `anchor`
- Reusable code plan:
  - Existing code reused as-is: generated shape helpers
  - Additions to existing reusable library/module: `impression.modeling.topology`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - deterministic rail generation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/topology.py`
- Chosen defaults / parameters:
  - rectangles get corner and side-midpoint rails; circles get authored start
    and quadrant rails; rounded rectangles get straight segment, arc, and
    tangent transition rails
- Test strategy:
  - unit tests for generated names, correspondence ids, anchor preservation,
    and override behavior
- Data ownership:
  - generated helpers create topology records with generated provenance
- Routes:
  - generated helper -> topology path records -> loft planner
- Reuse/extraction decision:
  - add generated topology helpers to existing topology module
- UI field/control inventory:
  - factory arguments: `width`, `height`, `radius`, `anchor`, `name_prefix`

Open questions / nuance discovered:
- No unresolved architecture questions remain.

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 4 x 1 = 4
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: generated shape rails share one provenance and naming
  contract even though individual helpers cover different primitives.

### Retired Original Candidate Note: Topology Path And Landmark Records

Discovery purpose:
- Define the reusable topology path layer that preserves authored starts,
  direction, curve segments, landmarks, and correspondence ids before topology
  is sampled.

Responsibilities:
- Functions/methods:
  - `TopologyPath.from_points`
  - `TopologyPath.closed`
  - topology path validation
- Data structures/models:
  - `TopologyPath`
  - `TopologySegment`
  - `TopologyLandmark`
- Dependencies/services:
  - `Path2D`
  - `BSpline2D`
  - generated shape helpers
- Returns/outputs/signals:
  - normalized topology path records
  - validation errors
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `Path2D`, `BSpline2D`, `Loop`, `Section`
  - Additions to existing reusable library/module: `impression.modeling.topology`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - deterministic sampling and validation for authored paths
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/topology.py`
- Chosen defaults / parameters:
  - authored point arrays default to authored start and authored direction
- Test strategy:
  - unit tests for path records, landmark preservation, closure, direction, and
    curve-parameter landmarks
- Data ownership:
  - topology path records own authored boundary intent before loft planning
- Routes:
  - `Path2D` / `BSpline2D` / helper inputs -> `TopologyPath` -> `Section`
- Reuse/extraction decision:
  - add to existing reusable topology module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Resolved: public constructor naming is defined in
  [Public API Naming](#public-api-naming).

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 4 x 0.5 = 2
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: topology path, segment, and landmark records form one shared
  boundary object family and should be specified together before consumers are
  split.

### Retired Original Candidate Note: Authored Rail Resolution Rules

Discovery purpose:
- Define how explicit ids, names, segment names, authored starts, generated
  rails, and high-confidence inference are prioritized for correspondence.

Responsibilities:
- Functions/methods:
  - correspondence rail resolver
  - rail-priority validation
- Data structures/models:
  - rail resolution result
  - rail source enum
- Dependencies/services:
  - topology path records
  - current loop correspondence planner
- Returns/outputs/signals:
  - resolved rail map
  - refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: existing loft ambiguity diagnostics
  - Additions to existing reusable library/module: `impression.modeling.loft`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - deterministic resolver order
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - explicit correspondence ids, names, segment names, authored start/direction,
    generated rails, inference, refusal
- Test strategy:
  - unit tests for each priority tier and mixed rail/inference outcomes
- Data ownership:
  - loft planner owns resolved rail maps derived from topology records
- Routes:
  - `TopologyPath` / `Section` -> `loft_plan_sections`
- Reuse/extraction decision:
  - add to existing loft planner module, extract later only if shared outside
    loft
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Resolved: high-confidence inference gates are defined in
  [High-Confidence Inference Acceptance](#high-confidence-inference-acceptance).

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 13.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Small.

### Retired Original Candidate Note: Point Birth And Death Correspondence Events

Discovery purpose:
- Define point lifecycle events inside stable loops, including synthetic support
  samples and refusal conditions.

Responsibilities:
- Functions/methods:
  - point birth resolver
  - point death resolver
  - synthetic support sample insertion
- Data structures/models:
  - point lifecycle state
  - point birth/death event
  - synthetic support sample record
- Dependencies/services:
  - topology landmark records
  - loft loop correspondence planner
- Returns/outputs/signals:
  - lifecycle events
  - refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current loop pairing and diagnostics records
  - Additions to existing reusable library/module: `impression.modeling.loft`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - local span lookup and deterministic support sample insertion
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - birth/death requires stable neighboring tracks or explicit parent span
- Test strategy:
  - rectangle-to-rounded-L fixture, point birth fixture, point death fixture,
    refusal fixture
- Data ownership:
  - loop correspondence map owns lifecycle events
- Routes:
  - topology landmarks -> loop correspondence map -> executor samples
- Reuse/extraction decision:
  - add to existing loft planner module
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Resolved: point birth/death span collapse routes through the
  `collapse_degeneracy` tolerance family as defined in
  [Span Collapse Tolerance](#span-collapse-tolerance).

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: birth and death are symmetric lifecycle operations over the
  same stable-loop correspondence map and should be specified together unless
  implementation reveals separate executors.

### Retired Original Candidate Note: Correspondence-Preserving Resampling And Executor Consumption

Discovery purpose:
- Define how resolved correspondence maps become equal-count samples consumed by
  mesh and surface loft executors.

Responsibilities:
- Functions/methods:
  - correspondence-preserving resampler
  - sample correspondence validator
  - executor sample consumption update
- Data structures/models:
  - resampled loop correspondence
  - sample correspondence record
- Dependencies/services:
  - loop correspondence map
  - mesh executor
  - surface executor
- Returns/outputs/signals:
  - source/target sample arrays
  - sample correspondence records
  - validation errors
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `resample_loop`, `RuledSurfacePatch`, mesh
    executor face emission
  - Additions to existing reusable library/module: topology/loft resampling
    helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - sample allocation must be deterministic and bounded by loft sample count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/loft.py`
- Chosen defaults / parameters:
  - preserve protected landmarks exactly when possible; allocate remaining
    samples by span length
- Test strategy:
  - unit tests comparing planned correspondence records to mesh/surface output
- Data ownership:
  - loop correspondence map owns semantic truth; executor owns emitted geometry
- Routes:
  - correspondence map -> samples -> `loft_execute_plan` /
    `_loft_execute_plan_surface`
- Reuse/extraction decision:
  - add to loft planner/executor boundary
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Resolved: protected landmark overflow policy is defined in
  [Protected Landmark Sample Overflow](#protected-landmark-sample-overflow).

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: resampling and executor consumption are the same boundary
  handoff; keep together until the sample record contract is stable.

### Retired Original Candidate Note: User-Facing Topology Helpers

Discovery purpose:
- Define lightweight helper APIs that compile to the heavier topology path,
  landmark, correspondence, and lifecycle records.

Responsibilities:
- Functions/methods:
  - `TopologyPath.from_points`
  - `TopologyPath.named_rect`
  - builder methods for `point`, `segment`, `birth_span`, `birth_arc`
- Data structures/models:
  - helper builder state
- Dependencies/services:
  - topology path records
  - generated shape helpers
- Returns/outputs/signals:
  - topology path records
  - validation errors
- UI surfaces/components:
  - public modeling API
- UI fields/elements:
  - helper arguments
- Reusable code plan:
  - Existing code reused as-is: current `Path2D.from_points`, generated shape
    helpers
  - Additions to existing reusable library/module: `impression.modeling.topology`
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - helper validation should avoid hidden expensive solving
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/topology.py`
- Chosen defaults / parameters:
  - simple helpers default to authored starts and generated default rails
- Test strategy:
  - public API unit tests and docs/example snippets
- Data ownership:
  - helpers create topology records; they do not own planner decisions
- Routes:
  - helper -> topology records -> section -> loft planner
- Reuse/extraction decision:
  - add to topology module; keep helpers thin over records
- UI field/control inventory:
  - helper arguments only

Open questions / nuance discovered:
- Resolved: helper naming style is defined in
  [Public API Naming](#public-api-naming).

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split.
- Cohesion reason: helper methods should be specified together as one public
  authoring surface over the same topology path record family.

## Change History

- 2026-05-25: Added authored-model scope and user-defined rail policy: array
  starts, path starts, named topology entities, and explicit correspondence ids
  should drive correspondence before automatic ambiguity solving.
- 2026-05-25: Clarified that authored rails do not forbid helpful automation;
  high-confidence automatic resolution should remain available when it creates a
  large user benefit and reports its inference status honestly.
- 2026-05-25: Added point birth/death as first-class stable-loop
  correspondence events with lifecycle states and synthetic support samples.
- 2026-05-25: Critically reviewed the Specification Manifest for Discovery
  against the shared manifest-entry template; corrected under-counted scores,
  restored readiness blockers for unresolved choices, and marked required
  candidate splits before promotion.
- 2026-05-25: Resolved manifest blocker questions for topology public API
  naming, high-confidence inference gates, point birth/death span collapse
  tolerance, protected landmark sample overflow, and helper naming style before
  splitting candidates into implementation-sized specs.
- 2026-05-25: Split the loft topology point-correspondence manifest into active
  implementation-sized candidates, including an additional split of the builder
  API into core and lifecycle helper specs so no active candidate remains at the
  mandatory split threshold.
- 2026-05-25: Initial architecture note created after identifying that loft
  manages region/loop correspondence but not topology-owned point
  correspondence.
