# Orientation Language And Contract Research

Date: 2026-04-28

## Question

How should Impression define and maintain a durable orientation contract across:

- internal modeling
- preview
- export
- example/modeling language
- verification/test tooling

And where in the current codebase can orientation mismatches already occur?

## Executive Findings

Impression currently has **partial orientation discipline**, not a single owned orientation contract.

The codebase already does some important things correctly:

- loft station frames are validated as right-handed orthonormal bases
- preview uses a stable default camera
- reference-image tooling defines a handedness and canonical-view contract

But these pieces are **not tied together by one production-level contract object**.

The most important concrete mismatch I found is this:

- preview home/reset view puts the camera in the `(+X, +Y, +Z)` octant with `+Z` as up
- canonical object-view tests define `"front"` from the `-Y` side and `"isometric"` from `(+X, -Y, +Z)`

So preview and verification are already using **different Y-direction assumptions** for named views.

That means we do not just have ambiguous human language. We already have at least one real **owned-system orientation drift**.

## Current Impression Orientation Surface

### 1. Preview

Preview has an implemented viewer convention, but it is not declared as a shared contract object.

Current documentation says:

- `docs/cli.md`: camera defaults are `+Z up`, `+X right`, `+Y toward the camera`

Current implementation does this in `src/impression/preview.py`:

- reset/home camera position is placed at approximately `(+X, +Y, +Z)` relative to model center
- `view_up = (0, 0, 1)`

So preview currently behaves like a stable **right / forward / up isometric** view, assuming `+Y` is the near-facing direction.

### 2. Loft And Internal Station Frames

Loft has strong **local frame correctness**, but no declared global semantic axis language.

`src/impression/modeling/loft.py` enforces:

- unit-length station basis vectors
- orthogonality
- positive handedness via `dot(cross(u, v), n) > 0`

This is good, but it only proves:

- the station frame is mathematically valid

It does **not** say:

- which axis is world-up
- which direction is forward
- whether `n` is expected to align with progression-forward or camera-forward naming
- how user words like `front`, `back`, `deep`, `high`, or `toward` should map to axes

### 3. Progression

`PathBackedProgression` is now a richer semantic object, but it still does not own orientation language.

`src/impression/modeling/progression.py` currently owns:

- path
- parameter domain
- provenance
- transport policy
- deferred twist semantics
- deferred scale semantics

It does **not** currently own:

- handedness declaration
- world basis
- canonical forward / right / up semantics
- canonical views
- local-vs-global orientation scope

This is a major architectural gap because progression is the natural place to carry path-relative orientation semantics.

### 4. Path3D

`Path3D` is geometric, not semantic.

`src/impression/modeling/path3d.py`:

- samples lines, arcs, and beziers
- constructs arc plane bases with `_plane_basis(normal)`

That helper chooses a basis heuristically:

- prefer Z as helper axis unless nearly parallel
- otherwise fall back to Y

This is acceptable for geometry generation, but it means:

- local orientation can be manufactured implicitly
- no shared naming contract exists for the resulting basis

### 5. Text

Text uses yet another implicit orientation convention.

`src/impression/modeling/text.py` assumes:

- the base extrusion/orientation direction is `+Z`

Then it rotates from that base vector toward the requested direction.

That is a perfectly reasonable local convention, but it is a separate one, and it is not explained through any shared global orientation contract.

### 6. Threading

Threading also constructs its own basis locally.

`src/impression/modeling/threading.py`:

- normalizes `axis_direction`
- builds an axis basis by choosing a helper axis and crossing into a local frame

Again, this is mathematically fine, but semantically isolated.

### 7. Verification Tooling

The strongest orientation-contract work in the repository currently lives in **tests**, not production code.

`tests/reference_images.py` defines:

- `HandednessSpaceAnchorContract`
- `modeling_basis`
- `export_basis`
- `viewer_basis`
- `canonical_view`
- `camera_contract`

This is important because it means the repository already discovered the problem and invented the beginnings of the right abstraction, but only in test tooling.

## Confirmed Mismatch Points

### Mismatch 01: Preview Home View vs Canonical Test Views

This is the clearest concrete mismatch.

Preview reset:

- camera from `(+X, +Y, +Z)`
- `+Z` up

Reference-image canonical views:

- `"front"` camera from `-Y`
- `"side"` camera from `+X`
- `"top"` camera from `+Z`
- `"isometric"` camera from `(+X, -Y, +Z)`

So at minimum:

- preview’s Y sign and canonical-view Y sign do not agree for named front/isometric understanding

This is likely one of the reasons the airframe exercise made it too easy to talk past the model visually.

### Mismatch 02: Production Has No Shared Handedness / Basis Object

Tests can express:

- modeling basis
- export basis
- viewer basis

But production systems cannot.

That means:

- preview cannot declare what basis it believes it is rendering
- export cannot declare what basis it writes
- models cannot declare which basis they were authored against

### Mismatch 03: Global Semantic Axes Are Missing

Loft validates local frames, but Impression does not currently declare:

- which world axis is right
- which world axis is forward
- which world axis is up

Without that, words like:

- front
- aft
- top
- side
- nose-up
- toward camera

are all still partly social rather than contractual.

### Mismatch 04: Canonical View Names Are Underspecified

Current names like:

- `front`
- `side`
- `top`

do not say:

- front relative to which declared forward axis?
- side meaning left side or right side?

This is a language-level mismatch waiting to happen even if the math is right.

### Mismatch 05: Local Basis Builders Are Repeated

The following modules each derive local bases independently:

- `loft.py`
- `path3d.py`
- `text.py`
- `threading.py`
- parts of `drafting.py`

That does not automatically mean they are numerically wrong, but it does mean:

- local basis generation policies can drift
- helper-axis fallbacks can differ
- semantics can differ even when the math remains valid

## External Research

## ROS REP 103

ROS REP 103 is one of the cleanest examples of a codified orientation contract.

It explicitly states:

- all systems are right-handed
- for body frames: `x forward`, `y left`, `z up`
- special alternate frames should use suffixes such as `_optical` and `_ned`
- rotation conventions should be constrained because Euler-angle language is otherwise ambiguous

Most useful ideas for Impression:

- pick one global default
- keep it right-handed
- allow special alternate frames, but name them explicitly
- do not let ambiguous rotation language float around without a declared order or representation

## OpenUSD

OpenUSD has a very useful pattern for scene-level orientation control.

It explicitly allows a stage to declare:

- `upAxis = Y or Z`

And it requires applications to consult that declaration when constructing cameras.

It also separates:

- up-axis declaration
- handedness

Most useful ideas for Impression:

- orientation should be a **scene / model-level declaration**
- preview camera behavior should consume that declaration instead of hardcoding its own assumptions
- asset/reference mismatches should be corrected at boundaries rather than hidden internally

## glTF 2.0

glTF is useful here as an example of a strongly declared interchange convention.

It explicitly declares:

- right-handed system
- `+Y up`
- `+Z forward`
- the front of an asset faces `+Z`

Most useful ideas for Impression:

- an interchange target should publish an explicit basis
- “front” should be defined as a named axis, not just inferred socially

## Unity

Unity is useful mostly as a cautionary contrast.

It explicitly uses:

- left-handed system
- `+X right`
- `+Y up`
- `+Z forward`

This matters because it shows that:

- two systems can use similar words like “forward” and “up”
- while still disagreeing on handedness and transform behavior

For Impression this reinforces that:

- export/viewer adapters must be explicit
- language alone is not enough

## Aerospace / Aircraft Body Axes

Aircraft body-axis conventions are also relevant because the airframe example triggered this research.

The NASA aircraft reference I reviewed uses:

- `x` forward
- `y` right
- `z` down

And it stresses something very important:

- layout coordinates, component coordinates, and body axes can differ
- but all internal calculations must use a declared consistent convention

This is the key lesson for Impression:

- a domain-local frame is fine
- but it must be layered on top of a declared global contract, not substituted for one implicitly

## Proposed Solution

Impression should adopt a production-level **Orientation Contract** with three layers.

### Layer 1: Global World Contract

Impression should declare one canonical world contract for modeling, preview, and default export.

Recommended default:

- handedness: `right-handed`
- world basis:
  - `+X = right`
  - `+Y = forward`
  - `+Z = up`

Why this choice:

- it matches current preview documentation best
- it matches current preview camera implementation best
- it keeps `+Z up`, which is already stable in preview and tests
- it gives a simple cardinal language for modeling examples

### Layer 2: Canonical View Contract

Canonical views should be derived from the world contract, not hardcoded separately.

Recommended canonical views:

- `front_view`: camera on `+forward`, looking toward origin
- `rear_view`: camera on `-forward`
- `right_view`: camera on `+right`
- `left_view`: camera on `-right`
- `top_view`: camera on `+up`
- `bottom_view`: camera on `-up`
- `front_right_top_isometric`: camera in `(+right, +forward, +up)` octant

This means the current test-side labels should be tightened:

- replace `side` with `right_view` or `left_view`
- stop using `front` unless forward is explicitly declared

### Layer 3: Local / Domain Frames

Impression should allow explicit local frames layered over the world contract.

Examples:

- `optical_frame`
- `aircraft_body_frame`
- `thread_axis_frame`
- `station_frame`

For example, an aircraft body frame could be declared as:

- `+body_x = forward`
- `+body_y = starboard`
- `+body_z = down`

while still being embedded in a global Impression world of:

- `+X right`
- `+Y forward`
- `+Z up`

This lets us preserve domain-correct language without sacrificing world consistency.

## Proposed Production Objects

### `OrientationContract`

Suggested shape:

```python
@dataclass(frozen=True)
class OrientationContract:
    handedness: Literal["right-handed"]
    world_right_axis: AxisToken
    world_forward_axis: AxisToken
    world_up_axis: AxisToken
    canonical_views: CanonicalViewMap
    export_basis: BasisContract
    viewer_basis: BasisContract
```

### `AxisToken`

Make axis tokens explicit and cardinal:

- `+X`
- `-X`
- `+Y`
- `-Y`
- `+Z`
- `-Z`

These should be the only canonical axis tokens.

### `NamedFrameContract`

Suggested local-frame object:

```python
@dataclass(frozen=True)
class NamedFrameContract:
    name: str
    x_semantic: str
    y_semantic: str
    z_semantic: str
    relative_to: str
```

Examples:

- `world_frame`
- `aircraft_body_frame`
- `optical_frame`
- `station_frame`

## Language Rules To Codify

### Canonical Language

Use these in contracts, docs, and tests:

- `+X`, `-X`, `+Y`, `-Y`, `+Z`, `-Z`
- `right`, `left`, `forward`, `aft`, `up`, `down`
- `front_view`, `rear_view`, `right_view`, `left_view`, `top_view`, `bottom_view`

### Allowed But Non-Canonical Human Language

These may appear in prose, but should not define contracts by themselves:

- deep
- shallow
- high
- low
- near
- far
- toward me
- toward camera

If these appear in specs or plans, they should be accompanied by a canonical translation.

Example:

- `deep (+Y / forward in this model)`

### Rotation Language

Impression should also declare one preferred rotation-description rule.

Recommended:

- store and move rotations internally as matrices or quaternions
- if axis-order angles must be named, use an explicit declared order every time

This follows the same discipline seen in ROS REP 103.

## Concrete Impression Changes Recommended

### 1. Promote The Test Contract Into Production

The abstractions in `tests/reference_images.py` should move into production code.

Especially:

- handedness / space anchor declaration
- canonical view camera contract
- viewer basis / export basis declaration

### 2. Make Preview Consume The Contract

`src/impression/preview.py` should no longer own the camera basis implicitly.

Instead:

- preview reset/home view should be derived from `OrientationContract`
- docs should be generated from or validated against the same contract

### 3. Fix The Current Preview/Test Drift

Choose one of these and normalize everything around it:

- keep preview as truth and update canonical test views to `+Y`-front
- or keep test views as truth and flip preview’s Y-sign assumptions

My recommendation:

- keep preview as truth
- update canonical test views to match the proposed world contract

Reason:

- preview is the user’s active visual workspace
- the current docs already describe preview in that direction
- examples and modeling language will be easier to stabilize around one lived viewer experience

### 4. Add Contract-Carrying Metadata To Progression

`PathBackedProgression` should carry or reference the model’s orientation contract.

That would make progression the semantic bridge between:

- path geometry
- transport semantics
- orientation language
- station attachment semantics

### 5. Add Explicit Domain Frame Overlays

Examples like aircraft bodies should declare a local domain frame rather than rely on implicit word meaning.

This would let a spec say:

- world frame is `+X right`, `+Y forward`, `+Z up`
- aircraft body frame is `+body_x forward`, `+body_y right`, `+body_z down`

### 6. Replace Ambiguous View Names

Prefer:

- `right_view`
- `left_view`

instead of:

- `side`

because `side` is semantically incomplete.

## Specific Code Hotspots To Revisit

These are the highest-value places to harden next.

### Preview

- `src/impression/preview.py`
- `docs/cli.md`

Reason:

- current lived orientation contract is here
- it must become explicit and shared

### Verification

- `tests/reference_images.py`

Reason:

- production-quality contract ideas already exist here
- current canonical views are out of sync with preview Y-direction assumptions

### Progression

- `src/impression/modeling/progression.py`

Reason:

- this is the best home for path-backed orientation semantics

### Loft

- `src/impression/modeling/loft.py`

Reason:

- local frame correctness already exists
- it needs connection to named world and domain frames

### Local Basis Builders

- `src/impression/modeling/path3d.py`
- `src/impression/modeling/text.py`
- `src/impression/modeling/threading.py`

Reason:

- each currently makes silent local basis choices
- they should either consume shared policy or declare that they are local-only basis builders

## Recommended Adoption Sequence

### Phase 1

- create production `OrientationContract`
- define canonical world basis
- define canonical named views
- update preview to consume it

### Phase 2

- move handedness/view-basis test abstractions into production
- update test-side canonical views to be generated from the shared contract

### Phase 3

- attach orientation contract to progression or model-level scene metadata
- add named local frame overlays for domain-specific modeling

### Phase 4

- update example-planning and modeling-spec language rules so every model plan declares:
  - world contract
  - any local frame overlays
  - canonical review views

## Recommendation

Impression should treat orientation as a **first-class architectural contract**, not just a viewer default and not just a spec-writing discipline.

The system we want is:

- one world basis
- one shared preview/test/export contract
- named local overlays for domain frames
- canonical axis tokens
- canonical view names
- human-language aliases only as translated convenience

That will fix both:

- the current concrete preview/test mismatch
- the broader modeling-language drift that showed up during the airframe exercise

## External References

- ROS REP 103: https://www.ros.org/reps/rep-0103.html
- OpenUSD `upAxis`: https://openusd.org/docs/api/group___usd_geom_up_axis__group.html
- glTF 2.0 specification: https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.html
- Unity rotation/orientation manual: https://docs.unity.cn/2022.3/Documentation/Manual/QuaternionAndEulerRotationsInUnity.html
- NASA aircraft body-axis reference: https://rotorcraft.arc.nasa.gov/Publications/files/Johnson%20TP-2015-218751.pdf
