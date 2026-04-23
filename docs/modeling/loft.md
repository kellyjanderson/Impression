# Modeling — Loft

Loft creates a surface between a series of profiles. If a path is provided,
profiles are translated to sampled
positions and rotated to follow the path direction (parallel-transport frames
to minimize twist).
Station frames are normalized to a right-handed orthonormal basis (`u`, `v`, `n`).

Correspondence is deterministic: hole loops are matched per adjacent section
with a minimum-cost assignment (centroid distance + area delta) so loop order
differences between profiles do not cause random pairing.
Ambiguous non-overlapping transitions are rejected early as unsupported
split/merge-like events.
Given identical inputs and parameters, loft output is deterministic
(identical vertex/face ordering).
When costs tie, correspondence falls back to deterministic index ordering.
Invalid profile containment (for example, holes outside the outer loop) is still
rejected as an unsupported topology transition.

Canonical API:

- `Loft(...)` is the canonical surfaced `SurfaceBody` entrypoint using explicit progression, stations, and topology.
- `loft(...)` is the profile/path convenience API over the same surface-first planner.
- `loft_sections(...)` is the explicit-station convenience form of that same planner/executor pipeline.
- `loft_plan_sections(...)` builds a deterministic `LoftPlan` without executing it.
- `loft_execute_plan(...)` executes a previously built `LoftPlan`.
- `loft_plan_ambiguities(...)` reports ambiguous intervals and candidate IDs without executing geometry.
- `loft_profiles(...)` remains as a compatibility alias for `loft(...)`.

```python
from impression.modeling import Loft, Station, loft, loft_execute_plan, loft_plan_ambiguities, loft_plan_sections, loft_sections
from impression.modeling.drawing2d import make_rect
from impression.modeling import Path3D
```

## Choosing a Loft Lane

Use the loft API that matches the representation you want to own:

- `Loft(...)`
  - canonical surface-first loft with explicit progression, stations, and topology
  - best when downstream work should stay in `SurfaceBody` form
  - returns `SurfaceBody`
- `loft(...)`
  - easiest path-oriented convenience API
  - best when you have profiles and optionally a `Path3D`
  - uses the same planner, then hands off to the current public consumer path
- `loft_sections(...)`
  - explicit station frames with topology-aware sections
  - best when you want to author station placement yourself
  - uses the same planner, then hands off to the current public consumer path
- `loft_plan_sections(...)` + `loft_execute_plan(...)`
  - best when you need to inspect or persist planner output before execution
- `loft_plan_ambiguities(...)`
  - best when you want a report of ambiguous intervals/candidate IDs before choosing interactive selections

## Ownership Boundary

`loft.py` owns:

- station sequencing across profile/path samples
- frame generation/transport along the loft path
- section placement in 3D
- section-to-section surface bridging
- loft/endcap orchestration and error propagation

`loft.py` does not own:

- generic loop winding/classification algorithms
- generic containment tests
- generic triangulation primitives
- generic profile inset topology policy

Those reusable topology concerns live in [`topology`](topology.md) and are
consumed by loft. See [Topology Spec 08](../../project/specifications/topology-08-loft-module-narrowing.md).

## loft(profiles, path=None)

`loft(...)` is the profile/path convenience API. It uses the same surface-first
planner as `Loft(...)`, then hands the result to the current public loft
consumer path.

```python
from impression.modeling import loft, Path3D, Line3D
from impression.modeling.drawing2d import make_rect

def build():
    profiles = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(0.6, 1.4)),
        make_rect(size=(0.8, 0.8)),
    ]
    path = Path3D.from_points([(0, 0, 0), (0, 0, 1.0), (0, 0, 2.0)])
    return loft(profiles, path=path, cap_ends=True)
```

Example: `docs/examples/loft/loft_example.py`

## loft_sections(stations)

`loft_sections(...)` accepts explicit station frames and topology-native sections.
It is the low-level explicit-station form of the same topology-aware loft pipeline
used by `loft(...)`, with the same current public handoff posture.

```python
from impression.modeling import Station, loft_sections, as_section
from impression.modeling.drawing2d import make_rect

def build():
    s0 = Station(
        t=0.0,
        section=as_section(make_rect(size=(1.0, 1.0))),
        origin=(0.0, 0.0, 0.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )
    s1 = Station(
        t=1.0,
        section=as_section(make_rect(size=(0.8, 1.2))),
        origin=(0.1, 0.0, 1.0),
        u=(1.0, 0.0, 0.0),
        v=(0.0, 1.0, 0.0),
        n=(0.0, 0.0, 1.0),
    )
    return loft_sections([s0, s1], cap_ends=True)
```

Example: `docs/examples/loft/loft_sections_example.py`

## Loft(progression, stations, topology)

`Loft(...)` is the surfaced loft lane. It runs the same planner concepts as the
public loft convenience APIs, but it returns a `SurfaceBody` so the result can stay app-owned
and platform-agnostic until a consumer explicitly tessellates it.

```python
from impression.modeling import Loft, tessellate_surface_body
from impression.modeling.drawing2d import make_rect

def build():
    progression = [0.0, 0.5, 1.0]
    stations = [
        (0.0, 0.0, 0.0),
        (0.0, 0.1, 1.0),
        (0.0, 0.0, 2.0),
    ]
    topology = [
        make_rect(size=(1.0, 1.0)),
        make_rect(size=(0.8, 1.3)),
        make_rect(size=(0.6, 0.9)),
    ]
    body = Loft(progression=progression, stations=stations, topology=topology, cap_ends=True)
    return tessellate_surface_body(body).mesh
```

`stations` may be provided as:

- `Station` objects
- 3D origins, letting loft derive station frames from the progression path
- `(origin, u, v, n)` tuples when you want to supply the full frame explicitly

`topology` accepts the same kinds of profile inputs as `loft(...)`: sections,
regions, paths, and other topology-normalizable profile objects.

Real-world surfaced example:

- `docs/examples/loft/real_world/loft_hourglass_vessel_example.py`
  - multi-region hourglass with inner cavity, three pillar regions, and explicit
    `loft_plan_ambiguities(...)` candidate selection fed back through
    `ambiguity_mode="interactive"`

## Planner / Executor / Ambiguity Tools

The loft planner is public. Use it when you want deterministic inspection or
interactive control instead of a single black-box loft call.

Build a plan without executing it:

```python
from impression.modeling import loft_execute_plan, loft_plan_sections

plan = loft_plan_sections(stations, samples=48, split_merge_mode="resolve")
loft_output = loft_execute_plan(plan, cap_ends=True)
```

Inspect ambiguity candidates without generating geometry:

```python
from impression.modeling import loft_plan_ambiguities

report = loft_plan_ambiguities(stations, samples=48, split_merge_mode="resolve")
for interval in report.intervals:
    print(interval.interval, [candidate.candidate_id for candidate in interval.candidates])
```

That ambiguity report is the input to interactive disambiguation:

- call `loft_plan_ambiguities(...)`
- choose one `candidate_id` per interval you want to control
- pass `ambiguity_mode="interactive"` and
  `ambiguity_selection={(start_index, end_index): candidate_id}` into
  `loft(...)`, `loft_sections(...)`, `Loft(...)`, or `loft_plan_sections(...)`

## Advanced Transition Controls

The public loft convenience APIs, surfaced `Loft(...)`, and planner helpers share the same
core transition controls.

Split / merge controls:

- `split_merge_mode="fail" | "resolve"`
- `split_merge_steps`
- `split_merge_bias`

Ambiguity controls:

- `ambiguity_mode="fail" | "auto" | "interactive"`
- `ambiguity_selection`
- `ambiguity_selection_policy="required" | "best_effort"`
- `ambiguity_cost_profile="balanced" | "distance_first" | "area_first"`
- `ambiguity_max_branches`
- `disambiguation_mode="deterministic" | "probabilistic"`
- `disambiguation_seed`
- `probabilistic_trials`
- `probabilistic_temperature`
- `probabilistic_min_confidence`
- `probabilistic_fallback="deterministic" | "fail"`

Fairness and skeleton controls:

- `fairness_mode="off" | "local" | "global"`
- `fairness_weight`
- `fairness_iterations`
- `skeleton_mode="off" | "auto" | "required"`

Current posture:

- `skeleton_mode="auto"` falls back cleanly when skeleton guidance is unavailable
- `skeleton_mode="required"` fails explicitly if skeleton guidance is unavailable
- `fairness_mode="local"` is the default posture for bounded ambiguity smoothing
- `fairness_mode="global"` enables multi-interval optimization over ambiguous transitions

Resolve-mode split/merge demo:

- `docs/examples/loft/loft_split_merge_resolve_example.py`

Current constraints:

- stations must be strictly ordered by `t`
- frames must be right-handed orthonormal bases
- `loft(...)` and `loft_sections(...)` support multiple regions with deterministic minimum-cost region matching
- region and hole birth/death are supported via bounded synthetic transition loops
- split/merge behavior is controlled by `split_merge_mode` on both `loft(...)` and `loft_sections(...)`:
  - `fail` (default): reject split/merge-like ambiguity early
  - `resolve`: allow deterministic 1->N / N->1 decomposition using bounded synthetic loops
- true many-to-many (`N->M`, where `N>1` and `M>1`) remains unsupported and fails explicitly
- non-flat cap shaping on `loft(...)` currently requires one connected region per profile
- surfaced `Loft(...)` is the canonical direct `SurfaceBody` lane
- `loft(...)` and `loft_sections(...)` remain convenience handoffs over the same planner

## Correspondence Regression Fixtures

The loft test suite includes simple correspondence fixtures intended to reveal
twist or deformation regressions quickly:

- a rectangular station sequence that should preserve strong cross-section anisotropy through the mid-body
- a circular station sequence that should remain round through the mid-body

Those fixtures live in the loft test/showcase layer and are paired with
reference images and STL artifacts so correspondence regressions show up both
numerically and visually.

Split/merge controls:

- `split_merge_mode="fail" | "resolve"` (default `"fail"`)
- `split_merge_steps` controls how many micro-stations are injected in the transition window (default `8`, must be `>= 1`)
- `split_merge_bias` centers the transition window inside each station interval (default `0.5`, range `[0, 1]`)

Supported split/merge classes in `split_merge_mode="resolve"`:

- region split: `1->N`
- region merge: `N->1`
- hole split: `1->N`
- hole merge: `N->1`

Unsupported split/merge classes (explicit failure):

- many-to-many ambiguity: `N->M` where `N>1` and `M>1`

## Runtime Error Reference

These strings match current runtime behavior and are useful when debugging CI or scripted runs:

- `split_merge_mode must be 'fail' or 'resolve'.`
- `split_merge_steps must be >= 1.`
- `split_merge_bias must be within [0.0, 1.0].`
- `Stations must be strictly ordered by t.`
- `Station frame at index <i> must be unit-length.`
- `Station frame at index <i> must be orthogonal.`
- `Station frame at index <i> must be right-handed.`
- `Unsupported topology transition: region split/merge ambiguity detected.`
- `Unsupported topology transition: hole split/merge ambiguity detected.`
- `Unsupported topology transition: region split/merge detected.`
- `Unsupported topology transition: hole split/merge detected.`

## End Caps

Use `start_cap` and `end_cap` to round or taper the ends of a loft:

- `none` (default): no extra cap geometry
- `flat`: cap with the base profile (same as `cap_ends=True`)
- `taper`: linearly shrink the profile to a tip
- `dome`: half‑circle profile (true dome)
- `slope`: steeper start, gentler finish (inverse‑dome)

If either `start_cap` or `end_cap` is not `none`, the loft is automatically
closed at both ends. `cap_ends=True` remains as a backward‑compatible shortcut
for a flat cap.

Cap length is additive by default: the cap extends beyond the path endpoints.
Use `start_cap_length` / `end_cap_length` (in model units) to control how far
the cap blends. The blend eases to the last **non‑degenerate** profile over the
specified length. Use `cap_scale_dims` to control which axes scale:

- `both` (default): uniform scale in X/Y until the smallest dimension collapses
- `smallest`: scale only the limiting dimension

Caps use eased profiles (linear, sine, or quadratic) rather than pure linear ramps.
If no length is provided, `cap_steps * path_step` is used.

```python
from impression.modeling import loft
from impression.modeling.drawing2d import make_rect

def build():
    profiles = [make_rect(size=(1.0, 1.0))] * 5
    return loft(
        profiles,
        start_cap="dome",
        end_cap="taper",
        start_cap_length=2.0,
        end_cap_length=3.0,
        cap_scale_dims="both",
    )
```

## loft_endcaps(profiles, ..., endcap_mode=...)

`loft_endcaps(...)` is an experimental alternate endcap pipeline for comparing cap strategies.

```python
from impression.modeling import loft_endcaps

def build():
    return loft_endcaps(
        profiles,
        endcap_mode="ROUND",
        endcap_depth=1.2,
        endcap_radius=0.9,
        endcap_parameter_mode="independent",
        endcap_steps=24,
        endcap_placement="BOTH",
    )
```

Parameters:

- `endcap_mode`: `"FLAT"`, `"CHAMFER"`, `"ROUND"`, `"COVE"`
- `endcap_depth`: axial cap extent (required for non-flat modes unless `endcap_amount` is provided)
- `endcap_radius`: radial inset amount (required for non-flat modes unless `endcap_amount` is provided)
- `endcap_parameter_mode`: `"independent"` (default) or `"linked"`
- `endcap_steps`: segment count for smooth modes (`ROUND`, `COVE`)
- `endcap_placement`: `"START"`, `"END"`, or `"BOTH"`

Compatibility:

- `endcap_amount` is still accepted as a legacy shorthand and maps to both depth and radius.
- In `linked` mode, depth and radius must match (quarter-circle interpretation).
- In `independent` mode (default), depth and radius are orthogonal and can produce elliptical cap profiles.

Examples:

- `docs/examples/loft/loft_endcaps_profiles_example.py`
- `docs/examples/loft/loft_endcaps_compare_example.py`

## Loft Showcase Director

Use the curated multi-scene demo:

```bash
impression preview docs/examples/loft/loft_showcase.py
```

Switch scenes in the same preview window:

```bash
python scripts/dev/loft_demo.py list
python scripts/dev/loft_demo.py set caps-lab
python scripts/dev/loft_demo.py set topology-transitions
```

Auto-cycle (timed demo experience):

```bash
python scripts/dev/loft_demo.py play --interval 6 --loops 3
```

The showcase covers:

- endcap modes (`FLAT`, `CHAMFER`, `ROUND`, `COVE`) on mixed profile topology
- path-oriented loft choreography with asymmetric caps
- `loft_sections` topology transitions (region/hole birth/death)
- `text_profiles -> loft` sculpt workflow
