# Loft Topology and Correspondence Audit

Date: 2026-09-19  
Repository baseline: `main@ad25e6cefa1df1617b7c20c6ab3ecd3545afbabf`

## Purpose

This document audits the current loft implementation with one narrow goal: identify every path by which Impression can invent, infer, guess, or automatically select topological correspondence, then compare that behavior with the intended loft model.

The desired direction is **explicit topological evolution**. Loft should execute authored relationships; it should not decide which region, hole, path, or point "probably" corresponds to another.

This document deliberately distinguishes:

- **deterministic identity / deterministic execution** — desirable when it makes the same authored input produce the same output;
- **deterministic correspondence inference** — undesirable when it invents relationships that were not authored.

The second category is the architectural problem.

---

# 1. Target model

The intended conceptual model is:

```text
3DBSpline
    spatial progression curve
    frame/orientation transport along the curve

Station
    parameter/location on 3DBSpline
    position and orientation/frame derived from 3DBSpline
    Topology
    station-specific geometric properties

Topology
    collection of TopologyPaths

TopologyPath
    explicit identity/name
    ordered collection of TopologyPoints
    open/closed semantics

TopologyPoint
    explicit identity/name
    2D location in the Station-local plane
    explicit relationship(s) to TopologyPoints on adjacent Stations
    geometric interpolation policy for each relationship
        default: linear
```

The important separations are:

1. **3DBSpline owns spatial progression.**
2. **Station owns placement on that progression and one cross-sectional Topology.**
3. **Topology owns the set of topological paths that exist at that station.**
4. **TopologyPath owns ordered 2D boundary structure.**
5. **TopologyPoint owns cross-station geometric correspondence.**
6. **Loft consumes this graph and emits surfaces. It does not discover the graph.**

The current `Section -> Region -> Loop` model is therefore not the target aggregate. It may remain as a 2D planar modeling representation elsewhere, but it should not be the semantic core of loft.

---

# 2. Current loft has multiple competing models

The current code contains at least four overlapping notions of correspondence:

1. **TopologyPath rail correspondence**
2. **Section/Region/Loop geometric pairing**
3. **Point/sample correspondence**
4. **Synthetic topology lifecycle planning**

They are layered rather than unified. A named or explicit correspondence can enter the system and still be surrounded by downstream automatic matching machinery.

This is the main reason the named correspondence layer does not feel authoritative: it is currently one input signal among several.

---

# 3. Automatic correspondence paths that should be removed from loft semantics

## 3.1 Authored-rail resolver continues past explicit identity

Location: `src/impression/modeling/loft.py`

`resolve_authored_rails()` resolves in this order:

```python
RailSource.EXPLICIT_ID
RailSource.LANDMARK_NAME
RailSource.SEGMENT_NAME
RailSource.AUTHORED_ORDER
RailSource.GENERATED_RAIL
```

This means explicit correspondence is not the complete contract.

### Problematic paths

- **LANDMARK_NAME** — name coincidence can create correspondence.
- **SEGMENT_NAME** — segment-name coincidence can create correspondence.
- **AUTHORED_ORDER** — ordinal position can create correspondence.
- **GENERATED_RAIL** — generated-shape metadata can create correspondence.

The first two can be useful authoring conveniences **only if they are converted into explicit topology before loft execution**. They should not remain runtime matching rules.

`AUTHORED_ORDER` is particularly dangerous. Two points being fourth in two different station paths does not establish that they represent the same evolving topological point.

`GENERATED_RAIL` has the same architectural issue. A rectangle generator may know enough to author explicit point identities and relationships. Once generated, loft should consume those relationships exactly; it should not contain a special "generated rail" correspondence tier.

### Recommendation

Reduce runtime correspondence resolution to exactly one semantic source:

> explicit point relationships authored into the station topology graph.

Names may be used by builders to create those relationships, but the builder must materialize them before loft is called.

---

## 3.2 Cyclic-shift / reversal geometric point inference

Location: `src/impression/modeling/loft.py`

Important symbols:

- `score_correspondence_candidates()`
- `accept_or_refuse_inferred_correspondence()`
- `InferenceCandidateScore`
- `InferenceResult`
- `InferenceRefusalDiagnostic`

The candidate scorer tries every cyclic shift and reversal for equal-size paths and scores them using normalized geometric distance plus protected-anchor agreement.

Even though the implementation has refusal rules, this remains automatic topology inference.

### Why this conflicts with the target model

Under the target model, there is no meaningful question of:

> "Which shift of these vertices gives the cheapest match?"

A `TopologyPoint` states which point or points it connects to at the next station. If it does not state that relationship, the topology is incomplete.

### Recommendation

Remove this route from loft planning/execution.

If retained at all, move it into an **optional authoring-assistance utility** whose output is a proposed explicit topology graph that the caller can inspect/accept. It must never be an implicit execution path.

---

## 3.3 Region identity falls back to geometric assignment

Location: `src/impression/modeling/loft.py`

The region transition machinery distinguishes exact identity matches and geometric matches. Unnamed/unresolved regions are paired using minimum-cost assignment.

Relevant structures/functions include:

- region identity transition resolution
- `_pair_sections_for_transition()`
- `_minimum_cost_subset_assignment()`
- assignment candidate enumeration
- `geometric_pairs`

The cost system uses geometry to decide which disconnected region continues into which region.

### Why this is incorrect for the target model

Disconnected components / islands are topology. Their continuity is not a geometric optimization problem.

Two nearby islands can cross, exchange positions, merge, split, disappear, or reappear. Centroid distance cannot tell which semantic feature continues.

### Recommendation

Delete geometric region matching from loft execution.

A disconnected component must be represented by one or more named `TopologyPath` objects whose point relationships explicitly describe continuation, merge, split, birth, and death.

---

## 3.4 Hole pairing uses minimum-cost geometric assignment

Location: `src/impression/modeling/loft.py`

Hole expansion/collapse paths use `_minimum_cost_subset_assignment()` when no explicit override is supplied.

The same pattern appears for:

- more holes on the target station;
- fewer holes on the target station;
- stable hole sets;
- ambiguous hole transitions.

### Problem

A hole is not a geometry sample to be paired by proximity. It is a topological feature.

This route is directly contrary to the desired requirement:

> Loft must be able to open and close any number of holes and create, merge, and kill any topological feature.

That behavior requires explicit lifecycle relationships, not inferred assignment.

### Recommendation

Remove hole-specific correspondence logic from loft.

In the target model, "hole" is a consequence of path nesting/orientation/topological classification at a station, while cross-station continuity is represented by path/point relationship edges. There should not be a separate hole correspondence subsystem.

---

## 3.5 `minimum_cost_loop_assignment()`

Location: `src/impression/modeling/topology.py`

This helper performs deterministic one-to-one loop assignment using:

- centroid distance;
- area delta;
- lexicographic tie-breaking.

It is exported as public topology functionality.

### Recommendation

It must not participate in loft execution.

It may remain as a standalone analysis/authoring-assistance utility if useful, but its name/API should make clear that it proposes a geometric assignment and does not establish topology.

---

## 3.6 `minimum_cost_subset_assignment()`

Location: `src/impression/modeling/topology.py` and a richer loft-local version in `src/impression/modeling/loft.py`

This performs source-to-subset matching for differing loop counts.

The loft-local variant adds:

- ambiguity cost profiles;
- branch limits;
- local/global fairness;
- fairness weights;
- candidate enumeration.

### Recommendation

Remove from loft semantics.

This is sophisticated machinery solving a problem that explicit topology makes unnecessary.

---

## 3.7 Deterministic ambiguity selection

Current public loft entrypoints expose:

- `ambiguity_mode`
- `ambiguity_selection`
- `ambiguity_selection_policy`
- `ambiguity_cost_profile`
- `ambiguity_max_branches`
- `disambiguation_mode="deterministic"`

These exist because loft planning admits under-specified topology and then tries to choose among possible interpretations.

### Recommendation

There should be no runtime disambiguation mode in the target loft.

There should be only:

- valid complete topology -> execute;
- invalid/incomplete topology -> fail with a precise missing relationship diagnostic.

An interactive topology editor or migration tool can offer candidate fixes outside loft.

---

## 3.8 Probabilistic disambiguation

Current public entrypoints also expose:

- `probabilistic_trials`
- `probabilistic_temperature`
- `probabilistic_min_confidence`
- `probabilistic_fallback="deterministic"`

The planner contains probabilistic candidate selection and deterministic fallback.

### Recommendation

Remove this entire concept from loft.

Probabilistic topology inference is particularly incompatible with a CAD/modeling kernel because identical geometry can represent different design intent. Reproducibility does not solve the semantic ambiguity.

Again, this could theoretically live in an external authoring assistant, never in the model executor.

---

## 3.9 Fairness-based correspondence optimization

Current loft exposes and stores:

- `fairness_mode`
- `fairness_weight`
- `fairness_iterations`
- fairness objective pre/post state;
- branch crossing scores;
- curvature continuity scores;
- closure stress.

Fairness is currently entangled with candidate assignment.

### Important distinction

**Fairness of geometry is legitimate.  
Fairness as evidence for topological correspondence is not.**

After point relationships are explicit, a geometric solver may optimize the actual surface interpolation while preserving the authored graph.

### Recommendation

Keep fairness only as a geometric interpolation/surface-quality stage after correspondence is fixed.

It must not choose which feature maps to which feature.

---

## 3.10 Skeleton-guided correspondence

Current loft exposes `skeleton_mode="auto"` and checks whether skeleton guidance is available.

### Recommendation

As with fairness, skeleton information may guide geometry generation **after explicit topology exists**.

It must not select topology.

If its only purpose is correspondence/disambiguation, remove it from loft.

---

## 3.11 Generated rails from generated shapes

Location: `src/impression/modeling/topology.py`

Generated rounded rectangles and other generated paths create `GeneratedRailProvenance`, generated names, landmarks, and correspondence IDs.

### This functionality is not inherently wrong

A rectangle generator actually knows its semantic corners and tangent transitions. It should encode them.

### The architectural mistake

The generated-rail status becomes a special correspondence source interpreted by loft.

### Recommendation

Shape generators should directly produce normal explicit:

- `TopologyPoint` identities;
- `TopologyPath` identities;
- cross-station relationship IDs/edges when constructing station families.

After construction, there should be no runtime distinction between "generated" and "authored" topology.

---

## 3.12 Ordinal-derived point identity

`TopologyPoint.__post_init__()` currently derives IDs from:

1. name, if supplied;
2. otherwise `point-{ordinal}`.

This is deterministic identity generation, but it becomes unsafe if the resulting ID is treated as semantic cross-station identity.

### Recommendation

Ordinal-derived IDs can exist only as **station-local object IDs**.

They must never imply cross-station correspondence.

A cross-station relationship must be separately explicit.

This distinction should be represented in the type system rather than left to convention.

---

# 4. Synthetic topology paths currently generated by loft

The planner currently creates synthetic entities for births, deaths, split/merge transitions, and support geometry.

Important concepts include:

- `SyntheticSupportReference`
- `SyntheticRegionLineage`
- `SyntheticStationLineage`
- `PointLifecycleState.SYNTHETIC_BIRTH_SUPPORT`
- `PointLifecycleState.SYNTHETIC_DEATH_SUPPORT`
- shrunken-loop seeds
- inserted split/merge stations
- synthetic region/path lineage

This machinery exists largely because the planner is trying to convert incomplete station descriptions into executable topology.

## Target posture

Synthetic **geometric construction** can still be useful internally.

Synthetic **semantic correspondence** should not be.

For example, if a hole is explicitly authored as being born between station A and B, the surface generator may need an internal degenerate curve, singular point, intermediate station, or transition patch. That is an implementation detail derived from an explicit lifecycle edge.

The implementation may create synthetic support geometry, but it must not infer that the birth exists.

---

# 5. Current point-lifecycle direction is closer to the desired architecture

The existing code already contains useful concepts:

- point lifecycle events;
- parent spans;
- birth/death events;
- synthetic support references;
- explicit correspondence IDs;
- protected topology points.

These should be simplified around the new `TopologyPoint` relationship model rather than discarded wholesale.

The key change is ownership:

> lifecycle is a property of explicit point relationships, not a planner inference result.

A point relationship can naturally encode:

```text
1 -> 1    continuation
1 -> N    split / feature birth
N -> 1    merge / feature death
1 -> 0    termination
0 -> 1    birth
N -> M    general junction, if explicitly authored
```

Loft then constructs the required transition surface from that graph.

---

# 6. Too many public and internal routes into loft

The current module exposes or maintains several overlapping entrypoints:

- `loft_profiles(...)`
- `loft(...)`
- `Loft(progression, stations, topology, ...)`
- `_loft_profiles_surface(...)`
- `loft_sections(...)`
- `loft_plan_sections(...)`
- `loft_execute_plan(...)`
- debug mesh executor variants
- profile normalization from `Section | Region | Path2D | object`
- station coercion from tuples and coordinate sequences

This has two costs:

1. multiple representations can enter at different semantic levels;
2. every compatibility path keeps old assumptions alive.

## Recommended public surface

Prefer one canonical constructor/executor:

```python
loft(
    path: BSpline3D,
    stations: Sequence[Station],
    ...
) -> SurfaceBody
```

where:

```python
class Station:
    parameter: float
    topology: Topology
    ...
```

Position/orientation come from the 3D B-spline frame policy.

Alternative convenience APIs should be **builders that produce these canonical types**, not alternate execution paths.

For example:

```python
station_from_region(...)
topology_from_path2d(...)
topology_from_polygon(...)
stations_from_profiles(...)
```

may be useful migration helpers, but each must return canonical objects before loft execution begins.

---

# 7. Python typing should enforce the semantic boundary

The current API accepts broad unions such as:

```python
Section | Region | Path2D | object
```

and station tuples.

This allows malformed or semantically incomplete topology to penetrate deep into the planner.

The desired architecture should make invalid states harder to express.

## Recommended direction

```python
@dataclass(frozen=True)
class TopologyPoint:
    id: TopologyPointId
    xy: Vec2
    next: tuple[PointRelationship, ...] = ()
    previous: tuple[PointRelationship, ...] = ()

@dataclass(frozen=True)
class TopologyPath:
    id: TopologyPathId
    points: tuple[TopologyPoint, ...]
    closed: bool = True

@dataclass(frozen=True)
class Topology:
    paths: tuple[TopologyPath, ...]

@dataclass(frozen=True)
class Station:
    parameter: float
    topology: Topology

@dataclass(frozen=True)
class LoftDefinition:
    path: BSpline3D
    stations: tuple[Station, ...]
```

Exact names can change, but the type boundary should not.

In particular:

- `Topology.paths` must contain `TopologyPath`, not tuples;
- `TopologyPath.points` must contain `TopologyPoint`, not coordinate tuples;
- loft should not accept `object`;
- loft should not infer topology from `Path2D`;
- station relationships should be validated before surface execution.

---

# 8. Relationship representation

The idea that correspondence belongs to `TopologyPoint` is sound.

One refinement is recommended: store explicit **relationship records** rather than direct Python object references.

Example:

```python
@dataclass(frozen=True)
class PointRelationship:
    target_station: StationId
    target_point: TopologyPointId
    interpolation: PointInterpolation = LinearPointInterpolation()
    role: Literal[
        "continue",
        "split",
        "merge",
        "birth",
        "death",
        "junction",
    ] = "continue"
```

Reasons:

- avoids circular object graphs;
- serializes cleanly to `.impress`;
- makes validation straightforward;
- supports one-to-many and many-to-one relationships;
- allows provenance and debugging;
- supports future nonlinear trajectories without changing point ownership.

"Linear" should be the default **geometric relationship/interpolation**, not the fallback correspondence algorithm.

---

# 9. TopologyPath identity versus TopologyPoint relationships

Path identity is still useful, but it should not secretly perform point matching.

A `TopologyPath` identity answers:

> What boundary/topological feature is this path at this station?

A `TopologyPoint` relationship answers:

> How does this specific location on that boundary evolve toward adjacent stations?

For a path that splits into two paths, path-level identity alone is insufficient. Point edges provide the actual junction graph.

Likewise, path identity makes holes/islands understandable and debuggable without needing a separate "hole matching" mechanism.

---

# 10. Holes, islands, births, merges, and deaths

The target graph should not have special-case correspondence algorithms for holes or islands.

At each station, topology can classify paths by containment:

- exterior boundary;
- hole boundary;
- island within a hole;
- nested hole/island levels;
- disconnected exterior region.

That classification describes the **station-local topology**.

Cross-station point relationships describe evolution.

Therefore the same graph can express:

- zero holes -> five holes;
- five holes -> zero holes;
- one hole -> three holes;
- three holes -> one hole;
- island birth inside a hole;
- hole merging into exterior boundary;
- disconnected component splitting;
- arbitrary feature death;
- arbitrary feature creation.

The executor's responsibility is to build valid transition surfaces for the explicit graph.

This removes the current explosion of separate cases for "hole assignment", "region assignment", "split/merge mode", and inferred synthetic lineage.

---

# 11. Stations and the 3D B-spline

Current `Station` stores an explicit frame:

- `origin`
- `u`
- `v`
- `n`

The target architecture should make the relationship between the station and 3D path explicit.

## Recommended semantic contract

A station stores at minimum:

- parameter/distance on the 3D B-spline;
- topology;
- optional explicit frame override/twist/scale policy.

The path/frame subsystem computes:

- origin;
- tangent;
- local U/V plane;
- transported orientation.

An explicit frame override can be supported when needed, but it should be an override of the path-derived frame rather than a separate unrelated placement model.

This prevents spatial progression logic from being mixed with topological correspondence.

---

# 12. Control-station inference

The repository contains separate:

- `control_station_inference.py`
- `control_stations.py`
- curve-intent and trajectory inference work.

The proposed architecture makes "control" a property of geometry/interpolation rather than a loft topology concept.

That is preferable.

A topological point may use a geometric relationship that internally has spline control data, tangent constraints, or other interpolation parameters. Loft does not need a separate semantic category of "control station" to establish correspondence.

Control-station inference may remain useful as an external curve-fitting/design-assistance feature, but it should not be a prerequisite for canonical loft execution.

---

# 13. What should remain deterministic

Removing deterministic correspondence does **not** mean removing deterministic behavior.

The following should remain deterministic:

- stable serialization;
- validation order;
- station ordering;
- path ordering where needed for canonical payloads;
- point relationship execution;
- patch ownership;
- seam naming;
- tessellation;
- transition decomposition of an already explicit graph;
- CSG;
- diagnostics;
- cache identities.

The distinction is:

> deterministic execution of authored intent is required; deterministic invention of intent is not.

---

# 14. Proposed loft validation invariants

Before any surface generation begins:

1. Every `Station` references exactly one canonical `Topology`.
2. Every `Topology` contains only `TopologyPath` values.
3. Every `TopologyPath` contains only `TopologyPoint` values.
4. Point IDs are unique within the required identity scope.
5. Path IDs are unique within a station.
6. Every relationship target exists on the immediately adjacent station unless explicitly declared as a multi-station geometric relation.
7. Relationship direction is reciprocal or can be deterministically normalized into reciprocal graph edges.
8. No relationship is inferred from coordinate proximity.
9. No relationship is inferred from ordinal position.
10. No relationship is inferred from equal names unless a builder has explicitly materialized that relationship.
11. No relationship is inferred from path containment class ("hole", "outer", "island").
12. Every topological feature that persists, births, dies, splits, or merges has enough explicit relationships to construct its transition.
13. Ambiguous/incomplete graphs fail before geometry generation.
14. Failure diagnostics identify the exact station, path, point, and missing/conflicting relationship.
15. Surface generation cannot mutate topology to make an incomplete graph executable.

---

# 15. Recommended removal / quarantine inventory

## Remove from canonical loft execution

- authored-order rail matching;
- generated-rail matching as a special runtime tier;
- cyclic shift correspondence scoring;
- reversal candidate scoring;
- inferred-correspondence acceptance;
- geometric region matching;
- geometric hole matching;
- minimum-cost loop/subset assignment;
- deterministic ambiguity selection;
- probabilistic disambiguation;
- probabilistic deterministic fallback;
- correspondence selection by fairness;
- correspondence selection by skeleton;
- implicit split/merge assignment;
- automatically inferred feature identity.

## Convert into authoring/migration helpers if useful

- name-based matching;
- generated shape semantic naming;
- geometric nearest/minimum-cost suggestions;
- cyclic-shift suggestions;
- ambiguity candidate generation;
- probabilistic suggestions;
- skeleton/fairness correspondence suggestions.

Their output must be an explicit `Topology` / point-relationship graph before canonical loft runs.

## Keep, but rebase on explicit relationships

- lifecycle validation;
- synthetic geometric support generation;
- junction construction;
- surface patch interpolation;
- cap construction;
- seam construction;
- point resampling for tessellation;
- fairness optimization of already-corresponded geometry;
- self-intersection validation.

---

# 16. Major architectural conclusion

The existing loft implementation is not merely missing a stronger named-correspondence priority rule.

It has two conflicting philosophies:

### Current philosophy

> Accept partially described station geometry, infer likely topology/correspondence, plan synthetic topology, then execute.

### Desired philosophy

> Accept a complete topological evolution graph, validate it, then execute geometry.

Trying to preserve both will continue producing cases where automatic behavior "grabs hold" despite explicit naming.

The clean repair is therefore not to tune inference priority. It is to **remove inference from the canonical loft executor entirely**.

---

# 17. Proposed canonical flow

```text
BSpline3D
    |
    +--> Station(parameter, Topology)
    |       |
    |       +--> TopologyPath
    |               |
    |               +--> TopologyPoint --explicit edge--> adjacent TopologyPoint
    |
    +--> Station(parameter, Topology)
    |
    +--> Station(parameter, Topology)

                |
                v

        validate topology graph
                |
                v
        derive transition cells
                |
                v
     evaluate geometric relations
       (linear by default)
                |
                v
         construct patches
                |
                v
       construct seams/caps
                |
                v
          SurfaceBody
```

There is intentionally no "find correspondence" phase.

---

# 18. Consequences for the eventual repair plan

A repair plan generated from this audit should treat the following as non-negotiable:

1. Introduce/finish the canonical `Topology` aggregate.
2. Make `Station.topology: Topology` the only canonical loft topology input.
3. Make station placement explicitly dependent on the 3D B-spline.
4. Put adjacent-station relationships on `TopologyPoint` through serializable relationship records.
5. Make linear relationship interpolation the default.
6. Remove the current control concept from loft topology.
7. Remove automatic correspondence from runtime loft planning.
8. Remove `Section/Region/Loop` as canonical loft semantics.
9. Collapse public loft entrypoints around one typed canonical route.
10. Preserve convenience builders only if they compile to the canonical graph first.
11. Make arbitrary feature birth/death/split/merge a normal graph operation, not a special ambiguity mode.
12. Do not allow "incomplete but inferable" topology through the execution boundary.

The resulting loft should be substantially smaller because much of the current planner exists to infer information the new data model requires the author/builders to provide explicitly.
