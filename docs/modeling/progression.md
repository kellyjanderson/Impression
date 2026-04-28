# Modeling — Progression

`PathBackedProgression` is the first `0.1.0.a` step toward treating progression
as a semantic object built on `Path3D`, rather than as loose scalar arrays.

```python
from impression.modeling import Path3D, PathBackedProgression

path = Path3D.from_points([(0.0, 0.0, 0.0), (0.2, 0.0, 1.0), (0.4, 0.1, 2.0)])
progression = PathBackedProgression(path=path)
```

## Ownership

The progression object owns:

- the underlying `Path3D` spine reference
- the parameter domain for travel along that spine
- explicit-vs-inferred provenance

It does **not** replace `Path3D`. The path remains the geometric carrier; the
progression object owns the loft-travel semantics around that carrier.

## Provenance

Use `ProgressionProvenanceRecord` to keep authored and inferred progression
explicit and inspectable:

```python
from impression.modeling import ProgressionProvenanceRecord

provenance = ProgressionProvenanceRecord(
    kind="inferred",
    source="dense_station_inference",
)
```

This first milestone keeps the contract intentionally small:

- progression identity is stable and replayable
- progression is distinct from the raw path primitive
- later station-attachment and transport semantics can build on this object

## Station Attachment

Stations can now attach to progression explicitly instead of only living beside
parallel scalar progression arrays:

```python
from impression.modeling import ProgressionStationAttachment, Station, as_section
from impression.modeling.drawing2d import make_rect

station = Station(
    t=0.5,
    section=as_section(make_rect(size=(1.0, 1.0))),
    origin=(0.0, 0.0, 0.5),
    u=(1.0, 0.0, 0.0),
    v=(0.0, 1.0, 0.0),
    n=(0.0, 0.0, 1.0),
)
attachment = ProgressionStationAttachment.from_station(
    progression=progression,
    station=station,
    station_index=0,
)
```

That attachment contract keeps:

- progression identity explicit
- attachment ordering durable
- station-owned topology truth intact

## Transport Semantics

Transport semantics now live explicitly on the progression object too:

```python
from impression.modeling import ProgressionTransportPolicy

progression = PathBackedProgression(
    path=path,
    transport_policy=ProgressionTransportPolicy(kind="parallel_transport"),
)
contract = progression.loft_transport_contract
```

In this milestone:

- transport ownership is explicit on progression
- loft-facing consumption is represented by an inspectable contract object
- transport policy stays separate from later twist and scale semantic slots

## Twist and Scale Slots

Twist and scale semantics now have explicit owned slots on progression, even
though first-milestone execution remains deferred:

```python
from impression.modeling import (
    ProgressionScaleSemanticSlot,
    ProgressionTwistSemanticSlot,
)

progression = PathBackedProgression(
    path=path,
    twist_semantics=ProgressionTwistSemanticSlot(status="deferred"),
    scale_semantics=ProgressionScaleSemanticSlot(status="deferred"),
)
```

That keeps the contract honest:

- twist semantics are not hidden inside transport policy
- scale semantics are not hidden inside transport policy
- deferred execution does not erase the owned semantic slots

## Hidden Control Stations

The planner can also carry hidden control stations as internal records without
promoting them to public authored API:

```python
from impression.modeling import (
    HiddenControlStationProvenanceRecord,
    HiddenControlStationRecord,
)

record = HiddenControlStationRecord(
    station_id="control-0",
    origin=(0.0, 0.0, 0.5),
    u=(1.0, 0.0, 0.0),
    v=(0.0, 1.0, 0.0),
    n=(0.0, 0.0, 1.0),
    topology_reference=None,
    provenance=HiddenControlStationProvenanceRecord(),
)
```

These records are planner-owned and keep provenance explicit, but they are not
the same thing as public topology stations.

Planner consumption stays explicit too:

```python
from impression.modeling import HiddenControlStationPlannerConsumption

consumption = HiddenControlStationPlannerConsumption(
    planner_stage="fit_guidance",
    topology_station_ids=("topo-0", "topo-1"),
    hidden_control_station_ids=("control-0",),
)
```

That contract keeps hidden control stations on the planner side of the boundary
without letting them override topology truth or become public authored inputs.

## Dense Loft Descriptor Preparation

Fit-backed loft analysis can prepare dense descriptor bands deterministically
from ordered stations:

```python
from impression.modeling import prepare_dense_loft_fit_descriptors

descriptor_band = prepare_dense_loft_fit_descriptors(stations)
```

The initial descriptor band keeps:

- station ordering intact
- simple structural meaning such as region counts
- replayable records for later candidate-fit comparison

Curve-intent inference then builds explicit descriptor families from that band:

```python
from impression.modeling import build_curve_intent_descriptor_families

families = build_curve_intent_descriptor_families(descriptor_band)
```

The initial families are:

- section descriptors
- loop descriptors
- correspondence-track descriptors

Span-local curve-intent evidence can then be assembled from those families:

```python
from impression.modeling import assemble_span_local_curve_intent_evidence

evidence = assemble_span_local_curve_intent_evidence(families)
```

That assembled evidence keeps ordering explicit and gives later candidate
classification a stable downstream shape to consume.

Curve-intent candidate posture can then be classified explicitly:

```python
from impression.modeling import classify_curve_intent_candidate

report = classify_curve_intent_candidate(evidence)
```

The initial posture stays conservative:

- strong evidence yields a candidate report
- weak or conflicting evidence yields an explicit indeterminate posture

Station-derived candidate fits can then be generated and compared explicitly:

```python
from impression.modeling import (
    compare_station_derived_curve_fit_candidates,
    generate_station_derived_curve_fit_candidates,
)

candidates = generate_station_derived_curve_fit_candidates(descriptor_band)
selected, assessment = compare_station_derived_curve_fit_candidates(candidates)
```

The comparison contract always returns either:

- an accepted candidate with residual diagnostics
- or an explicit refusal

There is also a parallel shared-trajectory candidate lane:

```python
from impression.modeling import (
    compare_shared_trajectory_curve_fit_candidates,
    generate_shared_trajectory_curve_fit_candidates,
)

candidates = generate_shared_trajectory_curve_fit_candidates(descriptor_band)
selected, assessment = compare_shared_trajectory_curve_fit_candidates(candidates)
```

This keeps the shared-trajectory fit path using the same explicit residual and
refusal posture as the station-derived lane.
