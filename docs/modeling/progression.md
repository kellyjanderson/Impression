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
