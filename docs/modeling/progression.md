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
