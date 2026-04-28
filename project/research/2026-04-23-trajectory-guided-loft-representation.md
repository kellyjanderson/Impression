# Trajectory-Guided Loft Representation

## Topic

Representation choices for trajectory-guided loft and how they should interact
with stations, planning, and control-station inference.

## Findings

### The first useful attachment level should be a whole-loft shared path

The existing codebase already has a meaningful `Path3D` abstraction and the
public loft convenience API already uses path guidance for station placement.
That makes a shared-path trajectory mode the easiest first extension because it:

- extends an existing user-visible concept
- preserves one coherent vertical intent for the whole loft
- avoids immediately inventing region or track identity APIs

Region-level and track-level trajectories are valuable, but they depend on
stable region or correspondence identifiers becoming first-class authoring
inputs.

### Station placement should remain the hard structural anchor in the first version

The first version should treat explicit station placement as authoritative.

That means trajectory guidance should primarily shape the evolution between
stations, not override station origins themselves.

In practical terms:

- stations define hard anchors
- trajectories shape in-between travel

This preserves the current meaning of stations and avoids surprising conflicts
between two different placement systems.

### First-pass trajectory guidance should influence interpolation, not replace topology planning

The most compatible initial posture is:

- preprocessing or planning aid
- not a replacement for the planner

Trajectory guidance should adjust how already-corresponded features travel
through space, while leaving:

- topology interpretation
- ambiguity handling
- structural planning

inside the existing planner boundary.

### Determinism is achievable if trajectories are sampled and attached deterministically

Nothing about trajectory guidance inherently breaks determinism, provided that:

- path sampling is deterministic
- attachment identities are stable
- tie-breaking and ordering remain explicit

This matches the current loft posture and should remain a hard requirement.

### Region-level guidance is the next logical step after whole-loft guidance

Once a shared-path version exists, the next useful layer is region-level
guidance because it:

- handles multi-lobe and multi-region forms
- still avoids the full complexity of per-track attachment
- aligns better with the planner's current region-based understanding than a
  raw node-level API would

### Track-level guidance is likely the richest but also the most expensive

Track-level trajectories are architecturally attractive for high-control loft
authoring, but they depend on a strong answer to:

- how correspondence identity is exposed durably
- how tracks survive topology changes

That makes them a poor first milestone but an excellent long-term branch.

### Trajectory guidance and control-station inference should eventually compose

These two future ideas are complementary:

- control-station inference reduces station density
- trajectory guidance preserves non-linear vertical behavior

A likely long-term workflow is:

```text
dense stations
-> infer control stations
-> infer or author shared / region trajectories
-> planner
-> executor
```

That suggests trajectory guidance should be designed so it can be added to a
reduced progression later, not only to a dense authored one.

## Implications

Recommended staging:

1. shared whole-loft trajectory guidance
2. region-level trajectory guidance
3. track-level trajectory guidance
4. coupling with control-station inference

Recommended first contract:

- stations remain hard anchors
- trajectories modify in-between evolution
- planner remains topology owner
- path sampling and attachment stay deterministic

## References

- `project/future-features/trajectory-guided-loft-architecture.md`
- `project/future-features/control-station-inference-architecture.md`
- `docs/modeling/path3d.md`
- `docs/modeling/loft.md`
- `src/impression/modeling/path3d.py`
- `src/impression/modeling/loft.py`
