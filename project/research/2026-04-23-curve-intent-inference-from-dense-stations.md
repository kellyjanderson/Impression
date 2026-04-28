# Curve Intent Inference From Dense Stations

## Topic

How dense faceted station input can communicate curved loft intent and how that
intent might be inferred algorithmically.

## Findings

### Dense stations can already act as a faceted description of a curve

The hourglass example is a good concrete case:

- many stations
- non-uniform station density
- visually smooth final result

That means the authored input is already carrying more information than "many
linear spans." It is communicating:

- where curvature matters
- where curvature changes quickly
- where a shape should linger or tighten

The main inference opportunity is to recover that higher-level curve intent
instead of preserving all facets literally.

### Station frequency over distance is a strong intent signal, but not enough by itself

High station density often means:

- higher curvature
- more delicate transition control
- ambiguity management

But station density alone is ambiguous. It might also mean:

- user caution
- manual experimentation
- topology event staging

So density should be treated as one signal among several, not as a complete
proof of curve intent.

### The best signals are density plus smooth descriptor continuity

The strongest likely inference stack is:

- station spacing over progression distance
- continuity of section descriptors across the run
  - area
  - centroid
  - anisotropy
  - hole offset / count stability
- stability of correspondence-track motion
- consistency of local second-derivative or acceleration-like change

In practical terms, the system should look for:

- many small changes
- but all pointing toward one smooth higher-level pattern

rather than:

- many unrelated abrupt shifts

### The right inferred unit is probably not raw point curvature

Because loft is topology-aware, the best signal is likely not:

- "point i follows a smooth cubic"

but instead:

- loop center moves smoothly
- loop size evolves smoothly
- corresponding features drift along a consistent path
- station density clusters where those trends bend sharply

That keeps curve-intent inference aligned with the rest of the loft system.

### Topology events should suppress or segment curve inference

A dense run that contains:

- hole birth/death
- split/merge transitions
- ambiguity intervals

should probably be segmented before curve intent is inferred.

Those are structural events first and smooth curve evidence second.

### The best first inference output is descriptive, not authoritative

A useful first tool would report:

- likely curved runs
- likely control stations
- candidate shared trajectory shapes or span classes

without immediately replacing the authored input.

That makes the feature safer and more inspectable while the project learns what
signals are actually reliable.

## Implications

Recommended first inference posture:

- segment out topology events first
- combine station density with smooth section/correspondence descriptor trends
- infer curve intent over loop / region evolution, not only over point samples
- return diagnostic candidate results before turning the inference into a hard
  authoring rewrite

This creates a path toward "the system can tell that a dense faceted run is
really a curve" without over-claiming certainty too early.

## References

- `project/future-features/trajectory-guided-loft-architecture.md`
- `project/future-features/control-station-inference-architecture.md`
- `docs/modeling/loft.md`
- `src/impression/modeling/loft.py`
- `docs/examples/loft/real_world/loft_hourglass_vessel_example.py`
