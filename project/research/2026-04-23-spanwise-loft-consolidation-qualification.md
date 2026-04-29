# Spanwise Loft Consolidation Qualification

## Topic

Qualification rules and success criteria for deciding when a run of loft
intervals should be treated as one larger coherent span.

## Findings

### The first exact qualification should require stable one-to-one structure

Current loft planning and surfaced execution are still strongly interval-based.
The private surfaced executor in `src/impression/modeling/loft.py` explicitly
maps loop pairs to ruled patches and notes that it does not yet attempt full
shell-level seam orchestration across all branch interactions.

That means the safest exact first-pass consolidation target is a run where:

- each interval stays one-to-one in region structure
- loop correspondence is stable across the run
- no hole birth/death or split/merge event occurs
- no ambiguity-selection event changes the branch structure

In other words, the first exact notion of "one coherent span" should be:

- stable topology
- stable correspondence
- compatible local surface family

rather than "looks smooth enough."

### First-pass consolidation should optimize for fewer patches before it optimizes for new patch families

The current surfaced kernel requires:

- planar
- ruled
- revolution

as the v1 required patch families in `src/impression/modeling/surface.py`.

Current loft sidewalls are realized primarily as ruled patches. That makes the
lowest-risk first target:

- fewer ruled patches

rather than:

- immediate refitting into a new higher-order patch family

So the initial objective should be:

1. fewer patches
2. no structural drift
3. seam reduction only when exact

Patch-family substitution and seam relocation are valuable future targets, but
they should not be the first qualification gate.

### Topology events should initially terminate exact consolidation spans

A topology event that is locally real but globally over-segmenting may still be
valuable in later approximate or repair-oriented tooling, but it should not be
folded into the first exact consolidation contract.

For a first exact lane, the following should break the span:

- region count changes
- hole count changes
- split/merge staging
- ambiguity intervals
- closure-cap behavior changes

Those are the places where the current planner is expressing real structural
facts, not just inconvenient density.

### Seam placement is a downstream optimization target, not the initial proof of coherence

Because the current surfaced loft executor is already conservative about seam
orchestration, a first consolidation program should use seam reduction as a
result of proven span coherence, not as the primary definition of coherence.

So the right order is:

1. prove the run is one exact structural span
2. then simplify patch/seam structure

not:

1. relocate seams and hope the span was actually coherent

## Implications

Recommended exact first-pass qualification rule:

- same region cardinality across the run
- same loop correspondence family across the run
- no ambiguity intervals
- no split/merge staging
- no hole birth/death
- same local patch family assumption

Recommended first success target:

- reduce patch count exactly
- preserve visible shape exactly within explicit numeric tolerance
- preserve structural planner truth

This favors honesty and a narrow initial lane over a broader but less
trustworthy consolidation story.

## References

- `project/future-features/spanwise-loft-consolidation-architecture.md`
- `project/future-features/spanwise-loft-inline-enhancement-architecture.md`
- `project/future-features/spanwise-loft-postprocessing-optimization-architecture.md`
- `src/impression/modeling/loft.py`
- `src/impression/modeling/surface.py`
- `docs/modeling/loft.md`
