# Control Station Inference Semantics

## Topic

Semantic contract for future control-station inference in loft.

## Findings

### The primary inferred truth should be loop / region evolution, not raw point trajectories

Current loft planning is already topology-first rather than point-cloud-first.
The main planner path in `src/impression/modeling/loft.py` converts sections
into region loops, plans correspondence over those loops, and only then
realizes geometry. The current public architecture says the same thing:

- loft is planner-driven and topology-aware
- `Loft(...)` is the canonical surfaced lane
- `loft_plan_sections(...)` and `loft_plan_ambiguities(...)` are the public
  inspection tools

That means a future inference tool should operate primarily on:

- region count and identity
- loop correspondence stability
- local evolution of region/loop descriptors
- span-local change patterns

It should not start from "fit every sampled point path independently."

Point trajectories are still useful, but they should be a downstream realization
detail derived from loop or correspondence-field intent.

### Topology-critical stations are the ones that change planner truth

A station should be treated as topology-critical when removing it would change
any of the following:

- region count at that progression location
- hole birth or hole death interpretation
- split / merge ambiguity class
- correspondence pairing that the planner would otherwise make
- required branch or closure structure in the loft plan

In other words, a topology station is not defined by visual importance alone.
It is defined by whether the planner's structural interpretation would change if
that station disappeared.

This aligns with the current planner behavior in `src/impression/modeling/loft.py`,
which already treats:

- region count transitions
- hole transitions
- split/merge staging
- ambiguity reporting

as first-class planning concerns.

### Shape-control stations should be defined by change in span behavior, not by arbitrary spacing

If topology stations preserve structural truth, then control stations should
preserve shape truth:

- curvature change
- growth or shrink rate change
- asymmetry change
- track drift that simple interpolation would not preserve well enough

This suggests that a control station is best understood as a retained
description of span behavior, not as merely "every nth station."

### Future control sections and inferred control stations should be related but not identical

The future-feature note correctly places this concept near user-authored
control sections, but they should stay distinct:

- control sections are authored user intent
- inferred control stations are machine-discovered structure from dense input

The cleanest long-term relationship is:

- inferred control stations become editable authored objects after acceptance
- but they retain provenance that they were inferred rather than originally
  authored

That gives the system a path from analysis to authoring without pretending the
two concepts are the same thing.

## Implications

Recommended semantic posture for a first architecture/spec pass:

- infer over region/loop/correspondence evolution
- classify retained stations into:
  - topology stations
  - control stations
- treat topology preservation as a hard requirement
- treat shape preservation as an optimized requirement with explicit error
  reporting
- define inferred control stations as promotable to authored control objects
  later, but do not require that UI/modeling layer in the first implementation

This keeps the feature aligned with the existing loft architecture rather than
turning it into a generic curve-fitting system.

## References

- `project/future-features/control-station-inference-architecture.md`
- `docs/modeling/loft.md`
- `docs/modeling/topology.md`
- `src/impression/modeling/loft.py`
