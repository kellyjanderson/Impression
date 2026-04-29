# Control Station Inference Workflow

## Topic

Workflow, determinism, and user-facing acceptance model for future
control-station inference.

## Findings

### The first implementation should be offline simplification plus review

The current loft architecture already exposes public planner inspection tools,
not just black-box execution:

- `loft_plan_sections(...)`
- `loft_plan_ambiguities(...)`
- `loft_execute_plan(...)`

That makes the cleanest first implementation an offline analysis step:

```text
dense stations
-> infer_control_stations(...)
-> reduced proposal + diagnostics
-> user review
-> accepted reduced progression
```

This is a better fit than immediate interactive editing because it:

- preserves the current planner/executor split
- keeps the feature useful without first inventing a new UI/editor model
- allows deterministic evaluation before any richer authoring surface exists

### Error metrics should be section- and span-oriented, not only mesh-oriented

The acceptance model should primarily measure drift in terms the loft system
already understands:

- section silhouette drift
- loop area / centroid drift
- correspondence drift
- region-count preservation
- progression interval residuals

Mesh-level comparison is still useful, but it should remain secondary and
diagnostic because the project posture is surface-first and mesh is downstream.

The strongest likely metric stack is:

- structural equality:
  - same region counts
  - same hole counts
  - same ambiguity intervals
- section-space residuals:
  - contour overlap
  - area delta
  - centroid delta
  - optional station-slice comparisons
- surfaced-result diagnostics:
  - bounding-box drift
  - watertightness after tessellation

### User-pinned stations should be hard constraints

If the tool supports user pinning, those pins should be non-negotiable:

- pinned topology stations remain retained
- pinned control stations remain retained
- inference may optimize around them, but not delete or reclassify them without
  explicit user action

That gives the tool a clean contract:

- inference proposes
- pinning constrains
- acceptance finalizes

### Inferred results should remain deterministic

The current loft system explicitly values determinism in correspondence and plan
construction. A future inference tool should preserve that posture:

- same dense input
- same configuration
- same deterministic reduction result

If ranking, clustering, or fitting heuristics are used, they should still end
with deterministic tie-breaking and stable ordering.

This matters because control-station inference is likely to become part of the
authoring pipeline, and unstable retention would be very hard to reason about.

### Inferred control stations should become editable after acceptance, but not require a new authoring object immediately

The most practical first implementation is:

- return retained stations plus classification metadata
- allow later workflows to serialize or promote them

It does not need to introduce a brand-new control-station authoring object in
the same milestone.

The first milestone can instead treat them as:

- ordinary retained stations
- plus explicit classification / provenance metadata

That is enough to validate the concept before designing a new surface-level
authoring API.

## Implications

Recommended first implementation posture:

- offline analysis tool
- deterministic output
- hard user pins
- section-space and structural residual metrics
- accepted output returned as ordinary loft input plus station classification
  metadata

The first milestone should optimize for legibility and trust, not maximum
compression.

## References

- `project/future-features/control-station-inference-architecture.md`
- `docs/modeling/loft.md`
- `project/research/2026-04-20-visual-output-verification-ideas.md`
- `src/impression/modeling/loft.py`
