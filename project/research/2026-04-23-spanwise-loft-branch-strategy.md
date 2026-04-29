# Spanwise Loft Branch Strategy

## Topic

Comparative research on where spanwise loft consolidation should live and how
its inline, postprocess, and repair branches should differ.

## Findings

### The first practical implementation should live outside the planner

The cleanest architectural branch is inline planner recognition, but the
current planner/executor split is still comparatively young and the surfaced
executor remains conservative. The lowest-risk experimental path is therefore:

- start with a postprocess exact optimizer
- use that to learn what compatibility evidence is actually reliable
- promote only the proven cases into inline planning later

This gives the project a way to learn from real artifacts without first
rewriting the planner boundary.

### Inline consolidation should require explicit plan-level representation, not hidden executor magic

If inline consolidation is added later, it should not be implemented as a
silent executor shortcut. The planner must emit something explicit, such as:

- grouped interval runs
- a larger-span transition record
- or another stable plan-level representation

That is necessary to keep:

- `loft_plan_sections(...)`
- `loft_execute_plan(...)`
- plan inspection and debugging

honest and reproducible.

### Postprocess consolidation should start as exact merge, not refit

The postprocess branch should begin with exact-only eligibility:

- merge compatible adjacent spans
- do not yet invent new higher-order fits
- report when nothing qualifies exactly

That avoids turning the first tool into a surface guesser and gives the project
an evidence-based path toward later approximate refit modes.

### Approximation reporting should be explicit and multi-layered

Whenever consolidation is not exact, the tool should report:

- whether the result is exact or approximate
- geometric drift metrics
- seam changes
- patch-count reduction
- any topology constraints that blocked stronger consolidation

This is especially important for the postprocess and repair branches, because
those branches will otherwise be hard to trust.

### The repair branch should share analysis logic, not necessarily result policy

The repair branch can and should reuse:

- span compatibility analysis
- adjacency tests
- error metrics

But it should not automatically share the same acceptance policy as the clean
inline or postprocess branches.

Repair has different goals:

- clean up noisy or damaged geometry
- tolerate bounded deviation
- prioritize salvage and interpretability

That means the repair branch should share logic where possible, but own its own
acceptance thresholds and diagnostic posture.

### Foreign geometry should be in scope for repair, but not for exact consolidation

The repair branch is the right place to allow:

- foreign mesh-derived surface reconstruction
- noisy or imperfect source spans

The exact inline/postprocess consolidation branches should stay narrower and
bias toward loft-authored or surfaced-native inputs first.

That keeps the clean simplification story separate from the messier repair
story.

## Implications

Recommended branch order:

1. exact postprocess optimizer
2. inline planner promotion of proven exact cases
3. repair-oriented reinterpretation branch using shared analysis tools but its
   own acceptance policy

Recommended reporting posture:

- exact vs approximate must always be explicit
- seam relocation must be reported, not hidden
- repair deviation bounds must be tighter and more visible than ordinary
  simplification drift

## References

- `project/future-features/spanwise-loft-inline-enhancement-architecture.md`
- `project/future-features/spanwise-loft-postprocessing-optimization-architecture.md`
- `project/future-features/spanwise-loft-repair-tool-architecture.md`
- `src/impression/modeling/loft.py`
- `src/impression/modeling/surface.py`
- `docs/modeling/loft.md`
