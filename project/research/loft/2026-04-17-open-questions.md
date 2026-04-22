# Loft Open Questions Research

## Topic

Open research questions for the next-generation loft architecture.

This document records what is already settled, what remains open, and what
evidence would be needed before those open questions should harden into loft
specifications or implementation policy.

## Findings

### 1. Core architectural direction is now stable enough

The current loft architecture appears settled on these major points:

- loft is a deterministic surface constructor, not an interpolation tool
- loft consumes placed topology states over progression
- authored correspondence is directional and relationship-first:
  - `predecessor_ids`
  - `successor_ids`
- standalone authored region `id` is not currently required for known loft
  problems
- ambiguity is a planning-time request for added directional correspondence,
  not an executor concern
- execution consumes only resolved plan output
- neighboring resolved structure may influence an interval with a default
  one-interval radius on either side
- `N -> M` / `M -> N` reduction order is currently:
  1. isolate unresolved subset
  2. apply predecessor/successor constraints
  3. resolve direct correspondences
  4. resolve clear births/deaths
  5. reduce remaining structure into deterministic `1 -> N` / `N -> 1`

These decisions appear strong enough to support later loft specification work.

### 2. Most remaining loft questions are not about philosophy

The remaining loft questions are now mostly about:

- decomposition gates
- tolerance families
- invalid-versus-underconstrained classification
- how much current implementation behavior should become durable contract

This is good. The architecture branch no longer appears blocked on first-order
conceptual disagreement.

### 3. Automatic decomposability gate is now simpler, but still needs runtime evidence

The current working rule is:

> a subset is automatically decomposable while the planner can keep
> deterministically reducing regions under the normal reduction order

The gate is reached when:

- unresolved regions still remain
- no additional regions can be decomposed without new directional constraints

This resolves most of the earlier conceptual uncertainty, but still leaves open
whether the current reduction order and neighboring-evidence policy are
sufficient in practice across real many-to-many fixtures.

### 4. Directional correspondence appears sufficient for authored control

Based on the current known loft problem set, directional correspondence seems
to solve the authored-input side cleanly.

The strongest concrete case discussed was:

- a middle topology has structures that must correspond backward differently
  than they correspond forward

That is naturally expressed by:

- `predecessor_ids`
- `successor_ids`

This weakens the need for:

- standalone authored region IDs
- authored lineage primitives

The remaining planner need is internal bookkeeping, not richer authored
identity.

### 5. Planner-internal graph structure should remain internal

A planner-owned directed graph is still useful for:

- decomposition bookkeeping
- synthetic branch references
- diagnostics
- optional debug visualization

But the graph does not appear to need architectural elevation into the authored
model.

Research should therefore treat graph representation as:

- an implementation-support structure
- optionally exportable for debugging
- not a user-facing loft concept

### 6. Invalid input and underconstrained input need a hard boundary

The current split is promising:

Invalid input:

- malformed topology
- broken containment
- contradictory predecessor/successor attachment
- invalid planner configuration

Underconstrained input:

- unrelated ambiguity
- ambiguity inside related regions
- multiple valid decompositions remain after deterministic reduction

This boundary should be preserved.

Current architectural direction is that ambiguity inside related regions does
not use a separate rule system. The planner should apply the same deterministic
reduction rules inside that related subset and only report the residual
ambiguity left after decomposition.

### 7. Tolerance policy can be seeded from current loft implementation, but is
not mature enough to freeze

The current loft implementation already provides useful baseline behavior:

- `samples >= 3`
- strictly increasing station ordering
- `split_merge_steps >= 1`
- `split_merge_bias in [0, 1]`
- `ambiguity_max_branches >= 1`
- deterministic synthetic birth/death seeding
- explicit closure ownership

What is not yet mature enough to freeze as architectural doctrine:

- one named numeric collapse epsilon family
- one named ambiguity-threshold family
- one named boundary/seam tolerance family
- how loft-local tolerances should compose with future surface-body tolerances

This is a real research area, not just editorial cleanup.

### 8. Operator-to-surface mapping should stay downstream of SurfaceBody work

The loft architecture intentionally did not fully define:

- exact patch ownership
- seam ownership
- trim responsibility
- patch-boundary semantics

Those are surface-kernel questions first.

Loft will need answers there, but loft should consume those rules, not define
them.

That means loft research should stop short of inventing surface-body law.

## Implications

### Immediate implication

Loft architecture is close to stable enough for the next specification wave,
but only if the following remain explicitly marked as research-backed rather
than frozen:

- tolerance families

### Near-term implication

The next major architectural dependency is SurfaceBody.

Loft can continue to refine planning concepts, but it should not harden
executor geometry or seam law until the surface-body architecture settles those
contracts.

### Implementation implication

When implementation resumes on next-gen loft, the safest order is:

1. preserve current planner/executor separation
2. keep directional correspondence as the only authored relationship primitive
3. treat graph bookkeeping as internal
4. avoid freezing numeric tolerance contracts beyond what is already exercised
   in current loft code

## Open Questions Requiring Further Research

### Automatic Decomposability

- Does the current reduction order actually consume regions as expected across
  the known many-to-many examples?
- Are there real cases where region consumption stalls too early under the
  one-interval neighboring-evidence policy?
- Which unresolved residuals still need user tie-breaking after the planner has
  exhausted all deterministic region consumption?

### Ambiguity Taxonomy

- How fine-grained should diagnostic labeling be once the planner has already
  applied the normal deterministic rules inside a related subset?
- Is there value in distinguishing:
  - symmetry-driven residual ambiguity
  - containment-driven residual ambiguity
  - split/merge residual ambiguity
  after decomposition, or is one residual-ambiguity class enough?

### Tolerance Families

- What collapse epsilon should be architectural versus implementation detail?
- Which current thresholds are stable enough to become named loft policy?
- How should loft-local thresholds interact with future surface-body seam and
  trim tolerances?

### Planner Graph Debugging

- What exported debug format would be most useful:
  - graph nodes/edges JSON
  - plan-embedded graph fragments
  - visualization-oriented intermediate structure
- Which internal graph details are useful to expose without turning them into
  public API?

## Suggested Research Tasks

1. Build a small fixture set of related-region ambiguities and verify that the
   normal deterministic reduction order behaves consistently inside those
   subsets before any ambiguity is reported.
2. Extract the current loft threshold behavior from `src/impression/modeling/loft.py`
   into one comparative research note before naming any permanent tolerance
   families.
3. Exercise many-to-many cases with and without neighboring interval evidence
   to determine whether the current one-interval propagation radius is
   sufficient in practice.
4. Prototype a planner-internal graph export for debugging only, then evaluate
   whether the information is actionable enough to keep.

## References

- `project/architecture/loft-evolution-system.md`
- `project/architecture/loft-planner-executor-architecture.md`
- `project/architecture/loft-ambiguity-and-diagnostics.md`
- `project/architecture/loft-plan-object-architecture.md`
- `project/architecture/loft-tolerance-and-degeneracy-architecture.md`
- `project/architecture/loft-nm-mn-decomposition-architecture.md`
- `project/specifications/loft-14-planner-executor-architecture-v1_0.md`
- `project/specifications/loft-15-region-1n-n1-planning-v1_4.md`
- `project/specifications/loft-16-region-nm-mn-branch-planner-v2_0.md`
- `project/specifications/loft-17-deterministic-ambiguity-resolver-v1_0.md`
- `src/impression/modeling/loft.py`
