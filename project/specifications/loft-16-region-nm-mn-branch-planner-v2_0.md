> Status: Deprecated historical spec.
> Superseded by the next-generation loft specification tree. Retained for
> project history only.

# Loft Spec 16: Region `N->M` / `M->N` Branch Planner (v2.0)

This specification defines deterministic planning/execution support for
many-to-many region topology transitions:

- `N->M` where `N>1`, `M>1`

This is the first branch-graph planner scope beyond `1->N` / `N->1`.

All statements are normative unless marked as future work.

---

## 1. Scope

In scope:

- planner branch-graph construction for many-to-many region transitions
- deterministic decomposition into stable/birth/death/split/merge branch actions
- executor support for plan-produced many-branch actions
- quality/failure guarantees

Out of scope:

- hole-level many-to-many in same milestone (can follow with dedicated spec)
- fairness optimization/skeleton smoothing beyond deterministic decomposition

---

## 2. Planner Model

Planner must construct a bipartite correspondence graph:

- left nodes: source regions
- right nodes: target regions
- weighted edges: correspondence compatibility cost

From this graph, planner produces a branch-transition DAG with ordered actions:

- stable branch actions
- split actions
- merge actions
- births
- deaths

---

## 3. Deterministic Optimization

Planner must be deterministic:

1. deterministic candidate edge generation
2. deterministic weighted matching/flow
3. deterministic tie-breaks for equal-cost branch assignments
4. deterministic action ordering in plan

Acceptable solver approach:

- deterministic min-cost flow or equivalent deterministic assignment cascade

No randomized search is allowed.

---

## 4. Decomposition Rules

For each interval:

1. allocate stable correspondences first (highest confidence / lowest cost)
2. classify remaining residual source/target mass into split/merge structures
3. materialize unresolved residuals as births/deaths where required
4. assign closure ownership per synthetic/residual branch

All branches must be representable as plan records consumable by executor.

---

## 5. Ambiguity Policy

Not all `N->M` cases are safely resolvable.

Planner must fail explicitly when:

- branch graph admits multiple equivalent decompositions with no deterministic tie-break
- geometric overlap/containment makes branch assignment unstable
- closure ownership cannot be assigned uniquely

Required failure category:

- `unsupported_topology_ambiguity` (with interval context)

---

## 6. Staging Strategy

Many-to-many transitions must be staged over micro-stations.

Planner defines:

- staging window (`split_merge_bias`)
- micro-station count (`split_merge_steps`)
- branch activation/deactivation schedule

Schedule must be monotone and deterministic.

---

## 7. Executor Contract for Many-Branch Plans

Executor must:

- execute branch actions in planner-specified order
- maintain canonical loop vertex reuse across branch actions
- enforce closure ownership exactly once per branch
- prevent interior duplicate caps and seam duplication

Executor must reject plan violations with explicit errors.

---

## 8. Mesh Quality Contract

Supported many-to-many outputs must satisfy:

- boundary edges = 0
- nonmanifold edges = 0
- degenerate faces = 0

If quality fails, execution must fail explicitly (no silent degraded mesh).

---

## 9. Acceptance Test Matrix

Required:

1. canonical `2->2` branch swap/redistribution case resolves and passes quality
2. canonical `2->3` and `3->2` cases resolve and pass quality
3. deterministic replay equality for representative `N->M` fixtures
4. staged density and bias controls produce deterministic, monotone structural effects
5. intentionally ambiguous `N->M` case fails with explicit ambiguity error

At least one real-world demo case in `docs/examples/loft/real_world/` must
exercise `N->M` successfully and pass analysis.

---

## 10. Definition of Done

Spec 16 is complete when:

1. planner can generate deterministic many-to-many region plans for supported cases
2. executor can execute those plans with quality guarantees
3. ambiguous `N->M` cases fail explicitly and deterministically
4. acceptance matrix is implemented and passing in CI
