# Loft Spec 15: Region `1->N` and `N->1` on Planner (v1.3)

This specification defines implementation of region split/merge on the
Planner/Executor architecture introduced in Spec 14.

Focus:

- region split: `1->N`
- region merge: `N->1`

All statements are normative unless marked as future work.

---

## 1. Scope

In scope:

- planner-native decomposition for region `1->N` and `N->1`
- deterministic branch ordering
- explicit closure ownership in plan artifacts
- executor consumption of planner-generated split/merge plans

Out of scope:

- many-to-many region ambiguity (`N->M`, `N>1`, `M>1`)
- hole-level many-branch planning

---

## 2. Inputs and Controls

Uses existing loft controls:

- `split_merge_mode: "fail" | "resolve"`
- `split_merge_steps: int`
- `split_merge_bias: float`

Planner behavior:

- `fail`: emit explicit error for `1->N` / `N->1`
- `resolve`: generate decomposed plan

---

## 3. Planner Decomposition Rules

### 3.1 Region Split (`1->N`)

Given one source region and `N` target regions:

1. select deterministic primary stable branch
2. create `N-1` synthetic birth branches
3. assign closure ownership (`prev`, loop/region as needed) for each birth branch
4. stage events over micro-stations according to controls

### 3.2 Region Merge (`N->1`)

Given `N` source regions and one target region:

1. select deterministic primary stable branch
2. create `N-1` synthetic death branches
3. assign closure ownership (`curr`) for each death branch
4. stage events over micro-stations according to controls

### 3.3 Deterministic Branch Selection

Primary branch selection must use deterministic minimum-cost rules and tie-breaks.
Branch order in plan must be stable across runs.

---

## 4. Plan Invariants

For every planned transition interval:

1. every planned branch has exactly one loop-pair path
2. each synthetic branch has exactly one closure owner side
3. no duplicate closure for same branch ref
4. all refs are resolvable in executor station context

Planner must validate invariants pre-execution.

---

## 5. Executor Requirements

Executor must:

- consume region split/merge branches directly from plan
- reuse canonical loop vertex starts across bridge and closure actions
- execute closures exactly once per planned closure record

Executor must not infer topology policy at runtime.

---

## 6. Error Contract

Required planner errors:

- unsupported split/merge mode for requested transition
- ambiguous correspondence in this scope
- invalid closure ownership map

Required executor errors:

- unresolved branch ref
- duplicate closure execution attempt
- closure mapping failure

---

## 7. Acceptance Tests

Required:

1. `1->N` region split in resolve mode passes mesh-quality gates
2. `N->1` region merge in resolve mode passes mesh-quality gates
3. `split_merge_steps` increases mesh density deterministically
4. deterministic replay equality for split and merge fixtures
5. fail mode rejects region split/merge

Mesh-quality gates:

- boundary edges = 0
- nonmanifold edges = 0
- degenerate faces = 0

---

## 8. Definition of Done

Spec 15 is complete when:

1. region `1->N` and `N->1` are planner-produced events (not ad hoc executor logic)
2. executor behavior is fully driven by plan records
3. acceptance tests pass in CI
4. docs and examples reflect planner-based split/merge execution
