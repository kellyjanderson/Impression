> Status: Deprecated historical spec.
> Superseded by the next-generation loft specification tree. Retained for
> project history only.

# Loft Spec 15: Region `1->N` and `N->1` on Planner (v1.4)

This specification defines planner-native handling of region split/merge for
`1->N` and `N->1` transitions, aligned with the current Planner/Executor loft
architecture.

All requirements are normative unless explicitly marked as future work.

---

## 1. Scope

In scope:

- deterministic planner decomposition for region `1->N` and `N->1`
- planner-native references and closure records
- planner metadata and plan validation contracts
- executor behavior driven only by planner records

Out of scope:

- many-to-many region ambiguity (`N->M`, `N>1`, `M>1`)
- hole many-branch planning beyond existing subset assignment behavior

---

## 2. Planner/Executor Controls

Loft control inputs:

- `split_merge_mode: "fail" | "resolve"`
- `split_merge_steps: int`
- `split_merge_bias: float`

Required behavior:

- `fail` supports deterministic, unambiguous `1->N` and `N->1` transitions
  without staged micro-station expansion
- `resolve` supports the same transitions and additionally enables staged
  micro-station decomposition controlled by `split_merge_steps` and
  `split_merge_bias`

---

## 3. Planner Data Contract

The planner output for this scope must use these records:

```text
LoftPlan
  samples: int
  stations: PlannedStation[]
  transitions: PlannedTransition[]
  metadata: dict
```

```text
PlannedTransition
  interval: (i, i+1)
  region_pairs: PlannedRegionPair[]
```

```text
PlannedRegionPair
  prev_region_ref: PlannedRegionRef
  curr_region_ref: PlannedRegionRef
  loop_pairs: PlannedLoopPair[]
  closures: PlannedClosure[]
```

```text
PlannedLoopPair
  prev_loop_ref: PlannedLoopRef
  curr_loop_ref: PlannedLoopRef
  prev_loop: Nx2
  curr_loop: Nx2
  role: stable | synthetic_birth | synthetic_death
```

```text
PlannedClosure
  side: prev | curr
  scope: loop | region
  loop_index: int | None
```

```text
PlannedRegionRef / PlannedLoopRef
  kind: actual | synthetic
  index: int
```

Planner-owned metadata contract:

- `plan_schema_version` is required
- `planner` is required
- split/merge control values are recorded

---

## 4. Decomposition Rules

### 4.1 Region Split (`1->N`)

Given one source region and `N` target regions:

1. choose deterministic stable correspondence for one branch
2. create `N-1` synthetic birth branches
3. assign `prev` closure ownership for synthetic birth branches
4. preserve deterministic branch ordering

### 4.2 Region Merge (`N->1`)

Given `N` source regions and one target region:

1. choose deterministic stable correspondence for one branch
2. create `N-1` synthetic death branches
3. assign `curr` closure ownership for synthetic death branches
4. preserve deterministic branch ordering

### 4.3 Deterministic Assignment

Correspondence and ordering must remain deterministic under repeated execution
for identical inputs and controls.

---

## 5. Plan Invariants

For every transition interval:

1. interval equals `(i, i+1)` for transition index `i`
2. each region pair contains at least one loop pair
3. each loop pair role matches loop-ref kinds:
`actual/actual -> stable`, `synthetic/actual -> synthetic_birth`,
`actual/synthetic -> synthetic_death`
4. loop closure records include a valid `loop_index`
5. region closure records set `loop_index = None`
6. loop vertex counts equal `plan.samples`
7. all refs use planner-native ref records

Invalid plans must fail before geometry emission.

---

## 6. Executor Contract

Executor requirements:

- consume only planner records (`Planned*` types)
- never infer split/merge policy at runtime
- execute bridge and closure actions exactly once per plan record
- raise interval-context errors on invalid closure/ref mappings

The executor may triangulate and allocate synthetic geometry, but only as
directly instructed by planner records.

---

## 7. Error Contract

Planner must raise:

- invalid split/merge control values
- ambiguous correspondence in this scope
- invalid plan structure or metadata contract failures

Executor must raise:

- unresolved planner refs
- invalid closure records
- out-of-range loop closure targets

Executor errors for plan-record violations must include interval context.

---

## 8. Acceptance Tests

Required:

1. `1->N` split in `resolve` mode passes mesh-quality gates
2. `N->1` merge in `resolve` mode passes mesh-quality gates
3. `split_merge_steps` increases mesh density deterministically
4. deterministic replay equality for split and merge fixtures
5. `fail` mode supports unambiguous split/merge and rejects ambiguous `N->M`
6. planner emits synthetic role markers for birth/death
7. planner emits required metadata contract keys
8. executor rejects malformed closure indices with interval-context errors

Mesh-quality gates:

- boundary edges = 0
- nonmanifold edges = 0
- degenerate faces = 0

---

## 9. Definition of Done

Spec 15 is complete when:

1. `1->N` and `N->1` are planner-produced branch events
2. planner emits only planner-native refs and closure records
3. executor consumes plan records without topology-policy inference
4. planner/executor validation contracts are enforced
5. acceptance tests pass in CI

---

## 10. Change Log From v1.3

This revision updates v1.3 to align with planner implementation details:

- replaces abstract refs with planner-native ref types
- formalizes closure record shape (`scope`, `side`, `loop_index`)
- adds metadata schema contract requirements
- adds explicit plan validation invariants
- adds interval-context executor error requirement
- clarifies that `fail` mode still supports unambiguous `1->N` / `N->1`
