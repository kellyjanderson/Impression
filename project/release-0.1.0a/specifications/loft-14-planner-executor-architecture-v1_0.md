> Status: Deprecated historical spec.
> Superseded by the next-generation loft architecture and specification tree
> beginning at [Loft Spec 21](loft-21-nextgen-loft-evolution-program-v1_0.md).
> Retained for project history only.

# Loft Spec 14: Planner/Executor Architecture (v1.0)

This specification defines the architectural split of lofting into:

- a **Planner** (topology reasoning + transition planning)
- an **Executor** (geometry emission from a deterministic plan)

The goal is to establish the foundation required for robust many-branch
topology transitions (`N->M`) while keeping current `1->N` / `N->1` behavior
stable.

All statements are normative unless marked as future work.

---

## 1. Purpose

Current loft transition logic interleaves:

- topology classification
- branch/event planning
- vertex/face emission
- closure policy

This coupling increases risk and blocks clean `N->M` planning.

Required outcome:

1. Planner resolves topology differences into an explicit plan.
2. Executor consumes that plan and emits mesh deterministically.

---

## 2. Scope

In scope:

- planner/executor data contracts
- loft orchestration changes to call planner then executor
- parity behavior for currently supported transitions
- deterministic output preservation

Out of scope:

- introducing new transition classes (handled by subsequent specs)

---

## 3. Architecture

### 3.1 Planner Responsibilities

Planner owns:

- station/section normalization checks
- region/hole correspondence and event classification
- transition decomposition policy (`stable`, `birth`, `death`, `split`, `merge`)
- synthetic branch creation policy
- closure ownership assignment
- explicit plan validation before execution

Planner does not emit vertices/faces.

### 3.2 Executor Responsibilities

Executor owns:

- station frame placement in 3D
- loop-to-loop bridge strip emission
- cap/closure face emission
- canonical vertex indexing and reuse
- final mesh assembly and quality-ready output

Executor does not decide topology policy.

---

## 4. Plan Data Model

Introduce a deterministic, serializable plan contract:

```text
LoftPlan
  stations: PlannedStation[]
  transitions: PlannedTransition[]
  metadata: dict
```

```text
PlannedStation
  station_index: int
  t: float
  origin/u/v/n: Vec3
  regions: PlannedRegion[]
```

```text
PlannedTransition
  interval: (i, i+1)
  region_pairs: PlannedRegionPair[]
```

```text
PlannedRegionPair
  prev_ref
  curr_ref
  loop_pairs: PlannedLoopPair[]
  closures: PlannedClosure[]
```

```text
PlannedLoopPair
  prev_loop_ref
  curr_loop_ref
  role: stable | synthetic_birth | synthetic_death
```

```text
PlannedClosure
  side: prev | curr
  scope: loop | region
  target_ref
```

All references must be deterministic and index-stable.

---

## 5. Determinism Contract

Planner determinism:

- deterministic assignment and tie-breaks
- deterministic event ordering
- deterministic synthetic branch ordering

Executor determinism:

- deterministic vertex allocation order
- deterministic face emission order
- deterministic closure emission order

Same input + params => identical vertices/faces.

---

## 6. Orchestration Contract

`loft_sections(...)` must become:

1. `plan = loft_plan_sections(...)`
2. `mesh = loft_execute_plan(plan, ...)`

Behavioral parity requirement:

- for existing supported cases, output remains functionally identical except
  where bug fixes are intentional and documented.

---

## 7. Error Contract

Planner emits topology/planning errors:

- invalid station ordering/frame
- unsupported transition class
- ambiguity failures
- invalid closure ownership

Executor emits geometry/indexing errors:

- unresolved loop refs
- duplicate/invalid closure execution
- index mapping/triangulation failures

Errors must include interval context.

---

## 8. Migration Plan

Phase 1:

- Introduce planner/executor interfaces and plan dataclasses.
- Keep current logic, but route through interfaces.

Phase 2:

- Move topology classification + decomposition into planner.
- Move geometry emission into executor.

Phase 3:

- Remove direct topology mutation from executor path.
- Make planner output the sole source of execution truth.

---

## 9. Test Requirements

Required tests for this refactor:

1. planner output determinism snapshot for stable case
2. executor determinism given fixed plan
3. end-to-end parity for stable lofts
4. end-to-end parity for birth/death cases
5. end-to-end parity for current resolve-mode `1->N` / `N->1`
6. unchanged failure on unsupported `N->M` ambiguity

All supported outputs must satisfy mesh quality gates:

- boundary edges = 0
- nonmanifold edges = 0
- degenerate faces = 0

---

## 10. Definition of Done

Spec 14 is complete when:

1. planner/executor contracts exist and are wired through `loft_sections(...)`
2. current supported transition behavior is preserved via planner output
3. executor consumes only plan data for transition execution
4. tests above pass in CI
