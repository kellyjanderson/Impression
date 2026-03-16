# Loft Spec 19: Global Fairness and Skeleton Optimization (v1.0)

This specification defines fairness-optimized and skeleton-informed planning for
loft branch decomposition, making smoothness and centerline quality a hard
requirement for supported split/merge transitions.

It builds on deterministic planner/executor contracts from Specs 14-17.

All statements are normative unless marked as future work.

---

## 1. Scope

In scope:

- global fairness optimization across multi-interval branch plans
- optional skeleton-informed branch centerline guidance
- objective terms that reduce kinks/twists/branch crossings
- deterministic optimization with quality constraints

Out of scope:

- unconstrained free-form shape optimization
- non-deterministic continuous optimization
- replacing hard topology validity checks

---

## 2. Fairness Objective

Planner must optimize branch plans with fairness terms:

1. curvature continuity penalty across interval sequence
2. branch crossing penalty
3. branch acceleration/sudden re-route penalty
4. synthetic birth/death harshness penalty
5. closure stress penalty (local area distortion near closure)

Objective is lexicographic with topology validity first, fairness second.

---

## 3. Skeleton Guidance

For eligible regions/holes, planner may compute local medial/skeleton hints:

- centerline candidates for branch routing
- correspondence hints for ambiguous assignments
- closure placement hints

Skeleton use must be deterministic and optional per control settings.

If skeleton extraction fails, planner falls back to non-skeleton deterministic
fairness mode.

---

## 4. New Control Surface

Add controls:

- `fairness_mode: "off" | "local" | "global"` (default `"local"`)
- `fairness_weight: float` (default `0.2`)
- `skeleton_mode: "off" | "auto" | "required"` (default `"auto"`)
- `fairness_iterations: int` (default `12`)

Behavior:

- `off`: current deterministic topology-first planner behavior.
- `local`: per-interval fairness only.
- `global`: optimize branch continuity across all intervals in station run.

---

## 5. Determinism Contract

Fairness and skeleton optimization must remain deterministic:

- fixed initialization
- deterministic iteration order
- deterministic convergence criteria
- no random restarts in default path

Same input + same controls => identical plan and mesh.

---

## 6. Optimization Guardrails

Optimizer must respect hard constraints:

- topology validity
- closure uniqueness
- containment consistency
- no self-intersection in generated branch staging loops

If fairness objective conflicts with hard constraints:

- hard constraints win
- fairness gracefully degrades

---

## 7. Failure Policy

Fail explicitly when:

- `skeleton_mode="required"` and skeleton extraction fails
- optimizer cannot find valid plan under hard constraints
- optimization exceeds bounded iteration/resource limits

Failure codes:

- `fairness_optimization_failed`
- `skeleton_required_unavailable`

---

## 8. Metadata Requirements

Plan metadata must include:

- `fairness_mode`
- `fairness_weight`
- `fairness_iterations`
- `skeleton_mode`
- objective term breakdown (pre/post optimization)
- optimization convergence status

---

## 9. Quality Contract

Outputs must satisfy existing watertightness requirements:

- boundary edges = 0
- nonmanifold edges = 0
- degenerate faces = 0

Additionally, fairness diagnostics must be reported:

- branch crossing count
- continuity score
- closure distortion score

---

## 10. Acceptance Tests

Required:

1. Global fairness reduces crossing/continuity penalties vs baseline on
   designated fixtures.
2. Deterministic replay equality under same controls.
3. `skeleton_mode="required"` fails correctly when unavailable.
4. `skeleton_mode="auto"` falls back and remains valid.
5. Quality constraints remain satisfied in fairness-optimized outputs.
6. At least one real-world branching demo shows visibly smoother transition
   pathing with fairness enabled.

---

## 11. Definition of Done

Spec 19 is complete when:

1. Global fairness optimization is integrated and deterministic.
2. Skeleton guidance is available with robust fallback policy.
3. Optimized plans preserve topology correctness and watertightness.
4. Fairness improvements are measurable in acceptance fixtures.
