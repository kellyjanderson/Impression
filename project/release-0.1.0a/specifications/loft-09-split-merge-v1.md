> Status: Deprecated historical spec.
> Superseded by the next-generation loft specification tree beginning at
> [Loft Spec 21](loft-21-nextgen-loft-evolution-program-v1_0.md). Retained for
> project history only.

# Loft Spec 09: Split/Merge Transitions (v1)

This specification defines how `loft_sections(...)` should handle topology transitions where region or hole correspondence is not one-to-one.

All statements are normative unless explicitly marked as future work.

---

## 1. Decision: Region Birth/Death

Yes, region birth/death is required and remains part of the supported contract.

Rationale:

- It is already useful and implemented for practical workflows (multi-island transitions).
- It is required as a building block for deterministic split/merge resolution in staged transitions.
- Removing it would reduce loft utility in exactly the real-world cases this kernel is meant to solve.

---

## 2. Supported Events in v1

Stable events:

- `region_stable` (1 -> 1)
- `hole_stable` (1 -> 1)

Supported topology-change events:

- `region_birth` (0 -> 1)
- `region_death` (1 -> 0)
- `hole_birth` (0 -> 1)
- `hole_death` (1 -> 0)
- `region_split` (1 -> N) via staged resolution
- `region_merge` (N -> 1) via staged resolution
- `hole_split` (1 -> N) via staged resolution
- `hole_merge` (N -> 1) via staged resolution

Not supported in v1:

- general `N -> M` split/merge ambiguity where `N > 1` and `M > 1` in one interval

---

## 3. API Contract

Add to `loft_sections(...)`:

- `split_merge_mode: "fail" | "resolve" = "fail"`
- `split_merge_steps: int = 8`
- `split_merge_bias: float = 0.5`

Behavior:

- `"fail"` preserves current strict behavior (explicit error on split/merge detection).
- `"resolve"` enables staged transition decomposition described below.

---

## 4. Detection

Detection is deterministic and performed before bridging.

Per station pair:

1. compute deterministic minimum-cost assignment among likely correspondences
2. build correspondence graph with cardinality and overlap checks
3. classify each source/target entity by degree:
   - degree 0: birth/death
   - degree 1<->1: stable
   - degree 1->N: split
   - degree N->1: merge
   - N->M: ambiguous

If `split_merge_mode="fail"`, any split/merge classification aborts immediately.
If `split_merge_mode="resolve"`, only `N->M` ambiguous cases abort.

---

## 5. Resolution Strategy (v1)

v1 does not attempt full branch skeleton reconstruction.
Instead, it decomposes split/merge into deterministic staged events over synthetic micro-stations.

### 5.1 Split (1 -> N)

1. keep one stable correspondence path to the best-matched target loop
2. spawn remaining `N-1` targets as births over `split_merge_steps`
3. births use bounded target-derived seed loops (same point count/indexing as target)

### 5.2 Merge (N -> 1)

1. keep one stable correspondence path from the best-matched source loop
2. collapse remaining `N-1` sources as deaths over `split_merge_steps`
3. deaths use constrained triangulated closure at last non-collapsed state

This approach preserves watertightness and determinism while making complex transitions usable now.

---

## 6. Placement of Synthetic Micro-Stations

For source station `A` and target station `B`:

- Let interval parameter be `u in [0, 1]`.
- Place synthetic stations in a bounded sub-interval around `split_merge_bias`.

Example default window:

- start: `u0 = clamp(split_merge_bias - 0.2, 0, 1)`
- end: `u1 = clamp(split_merge_bias + 0.2, 0, 1)`

Interpolate station frame/origin using existing deterministic station-frame interpolation rules.

---

## 7. Geometric Guarantees

Required in `"resolve"` mode:

- manifold mesh output
- watertight output
- deterministic vertex/face ordering for identical input
- no random choices in correspondence, anchoring, or event order

If guarantees cannot be met for a case, fail explicitly with a clear error.

---

## 8. Error Contract

Error messages must identify event type and interval index.

Required categories:

- `unsupported_topology_ambiguity` (N->M)
- `split_resolution_failed`
- `merge_resolution_failed`
- `constrained_triangulation_failed`

---

## 9. Test Plan

Add tests for:

1. region split (1 -> 2) in resolve mode
2. region merge (2 -> 1) in resolve mode
3. hole split (1 -> 2) in resolve mode
4. hole merge (2 -> 1) in resolve mode
5. ambiguous 2 -> 2 case fails in resolve mode
6. split/merge fail mode preserves current rejection behavior
7. deterministic output snapshot for split and merge cases

---

## 10. Out of Scope (v1)

- exact medial-axis/skeleton branch stitching
- minimizing surface fairness across branch events
- full many-to-many branch topology optimization

Those belong to a later split/merge v2 spec once v1 is stable.
