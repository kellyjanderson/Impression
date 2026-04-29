> Status: Deprecated historical spec.
> Superseded by the next-generation loft specification tree. Retained for
> project history only.

# Loft Spec 17: Deterministic Ambiguity Auto-Resolver (v1.0)

This specification defines deterministic automatic ambiguity resolution for
`N->M` / `M->N` loft transitions on top of the planner/executor foundation
introduced in Specs 14-16.

Goal: resolve as many ambiguous transitions as reasonably safe, while
preserving determinism, mesh quality, and explicit failure for truly
indeterminate topology.

All statements are normative unless marked as future work.

---

## 1. Scope

In scope:

- deterministic auto-resolution for ambiguous region many-to-many transitions
- deterministic auto-resolution for ambiguous hole many-to-many transitions
- canonical tie-break hierarchy that eliminates order-dependence
- explicit residual-failure category for unrecoverable ambiguity

Out of scope:

- probabilistic or randomized disambiguation
- user-interactive branch picking in core loft API
- global fairness/skeleton optimization as a hard requirement

---

## 2. New Control Surface

Existing controls remain:

- `split_merge_mode: "fail" | "resolve"`
- `split_merge_steps: int`
- `split_merge_bias: float`

Add:

- `ambiguity_mode: "fail" | "auto"` (default `"auto"` when
  `split_merge_mode="resolve"`)
- `ambiguity_cost_profile: "balanced" | "distance_first" | "area_first"`
  (default `"balanced"`)
- `ambiguity_max_branches: int` (default `64`)

Behavior:

- `split_merge_mode="fail"` always fails on split/merge transitions.
- `split_merge_mode="resolve", ambiguity_mode="fail"` resolves only
  non-ambiguous decompositions; ambiguity fails explicitly.
- `split_merge_mode="resolve", ambiguity_mode="auto"` applies Spec 17
  deterministic auto-resolution pipeline.

---

## 3. Ambiguity Classes

Planner must classify ambiguity source before resolution attempt.

Classes:

1. **Permutation ambiguity**
   Multiple assignments share equal or near-equal primary cost.
2. **Containment ambiguity**
   Overlapping/containing candidates produce unstable ownership.
3. **Symmetry ambiguity**
   Geometrically symmetric source/target layouts produce equivalent mappings.
4. **Closure ambiguity**
   Multiple closures satisfy constraints with equivalent score.

Each interval must expose ambiguity class metadata in plan diagnostics.

---

## 4. Deterministic Cost Model

Planner scoring vector for candidate branch edge `(source_i, target_j)`:

1. centroid distance
2. normalized area delta
3. overlap penalty / containment inconsistency penalty
4. loop-shape signature distance (anchored perimeter profile)
5. deterministic index tie-break key

Scoring is lexicographic, not stochastic.

`ambiguity_cost_profile` adjusts only deterministic weights (never random):

- `balanced`: equal emphasis distance/area/shape
- `distance_first`: stronger distance term
- `area_first`: stronger area/shape term

---

## 5. Deterministic Solver Contract

For each ambiguous interval:

1. Build bipartite branch graph (region level, then hole level).
2. Solve deterministic min-cost flow / assignment with lexicographic objective.
3. If multiple equivalent optima remain, apply tie-break stack (Section 6).
4. Emit fully ordered planner actions with explicit branch ownership.

Solver must be deterministic across:

- repeated runs
- input region ordering permutations
- equivalent station object identity differences

---

## 6. Tie-Break Stack (Mandatory Order)

When candidate decompositions remain equivalent at current tie-break stage:

1. Lower total centroid distance.
2. Lower total area delta.
3. Lower containment-penalty sum.
4. Lower total branch crossing score
   (based on center-line crossings in local section frame).
5. Lower action complexity
   (prefer stable > split/merge > synthetic birth/death).
6. Canonical branch-key lexicographic order:
   - source topology signature
   - target topology signature
   - source canonical index
   - target canonical index

If still tied after stage 6:

- classify as residual indeterminate ambiguity and fail explicitly.

---

## 7. Canonical Topology Signatures

Each loop/region must have deterministic signature fields:

- anchored contour hash (arc-length sampled)
- signed area magnitude bucket
- centroid quantized to tolerance
- bbox aspect ratio bucket
- hole count (for region signatures)

Signature tolerance must be fixed and versioned in planner metadata.

---

## 8. Auto-Resolvable Case Set (Required)

`ambiguity_mode="auto"` must resolve at minimum:

1. Symmetric `2->2` reassignment with equal distances.
2. `2->3` and `3->2` with central branch birth/death where two matches are equivalent.
3. `3->3` permutations with mirrored layout.
4. Hole-level `2->3`, `3->2`, and `2->2` symmetric mapping cases within a stable region pair.
5. Mixed region+hole ambiguous intervals where region assignment and hole assignment
   are each independently resolvable under deterministic rules.

---

## 9. Explicit Non-Resolvable Cases (Fail)

Planner must fail with
`unsupported_topology_ambiguity` when:

- decomposition count exceeds `ambiguity_max_branches`
- residual tie remains after full tie-break stack
- closure ownership not uniquely derivable
- assignment implies self-intersection or invalid containment during staging

Failure must include:

- interval indices
- ambiguity class
- tie-break stage reached
- candidate count after pruning

---

## 10. Executor Contract Extensions

Executor must consume planner ambiguity outputs without reinterpretation.

Required:

- execute planner action order exactly
- enforce one closure owner per synthetic branch
- reject duplicate closure ownership or conflicting branch refs
- preserve deterministic vertex indexing and face order

No executor-side reassignment heuristics are allowed.

---

## 11. Quality Contract

All auto-resolved outputs must pass:

- boundary edges = 0
- nonmanifold edges = 0
- degenerate faces = 0

If any interval fails quality:

- interval resolution fails explicitly
- no silent fallback to degraded mesh

---

## 12. Plan Metadata Requirements

Planner metadata must include:

- `ambiguity_mode`
- `ambiguity_cost_profile`
- `ambiguity_resolved_intervals_count`
- `ambiguity_failed_intervals_count`
- per-class counts:
  - `permutation`
  - `containment`
  - `symmetry`
  - `closure`
- solver version / signature version

---

## 13. Acceptance Test Matrix

Required tests:

1. Region `2->2` symmetric ambiguous case resolves deterministically.
2. Region `2->3`, `3->2` ambiguous cases resolve and remain watertight.
3. Hole `2->2`, `2->3`, `3->2` ambiguous cases resolve deterministically.
4. Deterministic replay equality across repeated runs.
5. Deterministic equality under source/target region reorder permutations.
6. Residual indeterminate case fails with structured ambiguity diagnostics.
7. `ambiguity_mode="fail"` reproduces strict rejection behavior.
8. At least one real-world `N->M` example with ambiguous branch assignment
   succeeds and passes mesh analysis.

---

## 14. Implementation Sequence

Step 1:

- Add API controls and planner metadata fields.

Step 2:

- Implement deterministic ambiguity classifier and tie-break stack.

Step 3:

- Implement hole-level many-to-many planner decomposition.

Step 4:

- Integrate auto resolver with branch DAG emission and closure ownership.

Step 5:

- Add acceptance tests and real-world example coverage.

---

## 15. Definition of Done

Spec 17 is complete when:

1. Ambiguous `N->M/M->N` transitions are auto-resolved deterministically for the required case set.
2. Region and hole many-to-many paths both operate under planner/executor contracts.
3. Residual unresolved ambiguity fails with explicit structured diagnostics.
4. Acceptance matrix passes in CI with watertightness guarantees.
