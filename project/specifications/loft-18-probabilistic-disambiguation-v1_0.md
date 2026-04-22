# Loft Spec 18: Probabilistic Ambiguity Disambiguation (v1.0)

This specification defines an optional probabilistic disambiguation mode for
ambiguous `N->M / M->N` loft transitions.

It extends deterministic planner behavior from Specs 16-17 by adding controlled
stochastic search when deterministic tie-breaks cannot confidently select a
single decomposition.

All statements are normative unless marked as future work.

---

## 1. Scope

In scope:

- optional probabilistic branch-selection pipeline for residual ambiguity
- reproducible seeded execution
- confidence scoring and candidate ranking
- deterministic fallback/guardrails when confidence is low

Out of scope:

- replacing deterministic default behavior
- non-reproducible random behavior
- unconstrained Monte Carlo exploration without mesh-quality gates

---

## 2. New Control Surface

Add planner controls:

- `disambiguation_mode: "deterministic" | "probabilistic"`  
  (default `"deterministic"`)
- `disambiguation_seed: int | None`  
  (default `None`, auto-derived deterministic seed)
- `probabilistic_trials: int`  
  (default `64`, min `1`, max bounded)
- `probabilistic_temperature: float`  
  (default `0.25`, controls candidate spread)
- `probabilistic_min_confidence: float`  
  (default `0.65`, below this => fail or deterministic fallback)
- `probabilistic_fallback: "fail" | "deterministic"`  
  (default `"deterministic"`)

Behavior:

- Only active when `split_merge_mode="resolve"` and deterministic ambiguity
  remains after Spec 17 tie-break stack.
- For non-ambiguous intervals, deterministic planner path is unchanged.

---

## 3. Candidate Model

For an ambiguous interval, planner builds candidate decompositions:

- each candidate is a complete branch action graph for the interval
- each candidate has a score vector:
  - geometric cost
  - closure consistency penalty
  - projected crossing/self-intersection penalties
  - branch complexity penalty
  - optional fairness term (if enabled)

Candidates are sampled/ranked via seeded stochastic strategy with bounded
candidate count.

---

## 4. Reproducibility Contract

Probabilistic mode must still be reproducible.

Required:

- every run uses an explicit seed
- seed and trial count are stored in plan metadata
- same input + same controls + same seed => identical output plan and mesh

No true nondeterminism is allowed in CI or release builds.

---

## 5. Confidence and Selection

After trial evaluation:

- compute normalized candidate confidence from score separation
- select highest-confidence candidate if confidence >= threshold

If confidence is below threshold:

- apply `probabilistic_fallback`
  - `"deterministic"` => deterministic best candidate if valid
  - `"fail"` => explicit ambiguity failure

---

## 6. Failure Policy

Planner fails explicitly when:

- no valid candidate survives quality pre-check filters
- confidence remains below threshold and fallback is `fail`
- selected candidate violates closure ownership invariants

Failure code:

- `probabilistic_disambiguation_failed`

Diagnostics must include:

- seed
- trials
- top-k candidate scores
- selected/failed confidence

---

## 7. Executor Contract

Executor remains deterministic and plan-driven:

- no stochastic execution behavior
- consumes selected plan exactly
- enforces existing closure/quality invariants

---

## 8. Quality Contract

Selected candidate must satisfy:

- boundary edges = 0
- nonmanifold edges = 0
- degenerate faces = 0

If failed, interval/case fails explicitly (no silent degrade).

---

## 9. Metadata Requirements

Plan metadata must include:

- `disambiguation_mode`
- `disambiguation_seed`
- `probabilistic_trials`
- `probabilistic_temperature`
- `probabilistic_min_confidence`
- `probabilistic_selected_confidence`
- `probabilistic_candidate_count`

---

## 10. Acceptance Tests

Required:

1. Same seed reproduces identical plan and mesh.
2. Different seeds may choose different candidates, each valid/watertight.
3. Low-confidence case triggers configured fallback behavior.
4. Probabilistic path outperforms deterministic fail-only on at least one
   ambiguous fixture.
5. CI deterministic mode remains unaffected.

---

## 11. Definition of Done

Spec 18 is complete when:

1. Probabilistic mode is optional and reproducible.
2. Confidence-based selection and fallback behavior are implemented.
3. All selected outputs remain watertight and manifold.
4. Deterministic default path remains backward compatible.

---

## Refinement Status

Final.

This standalone feature spec remains active and does not currently require
another refinement round.
