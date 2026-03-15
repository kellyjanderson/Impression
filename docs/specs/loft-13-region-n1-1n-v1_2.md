# Loft Spec 13: Region `N->1` and `1->N` Transitions (v1.2)

> Superseded for implementation by
> `docs/specs/loft-15-region-1n-n1-planning-v1_4.md`.
> Keep this document for historical context only.

This specification defines the deterministic contract for **region-only**
split/merge transitions in `loft_sections(...)`:

- region merge: `N->1`
- region split: `1->N`

This spec narrows scope intentionally to region transitions so behavior is
predictable, testable, and watertight for production use.

All statements are normative unless explicitly marked as future work.

---

## 1. Scope

In scope:

- deterministic detection and resolution of region `N->1` and `1->N`
- support for disconnected islands as region entities
- resolve-mode staged decomposition and closure ownership
- watertight/manifold guarantees for supported cases

Out of scope:

- many-to-many region ambiguity (`N->M`, `N>1`, `M>1`)
- hole-level split/merge policy (covered by separate specs/contracts)
- skeleton/fairness optimization beyond deterministic staged resolution

---

## 2. API Contract

This spec uses existing controls on `loft_sections(...)`:

- `split_merge_mode: "fail" | "resolve" = "fail"`
- `split_merge_steps: int = 8`
- `split_merge_bias: float = 0.5`

Behavior:

- `fail`: permit deterministic, unambiguous `N->1` and `1->N`; reject
  ambiguous `N->M`
- `resolve`: permit `N->1` and `1->N` with staged micro-station
  decomposition controls; reject ambiguous `N->M`

---

## 3. Event Classification

For each adjacent station pair `(S_i, S_{i+1})`:

1. Build deterministic region correspondence candidates using minimum-cost
   assignment over outer loops.
2. Classify by cardinality:
   - `1<->1`: stable
   - `1->N`: split
   - `N->1`: merge
   - `N->M`: ambiguous (unsupported)

Required failure:

- `Unsupported topology transition: region split/merge ambiguity detected.`

for unsupported many-to-many ambiguity.

---

## 4. Deterministic Planner

### 4.1 Split (`1->N`)

Given one source region and `N` target regions:

1. Select one primary stable branch by minimum deterministic cost.
2. Materialize remaining `N-1` branches as deterministic births using
   synthetic seed regions derived from each target region.
3. Decompose over staged micro-stations (`split_merge_steps`) within a bounded
   window around `split_merge_bias`.

### 4.2 Merge (`N->1`)

Given `N` source regions and one target region:

1. Select one primary stable branch by minimum deterministic cost.
2. Collapse remaining `N-1` branches as deterministic deaths using synthetic
   collapsed target-side regions.
3. Apply deterministic closure on the death side only once per branch.

### 4.3 Deterministic Tie-Breaks

When costs tie, resolve in order:

1. lower centroid distance
2. lower area delta
3. lower source index
4. lower target index

No random or iteration-order-dependent choice is allowed.

---

## 5. Staging and Placement

Synthetic micro-stations must be injected in a bounded interval:

- `u0 = clamp(split_merge_bias - 0.2, 0, 1)`
- `u1 = clamp(split_merge_bias + 0.2, 0, 1)`

with deterministic interpolation of:

- station `t`
- `origin`
- frame vectors (`u`, `v`, `n`) with right-handed orthonormalization

If `split_merge_steps <= 1`, behavior degenerates to minimal staging but must
remain deterministic.

---

## 6. Closure Ownership Rules

For each synthetic branch introduced by the planner:

- birth branch closure occurs on `prev` side only
- death branch closure occurs on `curr` side only
- region-level closure for the same branch must not be duplicated

Duplicate closure ownership for the same branch is a correctness bug.

---

## 7. Mesh Quality Contract

For supported `N->1` and `1->N` region transitions in resolve mode:

- `boundary_edges == 0`
- `nonmanifold_edges == 0`
- `degenerate_faces == 0`

If quality cannot be achieved, fail explicitly instead of returning partial
geometry.

---

## 8. Error Contract

Required explicit failures:

- invalid split/merge mode or controls
- unsupported many-to-many region ambiguity
- station frame/ordering violations
- closure index/loop mapping failures

Messages must remain deterministic and suitable for tests/docs matching.

---

## 9. Acceptance Test Matrix

Minimum required tests:

1. `1->N` region split resolves and passes mesh-quality gates
2. `N->1` region merge resolves and passes mesh-quality gates
3. deterministic replay: repeated identical input => identical vertices/faces
4. staged density effect: higher `split_merge_steps` increases mesh density
5. many-to-many `N->M` region case fails explicitly
6. fail mode permits unambiguous region split/merge and rejects ambiguous `N->M`

At least one documentation example for resolve mode must export with zero
boundary/nonmanifold edges.

---

## 10. Definition of Done

Region transition support is considered complete for v1.2 when:

1. all acceptance tests above are present and passing
2. resolve demos validate mesh quality (`0` boundary/nonmanifold/degenerate)
3. docs reflect supported classes (`N->1`, `1->N`) and unsupported class (`N->M`)
4. behavior remains deterministic across reruns
