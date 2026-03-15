# Loft Spec 11: Resolve-Mode Watertightness (v1.1)

This specification defines the required implementation changes to make
`loft_sections(..., split_merge_mode="resolve")` produce watertight output for
supported split/merge transitions (`1->N` and `N->1`), while preserving
determinism.

All statements are normative unless explicitly marked as future work.

---

## 1. Current Failure Modes

Resolve mode currently creates boundary edges in split/merge cases.

Root causes:

1. Synthetic loop vertices are emitted multiple times across sidewall and
   closure code paths instead of reusing a single canonical vertex block.
2. Region closure code assumes contiguous loop blocks (`closure_faces + start`)
   that do not hold once synthetic loops are involved.
3. Closure ownership for births/deaths is not strict enough to guarantee every
   boundary edge is sealed exactly once.

---

## 2. Scope

In scope:

- region split/merge decomposition (`1->N`, `N->1`)
- hole split/merge decomposition (`1->N`, `N->1`)
- micro-station resolve path
- watertightness and determinism guarantees

Out of scope:

- many-to-many (`N->M`, `N>1`, `M>1`) branch planning
- fairness optimization / skeleton-based blending

---

## 3. Required Data Model Changes

Introduce transition-local canonical loop identity.

### 3.1 Loop key

Use a deterministic key for every loop used in bridging/closure:

```text
LoopKey = (
  station_index,
  region_ref_kind, region_ref_index,
  loop_ref_kind, loop_ref_index
)
```

### 3.2 Vertex block registry

Maintain:

```text
loop_vertex_start: dict[LoopKey, int]
```

Behavior:

- if key exists: reuse stored start index
- if key missing: emit loop vertices once, store start index, return it

No synthetic loop may be emitted more than once per station/key.

---

## 4. Bridging Requirements

For each paired loop bridge:

1. resolve canonical `start_a` and `start_b` from registry
2. emit strip triangles as currently done

Bridging must never create ad hoc temporary vertex blocks outside the registry.

---

## 5. Closure Requirements

### 5.1 Indexed closure triangulation

Replace "base + offset" assumptions with explicit indexed loop closure.

Required helper:

```text
triangulate_indexed_loops(loop_points: list[np.ndarray], loop_starts: list[int]) -> faces
```

Contract:

- triangulation runs in local loop index space
- local indices are mapped back to canonical global indices using `loop_starts`
- works for outer + holes

### 5.2 Closure ownership

Closure ownership must be unique per event:

- synthetic birth loop: close on `prev` side only once
- synthetic death loop: close on `curr` side only once
- region-level closure (`prev`/`curr`) must not duplicate per-loop closure faces

Any duplicated closure for the same loop key is a correctness bug.

---

## 6. Micro-Station Resolve Semantics

For resolve mode with staged micro-stations:

- canonical loop keys are per effective station index
- closures are allowed only on interval boundaries where a birth/death event
  starts or terminates
- interior micro-station intervals should bridge only

This prevents accidental interior caps that create non-manifold seams.

---

## 7. Determinism Requirements

Watertightness fixes must preserve deterministic output:

- same input => same vertex order and face order
- registry insertion order must be deterministic
- closure emission order must be deterministic

---

## 8. Error Contract

Required explicit errors:

- `split_resolution_failed`
- `merge_resolution_failed`
- `closure_index_resolution_failed`
- `duplicate_closure_ownership`

Error messages must include station interval index and event side (`prev`/`curr`)
when applicable.

---

## 9. Acceptance Tests (Required)

Add/upgrade tests so resolve mode asserts watertightness, not only mesh existence.

Minimum set:

1. region split (`1->2`) resolve mode: `boundary_edges == 0`
2. region merge (`2->1`) resolve mode: `boundary_edges == 0`
3. hole split (`1->2`) resolve mode: `boundary_edges == 0`
4. hole merge (`2->1`) resolve mode: `boundary_edges == 0`
5. deterministic snapshot stability for at least one split and one merge case
6. many-to-many ambiguity still fails explicitly

Use `analyze_mesh(mesh)` as the source of truth for watertightness assertions.

---

## 10. Completion Criteria

Spec 11 is complete when:

1. all required resolve-mode watertight tests pass
2. no regressions in existing loft/topology tests
3. resolve demo model exports with zero boundary edges per mesh

This spec is the hard blocker before topology completion can be declared.

