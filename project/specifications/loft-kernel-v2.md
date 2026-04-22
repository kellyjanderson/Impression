> Status: Deprecated historical spec.
> Superseded by the next-generation loft architecture and specification tree.
> Retained for project history only.

# Impression Kernel Specification: Section-Based Lofting (v2)

This specification defines deterministic section-based lofting behavior for Impression.
All requirements in this document are normative.

---

## 1. Station Definition (3D-Aware)

A station is a section with explicit 3D placement and orientation.

```text
Station
  t: float
  section: Section
  origin: Vec3
  u: Vec3
  v: Vec3
  n: Vec3
```

Constraints:

- stations are strictly ordered by `t`
- `u`, `v`, `n` form a right-handed orthonormal basis
- equivalent transform-matrix storage is permitted if semantically identical

---

## 2. Deterministic Region/Hole Matching

Greedy nearest-centroid matching is not allowed.
Region and hole correspondence must use deterministic minimum-cost bipartite assignment.

Cost function baseline:

```text
cost(A, B) = ||centroid(A) - centroid(B)|| + w_area * abs(area(A) - area(B))
```

Tie-breakers (required, in order):

1. lower centroid distance
2. lower area difference
3. lower source index
4. lower target index

If correspondence cardinality implies split/merge (see Section 6), loft must abort.

---

## 3. Loop Anchoring

Loop anchoring must be deterministic.

Primary anchor rule:

```text
anchor = argmin(angle(vertex - centroid))
```

Tie-breakers (required):

1. larger radius from centroid
2. lower original vertex index

---

## 4. Closure Rule for Death Events

Triangle-fan closure is not allowed for general concave loops.

For hole/region death:

- close the last non-collapsed surviving loop by constrained triangulation
- triangulation must preserve hole semantics if holes still exist in the surviving region

If constrained triangulation fails, abort the affected operation with explicit error.

---

## 5. Birth/Death Geometry Heuristics

Unbounded seed heuristics are not allowed.

The following heuristic is prohibited:

```text
seed_radius = sqrt(area) * 0.05
```

Required approach:

- use target-loop-derived geometry (shrunken/expanded target loop)
- preserve vertex count and indexing compatibility with target loop
- enforce bounded, monotone progression toward target topology

---

## 6. Split/Merge Detection Criteria

Split/merge must be explicitly detected before bridging.

Unsupported transitions:

- region split
- region merge
- hole split
- hole merge

Detection requirements:

- correspondence cardinality must be one-to-one for stable transitions
- one-to-many or many-to-one implies split/merge
- containment and overlap checks must confirm ambiguous cases

On detection of unsupported transition, loft must terminate with explicit error prior to mesh output.

---

## 7. Cap Parameter Contract

`cap_depth` and `cap_radius` must have explicit relationship semantics.

Implementation supports both modes and sets the default explicitly:

1. Linked mode: `cap_depth == cap_radius` (true quarter-circle interpretation)
2. **Independent mode (default)**: `cap_depth` and `cap_radius` are independent, producing elliptical variants when unequal

Whichever mode is selected:

- behavior must be deterministic
- schedule equations must remain monotone
- documentation and CLI/API names must reflect the chosen contract

Default contract:

```text
cap_mode_default = independent
cap_depth and cap_radius are orthogonal parameters
```

---

## 8. Determinism Requirements

Given identical stations and parameters, output mesh must be identical:

- no random sampling
- deterministic assignment and anchoring
- deterministic tie-break ordering
- deterministic failure behavior

---

## 9. Acceptance Criteria

Implementation is accepted when all are true:

1. path lofting honors station frames (`origin`, `u`, `v`, `n`)
2. correspondence uses deterministic assignment (no greedy order dependence)
3. anchoring tie-breaks eliminate index ambiguity
4. concave closure uses constrained triangulation (no fan artifacts)
5. birth/death progression is bounded and correspondence-preserving
6. split/merge detection is explicit and fails early
7. cap parameter contract is explicit and tested
