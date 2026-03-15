# Loft Spec 12: Topology Completion Gate

This specification defines the final gate for declaring loft/topology transition
work "complete" for v1.

Completion means "production-ready for supported topology transitions", not
"all theoretical transitions implemented".

---

## 1. Completion Scope

Must be complete:

- deterministic section-based loft core
- deterministic station frames (`origin`, `u`, `v`, `n`)
- deterministic one-to-one matching
- region/hole birth/death
- resolve-mode split/merge for `1->N` and `N->1`
- watertight and manifold output for all supported events
- explicit failure for unsupported many-to-many ambiguity (`N->M`)

May remain out of scope:

- full many-to-many branch planner
- skeleton/fairness optimization

---

## 2. Mandatory Test Matrix

The following must exist and pass in CI:

### 2.1 Stable transitions

1. single-region no-hole loft
2. multi-region deterministic order invariance
3. multi-hole deterministic order invariance

### 2.2 Birth/death transitions

1. region birth
2. region death
3. hole birth
4. hole death

### 2.3 Resolve split/merge transitions

1. region split (`1->2`) resolve mode, watertight
2. region merge (`2->1`) resolve mode, watertight
3. hole split (`1->2`) resolve mode, watertight
4. hole merge (`2->1`) resolve mode, watertight
5. resolve staging density check (`split_merge_steps` affects output density)

### 2.4 Unsupported transitions

1. many-to-many region ambiguity fails with explicit message
2. many-to-many hole ambiguity fails with explicit message

### 2.5 Determinism

1. same input repeated => byte-identical vertices/faces
2. reordered equivalent input => identical output where order should be invariant

---

## 3. Quality Gates

For all supported transition tests:

- `analyze_mesh(mesh).boundary_edges == 0`
- `analyze_mesh(mesh).nonmanifold_edges == 0`
- `analyze_mesh(mesh).degenerate_faces == 0` (or explicitly justified tolerance)

If any gate fails, completion is blocked.

---

## 4. API/Docs Gate

Before completion is declared:

1. `docs/modeling/loft.md` documents `split_merge_mode`, `split_merge_steps`,
   `split_merge_bias`, and supported/unsupported classes.
2. At least one dedicated resolve demo exists in `docs/examples/loft/`.
3. Error messages in docs match runtime behavior.

---

## 5. Release Gate

Before release/tag:

1. full test suite passes in venv
2. resolve demo exports successfully
3. mesh analysis report for resolve demo confirms watertightness
4. changelog/release notes call out supported split/merge classes and explicit
   unsupported classes

---

## 6. Definition of Done

Topology transition work is "complete (v1)" only when all of the following are true:

1. Spec 11 acceptance criteria are satisfied
2. Spec 12 test/quality/doc/release gates are satisfied
3. no known watertightness defects remain for supported transition classes

If any of these are false, topology transition work is not complete.

