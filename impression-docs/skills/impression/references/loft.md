# Impression Loft Guide

Use this file whenever the task involves lofted geometry.

## Posture

Loft is surface-first.

- Prefer `Loft(...)` when authoring or editing lofted geometry.
- Treat `loft(...)` as a convenience profile/path API over the same planner.
- Treat `loft_sections(...)` as the explicit-station convenience API over that
  same planner.
- Do not describe loft as a separate mesh-first modeling lane.

## Which API To Use

### `Loft(...)`

Use when:

- you want explicit progression values
- you want explicit stations or frames
- you want topology-native ownership
- you want to keep the result as `SurfaceBody`

### `loft(...)`

Use when:

- profiles plus optional `Path3D` are the simplest authored form
- you want the shortest public convenience expression for the shape

### `loft_sections(...)`

Use when:

- station frames are part of the authored intent
- you already have `Station` objects

### `loft_plan_sections(...)`

Use when:

- you need to inspect the planner output before execution
- you need deterministic records for debugging or tooling

### `loft_plan_ambiguities(...)`

Use when:

- the transition may be ambiguous
- split/merge-like behavior is involved
- you need stable `candidate_id` values before choosing a branch

## Ambiguity Rules

- Do not guess through ambiguous intervals.
- Run `loft_plan_ambiguities(...)` first.
- If user intent matters, feed the chosen `candidate_id` back through:
  - `ambiguity_mode="interactive"`
  - `ambiguity_selection={(start_index, end_index): candidate_id}`
- When using split/merge resolution, make the chosen controls explicit in code:
  - `split_merge_mode`
  - ambiguity controls
  - fairness controls

## Consumer Boundary

- Keep loft results app-owned as long as possible.
- Tessellate only when preview, export, or another explicit consumer needs mesh.
- If an example or test needs mesh output, keep the tessellation step explicit.

## Read In The Repo

When the workspace contains these docs, read them:

- `docs/modeling/loft.md`
- `docs/examples/loft/*`
