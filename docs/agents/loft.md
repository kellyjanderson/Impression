# Loft For Agents

## Posture

Treat loft as a **surface-first** feature.

- Prefer `Loft(...)` when authoring or editing lofted geometry.
- Treat `loft(...)` as a convenience profile/path API over the same planner.
- Treat `loft_sections(...)` as the explicit-station convenience form of that
  same planner.
- Do not describe loft as a separate mesh-first modeling lane.

The canonical public reference is [`../modeling/loft.md`](../modeling/loft.md).

## Which Loft API To Use

- `Loft(...)`
  - canonical surface-first loft entrypoint
  - use when you want explicit progression, stations, and topology
  - keep the result as `SurfaceBody` until a consumer needs tessellation
- `loft(...)`
  - use when profiles plus optional `Path3D` are the simplest authored form
- `loft_sections(...)`
  - use when station frames are part of the authored intent
- `loft_plan_sections(...)`
  - use when you need to inspect planner output before execution
- `loft_plan_ambiguities(...)`
  - use when ambiguity needs to be surfaced before choosing a branch

## Ambiguity Rules

- Use `loft_plan_ambiguities(...)` before guessing through ambiguous
  split/merge-like transitions.
- When user intent matters, choose explicit `candidate_id` values and feed them
  back through:
  - `ambiguity_mode="interactive"`
  - `ambiguity_selection={(start_index, end_index): candidate_id}`
- Do not rely on accidental ordering in ambiguous intervals.
- When using resolution controls, state the chosen:
  - `split_merge_mode`
  - ambiguity controls
  - fairness controls

That keeps loft behavior reproducible.

## Consumer Boundary

- Keep loft results app-owned as long as possible.
- Tessellate only at preview, export, or another explicit consumer boundary.
- If a test or tool only accepts mesh, make the tessellation step explicit in
  the code instead of pretending loft itself is mesh-first.

## Reading Order

When a task centers on loft:

1. Read [`../modeling/loft.md`](../modeling/loft.md)
2. Open the closest runnable example under [`../examples/loft/`](../examples/loft/)
3. If transitions are ambiguous, inspect `loft_plan_ambiguities(...)` before coding through the case
