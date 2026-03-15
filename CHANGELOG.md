# Changelog

## Unreleased

### Loft Topology Transition v1

- Added deterministic station-based topology lofting via `loft_sections(...)` with explicit station frames (`origin`, `u`, `v`, `n`).
- Added deterministic split/merge controls:
  - `split_merge_mode` (`"fail"` or `"resolve"`)
  - `split_merge_steps`
  - `split_merge_bias`
- Supported transition classes in resolve mode:
  - region split (`1->N`)
  - region merge (`N->1`)
  - hole split (`1->N`)
  - hole merge (`N->1`)
- Supported stable/event transitions:
  - region/hole stable
  - region birth/death
  - hole birth/death
- Explicitly unsupported:
  - many-to-many ambiguity (`N->M`, where `N>1` and `M>1`) for regions or holes.
