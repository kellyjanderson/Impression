> Status: Deprecated historical spec.
> Active work now lives in the next-generation loft specification tree and the
> current implementation. Retained for project history only.

# Topology Spec 03: Loft Transition Surface

## Goal

Use topology-native sections to support deterministic loft transitions and robust endcap planning.

## Focus Areas

- Section correspondence and loop anchoring.
- Topology event classification between adjacent sections:
  - stable
  - hole collapse
  - hole birth/death
  - split/merge (planned hooks)
- Endcap strategy hooks on section-level data.
- Deterministic failure policy for invalid inset/collapse states.

## Deliverables

1. Loft transition helpers that operate on `Section`/`Region`/`Loop`.
2. Explicit strict-mode behavior for invalid transitions.
3. Test matrix covering:
   - convex
   - concave
   - holes
   - acute corners
   - tilted frames

## Completion Criteria

- Loft/endcaps consume topology-native transition helpers.
- Transition behavior is deterministic and test-covered.
