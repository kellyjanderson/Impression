# Spanwise Loft Postprocessing Optimization Architecture

## Status

Future feature branch.

## Idea

Loft first produces its usual local interval-by-interval surface result.

After that, a postprocessing tool analyzes the resulting `SurfaceBody` and
detects when multiple adjacent loft-generated spans can be consolidated into a
simpler, larger-span representation.

## Architectural Placement

This branch lives after loft execution:

```text
placed topology sequence
-> loft planner
-> loft executor
-> initial surface body
-> spanwise postprocessing optimizer
-> simplified / consolidated surface body
```

## Why It Matters

- can be added incrementally without first changing core loft planning
- gives the system a simplification/compression tool for over-segmented loft
  output
- provides a practical experimental path before committing to deeper planner
  changes

## Main Challenge

This is easier to introduce, but less canonical than inline planning.

It risks becoming:

- cleanup after a too-local loft

rather than:

- a native larger-span loft representation

So the branch needs clear honesty about:

- when the optimization is exact
- when it is approximate
- how much semantic loft intent is preserved after consolidation

## Open Questions

- What adjacency and compatibility evidence should permit consolidation?
- Should this tool only merge equivalent local spans, or also refit them?
- How are simplification error and seam relocation reported?

