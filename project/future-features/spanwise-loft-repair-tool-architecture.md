# Spanwise Loft Repair Tool Architecture

## Status

Future feature branch.

## Idea

The same wider-span reasoning used to consolidate dense loft output could also
be used as a repair-oriented tool.

In this mode, the goal is not only to simplify a valid loft, but to recover or
replace an overly segmented, noisy, or damaged span with a cleaner larger-span
surface interpretation.

## Architectural Placement

This branch lives as a repair/reconstruction tool:

```text
existing loft-like or damaged geometry
-> span analysis / compatible-run detection
-> larger-span repair or consolidation tool
-> repaired / simplified surface body
```

## Why It Matters

- connects loft span reasoning to future repair workflows
- allows the system to treat dense or messy span decomposition as something
  that can be improved, not just preserved
- creates a bridge between loft simplification and model-repair tooling

## Main Challenge

Repair is a broader and less constrained problem than simplification of clean
authored lofts.

So this branch needs to be honest about the difference between:

- exact consolidation of clean loft output
- repair or reinterpretation of imperfect geometry

The second case is likely less deterministic and more diagnostic-heavy.

## Open Questions

- Should repair target only loft-authored geometry, or also foreign mesh-derived
  surface reconstructions?
- How much deviation from the source span is acceptable in a repair result?
- Should this branch reuse the same consolidation logic as the other two
  branches, or only share part of it?

