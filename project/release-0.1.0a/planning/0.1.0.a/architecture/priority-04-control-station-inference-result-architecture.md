# Priority 04 — Control-Station Inference Result Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document covers Priority Read item `4` from
[Low-Level Construct Gap Report](../../../research/2026-04-25-low-level-construct-gap-report.md).

Related architecture:

- [B-Spline Control-Station Inference Architecture](b-spline-control-station-inference-architecture.md)
- [Priority 02 — Parameterization, Knot, And Fit Policy Architecture](priority-02-parameterization-knot-and-fit-policy-architecture.md)

## Purpose

This branch defines the result objects and durable diagnostics needed for
control-station inference to be more than “return fewer stations.”

## Core Need

The research has already converged on a strong distinction:

- topology stations preserve structural truth
- control stations preserve shape-driving flexibility

That means future inference needs a structured result that can explain:

- what was retained
- why it was retained
- what was reduced
- how much fit drift remains
- where user pins or hard constraints blocked reduction

## Required Result Families

The smallest honest result architecture should include:

- `RetainedStationClassification`
- `RetainedStationProvenance`
- `UserPinnedStationConstraint`
- `ReducedProgressionResult`
- `StructuralPreservationReport`
- `InferenceDiagnosticsBundle`

Optional later additions:

- retained correspondence evidence record
- user-review acceptance record
- replayable reduction decision log

## Result Ownership

The result branch should own:

- retained station ordering
- per-station classification
- retained-vs-dropped provenance
- pinned or forced retention causes
- residual summaries from supporting fit diagnostics
- structural preservation or refusal reasons

It should not own:

- raw spline fitting algorithms
- path trajectory semantics
- surfaced patch reconstruction

## Behavioral Rules

The architecture should enforce:

1. topology stations cannot disappear silently
2. every retained control station must have an explainable cause
3. dropped stations must still remain diagnosable
4. user pins must survive as durable result metadata
5. reduction acceptance must reference explicit diagnostics

## System Placement

```text
dense stations
-> descriptor and fit analysis
-> retained classification decision
-> reduced progression result
   -> topology stations
   -> control stations
   -> diagnostics
   -> preservation or refusal report
```

This keeps the result object as the durable artifact, while B-spline fits and
other diagnostics remain supporting evidence rather than hidden replacement
truth.

## Scope Boundary

This branch should not assume:

- automatic user acceptance
- a specific UI
- exact storage shape for later editing tools

It only defines the durable project-facing result semantics.

## Delivery Guidance

Recommended implementation order:

1. retained station classification
2. reduced progression result object
3. structural preservation report
4. user-pin and provenance metadata
5. richer inference diagnostics bundle

## Architectural Conclusion

Priority `4` turns control-station inference into an inspectable project feature
instead of a lossy decimation helper.
