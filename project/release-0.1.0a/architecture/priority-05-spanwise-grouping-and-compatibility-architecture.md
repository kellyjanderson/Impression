# Priority 05 — Spanwise Grouping And Compatibility Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document covers Priority Read item `5` from
[Low-Level Construct Gap Report](../../research/2026-04-25-low-level-construct-gap-report.md).

Related architecture:

- [Priority 06 — Reconstruction And Repair Intermediates Architecture](priority-06-reconstruction-and-repair-intermediates-architecture.md)
- [Priority 07 — Surfaced B-Spline Patch Family Architecture](priority-07-surfaced-b-spline-patch-family-architecture.md)

## Purpose

This branch defines the grouping and compatibility records needed for spanwise
consolidation to become an inspectable architectural feature rather than hidden
executor magic.

## Core Need

The research favors an early exact postprocess path for spanwise consolidation,
but that still requires the system to answer:

- which interval runs are eligible to be grouped
- why they are compatible
- where consolidation must stop
- whether the result stayed exact or became approximate

Those answers need first-class records.

## Required Record Families

The smallest honest architecture set is:

- `GroupedIntervalRun`
- `SpanCompatibilityReport`
- `ConsolidationResultClassification`
- `SeamRelocationReport`
- `PatchCountReductionReport`
- `ConsolidationRefusalDiagnostics`

Optional later additions:

- grouped-run provenance record
- exact-vs-approximate drift bundle
- planner-promotion eligibility record

## Compatibility Ownership

The compatibility layer should own evidence such as:

- patch family compatibility
- shared boundary consistency
- continuity or seam constraints
- topological safety for grouped spans
- refusal cause when a run cannot extend further

This should be durable enough that future planner-time promotion can reuse the
same contract rather than inventing a second hidden rule set.

## Behavioral Rules

This branch should enforce:

1. grouped runs are explicit ordered records
2. every extension of a run is justified by a compatibility report
3. refusal to extend is preserved as diagnostics, not discarded
4. exact vs approximate outcome is reported explicitly
5. seam relocation, if any, is inspectable after the fact

## System Placement

```text
adjacent loft intervals or surfaced patches
-> compatibility analysis
-> grouped interval run
-> consolidation attempt
-> exact or approximate result classification
-> seam and patch-count diagnostics
```

## Scope Boundary

This branch should not define the actual refit patch family.

It is the grouping and decision layer that later exact merges, approximate
refits, or repair branches can share.

## Delivery Guidance

Recommended implementation order:

1. grouped run record
2. compatibility report
3. refusal diagnostics
4. exact-vs-approximate result classification
5. seam and patch-count reporting

## Architectural Conclusion

Priority `5` gives spanwise consolidation a durable planning vocabulary. Without
it, later consolidation work would be difficult to audit, debug, or promote
from postprocess into planner-owned behavior.
