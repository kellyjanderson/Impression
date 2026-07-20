# Priority 06 — Reconstruction And Repair Intermediates Architecture

## Status

`0.1.0.a` feature-path architecture branch.

This document covers Priority Read item `6` from
[Low-Level Construct Gap Report](../../research/2026-04-25-low-level-construct-gap-report.md).

Related architecture:

- [B-Spline Surface And Reconstruction Architecture](b-spline-surface-and-reconstruction-architecture.md)
- [Priority 05 — Spanwise Grouping And Compatibility Architecture](priority-05-spanwise-grouping-and-compatibility-architecture.md)
- [Priority 07 — Surfaced B-Spline Patch Family Architecture](priority-07-surfaced-b-spline-patch-family-architecture.md)

## Purpose

This branch defines the intermediate records needed between raw repair evidence
and final surfaced repair or reconstruction results.

## Core Need

The research strongly suggests that reconstruction and repair cannot jump
directly from:

- raw section loops
- local mesh neighborhoods
- damaged surfaced spans

to:

- final repaired `SurfaceBody`

The missing piece is a set of canonical intermediate records that stabilize the
input, make local truth inspectable, and let later repair strategies share the
same evidence.

## Required Intermediate Families

This branch should cover at least two intermediate layers.

### Section Reconstruction Intermediates

- `CanonicalSectionContour`
- `CleanedSectionProfile`
- `SectionToStationConversion`
- `SectionConfidenceReport`
- `SparseCrossSectionReconstructionInput`

### Local Repair Intermediates

- `LocalBoundaryRing`
- `PatchNeighborhoodDescriptor`
- `LocalSeamIntent`
- `RepairNeighborhoodInput`
- `RepairedPatchIntegrationReport`

## Behavioral Rules

The branch should enforce:

1. raw extracted loops are not final repair truth
2. canonicalized section records are stable and reusable
3. local neighborhood repair must preserve explicit seam and boundary ownership
4. repaired integration back into a `SurfaceBody` is reported, not guessed
5. quality or confidence of extracted evidence is explicit

## System Placement

```text
raw mesh or surfaced evidence
-> canonical section or neighborhood intermediates
-> repair or reconstruction strategy
-> repaired patch or reconstructed span
-> integration report
```

This keeps the architecture honest about where interpretation happens and where
confidence can drop.

## Relationship To Exact And Approximate Truth

This branch should be built with exact-vs-approximate reporting in mind.

Some repair paths may reconstruct exact local structure.
Others may produce interpreted or fitted repair results.

The intermediates should therefore preserve enough context for later result
taxonomy without forcing that taxonomy into every low-level record directly.

## Scope Boundary

This branch does not define the eventual surfaced patch family used by repair.

It defines the shared evidence layer that makes later repair strategies
comparable and inspectable.

## Delivery Guidance

Recommended implementation order:

1. canonical section contour objects
2. section confidence and station-conversion records
3. local boundary-ring and neighborhood descriptors
4. seam-intent and integration reports
5. shared repair input bundle

## Architectural Conclusion

Priority `6` is the missing bridge between retained mesh evidence and future
surface repair or reconstruction. Without these intermediates, later repair
branches would stay ad hoc and difficult to trust.
