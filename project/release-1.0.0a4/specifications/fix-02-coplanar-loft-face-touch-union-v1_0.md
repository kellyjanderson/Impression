# Fix 02: Coplanar Loft Face-Touch Union

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Coplanar Loft Face-Touch Union ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Coplanar Loft Face-Touch Union ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [GitHub issue #243](https://github.com/kellyjanderson/Impression/issues/243)
Split provenance: none
Canonical status: Draft
Review Score: pending independent review

## Source Field Carryover

The issue's observed behavior, expected behavior, reproduction geometry, and a4 milestone are retained. This specification adds an implementation boundary and measurable acceptance contract without weakening the issue.

## Purpose

Make union of loft surface bodies sharing a coplanar face return one validated shell instead of overlapping shells or a rejected partial result.

## Scope

Surface union contact classification, coincident-patch comparison, interior-pair removal, boundary seam merging, shell reconstruction, diagnostics, and fixture tests.

## Split Coverage

This leaf owns the complete responsibility stated above. It does not claim adjacent leaves indexed by the release intake.

## Refinement History

Initial do-specs draft. Independent refinement has not yet occurred.

## Implementation Routing

Feature branch after canonical review; integrate through the future a4 working branch. Back-reference issue #243 and this specification in commits and PRs.

## Chosen Defaults / Parameters

Classify geometrically coincident patches with opposite orientation as an interior pair only when their trimmed domains match within the existing surface tolerance. Remove both, merge the remaining boundary graph, and validate exactly one closed shell.

## Data Ownership

The surface CSG executor owns contact classification and reconstruction. `SurfaceBody` remains the owner of final patches and topology; validation owns success eligibility.

## Dependencies And Routes

Existing loft patch metadata, surface tolerances, `execute_loft_pair_csg`, and public surface-union validation.

## Prerequisite Handling

No dependency on difference execution. The coincident classifier becomes reusable evidence for later boolean work.

## Application Integration

`boolean_union` and internal loft-pair CSG use the same result validator. There is no mesh conversion or silent multi-shell fallback.

## Reuse And Extraction Plan

Extract only the shared records and validators named here. Do not create a parallel execution stack or copy planner logic between public and internal routes.

## Required DTOs / Functions / Components

Coincident contact record with patch ids, orientation relation, and domain match; interior-pair filter; remaining-shell assembler; union diagnostic evidence.

## Performance Contract

Patch-pair classification must be bounded by candidate spatial overlap, not unconditional sampling of every patch pair.

## Error And State Behavior

Ambiguous orientation, partial domain overlap, open seams, or a result other than one closed shell produces an explicit unsupported/invalid result; operands remain unchanged.

## Test Strategy

Use two face-touching loft fixtures plus near-coplanar and partial-overlap negatives. Assert patch removal, seam closure, shell count, and deterministic diagnostics. The paired contract is [Fix 02 Test](../test-specifications/fix-02-coplanar-loft-face-touch-union-v1_0.md).

## Acceptance Criteria

- [ ] The reproduced coplanar loft union yields exactly one closed SurfaceBody shell.
- [ ] The shared opposite-oriented interior patch pair is absent from the result.
- [ ] Near-coplanar or partial-domain contacts are not misclassified as the exact face-touch route.
- [ ] No mesh conversion or overlapping-shell partial success is returned.

## Readiness Checklist

- [x] Source issue and release ownership recorded.
- [x] Architecture transition and paired test contract identified.
- [x] Ownership, failure behavior, and measurable acceptance drafted.
- [ ] Independent review specs completed.
- [ ] Valid Review Score assigned and canonical status confirmed.
- [ ] Final progression responsibility coverage verified.

## Review Score Calculation

Template source: /Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md

Prior score: none

- Intent and scope: pending independent review
- Architecture and ownership: pending independent review
- Dependencies and integration: pending independent review
- Error, performance, and test contracts: pending independent review
- Acceptance and implementability: pending independent review

Total: pending independent review

