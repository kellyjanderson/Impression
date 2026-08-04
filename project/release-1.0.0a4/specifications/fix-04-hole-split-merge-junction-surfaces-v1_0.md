# Fix 04: Hole Split/Merge Junction Surfaces

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Hole Split/Merge Junction Surfaces ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Architecture ancestor: [Hole Split/Merge Junction Surfaces ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Source artifact: [GitHub issue #245](https://github.com/kellyjanderson/Impression/issues/245)
Split provenance: Issue #245 is split by [Known-Issue Intake](../planning/known-issue-intake.md).
Canonical status: Draft
Review Score: pending independent review

## Source Field Carryover

The source failure, expected outcome, reproduction evidence, and a4 milestone are retained. This leaf owns only the responsibility stated below; its sibling leaf retains the rest of issue #245.

## Purpose

Replace synthetic hole birth/death closure caps with a topology-valid junction transition so split/merge lofts are closed and retain only terminal caps.

## Scope

Count-changing hole transition planning, junction surface construction, patch roles, orientation, seam closure, body validation, and regression fixtures.

## Split Coverage

The intake ledger records sibling ownership and collectively preserves 100% of issue #245. Neither leaf is optional.

## Refinement History

Initial do-specs draft. Independent refinement has not yet occurred.

## Implementation Routing

Feature branch after canonical review; integrate through the future a4 working branch. Back-reference issue #245, this leaf, and its sibling where sequencing applies.

## Chosen Defaults / Parameters

Represent a one-to-many or many-to-one hole change as an explicit junction event between real stations. Construct connecting transition patches around the junction; planar caps are permitted only at authored terminal boundaries.

## Data Ownership

The loft plan owns junction events and loop lineage. The loft surface executor owns their patches. Final body validation owns closure and cap-count eligibility.

## Dependencies And Routes

Fix 03 identity-aware loop refs and Fix 05 preservation through synthetic stations. Existing surface patch and seam records remain authoritative.

## Prerequisite Handling

Fixes 03 and 05 are hard prerequisites; refuse count-changing execution if required lineage is unresolved.

## Application Integration

The standard loft builder consumes junction events without a separate test-model route. Patch diagnostics expose terminal-cap versus junction-transition roles.

## Reuse And Extraction Plan

Extend the canonical planner/executor records and helpers. Do not add test-model-specific identity, junction, or configuration paths.

## Required DTOs / Functions / Components

`LoftJunctionEvent`; junction boundary rings; oriented transition-patch builder; cap-role validation; seam-incidence checker.

## Performance Contract

Junction construction must scale with participating loop segments and must not trigger unbounded branch search.

## Error And State Behavior

Unresolved lineage, self-intersection, non-manifold seam incidence, or an internal closure cap fails the loft with actionable junction diagnostics.

## Test Strategy

Exercise one-to-two and two-to-one hole transitions, reverse station order, terminal hole birth/death, and invalid crossing lineage. Assert closure, cap roles, and seam incidence. The paired contract is [Fix 04 Test](../test-specifications/fix-04-hole-split-merge-junction-surfaces-v1_0.md).

## Acceptance Criteria

- [ ] The reproduced split/merge hole loft is a closed SurfaceBody.
- [ ] Exactly two authored terminal caps remain; no internal closure cap is emitted.
- [ ] Every junction seam has valid manifold incidence and consistent orientation.
- [ ] Invalid or ambiguous junction geometry is refused without returning a partial body.

## Readiness Checklist

- [x] Source issue, split ledger, and release ownership recorded.
- [x] Architecture transition and paired test contract identified.
- [x] Ownership, failure behavior, and measurable acceptance drafted.
- [ ] Independent review specs completed.
- [ ] Valid Review Score assigned and canonical status confirmed.
- [ ] Split responsibility coverage re-verified at the review fixed point.

## Review Score Calculation

Template source: /Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md

Prior score: none

- Intent and scope: pending independent review
- Architecture and ownership: pending independent review
- Dependencies and integration: pending independent review
- Error, performance, and test contracts: pending independent review
- Acceptance and implementability: pending independent review

Total: pending independent review

