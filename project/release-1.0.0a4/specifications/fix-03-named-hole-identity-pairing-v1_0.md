# Fix 03: Named Hole Identity Pairing

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Named Hole Identity Pairing ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Architecture ancestor: [Named Hole Identity Pairing ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Source artifact: [GitHub issue #244](https://github.com/kellyjanderson/Impression/issues/244)
Split provenance: none
Canonical status: Draft
Review Score: pending independent review

## Source Field Carryover

The issue's observed behavior, expected behavior, reproduction geometry, and a4 milestone are retained. This specification adds an implementation boundary and measurable acceptance contract without weakening the issue.

## Purpose

Make authored hole identities control cross-section correspondence before geometric fallback.

## Scope

Planner-native loop references, hole identity lookup, validation of duplicates/missing names, deterministic fallback for unnamed residue, diagnostics, and pairing tests.

## Split Coverage

This leaf owns the complete responsibility stated above. It does not claim adjacent leaves indexed by the release intake.

## Refinement History

Initial do-specs draft. Independent refinement has not yet occurred.

## Implementation Routing

Feature branch after canonical review; integrate through the future a4 working branch. Back-reference issue #244 and this specification in commits and PRs.

## Chosen Defaults / Parameters

Resolve exact authored identity first. Run geometric minimum-cost assignment only for still-unpaired unnamed holes. Never let proximity override a valid unique name match.

## Data Ownership

Loft planning owns loop identity resolution. Section inputs own authored names; executor stages consume resolved `PlannedLoopRef` records rather than re-deriving correspondence.

## Dependencies And Routes

Existing region identity assignment, topology paths, `_minimum_cost_hole_assignment`, and loft plan diagnostics.

## Prerequisite Handling

None. Fix 05 must preserve these records when count-changing expansion later creates synthetic stations.

## Application Integration

All loft entry points that accept named loops route through the same identity-first planner. Anonymous models retain current geometric fallback behavior.

## Reuse And Extraction Plan

Extract only the shared records and validators named here. Do not create a parallel execution stack or copy planner logic between public and internal routes.

## Required DTOs / Functions / Components

Extend `PlannedLoopRef` with stable identity/path data; identity index for section holes; deterministic unresolved-loop assignment; mismatch diagnostics.

## Performance Contract

Identity lookup is linear in loop count with indexed names; fallback retains existing small assignment cost only for unnamed residue.

## Error And State Behavior

Duplicate identities, conflicting paths, or a referenced identity missing where topology requires it fail planning with named diagnostics. Unnamed ambiguity follows existing refusal rules.

## Test Strategy

Pair crossed-position named holes, mixed named/unnamed holes, duplicate names, missing names, and unchanged anonymous fixtures. The paired contract is [Fix 03 Test](../test-specifications/fix-03-named-hole-identity-pairing-v1_0.md).

## Acceptance Criteria

- [ ] Unique named holes pair by identity even when geometric cost prefers the opposite mapping.
- [ ] Only unnamed residue enters geometric assignment.
- [ ] Duplicate or structurally missing identities produce deterministic diagnostics.
- [ ] Existing unnamed-hole behavior remains compatible.

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

