# Fix 05: Count-Changing Region Identity Preservation

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Count-Changing Region Identity Preservation ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Architecture ancestor: [Count-Changing Region Identity Preservation ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Source artifact: [GitHub issue #246](https://github.com/kellyjanderson/Impression/issues/246)
Split provenance: Issue #246 is split by [Known-Issue Intake](../planning/known-issue-intake.md).
Canonical status: Draft
Review Score: pending independent review

## Source Field Carryover

The source failure, expected outcome, reproduction evidence, and a4 milestone are retained. This leaf owns only the responsibility stated below; its sibling leaf retains the rest of issue #246.

## Purpose

Preserve authored region and loop lineage when the planner expands a count-changing transition with synthetic stations.

## Scope

Synthetic section construction, predecessor/successor topology paths, region and loop ids, lineage records, diagnostics, and planner tests.

## Split Coverage

The intake ledger records sibling ownership and collectively preserves 100% of issue #246. Neither leaf is optional.

## Refinement History

Initial do-specs draft. Independent refinement has not yet occurred.

## Implementation Routing

Feature branch after canonical review; integrate through the future a4 working branch. Back-reference issue #246, this leaf, and its sibling where sequencing applies.

## Chosen Defaults / Parameters

A synthetic station must be derived from an explicit transition record and carry stable predecessor and successor references. It may not be rebuilt as an anonymous section from geometry alone.

## Data Ownership

The original sections own authored identity. The planner owns derived lineage and synthetic-station ids; executors consume but never replace them.

## Dependencies And Routes

Fix 03 establishes identity-bearing loop references. Existing `_expand_split_merge_stations` and transition pairing are revised rather than duplicated.

## Prerequisite Handling

Fix 03 is a hard prerequisite. Fix 04 consumes the preserved junction lineage.

## Application Integration

All count-changing loft planning, including test-model routes, uses the same expansion function and serialized diagnostics.

## Reuse And Extraction Plan

Extend the canonical planner/executor records and helpers. Do not add test-model-specific identity, junction, or configuration paths.

## Required DTOs / Functions / Components

Synthetic station identity record; predecessor/successor region and loop references; stable topology path propagation; lineage invariant validator.

## Performance Contract

Identity propagation is linear in expanded regions/loops and adds no combinatorial search.

## Error And State Behavior

A synthetic item without complete lineage, duplicate derived id, or conflicting predecessor/successor fails planning before surface execution.

## Test Strategy

Inspect expanded plans for split and merge fixtures with named regions and holes, reverse direction, and multiple synthetic stations. Assert exact lineage stability. The paired contract is [Fix 05 Test](../test-specifications/fix-05-count-changing-region-identity-preservation-v1_0.md).

## Acceptance Criteria

- [ ] Synthetic stations retain exact predecessor and successor region and loop identities.
- [ ] Reversing station order preserves equivalent lineage with direction correctly inverted.
- [ ] No geometry-only anonymous section replaces an identity-bearing transition.
- [ ] Incomplete or conflicting lineage fails before execution.

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

