# Fix 09: Surface Difference No-Op Result Gate

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Surface Difference No-Op Result Gate ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Surface Difference No-Op Result Gate ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [GitHub issue #248](https://github.com/kellyjanderson/Impression/issues/248)
Split provenance: Issue #248 is split by [Known-Issue Intake](../planning/known-issue-intake.md).
Canonical status: Draft
Review Score: pending independent review

## Source Field Carryover

The source failure, expected result, test-model evidence, and a4 milestone are retained. This draft defines a narrow implementation and validation boundary without weakening the issue.

## Purpose

Prevent every surface difference route from reporting success when the output is geometrically unchanged from the minuend.

## Scope

Public difference result validation, geometry-change witnesses, no-cut classification, tolerance policy, diagnostics, and cross-route tests.

## Split Coverage

Fixes 08 and 09 collectively preserve 100% of issue #248: execution constructs changed geometry; the shared result gate prevents false success. Neither leaf is optional.

## Refinement History

Initial do-specs draft. Independent refinement has not yet occurred.

## Implementation Routing

Feature branch after canonical review; integrate through the future a4 working branch. Back-reference issue #248 and this specification in commits and PRs.

## Chosen Defaults / Parameters

A successful difference requires both cutter-interaction evidence and at least one validated geometry-change witness: removed/trimmed base patch domain, new intersection boundary, cutter-derived result patch, or changed topology. Object identity or cloned patches are not evidence.

## Data Ownership

The public surface-difference result gate owns success eligibility across all executors. Executors provide evidence but cannot bypass the gate.

## Dependencies And Routes

Existing no-cut result semantics, patch provenance, shell validation, and Fix 08 evidence records. The gate lands before Fix 08 is claimed complete.

## Prerequisite Handling

None for the gate itself. Fix 08 must integrate with it and cannot declare completion until it passes.

## Application Integration

Every public surface difference result passes through one validator, including analytic, loft, and future routes. Legitimate spatially disjoint no-cut remains a classified no-op outcome, not fabricated cut success.

## Reuse And Extraction Plan

Extend canonical surface CSG evidence, reconstruction, and validation. Do not introduce a mesh fallback or test-model-only route.

## Required DTOs / Functions / Components

`GeometryChangeWitness`; normalized difference-result evidence; unchanged-result comparator using topology/domain provenance and bounded geometric checks; public validation hook.

## Performance Contract

Prefer provenance/topology checks; geometric comparison is bounded fallback. Validation may not require dense whole-body sampling.

## Error And State Behavior

Interaction evidence plus no geometry change is an internal-invalid result with diagnostics. A proven disjoint cutter returns the documented no-cut outcome. Ambiguous comparison refuses success.

## Test Strategy

Test cloned-minuend false success, true cut, disjoint no-cut, tangential contact, tolerance-near changes, and every registered surface difference executor. The paired contract is [Fix 09 Test](../test-specifications/fix-09-surface-difference-no-op-result-gate-v1_0.md).

## Acceptance Criteria

- [ ] An unchanged clone of the minuend can never be returned as successful difference geometry.
- [ ] Every successful difference includes an inspectable geometry-change witness.
- [ ] Proven disjoint no-cut behavior remains distinct and deterministic.
- [ ] All registered surface difference routes pass through the same gate.

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

