# Fix 07: Surface-Only Public Boolean API

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Surface-Only Public Boolean API ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Surface-Only Public Boolean API ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [GitHub issue #247](https://github.com/kellyjanderson/Impression/issues/247)
Split provenance: none
Canonical status: Draft
Review Score: pending independent review

## Source Field Carryover

The source failure, expected result, test-model evidence, and a4 milestone are retained. This draft defines a narrow implementation and validation boundary without weakening the issue.

## Purpose

Make the public modeling boolean contract surface-only while retaining mesh operations as explicitly separate compatibility or diagnostic utilities.

## Scope

Public type signatures, runtime guards, exports, documentation, examples, deprecation path, installed-package smoke tests, and API contract tests.

## Split Coverage

This leaf owns the complete public API boundary responsibility from issue #247.

## Refinement History

Initial do-specs draft. Independent refinement has not yet occurred.

## Implementation Routing

Feature branch after canonical review; integrate through the future a4 working branch. Back-reference issue #247 and this specification in commits and PRs.

## Chosen Defaults / Parameters

`boolean_union`, `boolean_difference`, and `boolean_intersection` accept surfaced modeling operands and return surfaced results. Mesh operations use separately named APIs and are never selected implicitly.

## Data Ownership

The public modeling API owns surface-only validation and typing. Mesh utilities own mesh inputs/outputs and cannot masquerade as modeling booleans.

## Dependencies And Routes

Fixes 02, 08, and 09 must establish viable surfaced union/difference behavior before compatibility is removed from the public names.

## Prerequisite Handling

Hard prerequisites: Fixes 02, 08, and 09 acceptance. Otherwise keep this leaf blocked rather than narrowing the API prematurely.

## Application Integration

Update package exports, annotations, generated/reference docs, examples, and installed wheel behavior together. Add explicit migration guidance for mesh callers.

## Reuse And Extraction Plan

Extend canonical surface CSG evidence, reconstruction, and validation. Do not introduce a mesh fallback or test-model-only route.

## Required DTOs / Functions / Components

Surface operand runtime validator; surface-only overloads/types; separately named mesh utility exports; deprecation or removal diagnostics; API inventory test.

## Performance Contract

Surface-only dispatch must not add conversion or materialization overhead. Reject mesh operands at the boundary before kernel work.

## Error And State Behavior

Mesh operands passed to public surface booleans fail with an actionable message naming the separate mesh utility. Mixed representations are never coerced silently.

## Test Strategy

Run static signature assertions, runtime operand matrices, docs/example scans, and clean-wheel imports. Verify mesh utilities remain explicit and separate. The paired contract is [Fix 07 Test](../test-specifications/fix-07-surface-only-public-boolean-api-v1_0.md).

## Acceptance Criteria

- [ ] Public modeling boolean signatures accept and return surfaced types only.
- [ ] Mesh and mixed operands are rejected before execution with migration guidance.
- [ ] No hidden mesh fallback or conversion occurs.
- [ ] Source, docs, examples, and installed wheel expose the same contract.

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

