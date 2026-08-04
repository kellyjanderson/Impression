# Fix 08: Loft Surface Difference Cut Execution

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Loft Surface Difference Cut Execution ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Loft Surface Difference Cut Execution ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [GitHub issue #248](https://github.com/kellyjanderson/Impression/issues/248)
Split provenance: Issue #248 is split by [Known-Issue Intake](../planning/known-issue-intake.md).
Canonical status: Draft
Review Score: pending independent review

## Source Field Carryover

The source failure, expected result, test-model evidence, and a4 milestone are retained. This draft defines a narrow implementation and validation boundary without weakening the issue.

## Purpose

Execute real surface difference for the reproduced loft cutters, including validated branch decomposition and recomposition where topology requires it.

## Scope

Loft/cutter intersection evidence, trim curves, patch fragmentation, kept-fragment classification, cutter-derived closure patches, branch decomposition/recomposition, body validation, and fixtures.

## Split Coverage

Fixes 08 and 09 collectively preserve 100% of issue #248: execution constructs changed geometry; the shared result gate prevents false success. Neither leaf is optional.

## Refinement History

Initial do-specs draft. Independent refinement has not yet occurred.

## Implementation Routing

Feature branch after canonical review; integrate through the future a4 working branch. Back-reference issue #248 and this specification in commits and PRs.

## Chosen Defaults / Parameters

Build explicit intersection/trim evidence, fragment affected patches, retain fragments by oriented inside/outside classification, add cutter boundary patches, then reconstruct and validate closed shells. Branch decomposition is allowed only with preserved topology lineage.

## Data Ownership

The CSG executor owns intersection and fragment records. The result assembler owns final patch topology. Validation—not the caller—decides whether the cut can report success.

## Dependencies And Routes

Fix 05 identity preservation supports branch decomposition. Fix 09 supplies the mandatory geometry-change gate. Existing surface patch evaluators and CSG evidence records are reused.

## Prerequisite Handling

Fix 09 is a hard prerequisite for success reporting; Fix 05 is required for branched loft fixtures. Unsupported intersection families remain explicit refusals.

## Application Integration

`boolean_difference` routes qualifying loft/cutter pairs through this executor and returns only validated `SurfaceBody` results. The test model uses no grouped-body workaround.

## Reuse And Extraction Plan

Extend canonical surface CSG evidence, reconstruction, and validation. Do not introduce a mesh fallback or test-model-only route.

## Required DTOs / Functions / Components

Intersection-curve evidence; trim-fragment record; inside/outside classifier; cutter-cap builder; branch decomposition/recomposition record; result-shell assembler.

## Performance Contract

Candidate patch pairs must use bounds pruning. Fixture cuts must avoid whole-body dense sampling and complete within the test timeout defined by the paired spec.

## Error And State Behavior

Missing closed trim loops, ambiguous classification, invalid branch recomposition, open seams, or failed body validation returns an unsupported/invalid result with preserved operands.

## Test Strategy

Cut USB, acoustic, and snap-pocket fixtures; include separated branch and topology-native notch cases plus tangential/no-cut negatives. Assert operand witnesses, new boundaries, closure, and deterministic evidence. The paired contract is [Fix 08 Test](../test-specifications/fix-08-loft-surface-difference-cut-execution-v1_0.md).

## Acceptance Criteria

- [ ] Each reproduced qualifying cutter removes material and yields the expected new boundary patches.
- [ ] Branched loft topology is decomposed and recomposed into validated closed result shells.
- [ ] No grouped-body, separated-rail, topology-native notch, or flat-rim workaround is required.
- [ ] Unsupported or invalid geometry is refused without unchanged or partial success.

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

