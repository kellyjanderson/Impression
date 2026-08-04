# Fix 06: Expanded Planner Configuration Propagation

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Expanded Planner Configuration Propagation ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Architecture ancestor: [Expanded Planner Configuration Propagation ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Source artifact: [GitHub issue #246](https://github.com/kellyjanderson/Impression/issues/246)
Split provenance: Issue #246 is split by [Known-Issue Intake](../planning/known-issue-intake.md).
Canonical status: Draft
Review Score: pending independent review

## Source Field Carryover

The source failure, expected outcome, reproduction evidence, and a4 milestone are retained. This leaf owns only the responsibility stated below; its sibling leaf retains the rest of issue #246.

## Purpose

Ensure every internal planning pass honors caller-supplied ambiguity and search configuration, especially during expanded split/merge transitions.

## Scope

Planner options object, internal expansion calls, ambiguity branch limit, tolerance propagation, diagnostics, and configuration tests.

## Split Coverage

The intake ledger records sibling ownership and collectively preserves 100% of issue #246. Neither leaf is optional.

## Refinement History

Initial do-specs draft. Independent refinement has not yet occurred.

## Implementation Routing

Feature branch after canonical review; integrate through the future a4 working branch. Back-reference issue #246, this leaf, and its sibling where sequencing applies.

## Chosen Defaults / Parameters

Create one immutable planner-options value at the public entry point and pass it explicitly through every pairing and expansion call. No internal call may silently rely on a default after caller configuration exists.

## Data Ownership

The top-level loft planning invocation owns configuration. Nested helpers borrow the same immutable options and report its effective values.

## Dependencies And Routes

Existing `ambiguity_max_branches` behavior and transition pairing helpers. No change to the default value itself is required.

## Prerequisite Handling

Independent of Fix 05 implementation but both must pass before Fix 04 can be claimed complete.

## Application Integration

Every public and internal loft planning route constructs or receives the same options record; diagnostics include the effective branch cap when refusal occurs.

## Reuse And Extraction Plan

Extend the canonical planner/executor records and helpers. Do not add test-model-specific identity, junction, or configuration paths.

## Required DTOs / Functions / Components

`LoftPlannerOptions`; explicit helper parameters; effective-configuration diagnostic payload; test hooks for attempted branch counts.

## Performance Contract

The configured branch limit is a hard upper bound across all expansion passes; nested planning cannot reset or multiply it.

## Error And State Behavior

Invalid option values fail at the public boundary. Limit exhaustion returns the existing ambiguity refusal enriched with effective configuration and transition location.

## Test Strategy

Set branch limits below, at, and above fixture needs across direct and expanded transitions. Assert attempted branches never exceed the caller cap. The paired contract is [Fix 06 Test](../test-specifications/fix-06-expanded-planner-configuration-propagation-v1_0.md).

## Acceptance Criteria

- [ ] Every nested transition pairing receives the caller's planner options.
- [ ] `ambiguity_max_branches` is never reset to the default during expansion.
- [ ] Observed branch attempts respect the configured hard bound.
- [ ] Limit refusal identifies the effective cap and transition location deterministically.

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

