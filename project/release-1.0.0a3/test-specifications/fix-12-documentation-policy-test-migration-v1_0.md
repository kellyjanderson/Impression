# Fix 12 Test: Documentation Policy Test Migration

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One mutation-oriented documentation test review proves current authority paths and obligations are meaningfully covered.

## Backlink

[Fix 12 specification](../specifications/fix-12-documentation-policy-test-migration-v1_0.md)

## Manual Smoke

Run `tests/test_documentation_rules.py` on a clean checkout and inspect failure
messages to ensure they name current managed skill/release files.

## Automated Smoke

Assert every path opened by the test exists in the current tree and no source
literal references retired `agents/` or `project/agents/` authorities.

## Automated Acceptance

- Run the migrated module as part of the normal suite.
- Against temporary copies, remove each required semantic obligation and assert
  the corresponding check fails.
- Change unrelated prose and assert semantic checks remain stable.
- Cover active release placement, archive immutability, durable documentation,
  and reference image/STL lifecycle obligations.

Tests inspect repository fixtures only and do not modify managed skill sources.
