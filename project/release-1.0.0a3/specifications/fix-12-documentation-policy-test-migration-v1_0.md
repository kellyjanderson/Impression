# Fix 12: Documentation Policy Test Migration (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: One documentation-governance test module is migrated from retired paths to the current skill-owned authority.

## Problem And Outcome

`tests/test_documentation_rules.py` asserts retired `agents/`, `project/agents/`,
and old project specification paths. The test must validate the current managed
skill and release-folder rules rather than requiring obsolete mirrors.

## Scope

- Replace retired path assertions with current `.agents/skills` authority checks.
- Assert the active release/reference lifecycle wording actually relied upon.
- Avoid duplicating full skill text in tests.

Not in scope: rewriting the documentation skills or restoring deprecated mirrors.

## Implementation Routing

- `tests/test_documentation_rules.py`.
- `.agents/skills/documentation/SKILL.md`, release lifecycle, and applicable
  reference-artifact skill files as read-only authorities.

## Contract

The test inputs are repository-managed authority files. Tests assert durable
semantic obligations and current paths. Missing authority or removal of the
required completion/reference rules fails clearly; harmless prose changes do not
force exact-copy maintenance.

## Acceptance Criteria

- The documentation-rule module passes on a clean current checkout.
- No assertion references retired `agents/` or `project/agents/` paths.
- Tests still fail if durable documentation or reference-artifact obligations are
  actually removed.
- The release lifecycle's active/archive boundary remains covered.

## Verification

[Paired test specification](../test-specifications/fix-12-documentation-policy-test-migration-v1_0.md)
