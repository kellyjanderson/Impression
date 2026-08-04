# Fix 12 Test: Documentation Policy Test Migration

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Verify documentation governance tests target real obligations

- Input: current authority files, retired-path search, and temporary mutated copies.
- Work: test live paths, remove required obligations, and alter unrelated prose to
  distinguish semantic checks from exact-copy checks.
- Output: mutation evidence for each active/archive/document/reference rule.
- Complete when: current-tree tests pass and every required-rule removal fails.

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
