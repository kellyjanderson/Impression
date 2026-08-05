# Fix 12: Documentation Policy Test Migration (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - repository governance test correction`
Source artifact: `tests/test_documentation_rules.py`
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `none` - current managed documentation/reference skills and release lifecycle exist.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted three governance test methods, three authority
  dependencies, one test-module output, three reused authority files, and one existing
  test-module addition. No application, persistence, concurrency, write, security,
  performance, prerequisite, readiness, or unresolved-gap responsibility remains.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 12.5
- Split decision: remain whole; the three assertions jointly validate one documentation
  authority boundary in one existing test module.

## Source Field Carryover

- Source purpose: stop tests from requiring retired governance mirrors.
- Source responsibilities by category:
  - Functions/methods: authority existence, documentation completion, reference lifecycle tests.
  - Dependencies/services: documentation skill, reference-artifact skill, release lifecycle.
  - Returns/outputs/signals: passing/failing pytest module.
  - Reusable code plan: read current authority text as-is.
  - Data models, UI, database, async, write, security, performance, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: semantic obligation checks replace exact prose copies.
- Source split/provenance notes: not applicable.

## Purpose

Align documentation governance tests with the repository's current managed authorities.

## Problem And Outcome

`tests/test_documentation_rules.py` asserts retired `agents/`, `project/agents/`,
and old project specification paths. The test must validate the current managed
skill and release-folder rules rather than requiring obsolete mirrors.

## Scope

- Replace retired path assertions with current `.agents/skills` authority checks.
- Assert the active release/reference lifecycle wording actually relied upon.
- Avoid duplicating full skill text in tests.

Not in scope: rewriting the documentation skills or restoring deprecated mirrors.

## Split Coverage

- Parent spec: `none`
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-040607.md` | 2 | a3 specs 01-12, 13A, 13B | none | reached |

## Implementation Routing

- `tests/test_documentation_rules.py`.
- `.agents/skills/documentation/SKILL.md`, release lifecycle, and applicable
  reference-artifact skill files as read-only authorities.

## Chosen Defaults / Parameters

- Current managed skill paths and release lifecycle are authoritative.
- Assert required concepts with narrow semantic phrases/structure, not full text copies.
- Retired `agents/` and `project/agents/` mirrors are not restored.

## Data Ownership

- Source of truth: managed skill files and `project/releases/README.md`.
- Read ownership: `tests/test_documentation_rules.py` reads repository text.
- Write ownership: none at runtime; tests are read-only.
- Derived/cache data: assertion strings only.
- Privacy/logging constraints: repository text only.

## Dependencies And Routes

- Domain/service dependencies: documentation skill; reference-artifact skill; release lifecycle.
- Database, GUI, and background/concurrency routes: not applicable.

## Prerequisite Handling

- Architecture feedback artifacts/status: none; not applicable.
- Already implemented prerequisites: current authority files.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none.
- Progression handling: current item may proceed before full suite qualification.

## Application Integration

- App type: workflow.
- User/caller surface: repository pytest/CI governance lane.
- Invocation route: pytest discovery -> documentation-rule module -> authority files.
- Wiring owner/module: `tests/test_documentation_rules.py`.
- Observable result: clear pass/failure naming the current authority/obligation.
- Integration validation: focused module plus normal CI suite.
- Incomplete status risk: path-existence-only checks can pass after obligation removal.

## Reuse And Extraction Plan

- Existing code to reuse: current authority files, read as-is.
- Current reuse readiness: reusable as-is; update existing test module only.
- Extraction/wrapping/new modules: none.
- Additions to existing library/modules: semantic tests in existing pytest module.
- One-off code justification: repository governance assertions belong in this module.

## Required DTOs / Functions / Components

- DTOs/models and UI components: not applicable.
- Functions/methods: authority existence, documentation completion, reference lifecycle tests.

## Performance Contract

- Read a fixed small set of text files once per focused module run.

## Error And State Behavior

- Missing authority or obligation fails with the current path/concept in the assertion.
- Unrelated prose changes do not fail semantic checks.

## Test Strategy

- Unit tests: the governance module itself plus mutation-style temporary copies.
- Integrated route tests: normal CI pytest invocation.
- Service/DB and GUI tests: not applicable.
- Production-data rule: repository fixtures only.

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

## Readiness Checklist

- [x] Ancestors, full score, carryover, canonical status, and terminal ledger are explicit.
- [x] Authority paths, semantic defaults, functions, workflow route, and errors are explicit.
- [x] No blocker, missing prerequisite, unresolved gap, or split coverage remains.
- [x] Focused and CI verification avoid production data.
