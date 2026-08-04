# Fix 08: Safe Documentation Archive Extraction (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - corrective containment at the existing CLI boundary`
Source artifact: `src/impression/cli.py::_extract_docs_archive`
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `none` - current ZIP extraction and CLI error boundaries exist.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted validator/extractor methods, archive-member model,
  zip/path dependencies, success/refusal outputs, two reused standard-library boundaries,
  one module addition, filesystem writes, untrusted-input security, and archive-size cost.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 19
- Split decision: remain whole after mandatory split review. Member validation and
  extraction must be one all-or-none trust-boundary transaction; independently shipping
  either half would preserve the vulnerability or break valid installation.

## Source Field Carryover

- Source purpose: prevent documentation archives from writing outside destination.
- Source responsibilities by category:
  - Functions/methods: member validator and `_extract_docs_archive`.
  - Data structures/models: normalized validated member path/type.
  - Dependencies/services: `zipfile` and `pathlib` filesystem resolution.
  - Returns/outputs/signals: extracted tree or `BadParameter` refusal.
  - Reusable code plan: standard ZIP metadata and resolved-path containment.
  - Destructive/write behavior: clean-mode removal and extraction writes.
  - Security-sensitive behavior: archive member names/types are untrusted.
  - Performance-sensitive behavior: one bounded validation pass plus extraction.
  - UI, database, async, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: validate all selected members before clean/removal/write.
- Source split/provenance notes: 19-point leaf retained for transaction cohesion.

## Purpose

Make documentation ZIP installation contained and atomic with respect to unsafe input.

## Problem And Outcome

`_extract_docs_archive` joins archive member names to the destination without
rejecting absolute paths, `..` traversal, or unsafe link-like members. A crafted
documentation ZIP can therefore target files outside the installation directory.

## Scope

- Normalize and validate every archive member path before any extraction write.
- Reject absolute, drive-qualified, traversal, NUL-containing, and link-like entries.
- Require the resolved target to remain within the resolved destination.
- Preserve clean extraction of the release-generated documentation archive.

Not in scope: supporting arbitrary third-party archive formats or repairing
malformed archives.

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

- `src/impression/cli.py::_extract_docs_archive`.
- Focused CLI/archive security regressions using in-memory ZIP fixtures.

## Chosen Defaults / Parameters

- Prevalidate all selected members before `clean` removal or any write.
- Permit regular files/directories only; reject absolute/traversal/drive/NUL/link forms.
- Resolve target and destination and require containment component-wise.

## Data Ownership

- Source of truth: untrusted ZIP metadata plus caller-selected destination.
- Read ownership: validator reads all member metadata before extraction.
- Write ownership: extractor writes only validated members below destination.
- Derived/cache data: normalized target list exists in memory for the operation.
- Privacy/logging constraints: errors name unsafe members but never archive contents.

## Dependencies And Routes

- Domain/service dependencies: Python `zipfile`; `pathlib` resolution/filesystem.
- Database and GUI dependencies: none.
- Console route: docs install command downloads/reads bytes then calls extractor.
- Background/concurrency route: not applicable; operation is synchronous.

## Prerequisite Handling

- Architecture feedback artifacts/status: none; not applicable for localized containment correction.
- Already implemented prerequisites: docs command and ZIP selection logic.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none.
- Progression handling: current item may proceed before release qualification.

## Application Integration

- App type: console.
- User/caller surface: documentation installation command.
- Invocation route: command -> archive bytes -> validator -> optional clean -> extractor.
- Wiring owner/module: `src/impression/cli.py`.
- Observable result: installed docs or nonzero actionable refusal with no mutation.
- Integration validation: CLI install of packaged ZIP and hostile in-memory archives.
- Incomplete status risk: helper-only validation not wired before `clean` can still destroy data.

## Reuse And Extraction Plan

- Existing code to reuse: `ZipInfo`, `Path.resolve`, and current docs-prefix selection.
- Current reuse readiness: add validator to existing CLI module.
- Extraction/wrapping needed: private member-validation helper in `cli.py`.
- Additions to existing library/modules: validation-first extraction ordering.
- New reusable modules to expose: none.
- One-off code justification: helper is private to the single archive boundary.

## Required DTOs / Functions / Components

- DTOs/models: validated member path/type tuple; no public DTO.
- Functions/methods: private member validator; `_extract_docs_archive(...)`.
- UI fields/elements/components: not applicable.

## Performance Contract

- O(n + bytes) for n archive members; no duplicate payload decompression during validation.

## Error And State Behavior

- Any unsafe member refuses before destination cleanup/create/write.
- Valid archives preserve current clean/non-clean behavior; partial output is forbidden.

## Test Strategy

- Unit tests: hostile member/path matrix and valid prefix selection.
- Integrated route tests: packaged docs ZIP through the CLI extraction path.
- Service/DB and GUI tests: not applicable.
- Production-data rule: temporary directories and generated ZIPs only.

## Contract

Input is untrusted ZIP bytes and a chosen destination. Validation is all-or-none:
if any member is unsafe, extraction fails before writing any member. Valid regular
files and directories are written only below the destination. The refusal names
the unsafe member without echoing file contents.

## Acceptance Criteria

- `../`, nested traversal, absolute, drive-qualified, NUL, and symlink-style
  members are rejected before filesystem mutation.
- Prefix-confusion paths cannot escape to a sibling directory.
- A normal release docs ZIP installs successfully with and without `clean`.
- Tests assert no sentinel outside the destination is created or changed.

## Verification

[Paired test specification](../test-specifications/fix-08-safe-docs-archive-extraction-v1_0.md)

## Readiness Checklist

- [x] Ancestors, full score, carryover, canonical status, and terminal ledger are explicit.
- [x] The 19-point split review documents all-or-none security/write cohesion.
- [x] Console route, clean ordering, ownership, reuse, errors, security, and bound are explicit.
- [x] No blocker, missing prerequisite, unresolved gap, or split coverage remains.
- [x] Temporary-fixture verification covers the real route without production data.
