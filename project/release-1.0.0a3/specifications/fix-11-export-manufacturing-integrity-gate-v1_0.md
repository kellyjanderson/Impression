# Fix 11: Export Manufacturing Integrity Gate (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `project/release-0.1.0a/architecture/mesh-execution-tessellation-boundary-architecture.md`
Source artifact: current CLI export source review
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `fix-10-surfacebody-preview-export-consumption-v1_0.md` - surface results must reach the export collector before QA.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted export, mesh-QA, and atomic-write methods; two
  QA/result records; CLI/tessellation/I-O dependencies; STL/refusal outputs; three
  reused boundaries; three module additions; output writes; and mesh-size cost.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
- Adding code to an existing library/module: 3 x 1 = 3
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 22.5
- Split decision: remain whole after mandatory split review. Export-policy tessellation,
  QA refusal, and atomic write form one console transaction; separately shipping them
  would allow invalid or partial manufacturing output.

## Source Field Carryover

- Source purpose: refuse invalid manufacturing STL before target mutation.
- Source responsibilities by category:
  - Functions/methods: CLI export, mesh QA, atomic STL write.
  - Data structures/models: tessellation/mesh-quality result and failure categories.
  - Dependencies/services: CLI collector, tessellation QA, STL I/O.
  - Returns/outputs/signals: STL artifact or nonzero refusal.
  - Reusable code plan: export request, existing QA, `write_stl` serialization.
  - Destructive/write behavior: target creation/replacement is atomic after validation.
  - Performance-sensitive behavior: one linear QA pass and write.
  - UI, database, async, security, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: default export requires watertight manifold output.
- Source split/provenance notes: 22.5-point leaf retained for transaction cohesion.

## Purpose

Make CLI STL export a validated, atomic manufacturing-output transaction.

## Problem And Outcome

The CLI export path can merge data and call `write_stl` without requiring a
watertight, non-degenerate manufacturing result. Export must fail before writing
when the candidate mesh violates the supported STL integrity contract.

## Scope

- Use export tessellation policy for surface inputs.
- Validate non-empty geometry, finite coordinates, zero degenerate faces, and
  watertight/manifold status before STL write.
- Emit an actionable refusal with measured failure categories.
- Preserve an explicit opt-in path only for intentionally open/non-manufacturing
  output if one already exists; do not silently weaken the default.

Not in scope: automatic repair, slicer simulation, or non-STL interchange.

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

- `src/impression/cli.py::export`.
- `src/impression/modeling/tessellation.py` existing export and QA records.
- `src/impression/io.py` write boundary and focused CLI export tests.

## Chosen Defaults / Parameters

- Default STL requires non-empty, finite, zero-degenerate, watertight manifold geometry.
- Export uses `export_tessellation_request`; ASCII/binary selection remains unchanged.
- Validate completely before creating/replacing target; retain current output naming rules.

## Data Ownership

- Source of truth: collected model geometry and requested output options.
- Read ownership: export collector and QA gate.
- Write ownership: atomic STL writer after successful validation.
- Derived/cache data: export mesh/QA report are recomputable.
- Privacy/logging constraints: diagnostics include geometry measurements/path, not model source.

## Dependencies And Routes

- Domain/service dependencies: CLI collector, surface tessellation/QA, STL I/O.
- Database and GUI routes: none.
- Console route: `impression export` args -> model -> collector -> QA -> atomic write -> stdout/exit.
- Background/concurrency route: not applicable; synchronous command.

## Prerequisite Handling

- Architecture feedback artifacts: none; existing mesh boundary architecture applies.
- Already implemented prerequisites: export request, QA records, STL serializer.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: Fix 10, linked above.
- Progression handling: implement Fix 10 before this item.

## Application Integration

- App type: console.
- User/caller surface: `impression export MODEL --output PATH [--overwrite] [--ascii]`.
- Invocation route: command -> loader/collector -> export tessellation -> QA -> atomic writer.
- Wiring owner/module: `src/impression/cli.py`.
- Observable result: success panel/zero exit and STL, or stderr/nonzero refusal with untouched target.
- Integration validation: Typer CLI tests against valid/invalid fixtures and target sentinels.
- Incomplete status risk: helper QA without the command write boundary can still corrupt output.

## Reuse And Extraction Plan

- Existing code to reuse: export request, mesh QA records, `write_stl` serialization.
- Current reuse readiness: add gate/atomic staging to existing modules.
- Extraction/wrapping needed: temporary-path atomic wrapper around final write.
- Additions to existing library/modules: CLI gate, QA invocation, I/O atomic placement.
- New reusable modules to expose: none.
- One-off code justification: wrapper is reusable by STL output boundary only.

## Required DTOs / Functions / Components

- DTOs/models: existing tessellation result and mesh-quality/failure categories.
- Functions/methods: `cli.export`, QA validator, atomic `write_stl` wrapper.
- UI fields/elements/components: not applicable.

## Performance Contract

- One O(v + f) QA pass and one serialization; no duplicate tessellation or full mesh copy beyond staging.

## Error And State Behavior

- Any QA or I/O failure returns nonzero and leaves preexisting/new target unchanged.
- Success writes one complete target and reports selected format/units.

## Test Strategy

- Unit tests: QA categories and atomic placement.
- Integrated route tests: console args/stdout/exit/side effects for valid/invalid fixtures.
- Service/DB and GUI tests: not applicable.
- Production-data rule: generated temporary geometry/targets only.

## Contract

Input is collected model geometry. Output is either an STL written atomically
after passing the integrity gate or a nonzero command failure with no new/partial
target file. Units and format behavior remain unchanged from the current CLI.

## Acceptance Criteria

- Valid watertight model export succeeds in binary and ASCII modes.
- Open, empty, non-finite, non-manifold, and degenerate fixtures fail pre-write.
- Failure reports the violated properties and leaves no partial output.
- The test-modeling release fixtures export with zero degenerates.

## Verification

[Paired test specification](../test-specifications/fix-11-export-manufacturing-integrity-gate-v1_0.md)

## Readiness Checklist

- [x] Ancestors, full score, carryover, canonical status, prerequisite, and ledger are explicit.
- [x] The 22.5-point split review documents indivisible validate/write transaction cohesion.
- [x] Console contract, defaults, ownership, reuse, functions/models, performance, and errors are explicit.
- [x] No blocker, missing prerequisite artifact, unresolved gap, or split coverage remains.
- [x] Integrated CLI tests cover output side effects without production data.
