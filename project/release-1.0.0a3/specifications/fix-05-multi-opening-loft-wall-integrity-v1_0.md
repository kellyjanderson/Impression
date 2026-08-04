# Fix 05: Multi-Opening Loft Wall Integrity (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `project/release-0.1.0a/architecture/loft-tolerance-and-degeneracy-architecture.md`
Source artifact: `testingImp/references/impression-issues.md` issue 5
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `none` - multi-loop section, loft cap, and tessellation contracts already exist.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted loop classification, cap/side assembly, and
  tessellation validation; loop/trim records; loft/tessellation dependencies;
  valid/refusal outputs; two reused contracts; two module additions; and mesh-QA cost.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 17
- Split decision: remain whole after mandatory split review. Loop ownership, cap/side
  assembly, and tessellation validation are one end-to-end loft output transaction;
  any subset would knowingly emit or accept invalid wall geometry.

## Source Field Carryover

- Source purpose: make the test wall's authored holes survive direct lofting.
- Source responsibilities by category:
  - Functions/methods: loop classification, cap/side assembly, trim validation.
  - Data structures/models: outer/inner loop ownership and trim topology.
  - Dependencies/services: loft surface executor and tessellation validator.
  - Returns/outputs/signals: valid wall body or invalid-nesting refusal.
  - Reusable code plan: existing region normalization and mesh QA.
  - Performance-sensitive behavior: validation is bounded by loops and emitted cells.
  - UI, database, async, write, security, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: holes are disjoint inner loops at every station.
- Source split/provenance notes: 17-point leaf retained for transaction cohesion.

## Purpose

Preserve multiple authored openings through loft cap, side, and tessellation execution.

## Problem And Outcome

A wall section containing multiple openings can loft into louver-like faces and
approximately 502 degenerate cells. The same intended wall currently requires a
solid loft followed by boolean cuts. Multiple inner loops must remain holes
through cap construction, side-wall orchestration, and tessellation.

## Scope

- Preserve outer/inner loop classification across the reproduced wall loft.
- Build caps and side surfaces without cross-connecting separate openings.
- Reject invalid loop nesting explicitly rather than emitting degenerate geometry.

Not in scope: arbitrary self-intersecting profiles or a general boolean-cut
replacement program.

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

- `src/impression/modeling/loft.py`: multi-region cap and side orchestration.
- `src/impression/modeling/tessellation.py`: trim-loop tessellation validation.
- Focused loft regression plus the test-modeling multi-opening wall fixture.

## Chosen Defaults / Parameters

- Normalized counter-clockwise outer and clockwise inner winding remains canonical.
- Disjoint inner loops are holes; nested/overlapping invalid loops refuse pre-emission.
- Zero degenerate cells is required at release QA tolerance.

## Data Ownership

- Source of truth: normalized station `Region` outer/inner loops.
- Read ownership: loft executor and trim tessellator.
- Write ownership: executor creates derived patches/mesh; stations remain immutable.
- Derived/cache data: cap/side topology and mesh are recomputable.
- Privacy/logging constraints: diagnostics may include loop IDs/counts only.

## Dependencies And Routes

- Domain/service dependencies: loft surface execution; trim-aware tessellation.
- Database, GUI, and concurrency routes: not applicable.

## Prerequisite Handling

- Architecture feedback artifacts: none; existing loft degeneracy architecture covers the behavior.
- Already implemented prerequisites: `Section`/`Region` loop ownership and mesh QA.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none.
- Progression handling: current item may proceed in the loft correction lane.

## Application Integration

- App type: library-only.
- User/caller surface: model authors lofting multi-opening sections.
- Invocation route: `Loft` -> cap/side executor -> trim tessellation.
- Wiring owner/module: `src/impression/modeling/loft.py`.
- Observable result: direct wall body with preserved openings and clean QA.
- Integration validation: original test-model wall without cut workaround.
- Incomplete status risk: cap-only or tessellation-only correction can still emit louvers/degenerates.

## Reuse And Extraction Plan

- Existing code to reuse: normalized region winding and current mesh QA.
- Current reuse readiness: add behavior to existing loft/tessellation modules.
- Extraction/wrapping/new reusable modules: none.
- Additions to existing library/modules: loop-aware cap/side assembly and validation.
- One-off code justification: none.

## Required DTOs / Functions / Components

- DTOs/models: existing `Region`/`Loop` and trim topology records.
- Functions/methods: loop classifier; cap/side assembler; trim tessellation validator.
- UI fields/elements/components: not applicable.

## Performance Contract

- Classification/validation is O(l + c) for loop count l and emitted cells c.

## Error And State Behavior

- Invalid nesting refuses before geometry emission; no partial body is returned.
- Valid single-opening and solid sections retain current behavior.

## Test Strategy

- Unit tests: loop ownership, invalid nesting, and cross-connection prevention.
- Integrated route tests: original wall plus one/several-opening controls.
- Service/DB and GUI tests: not applicable.
- Production-data rule: committed geometry only.

## Contract

Input is a valid wall section with one outer boundary and multiple disjoint inner
loops at each station. Output preserves the same opening count, contains no faces
bridging an opening, and reports zero degenerate cells under the release QA
tolerance. Invalid nesting is a diagnostic, not best-effort geometry.

## Acceptance Criteria

- The original multi-opening wall model works without solid-wall-plus-cuts.
- Opening count and loop ownership are stable at every station and cap.
- Tessellation has zero degenerate cells and passes the expected watertight check.
- Single-opening and solid-section loft regressions remain green.

## Verification

[Paired test specification](../test-specifications/fix-05-multi-opening-loft-wall-integrity-v1_0.md)

## Readiness Checklist

- [x] Ancestors, full template score, carryover, canonical status, and ledger are explicit.
- [x] The 17-point split review documents transaction cohesion; no blockers or gaps remain.
- [x] Routing, defaults, ownership, reuse, functions/models, performance, and errors are explicit.
- [x] Library route and integrated proof are explicit; tests avoid production data.
