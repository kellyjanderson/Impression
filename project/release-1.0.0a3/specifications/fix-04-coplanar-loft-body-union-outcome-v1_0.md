# Fix 04: Coplanar Loft-Body Union Outcome (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `project/release-0.1.0a/architecture/csg-coincident-contact-architecture.md`
Source artifact: `testingImp/references/impression-issues.md` issue 4
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `none` - the current public surface boolean path and validity records already exist.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted public union and result validator, boolean outcome
  record, CSG/validity dependencies, valid/refusal outputs, two reused boundaries, one
  module addition, and boolean validation cost. The word incomplete below describes a
  rejected runtime result, not unresolved specification work.
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
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 13
- Split decision: remain whole; validation is part of the public union transaction
  and a validator without the success/refusal handoff is not independently deliverable.

## Source Field Carryover

- Source purpose: prevent the enclosure body from disappearing during coplanar union.
- Source responsibilities by category:
  - Functions/methods: `boolean_union` and result validation.
  - Data structures/models: `SurfaceBooleanResult` success/refusal record.
  - Dependencies/services: surface CSG execution and body/mesh validity checks.
  - Returns/outputs/signals: complete union or typed refusal.
  - Reusable code plan: existing feature gate and result/validity records.
  - Performance-sensitive behavior: validation must remain bounded by result size.
  - UI, database, async, write, security, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: supported success is preferred; refusal is acceptable.
- Source split/provenance notes: not applicable.

## Purpose

Make silent operand loss impossible for the reproduced coplanar loft-body union.

## Problem And Outcome

Unioning a loft body into an enclosure across coplanar contact can collapse the
earlier enclosure, forcing the test model to return a group instead of one body.
The operation must either return a valid union containing both operands or refuse
with a specific unsupported/invalid-result diagnostic; operand loss is forbidden.

## Scope

- Add result validation for the confirmed coplanar loft-body union case.
- Return the valid combined body when the supported kernel path succeeds.
- Return a typed, actionable refusal before exposing an incomplete result.

Not in scope: universal coincident-face boolean support or automatic mesh repair.

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

- `src/impression/modeling/csg.py`: `boolean_union` result classification/gate.
- Surface boolean helpers already used by the public union boundary.
- Focused CSG regression plus the test-modeling enclosure composition.

## Chosen Defaults / Parameters

- Existing boolean tolerance default remains `1e-4`.
- Success requires both operand witnesses and validity; otherwise return typed refusal.
- Operand order must not alter the success/refusal class.

## Data Ownership

- Source of truth: input `SurfaceBody` operands and CSG outcome records.
- Read ownership: `boolean_union` and the result validator.
- Write ownership: immutable result construction; operands are not mutated.
- Derived/cache data: witness/validity measurements are recomputable.
- Privacy/logging constraints: diagnostics contain operation/geometry IDs, not source text.

## Dependencies And Routes

- Domain/service dependencies: surface CSG result path; body/tessellation validity helpers.
- Database, GUI, and concurrency routes: not applicable.

## Prerequisite Handling

- Architecture feedback artifacts: none; existing coincident-contact architecture applies.
- Already implemented prerequisites: `SurfaceBooleanResult` and feature gate.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none.
- Progression handling: current item may proceed after its fixture is committed.

## Application Integration

- App type: library-only.
- User/caller surface: model code calling `boolean_union(...)`.
- Invocation route: public union -> surface result -> validity/witness gate.
- Wiring owner/module: `src/impression/modeling/csg.py`.
- Observable result: complete union or typed refusal.
- Integration validation: test-model enclosure in both operand orders.
- Incomplete status risk: kernel-only success without public result validation can still lose an operand.

## Reuse And Extraction Plan

- Existing code to reuse: surface feature gate and existing result/validity records.
- Current reuse readiness: add validator to existing `csg.py` path.
- Extraction/wrapping/new reusable modules: none.
- Additions to existing library/modules: one result validation boundary.
- One-off code justification: none.

## Required DTOs / Functions / Components

- DTOs/models: existing `SurfaceBooleanResult` with typed failure reason.
- Functions/methods: `boolean_union(...)`; result witness/validity validator.
- UI fields/elements/components: not applicable.

## Performance Contract

- Validation is O(v + f) in result mesh/body evidence and runs once per union.

## Error And State Behavior

- Invalid results return refusal rather than partial geometry; operands stay unchanged.
- Existing unsupported-family refusal remains stable.

## Test Strategy

- Unit tests: witness/validity classification and refusal.
- Integrated route tests: minimal fixture and full enclosure, both operand orders.
- Service/DB and GUI tests: not applicable.
- Production-data rule: committed deterministic geometry only.

## Contract

Inputs are two valid bodies with the reproduced coplanar contact. A successful
output preserves the occupied volume and distinguishing bounds of both operands
and passes body validity checks. Otherwise the operation raises the documented
refusal; it never returns a body that silently omits either operand.

## Acceptance Criteria

- The test-model enclosure either forms one valid body or receives the declared
  coplanar-union refusal without data loss.
- Successful output includes both operand witness regions and is watertight.
- Operand order does not change success/refusal classification.
- Existing supported union fixtures remain green.

## Verification

[Paired test specification](../test-specifications/fix-04-coplanar-loft-body-union-outcome-v1_0.md)

## Readiness Checklist

- [x] Ancestors, template, full score, source carryover, canonical status, and ledger are explicit.
- [x] Runtime invalid-result wording is resolved behavior, not an unresolved gap marker.
- [x] No readiness blocker, missing prerequisite, or split coverage remains.
- [x] Routing, defaults, ownership, reuse, functions, performance, and errors are explicit.
- [x] Library route and integrated proof are explicit; tests avoid production data.
