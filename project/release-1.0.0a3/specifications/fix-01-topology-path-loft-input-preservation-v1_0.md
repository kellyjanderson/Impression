# Fix 01: TopologyPath Loft Input Preservation (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md`
Source artifact: `testingImp/references/impression-issues.md` issue 1
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `none` - this is the first topology-input correction in the a3 loft sequence.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; the obsolete IWU annotation was not a Review Score.
- Adversarial rescore basis: recounted both modified entrypoints, both topology records,
  both module dependencies, success/refusal outputs, existing adapter reuse, and both
  module additions; no UI, persistence, concurrency, write, security, performance,
  prerequisite, readiness, or unresolved-gap responsibility is hidden in this leaf.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
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
- Split decision: remain whole; one normalization boundary produces one identity-preserving
  section result and cannot be delivered meaningfully in smaller independent pieces.

## Source Field Carryover

- Source purpose: allow the named diagonal topology used by the audio-cube model to
  enter `Loft(...)` without erasing authored correspondence intent.
- Source responsibilities by category:
  - Functions/methods: extend `as_section(...)` and `Loft(...)` normalization.
  - Data structures/models: preserve `TopologyPath` into `Section`/`Loop` records.
  - Dependencies/services: topology normalization and loft planning modules.
  - Returns/outputs/signals: normalized section or specific invalid-path refusal.
  - Reusable code plan: reuse `TopologyPath.to_section_loop()` as-is.
  - UI, database, async, write, security, performance, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: none; direct public acceptance is chosen.
- Source split/provenance notes: not applicable.

## Purpose

Make a closed identity-bearing `TopologyPath` a first-class loft topology input
without changing the behavior of existing section-like inputs.

## Problem And Outcome

`Loft(...)` accepts section-like inputs but its normalization through
`as_section` rejects `TopologyPath`; manually converting the path discards the
named point and protection metadata required by correspondence planning. A
closed `TopologyPath` must become a loft section without losing its stable IDs,
correspondence IDs, landmarks, segment roles, anchor, direction, or protected
point intent.

## Scope

- Extend the `Loft`/`as_section` input boundary for closed `TopologyPath` values.
- Define a deterministic mapping from path records to section loop records.
- Refuse open paths or invalid topology with a specific input diagnostic.
- Preserve all existing `Section`, `Region`, `Path2D`, and planar-shape behavior.

Not in scope: tessellation enforcement of protected vertices (Fix 02) or
correspondence inference policy (Fix 03).

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

- `src/impression/modeling/topology.py`: `as_section`, path-to-section adapter.
- `src/impression/modeling/loft.py`: `Loft` normalization and diagnostics.
- `tests/test_loft_api.py` and a focused topology-path loft regression module.
- Reproduction: `testingImp/models/audio_cube_diagonal_halves.py`.

## Chosen Defaults / Parameters

- Closed paths are accepted; open paths are rejected.
- Authored coordinates and identity records are copied without resampling or renaming.
- Existing `segments_per_circle` and `bezier_samples` defaults remain unchanged for `Path2D`.

## Data Ownership

- Source of truth: the input `TopologyPath` owns authored point/segment identity.
- Read ownership: `as_section` reads the path through its topology adapter.
- Write ownership: normalization creates a new `Section`; it does not mutate the path.
- Derived/cache data: the section loop is recomputable from the path.
- Privacy/logging constraints: diagnostics may include IDs and path state, not model source text.

## Dependencies And Routes

- Domain/service dependencies: `impression.modeling.topology`, `impression.modeling.loft`.
- Database dependencies: none.
- GUI route: not applicable.
- Background/concurrency route: not applicable; normalization is synchronous.

## Prerequisite Handling

- Architecture feedback artifacts: none.
- Architecture feedback status: not applicable; existing topology correspondence architecture covers the boundary.
- Already implemented prerequisites: `TopologyPath.to_section_loop()`.
- Missing prerequisite architecture: none.
- Missing prerequisite specifications: none.
- Unimplemented prerequisite specifications: none.
- Progression handling: current item may proceed first in the loft correction lane.

## Application Integration

- App type: library-only.
- User/caller surface: model authors calling `Loft(...)`.
- Invocation route: `Loft` -> `as_section` -> topology adapter -> loft planner.
- Wiring owner/module: `src/impression/modeling/loft.py`.
- Observable result: `SurfaceBody` planning accepts the path and retains its identities.
- Integration validation: direct-call test plus audio-cube model smoke.
- Incomplete status risk: helper-only acceptance would miss the real `Loft` route.

## Reuse And Extraction Plan

- Existing code to reuse: `TopologyPath.to_section_loop()` - canonical identity adapter.
- Current reuse readiness: reusable as-is.
- Extraction/wrapping needed: none.
- Additions to existing library/modules: `topology.as_section`, `loft.Loft` type/diagnostic boundary.
- New reusable modules to expose: none.
- One-off code justification: none.

## Required DTOs / Functions / Components

- DTOs/models: `TopologyPath`, `Section`, and `Loop`; no new DTO.
- Functions/methods: `as_section(...) -> Section`; `Loft(...) -> SurfaceBody`.
- UI fields/elements/components: not applicable.

## Performance Contract

- Conversion is linear in authored path points and adds no geometric sampling pass.

## Error And State Behavior

- Open paths and invalid/duplicate topology fail before loft planning with stable diagnostics.
- The input path is unchanged after success or failure.

## Test Strategy

- Unit tests: field-for-field adapter preservation and invalid inputs.
- Service/DB tests: not applicable.
- GUI/controller tests: not applicable.
- Integrated route tests: direct `Loft` call and audio-cube diagonal-halves model.
- Production-data rule: tests use committed local geometry only.

## Contract

Input is a closed `TopologyPath`; output is the canonical loft planning input
with an identity-preserving loop. Conversion must not resample or rename authored
topology. No unresolved design choice remains: direct acceptance is the public
behavior, with the existing section representation carrying mapped identity.

## Acceptance Criteria

- The test-modeling path can be passed directly to `Loft(...)`.
- Authored point/correspondence IDs and protection flags are observable unchanged
  by loft correspondence planning.
- Invalid open-path input fails before planning with a stable, actionable error.
- Existing section-like input tests remain green.

## Verification

[Paired test specification](../test-specifications/fix-01-topology-path-loft-input-preservation-v1_0.md)

## Readiness Checklist

- [x] Primary and architecture ancestors are explicit.
- [x] Current implementation-spec template source and complete Review Score are recorded.
- [x] Rescore was adversarial; unresolved-gap markers are absent.
- [x] Source fields, canonical status, prerequisites, and split coverage are explicit.
- [x] Review ledger path and terminal new-leaf state are recorded.
- [x] Implementation routing, reuse, functions/models, defaults, and ownership are explicit.
- [x] UI, database, concurrency, performance, and privacy applicability are explicit.
- [x] App type, invocation route, observable result, and integrated validation are explicit.
- [x] Test strategy avoids production data and acceptance criteria are testable.
