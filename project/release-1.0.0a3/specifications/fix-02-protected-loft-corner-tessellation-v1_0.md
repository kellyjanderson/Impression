# Fix 02: Protected Loft Corner Tessellation (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `project/release-0.1.0a/architecture/loft-topology-point-correspondence-architecture.md`
Source artifact: `testingImp/references/impression-issues.md` issue 2
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `fix-01-topology-path-loft-input-preservation-v1_0.md` - must preserve the authored protection record into loft planning first.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted loft and tessellation methods, protection/boundary
  records, both modules, mesh output, two reused boundaries, both module additions,
  and sampling sensitivity; linked Fix 01 is sequenced rather than missing.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
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
- Total: 14
- Split decision: remain whole; propagation and mandatory sampling are two ends of
  one protected-vertex invariant and neither is independently releasable.

## Source Field Carryover

- Source purpose: stop the audio-cube diagonal corner from disappearing or drifting.
- Source responsibilities by category:
  - Functions/methods: loft surface execution and body tessellation.
  - Data structures/models: protected point identity and mandatory boundary samples.
  - Dependencies/services: loft planner/executor and tessellation engine.
  - Returns/outputs/signals: a mesh containing the protected vertex.
  - Reusable code plan: existing surface-body tessellation and request policies.
  - Performance-sensitive behavior: sample-density changes must not move mandatory vertices.
  - UI, database, async, write, security, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: tolerance uses the active tessellation request.
- Source split/provenance notes: not applicable.

## Purpose

Carry an authored protected point through loft surface construction and tessellation
so the corresponding vertex is stable under sampling-policy changes.

## Problem And Outcome

The protected diagonal corner in the audio-cube half can disappear or drift in
the tessellated body, and the result changes when sample count changes. A
protected loft point must survive planning and surface tessellation as a vertex
within the active geometric tolerance, independent of unrelated sampling density.

## Scope

- Propagate protected-point identity from loft plan through surface patches to
  the tessellation boundary.
- Constrain shared-boundary sampling so protected vertices are mandatory samples.
- Keep fairness disabled behavior deterministic; do not redesign fairness.

Not in scope: accepting `TopologyPath` input (Fix 01) or general adaptive
tessellation quality policy.

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

- `src/impression/modeling/loft.py`: protected point lifecycle in the plan/executor.
- `src/impression/modeling/tessellation.py`: mandatory boundary sample handling.
- Focused regression tests plus `testingImp/models/audio_cube_diagonal_halves.py`.

## Chosen Defaults / Parameters

- Active tessellation tolerance defines coordinate coincidence.
- Protected points are mandatory samples in preview and export policies.
- Extra samples may be inserted but may not move/delete protected samples.

## Data Ownership

- Source of truth: the loft plan owns protected-point identity and coordinates.
- Read ownership: surface execution and tessellation consume the record.
- Write ownership: tessellation writes derived mesh vertices only.
- Derived/cache data: mesh vertices are recomputable from body and request.
- Privacy/logging constraints: IDs/coordinates may appear in diagnostics; source text may not.

## Dependencies And Routes

- Domain/service dependencies: loft surface executor; surface tessellation engine.
- Database dependencies: none.
- GUI route: not applicable at this library boundary.
- Background/concurrency route: not applicable; deterministic synchronous execution.

## Prerequisite Handling

- Architecture feedback artifacts: none.
- Architecture feedback status: not applicable; existing correspondence architecture covers protected intent.
- Already implemented prerequisites: surface-body tessellation request pipeline.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: Fix 01, linked above.
- Progression handling: implement Fix 01 before this item.

## Application Integration

- App type: library-only.
- User/caller surface: loft consumers requesting preview or export tessellation.
- Invocation route: `Loft` -> surface executor -> `tessellate_surface_body`.
- Wiring owner/module: `src/impression/modeling/tessellation.py`.
- Observable result: mesh vertex at each protected authored point.
- Integration validation: both audio-cube halves under preview/export requests.
- Incomplete status risk: planner-only propagation would leave tessellation free to drop the point.

## Reuse And Extraction Plan

- Existing code to reuse: loft protected-point records; `tessellate_surface_body` request pipeline.
- Current reuse readiness: add to existing modules.
- Extraction/wrapping needed: none.
- Additions to existing library/modules: loft boundary metadata; mandatory tessellation samples.
- New reusable modules to expose: none.
- One-off code justification: none.

## Required DTOs / Functions / Components

- DTOs/models: existing protected-point metadata and boundary sample constraints.
- Functions/methods: loft surface executor path; `tessellate_surface_body(...)`.
- UI fields/elements/components: not applicable.

## Performance Contract

- Mandatory samples add O(p) work for p protected points and do not trigger global resampling.

## Error And State Behavior

- Missing protected-point propagation is a deterministic validity failure in tests.
- Invalid coordinates are rejected before mesh emission; no partial mesh is returned.

## Test Strategy

- Unit tests: mandatory sample insertion and coordinate stability.
- Service/DB and GUI/controller tests: not applicable.
- Integrated route tests: audio-cube halves across policies and densities.
- Production-data rule: committed deterministic geometry only.

## Contract

Input is a valid loft plan containing a protected authored point. Output is a
tessellated body with a corresponding vertex within tolerance and with bounds
that do not drift when only non-protected sampling density changes. The chosen
rule is vertex preservation, not proximity represented only by an edge crossing.

## Acceptance Criteria

- The diagonal corner is present in both lofted halves within declared tolerance.
- Sample-count changes add samples without moving or deleting protected vertices.
- Shared boundaries remain coincident and the closed result meets mesh QA.
- Non-protected loft behavior and performance remain within existing contracts.

## Verification

[Paired test specification](../test-specifications/fix-02-protected-loft-corner-tessellation-v1_0.md)

## Readiness Checklist

- [x] Ancestors, template source, full rescore, canonical status, and source carryover are explicit.
- [x] Deferral markers, blockers, missing architecture/specs, and split gaps are absent.
- [x] Fix 01 prerequisite and progression order are explicit.
- [x] Routing, reuse, functions/models, defaults, data ownership, and performance bound are explicit.
- [x] UI/database/concurrency/privacy applicability and library integration route are explicit.
- [x] Integrated verification avoids production data and acceptance criteria are testable.
