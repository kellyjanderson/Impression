# Fix 09: Surface Difference No-Op Result Gate

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [GitHub issue #248](https://github.com/kellyjanderson/Impression/issues/248)
Split provenance: Issue #248 is split by `../planning/known-issue-intake.md`; this leaf owns shared difference success eligibility while Fix 08 owns cut construction.
Canonical status: Archived
Review Score: 26
Prerequisites:
- none - existing result envelope, patch provenance, and no-cut semantics are the baseline

## Source Field Carryover

- Source purpose: Prevent every surfaced difference executor from reporting success when the returned geometry is unchanged from the minuend.
- Source responsibilities by category:
  - Functions/methods: normalize executor evidence, compare result to minuend, classify no-cut, gate success
  - Data structures/models: `GeometryChangeWitness` and normalized difference-result evidence
  - Dependencies/services: patch provenance/domains, topology, operand interaction evidence, CSG result envelope
  - Returns/outputs/signals: successful changed body, documented disjoint no-cut outcome, or invalid/unsupported result
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: one shared gate for analytic, loft, and future surface difference executors
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive writes
  - Security/privacy-sensitive behavior: not applicable
  - Performance-sensitive behavior: provenance/topology checks first; bounded geometric comparison only as fallback
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none hidden; independent review may refine split cohesion and exact symbol names.
- Source split/provenance notes: Issue #248 is split by `../planning/known-issue-intake.md`; this leaf owns shared difference success eligibility while Fix 08 owns cut construction.

## Purpose

Prevent every surfaced difference executor from reporting success when the returned geometry is unchanged from the minuend.

## Scope

- Owns:
  - normalized geometry-change witness contract across all surface difference routes
  - shared postcondition distinguishing true cut, proven disjoint no-cut, invalid unchanged overlap, and ambiguity
  - provenance/topology-first comparison with bounded geometric fallback
  - registry-wide public-route tests including cloned-minuend false success

- Does not own:
  - constructing loft trim/fragment geometry, owned by Fix 08
  - changing union/intersection success criteria except shared evidence reuse

## Split Coverage

- Split parent: this specification
- Parent coverage status: 100% covered
- Coverage matrix:
  - `fix-09a-difference-geometry-change-evidence-v1_0.md` - Covered: witness model, evidence normalization, unchanged comparison.
  - `fix-09b-difference-public-success-gate-v1_0.md` - Covered: success/no-cut classification, registry-wide public gate.
- Parent responsibilities still missing from children:
  - none
- Parent disposition: Archived after both children completed fresh review and canonicalization.

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 1 | Fixes 01-09 | Fix 09a and Fix 09b | continue |

Pass 1 split decision: forced split into Fix 09a and Fix 09b.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - shared difference result gate and evidence normalization
- Supporting modules/files:
  - none
- GUI/QML files, if applicable:
  - none; no QML is involved
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - shared difference result gate and evidence normalization
- Tests:
  - `tests/test_surface_csg.py` - true cut, cloned minuend, disjoint, tangent, tolerance-near, and executor registry coverage
  - `tests/csg_reference_fixtures.py` - rotated snap-groove false-success regression

## Chosen Defaults / Parameters

- success requires cutter-interaction evidence plus at least one validated geometry-change witness
- valid witnesses include changed/removed base domain, new intersection boundary, cutter-derived result patch, or changed topology
- cloned objects and object identity differences are not evidence
- proven disjoint input uses documented no-cut semantics; overlap plus unchanged result is invalid

## Data Ownership

- source of truth: immutable minuend/cutter/result topology and normalized executor evidence
- read ownership: shared public result gate
- write ownership: none; gate classifies the executor result
- derived/cache data: comparison summaries are recomputable
- privacy/logging: not applicable

## Dependencies And Routes

- Domain/service dependencies:
  - existing no-cut classification, patch provenance, topology/body validators, CSG route registry
  - library route: every surfaced difference executor -> normalized evidence -> shared postcondition -> public result
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-surface-boolean-correctness-and-api-boundary.md` - owns surfaced difference success eligibility
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - structured result envelope, no-cut semantics, patch provenance, and body validity evidence
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - none
- Progression handling:
  - implement before Fix 08 may claim completion

## Application Integration

- App type: library-only
- User/caller surface: public `boolean_difference` and every registered surfaced executor
- Invocation route: executor result/evidence -> shared geometry-change/no-cut gate -> public `SurfaceBooleanResult`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: truthful changed success or explicit disjoint/invalid/unsupported outcome
- Integration validation: public route matrix across all registered executors
- Incomplete status risk: completion requires the declared integrated route and prerequisite sequence to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: public `boolean_difference` and every registered surfaced executor is the consuming public route and public route matrix across all registered executors

## Reuse And Extraction Plan

- Existing code to reuse:
  - existing code: result envelope, patch provenance, no-cut semantics, topology/body validity
- Current reuse readiness:
  - readiness: add one shared postcondition to existing CSG module
- Extraction/wrapping needed:
  - extraction: `GeometryChangeWitness` and evidence normalizer reused by every executor
- Additions to existing library/modules:
  - readiness: add one shared postcondition to existing CSG module
- New reusable modules to expose:
  - new reusable modules: none
- One-off code justification, if any:
  - one-off justification: none

## Required DTOs / Functions / Components

- DTOs/models:
  - `GeometryChangeWitness(kind, operand_ids, patch_ids, boundary_ids, tolerance_evidence)`
- Functions/methods:
  - normalized difference-result evidence
  - unchanged-result comparator
  - shared public difference postcondition hook
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- use provenance/domain/topology checks before geometric evaluation
- geometric comparison is bounded and localized to candidate patches
- no dense whole-body sampling or tessellation

## Error And State Behavior

- interaction plus no validated change returns invalid with diagnostic
- proven disjoint returns documented no-cut, not fabricated cut success
- ambiguous comparison refuses success
- no executor may bypass the shared gate

## Test Strategy

- Unit tests:
  - each witness kind, cloned minuend, disjoint, tangent, tolerance-near change, ambiguous evidence
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - public `boolean_difference` routes for every registered surfaced executor assert shared gate participation and correct result classification
- Production-data rule:
  - tests use project fixtures and temporary directories; they do not require user production data

## Acceptance Criteria

- An unchanged clone of the minuend cannot report `status=succeeded`.
- Every successful surface difference carries at least one inspectable geometry-change witness plus cutter-interaction evidence.
- Proven disjoint no-cut behavior remains deterministic and distinct from invalid unchanged overlap.
- Tangential, tolerance-near, and ambiguous cases follow explicit classifications.
- Every registered surfaced difference executor passes through the same public postcondition.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [ ] Review Score appears in front matter and matches a completed independent calculation.
- [x] Current implementation-spec template was loaded; its path is recorded below.
- [ ] Independent adversarial recount completed.
- [x] No unresolved placeholder is hidden as implementation-ready behavior.
- [x] Source responsibilities are carried into durable sections.
- [x] Canonical status is Draft.
- [x] Prerequisites are linked or marked not applicable.
- [x] Missing/stale architecture is tracked in the active ACD.
- [x] Missing prerequisite behavior is linked or marked not applicable.
- [x] Split coverage is recorded for issue-level splits.
- [x] Review ledger is marked not applicable before review.
- [x] Implementation owner/module and reuse/extraction decisions are named.
- [x] UI fields/elements and concurrency are explicit or not applicable.
- [x] Defaults, data ownership, app type, route, performance, privacy, and test strategy are explicit.
- [x] Acceptance criteria are observable and testable.
- [ ] Independent `review specs` confirms cohesion, scoring, canonical status, and final progression coverage.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: pending independent review; rejected as nonnumeric creation placeholder.
- Adversarial rescore basis: recounted every category from the current text; checked hidden route wiring, reuse, prerequisites, write behavior, concurrency, and performance.
- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 4 x 1 = 4
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 4 x 0.5 = 2
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 3 x 2 = 6
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 26
- If total matches prior score, adversarial survival reason: not applicable; prior score was nonnumeric.
