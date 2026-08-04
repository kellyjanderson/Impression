# Fix 04: Hole Split/Merge Junction Surfaces

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Source artifact: [GitHub issue #245](https://github.com/kellyjanderson/Impression/issues/245)
Split provenance: none
Canonical status: Draft
Review Score: pending independent review
Prerequisites:
- `fix-03-named-hole-identity-pairing-v1_0.md` - identifies continuing, born, and closing holes
- `fix-05-count-changing-region-identity-preservation-v1_0.md` - preserves junction lineage through synthetic stations

## Source Field Carryover

- Source purpose: Replace interior closure caps in one-to-many and many-to-one hole transitions with topology-valid junction surfaces so the published examples produce closed bodies with only terminal caps.
- Source responsibilities by category:
  - Functions/methods: plan junction events, build branch transition patches, orient loops, assemble seams, validate cap roles and closure
  - Data structures/models: `LoftJunctionEvent` plus participating loop lineage and junction boundary rings
  - Dependencies/services: identity-bearing plan, surface patch builder, seam graph, cap/closure evidence
  - Returns/outputs/signals: closed `SurfaceBody`, `cap_valid=True`, `closed_valid=True`, exactly two terminal caps
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: extend existing loft plan/executor and closure evidence rather than a test-only path
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive writes
  - Security/privacy-sensitive behavior: not applicable
  - Performance-sensitive behavior: work scales with participating loop segments and respects caller branch limits
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none hidden; independent review may refine split cohesion and exact symbol names.
- Source split/provenance notes: none

## Purpose

Replace interior closure caps in one-to-many and many-to-one hole transitions with topology-valid junction surfaces so the published examples produce closed bodies with only terminal caps.

## Scope

- Owns:
  - explicit hole junction event planning in both directions
  - junction patch construction, orientation, seam incidence, and patch roles
  - terminal-cap versus interior-junction validation
  - published many-to-one regression, reversed one-to-many regression, and no-mesh proof

- Does not own:
  - identity matching rules, owned by Fix 03 and Fix 05
  - general self-intersecting or arbitrary higher-order junction topology

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none
- Issue-level split disposition: none

## Refinement History

Not applicable before review. No request review ledger exists; this is a do-specs creation draft.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - junction planning, patch execution, roles, closure evidence
- Supporting modules/files:
  - none
- GUI/QML files, if applicable:
  - none; no QML is involved
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - junction planning, patch execution, roles, closure evidence
- Tests:
  - `tests/test_loft_point_birth_death_resolution.py` - transition planning and directionality
  - `tests/test_loft_surface_executor_correspondence.py` - real surface junction execution
  - `tests/test_loft_showcase.py` - published many-to-one and reversed one-to-many examples

## Chosen Defaults / Parameters

- represent birth/death as an explicit interior junction, not a shrunken loop plus planar cap
- allow planar caps only at authored terminal boundaries
- require manifold seam incidence and consistent outer/hole orientation
- refuse unresolved lineage or invalid junction geometry without partial body

## Data Ownership

- source of truth: resolved plan lineage from authored sections
- read ownership: junction planner/executor reads immutable participating loop refs
- write ownership: surface executor creates junction patches and final seam graph
- derived/cache data: junction boundary rings and patch role evidence are recomputable
- privacy/logging: not applicable

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 03 named holes, Fix 05 synthetic lineage, existing patch/seam/cap validators
  - library route: `Loft` -> resolved plan -> junction-aware executor -> closure validation
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-loft-identity-and-junction-correctness.md` - owns junction event and surface execution transition
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - existing planned-loop, surface-patch, seam, and closure-evidence records
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - Fix 03 and Fix 05
- Progression handling:
  - mark this leaf `Missing prerequisite` until Fix 03 and Fix 05 are implemented

## Application Integration

- App type: library-only
- User/caller surface: published split/merge example and `Loft(...)` callers
- Invocation route: sections -> identity-aware plan -> junction event -> surface executor -> closure evidence
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: closed surfaced loft with valid orientation, seams, and exactly two terminal caps
- Integration validation: public example and executor regression in both transition directions
- Incomplete status risk: drafted but prerequisite-blocked

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: published split/merge example and `Loft(...)` callers is the consuming public route and public example and executor regression in both transition directions

## Reuse And Extraction Plan

- Existing code to reuse:
  - existing code: planned loops, surface patches, seam graph, cap roles, closure diagnostics
- Current reuse readiness:
  - readiness: add junction event/patch roles to existing loft module
- Extraction/wrapping needed:
  - extraction: shared junction builder used by both transition directions
- Additions to existing library/modules:
  - readiness: add junction event/patch roles to existing loft module
- New reusable modules to expose:
  - new reusable modules: none unless implementation requires a distinct surface-junction boundary
- One-off code justification, if any:
  - one-off justification: none

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftJunctionEvent(direction, continuing, born, closing, station_span)`
- Functions/methods:
  - junction boundary-ring record
  - oriented junction transition patch builder
  - terminal-cap/junction-role and manifold seam validators
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- construction scales with participating loop segments
- no unbounded ambiguity search in execution
- published fixtures complete within normal loft focused-test timeout

## Error And State Behavior

- unresolved lineage, self-intersection, invalid orientation, or non-manifold incidence fails before success
- internal planar closure cap is an invalid result
- failure returns no partial body and preserves deterministic diagnostics

## Test Strategy

- Unit tests:
  - junction event direction, patch roles, orientation, seam incidence, invalid/crossing lineage
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - published many-to-one and reversed one-to-many through `Loft(...)`, asserting closure and terminal cap count
- Production-data rule:
  - tests use project fixtures and temporary directories; they do not require user production data

## Acceptance Criteria

- Both published many-to-one and reversed one-to-many fixtures have `cap_valid=True` and `closed_valid=True`.
- Every result has complete seam coverage and exactly two authored terminal caps.
- Outer and hole loop orientations validate; no interior closure cap is emitted.
- Named identities determine continuation, birth, and closure consistently in plan and execution.
- No mesh fallback or partial body is returned.

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
- Prior recorded score: none
- Adversarial rescore basis: pending independent `review specs`; this creation action does not count or certify categories.
- Functions/methods: pending independent review
- Data structures/models: pending independent review
- Dependencies/services: pending independent review
- Returns/outputs/signals: pending independent review
- UI surfaces/components: pending independent review
- UI fields/elements: pending independent review
- Existing reusable code reused as-is: pending independent review
- Adding code to an existing library/module: pending independent review
- Creating a new reusable library/module: pending independent review
- Database queries/tables/migrations: pending independent review
- Async/concurrency behavior: pending independent review
- Destructive/write behavior: pending independent review
- Security/privacy-sensitive behavior: pending independent review
- Performance-sensitive behavior: pending independent review
- Cross-screen reusable behavior: pending independent review
- Readiness blockers: pending independent review
- Missing prerequisites: pending independent review
- Unresolved deferral/gap markers: pending independent review
- Total: pending independent review
- If total matches prior score, adversarial survival reason: not applicable until independent review calculates a score.
