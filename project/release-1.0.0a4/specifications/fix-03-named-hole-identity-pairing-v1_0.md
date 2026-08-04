# Fix 03: Named Hole Identity Pairing

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Source artifact: [GitHub issue #244](https://github.com/kellyjanderson/Impression/issues/244)
Split provenance: none
Canonical status: Draft
Review Score: pending independent review
Prerequisites:
- none - existing section `TopologyPath` metadata and loft planning records are the baseline

## Source Field Carryover

- Source purpose: Make stable authored hole identities determine loft correspondence before geometric proximity, while preserving documented fallback for unnamed residue.
- Source responsibilities by category:
  - Functions/methods: index identities, resolve exact pairs, assign unnamed residue, validate and report correspondence
  - Data structures/models: extended `PlannedLoopRef` carrying stable identity/topology path
  - Dependencies/services: `Station`/section normalization, `TopologyPath`, `_minimum_cost_hole_assignment`, plan diagnostics
  - Returns/outputs/signals: deterministic planned loop pairs consumed unchanged by execution
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: extend canonical loft planner records and assignment pipeline
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive writes
  - Security/privacy-sensitive behavior: not applicable
  - Performance-sensitive behavior: identity indexing linear in loop count; assignment only sees unnamed residue
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none hidden; independent review may refine split cohesion and exact symbol names.
- Source split/provenance notes: none

## Purpose

Make stable authored hole identities determine loft correspondence before geometric proximity, while preserving documented fallback for unnamed residue.

## Scope

- Owns:
  - identity-bearing planner-native loop references and section identity indexes
  - identity-first named-hole pairing and deterministic geometric fallback for unnamed residue
  - duplicate, missing, and contradictory identity diagnostics
  - agreement between plan metadata and executed loft pairing

- Does not own:
  - region identity preservation through synthetic count-changing stations, owned by Fix 05
  - junction surface construction, owned by Fix 04

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
  - `src/impression/modeling/loft.py` - loop reference, identity index, matching, diagnostics, and executor handoff
- Supporting modules/files:
  - none
- GUI/QML files, if applicable:
  - none; no QML is involved
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - loop reference, identity index, matching, diagnostics, and executor handoff
- Tests:
  - `tests/test_loft_identity_first_pairing.py` - crossed named holes and mixed residue
  - `tests/test_topology_path_loft_input.py` - public authored identity route
  - `tests/test_loft_surface_executor_correspondence.py` - plan/execution agreement

## Chosen Defaults / Parameters

- unique matching identity always wins before geometry
- geometric minimum-cost assignment is limited to still-unpaired unnamed holes
- duplicate or contradictory identities fail deterministically
- no implicit renaming or proximity override of a valid identity

## Data Ownership

- source of truth: authored `TopologyPath` values on normalized section loops
- read ownership: planner identity index and matching pass
- write ownership: planner creates immutable resolved loop references; executor consumes them
- derived/cache data: identity indexes are recomputable per plan
- privacy/logging: not applicable

## Dependencies And Routes

- Domain/service dependencies:
  - existing `PlannedLoopRef`, section normalization, topology paths, geometric assignment, and diagnostics
  - library route: `loft_plan_sections`/`Loft` -> normalize -> identity-first pairing -> plan -> executor
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-loft-identity-and-junction-correctness.md` - owns planner-native loop identity
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - authored topology-path metadata and planner/executor correspondence records
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - none
- Progression handling:
  - canonicalize and implement before Fix 05 and Fix 04

## Application Integration

- App type: library-only
- User/caller surface: `loft_plan_sections(...)` and `Loft(...)` callers
- Invocation route: authored sections -> normalization -> identity-first loop pairing -> plan -> surface execution
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: plan metadata and output surface follow the named hole paths
- Integration validation: public planner and executor tests with crossed names and anonymous controls
- Incomplete status risk: drafted; metadata-only proof is insufficient without executor agreement

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: `loft_plan_sections(...)` and `Loft(...)` callers is the consuming public route and public planner and executor tests with crossed names and anonymous controls

## Reuse And Extraction Plan

- Existing code to reuse:
  - existing code: `TopologyPath`, normalized section loops, planned refs, minimum-cost assignment, diagnostic records
- Current reuse readiness:
  - readiness: add identity fields and resolution to existing loft module
- Extraction/wrapping needed:
  - extraction: no new module; identity-first helper shared by planner entry points
- Additions to existing library/modules:
  - readiness: add identity fields and resolution to existing loft module
- New reusable modules to expose:
  - new reusable modules: none
- One-off code justification, if any:
  - one-off justification: none

## Required DTOs / Functions / Components

- DTOs/models:
  - extended `PlannedLoopRef(kind, index, identity, topology_path)`
- Functions/methods:
  - section loop identity index
  - identity-first pairing result separating named pairs and unnamed residue
  - stable duplicate/missing/contradictory diagnostics
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- identity resolution is O(n) with indexed names
- geometric assignment receives only unnamed residue
- no new branch enumeration for exact named pairs

## Error And State Behavior

- duplicate or contradictory identities fail before geometry scoring
- a structurally required missing identity produces a named diagnostic
- unnamed ambiguity retains existing refusal policy
- execution may not silently re-pair an already resolved plan

## Test Strategy

- Unit tests:
  - crossed named holes, mixed named/unnamed loops, duplicate, missing, contradictory, and anonymous compatibility cases
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - public planning and `Loft` execution assert the same selected pairs and resulting topology
- Production-data rule:
  - tests use project fixtures and temporary directories; they do not require user production data

## Acceptance Criteria

- Two named holes that exchange positions pair by authored identity, not proximity.
- Only unnamed residue enters geometric matching and existing anonymous behavior remains compatible.
- Duplicate, missing, or contradictory identities produce explicit stable diagnostics.
- Plan metadata and executed loft agree on every selected named-hole pair.
- No mesh fallback or execution-time re-pairing is introduced.

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
