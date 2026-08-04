# Fix 05: Count-Changing Region Identity Preservation

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Source artifact: [GitHub issue #246](https://github.com/kellyjanderson/Impression/issues/246)
Split provenance: Issue #246 is split by `../planning/known-issue-intake.md`; this leaf owns identity resolution and synthetic lineage while Fix 06 owns caller configuration propagation.
Canonical status: Draft
Review Score: pending independent review
Prerequisites:
- `fix-03-named-hole-identity-pairing-v1_0.md` - supplies identity-bearing loop references for synthetic stations

## Source Field Carryover

- Source purpose: Resolve stable exact region identities before ambiguity enumeration and preserve region/loop lineage through every synthetic station in count-changing loft expansion.
- Source responsibilities by category:
  - Functions/methods: resolve exact region pairs, create lineage-bearing synthetic stations, validate derived identities, feed executor
  - Data structures/models: synthetic station identity/lineage record with predecessor and successor region/loop refs
  - Dependencies/services: Fix 03 loop refs, existing region identity assignment, expansion and transition pairing
  - Returns/outputs/signals: expanded plan retaining exact `shell`/`guide-a` pairs and explicit birth for `guide-b`
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: replace anonymous geometry-only synthetic sections inside existing expansion
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive writes
  - Security/privacy-sensitive behavior: not applicable
  - Performance-sensitive behavior: exact matches are removed before permutation search; lineage propagation is linear
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none hidden; independent review may refine split cohesion and exact symbol names.
- Source split/provenance notes: Issue #246 is split by `../planning/known-issue-intake.md`; this leaf owns identity resolution and synthetic lineage while Fix 06 owns caller configuration propagation.

## Purpose

Resolve stable exact region identities before ambiguity enumeration and preserve region/loop lineage through every synthetic station in count-changing loft expansion.

## Scope

- Owns:
  - exact region pairing before geometric permutation generation
  - synthetic station predecessor/successor identity and topology-path lineage
  - stable derived ids, reverse-direction equivalence, and conflict validation
  - named 2-to-3 and rail-pair public planning/execution regressions

- Does not own:
  - planner option/branch-limit propagation, owned by Fix 06
  - junction patch construction, owned by Fix 04

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none
- Issue-level split disposition: Issue #246 is split by `../planning/known-issue-intake.md`; this leaf owns identity resolution and synthetic lineage while Fix 06 owns caller configuration propagation.

## Refinement History

Not applicable before review. No request review ledger exists; this is a do-specs creation draft.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - identity-first region assignment, `_expand_split_merge_stations`, lineage validation
- Supporting modules/files:
  - none
- GUI/QML files, if applicable:
  - none; no QML is involved
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - identity-first region assignment, `_expand_split_merge_stations`, lineage validation
- Tests:
  - `tests/test_loft_identity_first_pairing.py` - named 2-to-3 exact pairs and birth
  - `tests/test_loft_point_lifecycle_records.py` - synthetic lineage records
  - `tests/test_loft.py` - selected microphone rail-pair public execution

## Chosen Defaults / Parameters

- pair unique exact region identities before geometry scoring
- synthetic stations derive ids and lineage from an explicit transition record, never geometry alone
- preserve predecessor/successor region and loop references through direction reversal
- fail conflicting or incomplete lineage before execution

## Data Ownership

- source of truth: authored region/loop `TopologyPath` identities
- read ownership: planner identity resolution and expansion
- write ownership: planner creates immutable derived lineage records
- derived/cache data: synthetic station geometry and identity indexes are recomputable from plan inputs
- privacy/logging: not applicable

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 03, current `_identity_first_region_assignment`, expansion helpers, plan metadata, executor
  - library route: public planner/`Loft` -> exact region resolution -> synthetic expansion -> execution
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-loft-identity-and-junction-correctness.md` - owns synthetic-station lineage
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - existing region identity assignment, station records, transition expansion, and plan metadata
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - Fix 03
- Progression handling:
  - implement after Fix 03 and before Fix 04

## Application Integration

- App type: library-only
- User/caller surface: `loft_plan_sections(...)`, ambiguity inspection, and `Loft(...)`
- Invocation route: named stations -> exact pairing -> count-changing expansion -> plan/executor
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: stable plan lineage and executable surfaced rail transition
- Integration validation: public planner metadata inspection plus real `Loft` execution
- Incomplete status risk: drafted and prerequisite-linked

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: `loft_plan_sections(...)`, ambiguity inspection, and `Loft(...)` is the consuming public route and public planner metadata inspection plus real `Loft` execution

## Reuse And Extraction Plan

- Existing code to reuse:
  - existing code: region identity assignment, `Station`, topology paths, transition pairing, plan metadata
- Current reuse readiness:
  - readiness: extend existing records and expansion; no parallel planner
- Extraction/wrapping needed:
  - extraction: shared lineage constructor/validator for split and merge
- Additions to existing library/modules:
  - readiness: extend existing records and expansion; no parallel planner
- New reusable modules to expose:
  - new reusable modules: none
- One-off code justification, if any:
  - one-off justification: none

## Required DTOs / Functions / Components

- DTOs/models:
  - synthetic station lineage record
- Functions/methods:
  - predecessor/successor region and loop references
  - stable derived identity/path constructor
  - lineage completeness/conflict validator
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- exact matches reduce ambiguity candidates before enumeration
- lineage propagation is O(regions + loops) per synthetic station
- no geometry-only permutation for already resolved identities

## Error And State Behavior

- duplicate, contradictory, or unmatched required exact ids fail with source/target diagnostics
- synthetic items with missing lineage fail before surface execution
- no anonymous rebuilding of identity-bearing sections

## Test Strategy

- Unit tests:
  - 2-to-3 exact pairs plus birth, reverse direction, multiple synthetic stations, conflict and incomplete-lineage cases
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - selected plan executes through `Loft` and retains identities in output diagnostics for the microphone rail-pair transition
- Production-data rule:
  - tests use project fixtures and temporary directories; they do not require user production data

## Acceptance Criteria

- A named 2-to-3 transition resolves stable exact pairs before geometry scoring and treats only the new identity as a birth.
- Every synthetic station retains exact predecessor and successor region/loop identities and topology paths.
- Reverse planning preserves equivalent lineage with direction correctly inverted.
- Conflicting or incomplete identities fail before execution with precise diagnostics.
- The selected microphone rail-pair transition is representable without mesh fallback.

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
