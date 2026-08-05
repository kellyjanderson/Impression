# Fix 06: Expanded Planner Configuration Propagation

Date: 2026-08-04
Status: Final
Primary ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Architecture ancestor: [Active ACD](../architecture/acd-loft-identity-and-junction-correctness.md)
Source artifact: [GitHub issue #246](https://github.com/kellyjanderson/Impression/issues/246)
Split provenance: Issue #246 is split by `../planning/known-issue-intake.md`; this leaf owns configuration propagation while Fix 05 owns identity and lineage.
Canonical status: Canonical
Review Score: 16
Prerequisites:
- none - existing public loft planner parameters and default validation are the baseline

## Source Field Carryover

- Source purpose: Ensure every direct, nested, and synthetic transition-planning call uses the caller's effective configuration, particularly `ambiguity_max_branches`.
- Source responsibilities by category:
  - Functions/methods: construct/validate options, pass through helpers, enforce branch cap, report effective config
  - Data structures/models: `LoftPlannerOptions` immutable value
  - Dependencies/services: public planner functions, expansion/pairing helpers, ambiguity diagnostics
  - Returns/outputs/signals: bounded candidate enumeration or precise refusal carrying effective cap/location
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: centralize existing planner parameters rather than adding helper defaults
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive writes
  - Security/privacy-sensitive behavior: not applicable
  - Performance-sensitive behavior: caller branch cap is a hard global bound through nested expansion
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none hidden; independent review may refine split cohesion and exact symbol names.
- Source split/provenance notes: Issue #246 is split by `../planning/known-issue-intake.md`; this leaf owns configuration propagation while Fix 05 owns identity and lineage.

## Purpose

Ensure every direct, nested, and synthetic transition-planning call uses the caller's effective configuration, particularly `ambiguity_max_branches`.

## Scope

- Owns:
  - one immutable planner-options record at the public boundary
  - explicit propagation through expansion and nested pairing
  - hard enforcement and diagnostics for the effective branch cap
  - 1-to-4-to-7 non-default-limit regression

- Does not own:
  - identity preservation or synthetic lineage, owned by Fix 05
  - changing the public default value or adding new search heuristics

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none
- Issue-level split disposition: Issue #246 is split by `../planning/known-issue-intake.md`; this leaf owns configuration propagation while Fix 05 owns identity and lineage.

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one immutable options record is propagated through one planner call graph.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - public option construction and explicit propagation through every pairing/expansion helper
- Supporting modules/files:
  - none
- GUI/QML files, if applicable:
  - none; no QML is involved
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - public option construction and explicit propagation through every pairing/expansion helper
- Tests:
  - `tests/test_loft_identity_first_pairing.py` - direct/expanded configured limits
  - `tests/test_loft.py` - staged 1-to-4-to-7 public regression

## Chosen Defaults / Parameters

- retain public default `ambiguity_max_branches=64` unless independently changed elsewhere
- construct one immutable options value per public planning invocation
- nested helpers receive options explicitly and define no shadow defaults
- invalid values fail at the public boundary

## Data Ownership

- source of truth: caller-supplied public planner arguments normalized into `LoftPlannerOptions`
- read ownership: every planning helper reads the same immutable value
- write ownership: none after construction
- derived/cache data: effective diagnostic payload is derived from options
- privacy/logging: not applicable

## Dependencies And Routes

- Domain/service dependencies:
  - public `loft_plan_*` functions, transition expansion, pairing/ambiguity enumeration, diagnostics
  - library route: public args -> options -> all nested planning calls
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-loft-identity-and-junction-correctness.md` - owns caller configuration through expansion
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - public option validation, ambiguity counters, and diagnostic metadata
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - none
- Progression handling:
  - this leaf and Fix 05 may proceed independently after review; both precede Fix 04 completion

## Application Integration

- App type: library-only
- User/caller surface: all public loft planning and `Loft(...)` calls exposing planner limits
- Invocation route: caller args -> immutable options -> direct/nested expansion and pairing
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: candidate enumeration respects requested cap and reports it on refusal
- Integration validation: public route tests below/at/above required search size
- Incomplete status risk: completion requires the declared integrated route and prerequisite sequence to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: all public loft planning and `Loft(...)` calls exposing planner limits is the consuming public route and public route tests below/at/above required search size

## Reuse And Extraction Plan

- Existing code to reuse:
  - existing code: planner parameters, validation, ambiguity counters, diagnostic metadata
- Current reuse readiness:
  - readiness: group and thread through existing module
- Extraction/wrapping needed:
  - extraction: `LoftPlannerOptions` in the loft module
- Additions to existing library/modules:
  - readiness: group and thread through existing module
- New reusable modules to expose:
  - new reusable modules: none
- One-off code justification, if any:
  - one-off justification: none

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftPlannerOptions(ambiguity_max_branches, tolerances, split_merge_mode, fairness_mode, ...)`
- Functions/methods:
  - public option normalizer/validator
  - effective-configuration diagnostic payload
  - branch-attempt instrumentation for tests
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- visited/retained candidates never exceed the caller cap
- nested expansion cannot reset or multiply the cap
- options propagation adds constant overhead

## Error And State Behavior

- invalid options fail before planning
- limit exhaustion names effective cap, candidate count, and transition location
- no nested call silently falls back to 64 when caller supplied another value

## Test Strategy

- Unit tests:
  - options construction, explicit helper propagation, invalid values, below/at/above cap behavior
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - staged 1-to-4-to-7 public planning with `ambiguity_max_branches=4096` does not report 64; a smaller cap refuses at that exact cap
- Production-data rule:
  - tests use project fixtures and temporary directories; they do not require user production data

## Acceptance Criteria

- Every direct and nested planning helper receives the caller's effective options.
- `ambiguity_max_branches` is never reset to 64 during synthetic expansion when the caller supplied another value.
- Observed branch attempts honor the configured hard bound.
- The staged 1-to-4-to-7 regression succeeds or refuses according to the supplied limit, never a hidden default.
- Limit diagnostics deterministically identify the effective cap and transition location.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Review Score appears in front matter and matches a completed independent calculation.
- [x] Current implementation-spec template was loaded; its path is recorded below.
- [x] Independent adversarial recount completed.
- [x] No unresolved placeholder is hidden as implementation-ready behavior.
- [x] Source responsibilities are carried into durable sections.
- [x] Canonical status is Canonical.
- [x] Prerequisites are linked or marked not applicable.
- [x] Missing/stale architecture is tracked in the active ACD.
- [x] Missing prerequisite behavior is linked or marked not applicable.
- [x] Split coverage is recorded for issue-level splits.
- [x] Review ledger records the completed request-scoped passes.
- [x] Implementation owner/module and reuse/extraction decisions are named.
- [x] UI fields/elements and concurrency are explicit or not applicable.
- [x] Defaults, data ownership, app type, route, performance, privacy, and test strategy are explicit.
- [x] Acceptance criteria are observable and testable.
- [x] Independent `review specs` confirms cohesion, scoring, canonical status, and release responsibility coverage.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 16; adversarial input, not trusted.
- Adversarial rescore basis: fresh terminal recount checked split lineage, UI/control inventory, routes, reuse, prerequisites, writes, concurrency, performance, and deferral markers.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 2 x 2 = 4
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 16
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
