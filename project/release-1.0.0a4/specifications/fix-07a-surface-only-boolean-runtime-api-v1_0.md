# Fix 07A: Surface-Only Boolean Runtime API

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [Split parent](./fix-07-surface-only-public-boolean-api-v1_0.md)
Split provenance: `fix-07-surface-only-public-boolean-api-v1_0.md` from GitHub issue #247
Canonical status: Canonical
Review Score: 22
Prerequisites:
- Fix 02 surfaced union
- Fix 08c surfaced difference reconstruction
- Fix 09b public difference gate

## Source Field Carryover

- Source purpose: Restrict public modeling boolean signatures, parameter names, exports, return types, and runtime guards to surfaced representations.
- Source responsibilities by category:
  - Functions/methods: surface operand validator, public boolean boundary update, mesh utility export separation
  - Data structures/models: surface-only operand/result type contract
  - Dependencies/services: public modeling export table, surface CSG result envelope
  - Returns/outputs/signals: surfaced result or actionable representation error
  - UI surfaces/components: not applicable
  - UI fields/elements: not applicable
  - Reusable code plan: add the named responsibilities to existing owner modules
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive runtime writes
  - Security/privacy-sensitive behavior: local paths may appear in diagnostics where applicable; source contents are not logged
  - Performance-sensitive behavior: representation validation is constant-time per operand before kernel work
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none.
- Source split/provenance notes: this child owns only the parent responsibilities listed below.

## Purpose

Restrict public modeling boolean signatures, parameter names, exports, return types, and runtime guards to surfaced representations.

## Scope

- Owns:
  - surface-only annotations, naming, exports, and result type
  - early mesh/mixed operand guard with migration message
  - separate explicit export for intentionally retained mesh utilities

- Does not own:
  - docs/examples and clean-package conformance, owned by Fix 07b

## Split Coverage

- Parent spec: `fix-07-surface-only-public-boolean-api-v1_0.md`
- Parent coverage status: 100% covered collectively by the parent's listed children
- Parent responsibilities owned by this child:
  - surface-only annotations, naming, exports, and result type
  - early mesh/mixed operand guard with migration message
  - separate explicit export for intentionally retained mesh utilities
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/a4-open-issues-20260804-165103.md` | 4 | nineteen-leaf active set | none | reached |

Pass 4 split decision: retained. Cohesion reason: one installed runtime representation boundary across coordinated public export modules.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py` - implementation and wiring owner
- Supporting modules/files:
  - use only the dependencies named below
- GUI/QML files, if applicable:
  - not applicable
- Reusable library/module files:
  - `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py` - extend the existing reusable boundary
- Tests:
  - `tests/test_surface_csg.py` runtime/signature matrix

## Chosen Defaults / Parameters

- no implicit mesh conversion
- invalid representation fails before kernel dispatch

## Data Ownership

- Source of truth: `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py`
- Read ownership: the named caller route reads immutable request/plan/surface evidence through the existing public boundary.
- Write ownership: `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py` creates only its derived records/results; input source and operands remain unchanged.
- Derived/cache data: all records are recomputable from the declared inputs.
- Privacy/logging constraints: do not log model source contents; otherwise not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - public modeling export table
  - surface CSG result envelope
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-surface-boolean-correctness-and-api-boundary.md` - target ownership and route are resolved
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - existing records/routes named under Dependencies And Routes
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - Fix 02 surfaced union
  - Fix 08c surfaced difference reconstruction
  - Fix 09b public difference gate
- Progression handling:
  - prerequisites listed above run first; otherwise this child may proceed after canonical review

## Application Integration

- App type: library-only
- User/caller surface: public `impression.modeling` boolean functions
- Invocation route: public `impression.modeling` boolean functions -> `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py` -> surfaced result or actionable representation error
- Wiring owner/module: `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py`
- Observable result: surfaced result or actionable representation error
- Integration validation: `tests/test_surface_csg.py` runtime/signature matrix
- Incomplete status risk: completion requires the caller route and paired test specification to pass

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: public `impression.modeling` boolean functions is the consuming route

## Reuse And Extraction Plan

- Existing code to reuse:
  - public modeling export table
  - surface CSG result envelope
- Current reuse readiness:
  - add responsibilities to the existing owner module
- Extraction/wrapping needed:
  - only the named DTO/helper boundary; no parallel subsystem
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py` - surface operand validator, public boolean boundary update, mesh utility export separation
- New reusable modules to expose:
  - none
- One-off code justification, if any:
  - none

## Required DTOs / Functions / Components

- DTOs/models:
  - surface-only operand/result type contract
- Functions/methods:
  - surface operand validator
  - public boolean boundary update
  - mesh utility export separation
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- representation validation is constant-time per operand before kernel work

## Error And State Behavior

- mesh/mixed inputs identify the separate non-modeling utility

## Test Strategy

- Unit tests:
  - `tests/test_surface_csg.py` runtime/signature matrix
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - public `impression.modeling` boolean functions must exercise surfaced result or actionable representation error
- Production-data rule:
  - deterministic project fixtures and temporary directories only

## Acceptance Criteria

- surface-only annotations, naming, exports, and result type is implemented and asserted through the declared route.
- early mesh/mixed operand guard with migration message is implemented and asserted through the declared route.
- separate explicit export for intentionally retained mesh utilities is implemented and asserted through the declared route.
- The paired test specification [Surface-Only Boolean Runtime API Test](../test-specifications/fix-07a-surface-only-boolean-runtime-api-v1_0.md) passes without helper-only substitution.

## Readiness Checklist

- [x] Ancestors, source parent, split provenance, and 100% collective parent coverage are explicit.
- [x] Numeric Review Score was supplied by the terminal fresh child-review pass.
- [x] Current template path is recorded below.
- [x] Architecture, owner, routes, defaults, data, reuse, prerequisites, performance, privacy, tests, and acceptance are explicit.
- [x] No parent responsibility owned by this child remains only in the parent.
- [x] Child was independently rescored and canonicalized after ledger pass 1 readback.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: 22; adversarial input, not trusted.
- Adversarial rescore basis: fresh terminal recount checked split lineage, UI/control inventory, routes, reuse, prerequisites, writes, concurrency, performance, and deferral markers.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
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
- Missing prerequisites: 3 x 2 = 6
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 22
- If total matches prior score, adversarial survival reason: the score survived a complete terminal recount; no omitted responsibility, blocker, or route was found.
