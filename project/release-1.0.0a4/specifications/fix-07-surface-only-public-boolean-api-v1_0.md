# Fix 07: Surface-Only Public Boolean API

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [GitHub issue #247](https://github.com/kellyjanderson/Impression/issues/247)
Split provenance: none
Canonical status: Draft
Review Score: pending independent review
Prerequisites:
- `fix-02-coplanar-loft-face-touch-union-v1_0.md` - proves surfaced union replacement behavior
- `fix-08-loft-surface-difference-cut-execution-v1_0.md` - proves surfaced cut behavior
- `fix-09-surface-difference-no-op-result-gate-v1_0.md` - protects surfaced difference truthfulness

## Source Field Carryover

- Source purpose: Make public modeling booleans surface-only in annotations, runtime behavior, exports, docs, examples, and installed packages while retaining mesh work only behind explicitly non-modeling utilities.
- Source responsibilities by category:
  - Functions/methods: change boolean signatures/guards/exports and separate mesh utility names
  - Data structures/models: public operand/result typing restricted to `SurfaceBody`/`SurfaceBooleanResult`
  - Dependencies/services: modeling package exports, docs/examples, type/runtime tests, packaging smoke
  - Returns/outputs/signals: surfaced result or early actionable `TypeError`; explicit separate mesh utility where retained
  - UI surfaces/components: not applicable
  - UI fields/elements: public parameter names use surface/body terminology
  - Reusable code plan: reuse the existing surface CSG routes and separate existing mesh operations rather than coercing
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: documentation/example edits and package export changes; no runtime destructive writes
  - Security/privacy-sensitive behavior: not applicable
  - Performance-sensitive behavior: reject invalid representations before kernel work; no implicit conversion
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none hidden; independent review may refine split cohesion and exact symbol names.
- Source split/provenance notes: none

## Purpose

Make public modeling booleans surface-only in annotations, runtime behavior, exports, docs, examples, and installed packages while retaining mesh work only behind explicitly non-modeling utilities.

## Scope

- Owns:
  - surface-only public signatures, parameter names, return types, and runtime guards
  - separate names/exports and migration guidance for intentionally retained mesh utilities
  - documentation, examples, type tests, and installed-wheel API agreement
  - guard against mesh operands reappearing in the public modeling surface

- Does not own:
  - implementing surfaced union/difference algorithms, owned by Fixes 02, 08, and 09
  - removing mesh from preview, export, diagnostics, or explicit mesh-analysis utilities

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
  - `src/impression/modeling/csg.py` - surface-only public functions and guards
  - `src/impression/modeling/__init__.py` - public exports
- Supporting modules/files:
  - `docs/modeling/csg.md` - API contract
  - `docs/examples/csg/` and relevant tutorials - surfaced examples
  - `pyproject.toml`/packaging configuration only if export metadata requires it
- GUI/QML files, if applicable:
  - none; no QML is involved
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - surface-only public functions and guards
- Tests:
  - `tests/test_surface_csg.py` - runtime operand/result matrix
  - `tests/test_surface_csg_docs.py` - docs/examples guard
  - installed-wheel smoke - exported signatures and imports

## Chosen Defaults / Parameters

- public `boolean_union`, `boolean_difference`, and `boolean_intersection` accept surfaced modeling operands only
- public surface booleans return `SurfaceBooleanResult` consistently
- mesh operands fail before route selection with migration guidance
- retained mesh operations have distinct non-modeling names and no implicit dispatch

## Data Ownership

- source of truth: public modeling API annotations, runtime validators, and export table
- read ownership: callers and docs consume that contract
- write ownership: modeling package maintainers change exports/docs together
- derived/cache data: generated API docs, if any, derive from source signatures
- privacy/logging: not applicable

## Dependencies And Routes

- Domain/service dependencies:
  - Fixes 02, 08, 09, public modeling exports, docs/examples, build and clean-install smoke
  - library route: caller -> public representation guard -> surface CSG route -> `SurfaceBooleanResult`
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-surface-boolean-correctness-and-api-boundary.md` - owns the surfaced public API boundary
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - current public export mechanism, surface result envelope, and explicit mesh operations
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - Fix 02, Fix 08, and Fix 09
- Progression handling:
  - mark this leaf `Missing prerequisite` until surfaced union and difference replacements pass acceptance

## Application Integration

- App type: library-only
- User/caller surface: installed public `impression.modeling` API consumed by scripts, docs, preview, and export
- Invocation route: import/call -> representation guard -> surfaced solver -> result envelope
- Wiring owner/module: `src/impression/modeling/csg.py` and `src/impression/modeling/__init__.py`
- Observable result: consistent surface-only signature/runtime behavior and actionable mesh migration error
- Integration validation: source and clean-wheel signature/runtime matrix plus docs/example scan
- Incomplete status risk: drafted but prerequisite-blocked

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: installed public `impression.modeling` API consumed by scripts, docs, preview, and export is the consuming public route and source and clean-wheel signature/runtime matrix plus docs/example scan

## Reuse And Extraction Plan

- Existing code to reuse:
  - existing code: surface CSG result envelope/routes and explicit mesh operations
- Current reuse readiness:
  - readiness: narrow existing public boundary; do not create a second surfaced API
- Extraction/wrapping needed:
  - extraction: separate retained mesh utilities behind existing/internal mesh module or explicit names
- Additions to existing library/modules:
  - readiness: narrow existing public boundary; do not create a second surfaced API
- New reusable modules to expose:
  - new reusable modules: none
- One-off code justification, if any:
  - one-off justification: none

## Required DTOs / Functions / Components

- DTOs/models:
  - surface operand validator
- Functions/methods:
  - surface-only public function annotations and parameter names
  - explicit retained mesh utility exports, if any
  - migration/error message and API inventory guard
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- reject mesh/mixed operands before kernel work
- no mesh-to-surface conversion or tessellation
- surface dispatch overhead remains constant

## Error And State Behavior

- mesh or mixed operands raise actionable boundary error naming the separate utility
- unsupported surface topology returns normal `SurfaceBooleanResult`, not a mesh fallback
- source/docs/wheel contract mismatch fails tests

## Test Strategy

- Unit tests:
  - signature/type assertions, runtime representation matrix, export inventory, error messages
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - clean installed wheel imports and calls public APIs; docs/examples execute or static-check against the same contract; preview/export consume surfaced results
- Production-data rule:
  - tests use project fixtures and temporary directories; they do not require user production data

## Acceptance Criteria

- Public modeling boolean annotations and runtime behavior exclude `Mesh` and `MeshGroup`.
- Surface modeling booleans return surfaced result types only and use surface/body parameter terminology.
- Mesh or mixed operands fail before kernel work with clear migration guidance; no implicit conversion occurs.
- Docs, examples, type tests, exports, and a clean installed wheel expose the same surface-only contract.
- An automated guard prevents mesh operands from reappearing in the public modeling API.

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
