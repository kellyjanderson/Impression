# Fix 09: User-Model Loader Module Identity (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - corrective ownership inside the existing CLI loader`
Source artifact: `src/impression/cli.py::_load_module`
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `none` - model-owned module tracking helpers already exist.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted load, ownership tracking, and cleanup methods;
  module ownership data; importlib/sys.modules dependencies; loaded-module and error
  outputs; existing tracking reuse; and one CLI-module addition.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
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
- Total: 14.5
- Split decision: remain whole; module ownership, cleanup, and reload are one loader
  isolation transaction and must be reviewed through sequential loads.

## Source Field Carryover

- Source purpose: prevent split class identities when user models reload.
- Source responsibilities by category:
  - Functions/methods: load, model-owned module tracking, prior-load cleanup.
  - Data structures/models: owned module-name/path set.
  - Dependencies/services: `importlib` and `sys.modules`.
  - Returns/outputs/signals: loaded module or model-load error.
  - Reusable code plan: `_tracked_preview_module_paths` ownership filtering.
  - UI, database, async, write, security, performance, and cross-screen behavior: not applicable.
- Source open questions / nuance discovered: local model helpers reload; package/third-party modules do not.
- Source split/provenance notes: not applicable.

## Purpose

Reload user-owned model code without reloading Impression or unrelated modules.

## Problem And Outcome

The user-model loader deletes `impression.modeling` and its submodules from
`sys.modules`. Existing objects can then belong to old class definitions while a
newly loaded model imports replacements, breaking `isinstance`, dispatch, and
serialization. Reloading a user model must not reload installed Impression code.

## Scope

- Give each loaded user model a controlled module namespace and cleanup set.
- Retain canonical `impression` package/module objects across loads.
- Refresh changed user-model code and its owned local helper modules.
- Preserve preview isolation and repeat-load behavior.

Not in scope: a general Python plugin sandbox or process isolation redesign.

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

- `src/impression/cli.py`: model load, module tracking, cleanup, and finish path.
- `tests/test_preview_isolation.py`, CLI preview tests, and focused identity tests.

## Chosen Defaults / Parameters

- Canonical `impression` and third-party modules remain loaded.
- Model file and local imported helpers are owned by the current load and refresh next load.
- Failure cleanup removes only names introduced/owned by the failed load.

## Data Ownership

- Source of truth: module object/path plus the loader's owned-name set.
- Read ownership: CLI model loader.
- Write ownership: loader registers/removes only model-owned `sys.modules` entries.
- Derived/cache data: owned-name/path set is rebuilt per load.
- Privacy/logging constraints: errors may show model paths/tracebacks per current CLI behavior.

## Dependencies And Routes

- Domain/service dependencies: Python `importlib`; `sys.modules` registry.
- Database and GUI routes: none.
- Console route: preview/export model path -> loader -> model factory.
- Background/concurrency route: not applicable; load is synchronous.

## Prerequisite Handling

- Architecture feedback artifacts/status: none; not applicable for localized ownership correction.
- Already implemented prerequisites: model-path tracking/filter helpers.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: none.
- Progression handling: current item may proceed before surface consumer fixes.

## Application Integration

- App type: console.
- User/caller surface: preview and export commands loading a model file.
- Invocation route: command -> `_load_module` -> model factory -> scene result.
- Wiring owner/module: `src/impression/cli.py`.
- Observable result: edited user code reloads and returns canonical Impression objects.
- Integration validation: sequential CLI/model loads plus class identity assertions.
- Incomplete status risk: isolated helper tests can miss actual `sys.modules` cleanup order.

## Reuse And Extraction Plan

- Existing code to reuse: `_tracked_preview_module_paths` filtering and current module registration.
- Current reuse readiness: add ownership cleanup to existing CLI module.
- Extraction/wrapping/new reusable modules: none.
- Additions to existing library/modules: owned-name tracking around `_load_module`.
- One-off code justification: loader-local ownership is not a public API.

## Required DTOs / Functions / Components

- DTOs/models: set/map of model-owned module names and resolved paths.
- Functions/methods: `_load_module`, ownership tracker, prior-load cleanup.
- UI fields/elements/components: not applicable.

## Performance Contract

- Scan only newly imported/tracked module names; no filesystem tree walk.

## Error And State Behavior

- Syntax/import/runtime failures clean only load-owned entries and preserve canonical modules.
- A subsequent corrected load succeeds without process restart.

## Test Strategy

- Unit tests: ownership filtering and failure cleanup.
- Integrated route tests: sequential edited model/helper loads through CLI loader.
- Service/DB and GUI tests: not applicable.
- Production-data rule: temporary source trees only.

## Contract

Inputs are a model path and the already imported Impression runtime. Output is a
loaded user-model module whose Impression classes are object-identical to the
runtime's classes. Cleanup may remove only names owned by the prior user-model
load. A changed model/helper is re-executed on the next load.

## Acceptance Criteria

- Class identity from model output matches the caller's canonical imports.
- Two sequential model loads reflect edited user code without reloading Impression.
- Cleanup does not remove unrelated application or third-party modules.
- Existing dataclass, preview isolation, and error cleanup tests remain green.

## Verification

[Paired test specification](../test-specifications/fix-09-user-model-loader-module-identity-v1_0.md)

## Readiness Checklist

- [x] Ancestors, full score, carryover, canonical status, and terminal ledger are explicit.
- [x] Ownership, defaults, functions, reuse, error cleanup, and real console route are explicit.
- [x] No blocker, missing prerequisite, unresolved gap, or split coverage remains.
- [x] Temporary fixtures prove refresh/identity without production data.
