# Fix 10: SurfaceBody Preview and Export Consumption (v1.0)

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `project/release-0.1.0a/architecture/surface-first-internal-model.md`
Source artifact: current scene-consumer source review
Split provenance: `none`
Canonical status: `Canonical`
Prerequisites:
- `fix-09-user-model-loader-module-identity-v1_0.md` - canonical classes must survive model loading for reliable `SurfaceBody` dispatch.

## Review Score

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; obsolete IWU metadata removed.
- Adversarial rescore basis: counted scene collection, surface tessellation, and CLI
  handoff; three payload models; preview/CLI/tessellation dependencies; two route outputs;
  GUI preview surface; two reused adapters; two module additions; tessellation cost; and
  one cross-route reusable collector. Fix 09 is linked and sequenced, not missing.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 23
- Split decision: remain whole after mandatory split review. The sole implementation
  artifact is the shared scene-consumer adapter; preview and export are separately
  failing integration proofs of that same boundary, not independent implementations.

## Source Field Carryover

- Source purpose: make direct surface-first model results usable by primary consumers.
- Source responsibilities by category:
  - Functions/methods: scene traversal, `tessellate_surface_body`, CLI handoff.
  - Data structures/models: `SurfaceBody`, `Mesh`, `Polyline`/group payloads.
  - Dependencies/services: preview collector, CLI, tessellation policy.
  - Returns/outputs/signals: preview dataset and export dataset.
  - UI surfaces/components: preview viewport.
  - Reusable code plan: existing surface tessellation and group traversal.
  - Performance-sensitive behavior: each surface tessellates exactly once per route.
  - Cross-screen reusable behavior: one collector serves preview and export.
  - Database, async, write, and security behavior: not applicable in this leaf.
- Source open questions / nuance discovered: preview/export policies remain distinct inputs.
- Source split/provenance notes: 23-point leaf retained because one shared adapter owns both proofs.

## Purpose

Connect canonical `SurfaceBody` results to the shared scene-consumer boundary used
by preview and export without hidden model-side mesh conversion.

## Problem And Outcome

The normal scene collector is typed and implemented around mesh/polyline data,
while current modeling APIs produce `SurfaceBody`. A model that returns a surface
body must preview and export through the documented CLI path without a model-side
manual tessellation workaround.

## Scope

- Recognize `SurfaceBody` in primary scene/result traversal.
- Tessellate once at the consumer boundary with preview or export policy.
- Preserve group ordering, transforms, and existing mesh/polyline support.
- Keep the adapter explicit; do not restore hidden mesh-first modeling fallbacks.

Not in scope: migration of every secondary development tool or new scene API.

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

- `src/impression/preview.py::_collect_datasets_from_scene` and its callers.
- `src/impression/cli.py` preview/export handoff.
- Existing surface tessellation consumer utilities and focused CLI tests.

## Chosen Defaults / Parameters

- Preview uses `preview_tessellation_request`; export uses `export_tessellation_request`.
- Traverse supported groups in stable authored order; apply transforms exactly once.
- Unsupported values raise a named consumer error; no hidden fallback.

## Data Ownership

- Source of truth: model result/scene payload and its transforms.
- Read ownership: shared scene collector reads supported payloads.
- Write ownership: collector creates derived datasets only; source objects stay immutable.
- Derived/cache data: datasets are recomputable from payload and route policy.
- Privacy/logging constraints: diagnostics name unsupported types, not model source.

## Dependencies And Routes

- Domain/service dependencies: preview scene collector; CLI consumer; surface tessellation.
- Database dependencies: none.
- GUI route: preview command -> collector with preview policy -> viewport.
- Console route: export command -> same collector with export policy -> downstream export gate.
- Background/concurrency route: not applicable inside collector.

## Prerequisite Handling

- Architecture feedback artifacts: none; surface-first architecture already defines consumer tessellation.
- Already implemented prerequisites: surface tessellation requests and group traversal.
- Missing prerequisite architecture/specifications: none.
- Unimplemented prerequisite specifications: Fix 09, linked above.
- Progression handling: implement Fix 09 before this item.

## Application Integration

- App type: mixed.
- User/caller surface: GUI preview viewport; console export command.
- Invocation route: model result -> shared collector -> route-specific tessellation -> viewport/export.
- Wiring owner/module: `src/impression/preview.py`, invoked by `src/impression/cli.py`.
- Observable result: rendered dataset and export-ready mesh collection.
- Integration validation: separate preview collection and CLI export route tests.
- Incomplete status risk: either route can be unwired even if the shared helper passes.

## Reuse And Extraction Plan

- Existing code to reuse: `tessellate_surface_body`; current recursive group traversal.
- Current reuse readiness: add surface handling/policy parameter to existing collector.
- Extraction/wrapping needed: route-policy argument/wrapper around shared collector.
- Additions to existing library/modules: preview collector and CLI handoff.
- New reusable modules to expose: none.
- One-off code justification: none.

## Required DTOs / Functions / Components

- DTOs/models: existing `SurfaceBody`, `Mesh`, and `Polyline` payloads.
- Functions/methods: `_collect_datasets_from_scene`, `tessellate_surface_body`, CLI handoff.
- UI components: existing preview viewport; no new fields/controls.

## Performance Contract

- Each surface is tessellated once per consumer call; traversal is O(payload nodes + output cells).

## Error And State Behavior

- Empty/unsupported results produce named route errors; no partial hidden fallback.
- Preview/export policy selection is deterministic and route-local.

## Test Strategy

- Unit tests: payload traversal, policy selection, ordering, transforms, unsupported types.
- GUI/controller tests: noninteractive preview collection to viewport payload.
- Integrated route tests: preview and console export separately.
- Service/DB tests: not applicable; temporary model fixtures only.

## Contract

Input is a model result/scene containing `SurfaceBody`, mesh, polyline, or supported
groups. Output is ordered render/export data; surfaces use the policy appropriate
to the consuming mode and transforms are applied exactly once. Unsupported values
produce a specific diagnostic.

## Acceptance Criteria

- A model returning one `SurfaceBody` previews successfully.
- The same model exports STL without model-authored tessellation code.
- Preview and export choose their respective tessellation policies.
- Mixed supported groups preserve order and existing mesh behavior.

## Verification

[Paired test specification](../test-specifications/fix-10-surfacebody-preview-export-consumption-v1_0.md)

## Readiness Checklist

- [x] Ancestors, full score, carryover, canonical status, prerequisite, and ledger are explicit.
- [x] The 23-point split review documents one shared adapter and separate route proofs.
- [x] Mixed app routes, ownership, defaults, reuse, functions/models, performance, and errors are explicit.
- [x] No blocker, missing prerequisite artifact, unresolved gap, or split coverage remains.
- [x] Preview and export integration tests avoid production data.
