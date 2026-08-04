# Fix 10: SurfaceBody Preview and Export Consumption (v1.0)

Date: 2026-08-04
Status: Final

## Work Units

Count: 1 IWU.

### IWU 1 — Connect SurfaceBody results to primary preview and export

- Input: direct and grouped model results containing `SurfaceBody` and legacy payloads.
- Work: recognize surfaces during traversal and tessellate exactly once at the
  consumer boundary using the selected preview or export policy.
- Output: ordered preview/export data with transforms applied once.
- Complete when: direct and mixed results preview/export without model-side adapters.

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

## Implementation Routing

- `src/impression/preview.py::_collect_datasets_from_scene` and its callers.
- `src/impression/cli.py` preview/export handoff.
- Existing surface tessellation consumer utilities and focused CLI tests.

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
