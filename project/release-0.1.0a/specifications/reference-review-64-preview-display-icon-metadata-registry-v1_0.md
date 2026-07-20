# Reference Review Spec 64: Preview Display Icon Metadata Registry (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one metadata registry for icon ids, resource paths, and labels.

## Overview

Expose preview display-control icon metadata through a reusable registry.

## Backlink

- [Reference Review Spec 63: Preview Display Icon Asset Packaging](reference-review-63-preview-display-icon-asset-packaging-v1_0.md)
- [Architecture: Reference Review Preview Display Controls Product Definition](../architecture/reference-review-preview-display-controls-product-definition.md)

## Scope

This specification covers:

- stable icon ids
- resource path lookup
- tooltip and accessible-name defaults

This specification does not cover SVG packaging validation or button state.

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/packaging.py` or adjacent UI resource module

Reuse/extraction decision:

- Add a small exported registry to the existing UI resource layer.

## Behavior

The implementation must:

- expose one record per preview display-control icon
- include stable ids for all controls described in the product definition
- provide resource paths without consumers hard-coding filesystem layout
- keep metadata independent from renderer state

## Verification

Test strategy:

- registry tests assert all required ids exist
- label tests assert every record has tooltip and accessible-name text
- path tests assert every registry path points to a packaged SVG

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Split decision:

- No split needed. Cohesive metadata lookup leaf; score is in the split-review band because it touches labels, ids, and paths, but all belong to one registry.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when controls can resolve icon metadata by stable id.
