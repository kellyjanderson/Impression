# Reference Review Spec 63: Preview Display Icon Asset Packaging (v1.0)

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, research, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one packaging leaf for generated preview display-control SVG assets.

## Overview

Package the generated preview display-control SVG files as durable UI resources.

## Backlink

- [Architecture: Reference Review Preview Display Controls Product Definition](../architecture/reference-review-preview-display-controls-product-definition.md)

## Scope

This specification covers:

- adding the generated SVG icon files to the QML resource tree
- including those files in package data
- validating that packaged files exist and parse as SVG

This specification does not cover icon metadata, button behavior, or toolbar composition.

## Implementation Boundary

Owner/module:

- `src/impression/devtools/reference_review/ui/qml/icons/preview-display/`
- `pyproject.toml`
- `src/impression/devtools/reference_review/ui/packaging.py`

Reuse/extraction decision:

- Add to the existing UI packaging/resource policy.

## Behavior

The implementation must:

- include exactly the generated SVG assets needed by preview display controls
- keep SVG assets under a stable preview-display resource folder
- include the SVG glob in setuptools package data
- avoid putting rendering behavior in asset files or packaging helpers

## Verification

Test strategy:

- package resource tests assert every SVG exists
- XML parse tests assert each asset is SVG with `viewBox="0 0 24 24"`
- package-data tests assert the SVG glob is present

## Manifest Assessment

Score:

- Functions/methods: 1 x 2 = 2
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 0 x 2 = 0
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 1 x 1 = 1
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 1 x 2 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 11.5

Split decision:

- No split needed. One static packaging boundary.

## Refinement Status

Final leaf.

## Acceptance

This specification is complete when SVG assets are packaged and resource validation passes.
