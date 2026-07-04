# Reference Review Spec 41: Workbench UI Dependency And Packaging Policy (v1.0)

## Overview

Keep UI dependencies optional and packageable while avoiding accidental core
Impression dependency leaks.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Workbench UI Dependency And Packaging Policy`.
- Manifest score: 21.5

## Scope

This specification covers:

- optional extra declaration checker
- QML resource layout verifier
- dependency policy record
- package resource manifest
- dependency policy report
- packaging smoke result

## Behavior

This leaf must define:

- workbench lives behind optional extra; WebEngine remains optional
- package metadata to import checks to smoke result

## Constraints

- Destructive/write behavior: packaging smoke writes build artifacts only
- Security/privacy-sensitive behavior: optional extras must not expose
  sandbox bypasses
- Performance-sensitive behavior: package smoke remains bounded
- Cross-screen reusable behavior: dependency policy protects all workbench
  UI modules

## Dependencies And Reuse

Dependencies/services:

- PySide6 deployment tooling
- import boundary checks

Reusable code plan:

- Existing code reused as-is: packaging metadata
- Additions to existing reusable library/module: devtool dependency checks
- New reusable library/module to create: none

Implementation owner/module:

- future packaging/devtool checks

## Data Ownership And Routes

Data ownership:

- packaging policy owns optional dependency boundaries

Routes:

- package metadata to import checks to smoke result

## UI Contract

- none

## Test Strategy

- import-boundary check and package smoke test

## Open Questions And Prerequisites

Open questions / nuance discovered:

- exact deployment tool can be selected later

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Workbench UI Dependency And Packaging Policy boundary is implemented
  as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
