# Reference Review Spec 24: Bridge Registration (v1.0)

## Overview

Register workbench bridge objects for QML without implementing individual
panels.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `PySide QML Shell Bootstrap`.
- Manifest score: 13

## Scope

This specification covers:

- Python-to-QML bridge object registration
- bridge naming contract
- bridge availability diagnostics

## Behavior

This leaf must define:

- register only allowlisted bridge objects
- make missing bridge objects visible in launch tests
- avoid exposing filesystem or promotion authority directly to QML

## Constraints

- Async/concurrency behavior: shell submits work through dispatcher only
- Security/privacy-sensitive behavior: no unrestricted file controls in
  shell
- Performance-sensitive behavior: startup avoids model loading on UI thread
- Cross-screen reusable behavior: shell hosts every panel

## Dependencies And Reuse

Dependencies/services:

- PySide6
- async dispatcher protocol

Reusable code plan:

- Existing code reused as-is: none
- Additions to existing reusable library/module: none
- New reusable library/module to create: workbench UI package

Implementation owner/module:

- future `src/impression/devtools/reference_review/ui/shell`

## Data Ownership And Routes

Data ownership:

- UI shell owns visible application state only

Routes:

- launcher to shell to QML engine

## UI Contract

- Surface/component: main workbench window
- Field/element: top-level split layout container

## Test Strategy

- launch smoke test and QML bridge registration test

## Open Questions And Prerequisites

Open questions / nuance discovered:

- none

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Bridge Registration boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
