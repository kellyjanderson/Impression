# Reference Review Spec 23: QML Launcher/Bootstrap (v1.0)

## Overview

Create the launcher and QML shell bootstrap for the Reference Review
Workbench.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `PySide QML Shell Bootstrap`.
- Manifest score: 13

## Scope

This specification covers:

- application launch entrypoint
- QML engine setup
- top-level workbench window construction
- startup diagnostics

## Behavior

This leaf must define:

- start the workbench without loading a fixture on the UI thread
- create the QML engine and load the shell chrome
- report startup failures as shell diagnostics

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

- the QML Launcher/Bootstrap boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
