# Reference Review Spec 36: Candidate List/Adoption UI (v1.0)

## Overview

Implement the Codex candidate list and human adoption UI.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Codex Sidecar Panel`.
- Manifest score: 14

## Scope

This specification covers:

- candidate list view model
- candidate path display
- regenerate action
- adopt action

## Behavior

This leaf must define:

- show only broker-approved candidate paths
- require human adoption before candidates affect review source
- route adoption requests through the broker

## Constraints

- Async/concurrency behavior: streams are cancellable and stale-guarded per
  fixture
- Destructive/write behavior: UI never writes candidate files directly
- Security/privacy-sensitive behavior: UI displays broker refusals and
  cannot bypass policy
- Performance-sensitive behavior: streamed text updates are throttled
- Cross-screen reusable behavior: sidecar uses selected fixture and updates
  candidate/adoption state

## Dependencies And Reuse

Dependencies/services:

- Codex sidecar broker
- async dispatcher

Reusable code plan:

- Existing code reused as-is: shared components
- Additions to existing reusable library/module: none
- New reusable library/module to create: none

Implementation owner/module:

- future `ui/panels/codex_sidecar`

## Data Ownership And Routes

Data ownership:

- broker owns authority; UI owns visible stream and actions

Routes:

- selected fixture to sidecar panel to broker request

## UI Contract

- Surface/component: Codex panel
- Surface/component: candidate list
- Surface/component: refusal banner
- Field/element: chat input, response stream, candidate path, regenerate,
  adopt, refusal

## Test Strategy

- stream, cancellation, refusal, candidate selection, and adopt action tests

## Open Questions And Prerequisites

Open questions / nuance discovered:

- none

Prerequisites before implementation:

- Codex broker protocol must exist

## Refinement Status

Final, with prerequisite dependencies listed below before implementation starts.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Candidate List/Adoption UI boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
