# Reference Review Spec 79: Review Workflow Persistence Smoke (v1.0)

Status: Split parent - superseded by child specs 79a, 79b, and 79c

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own
verification surface.
Draft count: 1 IWU.
Basis: one review workflow persistence smoke leaf. This count is a draft
creation estimate and must be verified by `review specs`.

## Overview

Confirm the stabilization work preserves the Reference Review app's core review
workflow: notes, approve, decline, unreviewed status, status badge, and
show-approved filtering.

## Backlink

- [ACD: Reference Review Hybrid Stabilization](../architecture/acd-reference-review-hybrid-stabilization.md)
- [Reference Review Promotion And Notes Lifecycle](../architecture/reference-review-promotion-and-notes-lifecycle.md)
- [Reference Review Hybrid Stabilization Plan](../planning/reference-review-hybrid-stabilization-plan.md)

## Source Manifest

- Source candidate: `Review Workflow Persistence Smoke`
- Source artifact: `project/release-0.1.0a/architecture/acd-reference-review-hybrid-stabilization.md`

## Scope

This specification covers:

- selected fixture notes load
- real-time notes persistence
- approve status persistence and dirty-to-gold artifact move
- decline status persistence without artifact move
- unreviewed fixture visibility
- status badge state
- show-approved filtering behavior

## Responsibilities

- Functions/methods:
  - selected fixture notes load route
  - notes save route
  - approve action route
  - decline action route
  - show-approved filter route
  - status badge state update route
- Data structures/models:
  - fixture review status
  - selected fixture notes
  - fixture list filter state
- Dependencies/services:
  - fixture record file or database store
  - durable write lane
  - artifact promotion service
- Returns/outputs/signals:
  - notes save completion
  - approve/decline completion
  - filtered fixture list update
  - visible status badge state
- UI surfaces/components:
  - fixture list
  - notes tab/panel
  - approve/decline controls
  - status badge
  - show-approved checkbox
- UI fields/elements:
  - notes text
  - status label/badge
  - checkbox checked state
  - fixture row visibility
- Reusable code plan:
  - Existing code reused as-is:
    - current promotion and notes lifecycle modules
  - Additions to existing reusable library/module:
    - optional durable write helper imported from kit if Spec 76 proves safe
  - New reusable library/module to create:
    - none expected
- Database queries/tables/migrations:
  - none expected; existing file/db fixture persistence route is used
- Async/concurrency behavior:
  - writes are serialized
  - stale selected fixture completions cannot update the wrong visible fixture
- Destructive/write behavior:
  - approve can move artifacts from dirty to gold
  - notes and status write to fixture record or database
- Security/privacy-sensitive behavior:
  - notes contents stay in the fixture persistence path; no new external sink
- Performance-sensitive behavior:
  - notes writes must not block the UI thread
- Cross-screen reusable behavior:
  - fixture status drives list, badge, and filter behavior

## Implementation Boundary

Owner/module:

- Reference Review lifecycle modules for notes, status, and promotion
- `src/impression/devtools/reference_review/ui/shell.py`
- relevant fixture store modules and tests

Routes:

- notes edit/save
- approve action
- decline action
- fixture filter toggle
- selected fixture change

Reuse/extraction decision:

- keep review persistence and promotion policy local
- use shared durable write helper only if import-safe and API-compatible

## Data And Defaults

Chosen defaults / parameters:

- approved fixtures are hidden by default unless show-approved is checked
- declined and unreviewed fixtures remain visible
- decline does not move dirty artifacts
- approve moves dirty artifact paths to matching gold paths

Data ownership:

- fixture store owns persisted notes and status
- promotion service owns artifact movement
- shell owns visible selected fixture state

Open questions and resolved assumptions:

- resolved: status has exactly approved, declined, and unreviewed states for
  this stabilization pass

Implementation prerequisites:

- Spec 77 task handoff for UI safety

## Behavior

The implementation must:

- load notes when selected fixture changes;
- save notes in real time to the selected fixture record or database;
- avoid applying stale notes completions to a newer selected fixture;
- approve a fixture by persisting approved status and moving dirty artifacts to
  gold with matching folder structure;
- decline a fixture by persisting declined status without moving artifacts;
- keep unreviewed fixtures visible;
- hide approved fixtures when show-approved is unchecked and show them when it
  is checked;
- show a wide status badge whose state matches the selected fixture.

## Verification

Test strategy:

- focused fixture-store tests for notes/status persistence;
- focused promotion tests for approve artifact movement;
- focused decline test proving artifacts are not moved;
- focused UI/shell model test for show-approved filtering and badge state.

Additional verification requirements:

- manual smoke on a real fixture file
- run `git diff --check`

## Readiness Fields

App type:

- GUI route with durable file side effects

User/caller surface:

- notes panel, approve/decline controls, fixture list, status badge

Invocation route:

- text edit, button click, checkbox toggle, fixture selection

Wiring owner/module:

- Reference Review shell and lifecycle modules

Observable result:

- review notes and statuses persist and are reflected in the visible fixture
  list and selected fixture badge

Integration validation:

- persistence tests, shell model tests, and manual GUI smoke

Readiness blockers:

- depends on Spec 77 for non-blocking durable write handoff

## Review Score

- Fresh review score: 61
- IWU recount: 3 IWU
- Split decision: split required. Notes persistence, approve/decline
  promotion, and visible status projection are independently failing routes.

## Refinement Status

Split parent. Do not implement directly.

## Child Specifications

- [Reference Review Spec 79a: Notes Persistence Route](reference-review-79a-notes-persistence-route-v1_0.md)
- [Reference Review Spec 79b: Approve/Decline Persistence And Promotion Route](reference-review-79b-approve-decline-persistence-and-promotion-route-v1_0.md)
- [Reference Review Spec 79c: Status Badge And Approved Filter Route](reference-review-79c-status-badge-and-approved-filter-route-v1_0.md)

## Split Coverage

| Parent responsibility | Child owner | Status |
| --- | --- | --- |
| selected fixture notes load | Spec 79a | Covered |
| real-time notes persistence | Spec 79a | Covered |
| approve status persistence and dirty-to-gold artifact move | Spec 79b | Covered |
| decline status persistence without artifact move | Spec 79b | Covered |
| unreviewed fixture visibility | Spec 79c | Covered |
| status badge state | Spec 79c | Covered |
| show-approved filtering behavior | Spec 79c | Covered |

## Acceptance

This specification is complete when the stabilized app still supports the core
review workflow through persistent notes, approve/decline state, status badge,
and show-approved filtering.
