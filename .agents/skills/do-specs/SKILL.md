---
name: do-specs
description: Use when the user says "do specs", "do-specs", or asks to create implementation specifications. Creates draft implementation specs from architecture, ACDs, parent specs, issues, or supplied notes, creates paired test specs when required, then stops with a handoff to review-specs; it must not review or certify its own specs.
---

# Do Specs

Use this skill as the creation action for draft implementation specs.

## Required Companion Skills

Immediately load and follow these skills:

- `test-specifications-core` when the project workflow requires paired test
  specifications for final feature leaves.
- `architecture` and `architectural-change-document` when the source material
  exposes missing, stale, or aspirational architecture that must be captured
  before the spec can be a faithful handoff.

If this skill conflicts with a companion skill, the companion skill with the
more specific rule is authoritative.

Do not load or execute `review-specs` or `specification-sizing` as a review
pass during this same action unless the user separately asks for that review
after spec creation completes.

## Workflow

1. Locate the source architecture, ACD, parent spec, issue, or user-supplied
   notes in scope.
2. Load the `implementation-spec` template from the selected process registry.
3. If required architecture is missing or stale, create or update the
   architecture feedback artifact before hiding that gap inside the spec:
   - once work has moved past the architecting phase into spec creation, do not
     update canonical architecture directly;
   - treat architecture files as sacrosanct after specs start;
   - create or update an ACD for every missing, stale, desired, transitional,
     uncertain, or not-yet-implemented architecture change discovered during
     `do specs`;
   - leave canonical architecture reconciliation until implementation is
     complete and the reconciliation process is explicitly run.
4. Create or update implementation specs from the selected architecture, ACD,
   parent spec, issue, or user-supplied source material. Preserve source
   responsibilities by mapping them into durable spec sections or explicit
   source provenance/history.
5. Link any architecture feedback artifact in the spec's ancestry,
   prerequisites, source field carryover, or readiness fields as appropriate.
6. Create paired test specifications when required by the project workflow.
7. Stop after writing the spec artifacts. Do not review, refine, calculate a
   Review Score as a check, or certify readiness in the same action.

## Reporting

Report the source artifacts used, ACD feedback artifacts created or updated,
created or updated specs, created or updated test specs, and the
exact follow-up command: `review specs`.
