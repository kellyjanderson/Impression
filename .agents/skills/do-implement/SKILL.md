---
name: do-implement
description: Use when the user asks Codex to execute implementation work from a progression document by finding the relevant progression file, selecting unchecked items as the body of work, following the linked specification and test specification, delivering the work through a feature branch and pull request, and updating progression state after verification.
---

# Do Implementation

Use this skill when the user asks to implement from a progression document.

When you finish implementing a spec, continue to the next one in the progression.

## Progression Source

Use the progression document identified by the user or by the current project context.

Do not hard-code the shortest `project/planning/progression.md` path. Workspaces may contain nested `project/` folders, nested planning areas, release folders, or project-local progression documents where the relevant progression is deeper than a top-level fallback.

Selection order:

1. If the user names or links a progression document, use that exact file.
2. If the active file, current task, or named architecture/specification/test-specification belongs to a nested project area, find the nearest related `progression.md` in that same area.
3. Otherwise, search the workspace for progression documents and choose the one whose directory and linked items match the requested body of work.
4. Ask the user only if multiple plausible progression documents remain.

Prefer the semantically relevant progression over the shorter path.

If no relevant progression document exists, do not invent implementation work. Create or request a planning progression anchor first.

## Body Of Work

Treat all unchecked implementation items in the progression document as the body of work for this prompt.

Follow the progression document in order. It is the execution punch list and progress indicator.

Implementation work should be anchored to final leaf specifications and paired test specifications. If a progression item links to a specification or test specification, read the linked document before editing code.

Before coding, verify the linked specification is an active canonical leaf. If
the item links to a parent, umbrella, superseded, archived, or incomplete split
spec, do not implement from it. Follow the canonical children when they exist,
or complete the split/canonicalization work before coding.

## Branch And Delivery Setup

Before editing code:

1. Identify the working branch that implementation should merge back into.
2. If the current branch is a non-main working branch, use it as the PR base.
3. If the current branch is `main`, `master`, detached, or otherwise not clearly a working branch, ask the user which working branch to use before continuing to merge behavior.
4. Create a feature branch from the working branch for the selected spec or coherent spec slice.

Use a clear feature branch name derived from the spec or feature area.

Do not implement directly on `main`, `master`, or the working branch.

## Execution Loop

For each unchecked implementation item:

1. Read the linked final leaf specification.
2. Read the paired test specification when present.
3. Verify the specification is canonical and not an incomplete parent split.
4. Inspect the existing code and tests needed to implement the specification, including existing abstractions, repeated logic, nearby helpers, and local architectural patterns that the implementation should reuse or repair.
5. Check whether the item depends on prerequisite behavior, architecture, or specs that are missing or unimplemented.
6. Implement the requested behavior according to the linked specification while preserving or improving the local code structure touched by the work.
7. Add or update focused tests required by the paired test specification.
8. Run the smallest meaningful verification command for the item.
9. Mark the item complete only when implementation and relevant verification are complete.

If the progression separates implementation and verification items, keep them separate:

* check the implementation item only when the feature behavior is implemented
* check the verification item only when the required tests or manual checks have passed

## Pull Request Delivery

When the selected implementation body is complete:

1. Commit the implementation, tests, and progression updates on the feature branch.
2. Push the feature branch to its remote tracking branch.
3. Open a pull request from the feature branch into the working branch.
4. Wait for the pull request to become stable, then merge it into the working branch after required checks, review expectations, and repository policy gates are satisfied.

If no working branch was identified before implementation, do not auto-merge into `main` or `master`. Ask the user for direction.

Do not merge unrelated work. If the branch contains changes outside the selected progression/spec body, stop and ask for direction before opening or merging the PR.

## Completion Discipline

Progression checkboxes are truth markers.

Never check an item when work is incomplete, intentionally deferred, stubbed, partially implemented, blocked, skipped, or knowingly broken.

If an item was already checked and you discover it is incomplete, uncheck it and append a short note naming the blocker.

If an unchecked item cannot be completed in the current pass, leave it unchecked and add a short note explaining what blocked it.

When the reason is a prerequisite gap, use `Missing prerequisite` status rather
than `blocked`.

## End Of Prompt

Before finishing:

* ensure every completed item has its checkbox updated
* ensure every incomplete item remains unchecked
* summarize completed items, blocked items, files changed, and verification run
* do not claim the whole body of work is complete unless every unchecked item from the start of the prompt has been completed or explicitly removed from scope by the user

## SkillsKeeper Directives

<!-- skillskeeper-directive: app-integration-completion-discipline -->
### App Integration Completion Discipline

## App Integration Completion Discipline

Progression checkboxes represent user-accessible completion, not local code existence. Do not check off a progression item unless the functionality is wired into the intended app surface, command, API, or workflow and validated through that route.

Use explicit status language when the work is short of complete:

- `Implemented in isolation; not complete.`
- `Wired; awaiting integration validation.`
- `Complete; reachable and validated.`

The final implementation note must name any selected specs or progression items that remain unwired, hidden behind unused code, or validated only at unit/helper level.
<!-- /skillskeeper-directive: app-integration-completion-discipline -->

<!-- skillskeeper-directive: app-type-completion-refusal -->
### App-Type Completion Refusal

## App-Type Completion Refusal

Do not mark a selected progression item complete when the app-type-specific route is missing or unvalidated.

Refuse completion when:

- GUI work lacks the visible control/event, visible state coverage, UI-thread handoff where applicable, or GUI route validation.
- Console work lacks the command/subcommand, flags/args/stdin/config behavior, stdout/stderr/exit-code contract, side-effect proof, or CLI validation.
- API/service work lacks the endpoint/caller contract, auth/permission/error behavior, side-effect proof, observability, or route validation.
- Mixed app work validates only one surface while another independently failing surface remains unwired or unvalidated.
- Library-only work does not name and validate the consuming module or downstream caller.
- Only helper/unit tests passed while the intended user/caller route remains designed, isolated, or merely wired.

When refusing completion, leave the checkbox unchecked and name the missing route or validation in the implementation note.
<!-- /skillskeeper-directive: app-type-completion-refusal -->

<!-- skillskeeper-directive: progression-template-awareness -->
### Progression Template Awareness

## Progression Template Awareness

When a progression document omits route wiring or route validation for feature-bearing work, compare it with the selected `progression` template from the process registry before checking items complete.

Do not silently treat a collapsed helper implementation checkbox as full product completion. If the route wiring or validation task is missing, leave the item unchecked or add explicit unchecked follow-up tasks that match the template structure.
<!-- /skillskeeper-directive: progression-template-awareness -->

<!-- skillskeeper-directive: canonical-spec-implementation -->
### Canonical Spec Implementation

## Canonical Spec Implementation

Do not implement directly from parent or umbrella specs after split work has
begun. If a progression item points at a non-canonical spec:

- find the canonical child specs and update the progression when unambiguous;
- leave the item unchecked when split coverage is incomplete;
- complete split coverage, parent archival, and index/progression cleanup before
  claiming implementation readiness.

Completed source notes in architecture docs are not implementation anchors.
Once canonical specs exist, active implementation should refer to the specs,
not the source notes.
<!-- /skillskeeper-directive: canonical-spec-implementation -->

<!-- skillskeeper-directive: implementation-stewardship -->
### Implementation Stewardship

When implementation work exposes bad code in the path of the selected specification, treat reasonable cleanup as part of implementing the specification, not as optional polish.

This includes duplicated inline logic, bypassed shared abstractions, inconsistent local patterns, dead-end helper creation, missing integration into existing extension points, and code that would make the new behavior harder to maintain or validate.

Before adding new helpers, routes, adapters, or one-off code paths:

- search for existing call sites and nearby equivalents;
- identify the abstraction or pattern the codebase already expects;
- update existing call sites when centralization is the correct fix;
- keep behavior changes covered by focused tests.

Do not create a separate helper while leaving equivalent inline logic scattered through the codebase. Do not preserve obviously broken local structure merely because the specification names only the new behavior.

If the cleanup is materially larger than the selected specification scope, risky without broader review, or likely to change unrelated behavior, document it as a `codeimprovement` issue using the `coding` skill's Code Improvement Issues process. Leave the progression item unchecked when the implementation depends on that cleanup, or add an explicit unchecked follow-up task that links to the code improvement issue.
<!-- /skillskeeper-directive: implementation-stewardship -->

<!-- skillskeeper-directive: prerequisite-discovery-during-implementation -->
### Prerequisite Discovery During Implementation

During implementation, if the selected item requires behavior, structure, or
contracts that are not ready, classify the gap before coding around it:

- If the prerequisite is not architected, create or update an ACD that defines
  the necessary target implementation. Leave the current progression item
  unchecked with `Status: Missing prerequisite - <ACD path>`. Do not mark it
  blocked and do not check it complete.
- If architecture already fully defines the prerequisite but no final
  specification exists, write or update the prerequisite specification and link
  it from the current item. Leave the current progression item unchecked with
  `Status: Missing prerequisite - <spec path>`. Use the same progression
  handling as missing architecture.
- If the prerequisite specification exists but is not implemented, pause the
  current item, ensure the prerequisite appears before it in progression, and
  implement the prerequisite first. Return to the paused item only after the
  prerequisite implementation and verification are complete.

When adding or updating prerequisite artifacts, also update the relevant
specification `Prerequisites` field when that document exists.
Do not satisfy missing prerequisites with local stubs, speculative code, or
unchecked assumptions.
<!-- /skillskeeper-directive: prerequisite-discovery-during-implementation -->

<!-- skillskeeper-directive: pull-request-stability-before-merge -->
### Pull request stability before merge

Before merging any pull request, require the PR itself to be stable. Treat a PR
as unstable when any part of the PR is still in process or non-terminal,
including required or optional checks, CI workflows, status contexts,
mergeability calculation, branch update checks, merge queue state, deployment
statuses, requested-review gates, or repository policy evaluation.

If GitHub reports any PR state as pending, queued, in progress, expected,
waiting, blocked, unknown, stale, unstable, or otherwise non-terminal, wait
before merging. Use the available GitHub surface, such as `gh pr checks
--watch`, `gh run watch`, `gh pr view`, or equivalent API polling, to observe
the PR until it reaches a stable terminal state.

Merge only after the PR is stable: required repository gates have passed, no
visible in-process PR check or status remains, mergeability is settled, and
review or policy expectations are satisfied. If any PR check, status, queue,
review, or policy state fails, is cancelled, remains blocked, or cannot reach a
stable result, report the result and do not merge unless the user explicitly
instructs an override that is allowed by repository policy.
<!-- /skillskeeper-directive: pull-request-stability-before-merge -->
