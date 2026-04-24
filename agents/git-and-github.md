# Git And GitHub Guidance

This document defines the repository branching and pull request rules for implementation work.

---

## Core Rule

No code changes should be made directly on `main`.

No implementation changes should be made without one of these durable planning anchors:

* an issue
* a specification
* an ad hoc work document

Implementation work must happen on one of these branch types:

* a bug-fix branch
* a feature branch

Documentation-only changes may follow the same rule when they are part of implementation work or PR preparation.

Before implementation begins, the path should be explicit:

* feature path
* ad hoc path

If the user has not specified which path to use, agents should ask a short
disambiguating question before proceeding with implementation.

---

## Workspace Repository Policy

For this workspace, active work should happen on a feature branch.

The working rule is:

* do not work directly on `main`
* create or switch to a feature branch before making changes
* keep that feature branch focused on one named unit of work

If a bug fix is small enough that the team would otherwise use a bug-fix branch,
this workspace still prefers a feature branch unless the user explicitly asks
for a different branch shape.

---

## Bug-Fix Branches

Use a bug-fix branch when the work is correcting broken or incorrect behavior.

Bug-fix branches must have an associated issue.

That issue is the durable record of:

* the defect being fixed
* the intended scope of the fix
* any relevant reproduction or acceptance notes

Recommended naming pattern:

```text
bugfix/<issue-id>-<short-description>
```

Example:

```text
bugfix/42-fix-for-pronunciation
```

Bug-fix work must also be back-referenced into the architecture/specification tree.

That means the issue-driven fix must update at least one of these, as appropriate:

* the affected architecture document
* the affected specification document
* a new specification if the bug exposed missing durable behavior definition

Issue work must not remain only in GitHub issue text and code history.

---

## Feature Branches

Use a feature branch when the work is implementing a new capability, significant enhancement, or planned structural addition.

Feature-path work on a feature branch must have an associated specification.

That specification is the durable record of:

* the feature scope
* the implementation boundary
* the expected behavior

Recommended naming pattern:

```text
feature/<short-spec-slug>
```

Example:

```text
feature/voice-session-realization
```

## Ad Hoc Work On Feature Branches

This workspace still prefers feature branches for active implementation work,
including ad hoc work.

When the work is on the ad hoc path rather than the feature path, the feature
branch should be anchored by an ad hoc work document under:

```text
project/adhoc/
```

That document is the durable record of:

* the bounded unit of work
* why it is being done on the ad hoc path
* the intended scope
* expected verification

---

## Pull Requests

All branch work should be merged through a pull request.

For the current team shape:

* pull requests are still required
* formal review is not required
* anyone may merge the pull request

This keeps branch history, issue or spec linkage, and merge boundaries visible without adding unnecessary process overhead.

---

## Commit Cadence

Commit frequently enough to protect meaningful progress.

The goal is not:

* hundreds of tiny noise commits

The goal is also not:

* waiting so long that significant work exists only in the working tree

Preferred cadence:

* commit on reasonable units of work
* commit when a meaningful sub-part is stable
* commit earlier when the user explicitly asks for it

---

## Push Cadence

Push when a reasonable unit of work has been completed.

Push earlier when the user explicitly asks for it.

Agents should not leave completed work stranded only in local commits longer
than necessary.

---

## Pull Request Cadence

When asked to create a pull request, agents should:

1. ensure the intended unit of work is committed
2. push the branch
3. create the pull request

Agents should not merge the pull request unless the user explicitly asks for
that merge.

---

## Merge Cadence

Merge pull requests when the user explicitly asks for the merge.

Merging a pull request is the completion of a feature branch for that unit of
work.

Until the pull request is merged, the work should be treated as branch work in
progress, not finished repository integration.

---

## Delivery Meaning

To deliver a named group of work means to follow the repository policy to
completion for that unit of work.

That means:

1. perform the work on a feature branch
2. commit it in reasonable units
3. push the branch when the unit is ready
4. create the pull request when asked
5. merge the pull request when asked

Agents should not describe work as delivered if it exists only as uncommitted
changes, only as local commits, or only as an open unmerged pull request.

---

## Agent Expectations

Before making code changes, agents should ensure that:

1. the current branch is not `main`
2. the branch is clearly a feature branch for this workspace unless the user explicitly directs otherwise
3. the work is anchored by an issue, a specification, or an ad hoc work document
4. bug-fix work links to an issue
5. feature-path work links to a specification
6. ad-hoc-path work links to an ad hoc work document
7. issue-driven work is back-referenced into the architecture/specification tree when appropriate

If those conditions are not met, agents should stop and create or switch to the correct branching context before proceeding with code changes.

When the user asks to "deliver" work, agents should also ensure that:

8. the intended change set is committed in reasonable units
9. the branch is pushed
10. a pull request is created when requested
11. the pull request is merged when requested

---

## Guiding Principle

Use branches and pull requests to keep implementation intent visible.

Issues define fixes.
Specifications define feature-path work.
Ad hoc documents define bounded ad-hoc-path work.
Architecture/specification/adhoc documents preserve durable project truth.
Pull requests define merge boundaries.
