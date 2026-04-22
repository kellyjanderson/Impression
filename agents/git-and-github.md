# Git And GitHub Guidance

This document defines the repository branching and pull request rules for implementation work.

---

## Core Rule

No code changes should be made directly on `main`.

No implementation changes should be made without one of these durable planning anchors:

* an issue, or
* a specification

Implementation work must happen on one of these branch types:

* a bug-fix branch
* a feature branch

Documentation-only changes may follow the same rule when they are part of implementation work or PR preparation.

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

Feature branches must have an associated specification.

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

---

## Pull Requests

All branch work should be merged through a pull request.

For the current team shape:

* pull requests are still required
* formal review is not required
* anyone may merge the pull request

This keeps branch history, issue or spec linkage, and merge boundaries visible without adding unnecessary process overhead.

---

## Agent Expectations

Before making code changes, agents should ensure that:

1. the current branch is not `main`
2. the branch is clearly a bug-fix branch or a feature branch
3. the work is anchored by either an issue or a specification
4. bug-fix work links to an issue
5. feature work links to a specification
6. issue-driven work is back-referenced into the architecture/specification tree

If those conditions are not met, agents should stop and create or switch to the correct branching context before proceeding with code changes.

---

## Guiding Principle

Use branches and pull requests to keep implementation intent visible.

Issues define fixes.
Specifications define features.
Architecture/specification documents preserve durable project truth.
Pull requests define merge boundaries.
