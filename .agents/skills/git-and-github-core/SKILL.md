---
name: git-and-github-core
description: Follow shared Git and GitHub process rules for branching, durable planning anchors, issue or specification linkage, and delivery cadence.
---

# Git And GitHub Core

Use this Skill for shared repository process around branches, anchors, commits, pushes, and pull requests.

## Core Rules

* do not make implementation changes directly on `main`
* do not begin implementation without a durable planning anchor
* use a named branch for implementation work
* keep GitHub issue or specification linkage durable rather than leaving behavior defined only in issues, PRs, or code history

## Durable Planning Anchors

Implementation work should be anchored by one of:

* an issue for bug-fix work
* a specification for feature work
* another durable planning artifact explicitly allowed by the workspace overlay

Workspace overlays may narrow the allowed anchor set or add workspace-specific path rules.

## Durable Back-Reference Rule

Issue-driven fixes must be back-referenced into the durable architecture or specification tree when appropriate.

Bug-fix work should not live only in:

* issue text
* pull request text
* branch history
* code changes

## Commit And Push Cadence

* commit on meaningful, stable units of work
* do not let important progress live only in the working tree
* push when a reasonable unit of work is complete or when explicitly asked
* when a feature branch is done, push the completed branch to its remote tracking branch
  before treating the branch work as complete

## Pull Requests And Delivery

When asked to create a pull request:

1. ensure the intended unit of work is committed
2. push the branch
3. create the pull request

Do not merge unless explicitly asked.

Do not describe work as delivered if it exists only as:

* uncommitted changes
* local commits
* an open unmerged pull request
