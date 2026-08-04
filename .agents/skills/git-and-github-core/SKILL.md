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

When a merge is explicitly requested, do not merge while the pull request is
unstable. Wait for the PR to become stable before merging.

Do not describe work as delivered if it exists only as:

* uncommitted changes
* local commits
* an open unmerged pull request

## SkillsKeeper Directives

<!-- skillskeeper-directive: github-planning-releases-and-change-artifacts -->
### GitHub planning releases and change artifacts

Use a `p` suffix, such as `2.0.0p` or `v2.0.0p`, only as a GitHub/repository planning-release convention. A planning release gives release-level planning artifacts a clean place in the repository and, for public projects, may be used to ask for feedback before implementation release work begins. Planning releases may contain release definitions, architecture documents, planning documents, ACDs, specifications, test specifications, research, and other durable process artifacts for the target version.

Do not treat a `p` suffix release as a Python package release or implementation delivery release. Do not update package metadata, build distributions, publish to a package registry, or claim implemented behavior only because a planning release exists.

Specifications and ACDs can be created or updated in any release, including but not limited to planning releases. They belong with the change they define. When a spec or ACD defines behavior, architecture, migration work, or conformance work for a release branch, commit it with the branch, pull request, or release work that carries that change. Do not leave the durable definition behind in a separate planning-only branch when the implementation or architecture change it defines lands elsewhere.
<!-- /skillskeeper-directive: github-planning-releases-and-change-artifacts -->

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
