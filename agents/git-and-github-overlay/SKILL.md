---
name: git-and-github-overlay
description: Apply Impression-specific Git and GitHub rules for allowed planning anchors, feature-branch preference, and planning-structure or ad hoc path handling.
---

# Impression Git And GitHub Overlay

This Skill narrows the shared `git-and-github-core` rules for the Impression workspace.

## Allowed Planning Anchors

Implementation work may be anchored by:

* an issue
* a specification
* an ad hoc work document under `project/adhoc/`
* a version-planning structure under `project/planning/` when the work is creating durable planning containers

Before implementation begins, the chosen path should be explicit:

* feature path
* ad hoc path
* planning-structure path

If the path is not clear, ask a short disambiguating question first.

## Workspace Branch Policy

For this workspace, active implementation work should happen on a feature branch.

Even small fixes should prefer a feature branch unless the user explicitly asks for a different branch shape.

## Planned Release Working Branches

When a planned release has been established and the team chooses to integrate
through a release working branch:

* merge the release planning branch into `main` first so the release scaffold is durable
* create a working branch from updated `main`
* name the working branch for the release, preferably `working/<release>`
* merge feature branches into the working branch, not directly into `main`
* keep using feature branches and pull requests for all feature work
* merge the working branch into `main` only when the planned release is ready

In that mode:

* `main` holds completed integrated release states
* the working branch holds in-progress release integration
* feature branches remain the unit of isolated implementation work

## Planning-Structure Path

Planning-structure work may create:

* new version-planning subtrees
* version-scoped architecture or specification containers
* other durable planning scaffolds

That path still uses a feature branch, but the new planning structure itself is the durable planning record.
