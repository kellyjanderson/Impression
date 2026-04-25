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

## Planning-Structure Path

Planning-structure work may create:

* new version-planning subtrees
* version-scoped architecture or specification containers
* other durable planning scaffolds

That path still uses a feature branch, but the new planning structure itself is the durable planning record.
