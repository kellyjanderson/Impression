---
name: workflow-impression-overlay
description: Apply Impression-specific workflow variants for feature, ad hoc, and planning-structure paths plus staged repository delivery on feature branches.
---

# Impression Workflow Overlay

This Skill adds Impression-specific workflow detail on top of `workflow-core`.

## Path Variants

Impression supports three implementation paths:

* feature path
* ad hoc path
* planning-structure path

Use the path explicitly and keep the durable anchor aligned with it.

## Delivery Meaning

For this workspace, repository delivery is staged:

* work on a feature branch, not `main`
* commit in reasonable units as progress stabilizes
* push when a unit is complete or when explicitly asked
* create a pull request when explicitly asked
* merge when explicitly asked

Merging the pull request is the completion of that feature branch's work.

## Delegation Link

When sub-agents are used, follow the local `delegation` Skill for ownership boundaries, waiting, and review.
