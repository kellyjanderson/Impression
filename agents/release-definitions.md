# Release Definition Guidance

Release definitions are the holistic version-level planning documents for the project.

They define the intent of a planned version and connect that version to the relevant UI definitions, architecture, and specifications.

---

## Purpose

Release definitions exist to answer:

* what this version is trying to accomplish
* what kind of user-facing improvement the version is meant to deliver
* which architecture and UI definition branches are in scope
* which specifications are planned to land in that version

They provide a version-level planning layer above progression.

---

## Relationship To Other Documents

Release definitions sit between broad project understanding and implementation sequencing.

The relationship is:

* product definition explains the product direction
* UI definitions explain visible behavior and interaction structure
* architecture explains system structure
* specifications define implementable units
* release definitions group a planned version holistically
* progression sequences final leaf specifications for execution

Release definitions do not replace progression.

Progression still answers:

* what order final leaf specifications should be implemented in
* what has been completed

Release definitions answer:

* why this version exists
* what coherent set of work belongs together in the version
* which user-visible outcomes the release is expected to surface

---

## Location

Project release definitions should live in folders named:

```text
project/release-x.y.z/
```

The primary document for a release should be:

```text
project/release-x.y.z/README.md
```

---

## Recommended Structure

A release definition should generally include:

### Intent

A concise description of what the version is meant to accomplish.

### Planned Architecture

Links to the architecture documents that define the structural work in scope for the release.

### Planned UI Definitions

Links to the UI definition documents that define the visible behavior in scope for the release.

### Planned Specifications

Links to the final or near-final specifications intended to ship in the release.

### Notes

Any release-level scoping notes, exclusions, or version-shaping context.

---

## When To Create Or Update A Release Definition

Create or update a release definition when:

* a new version is being planned
* a feature set needs a coherent version home
* the project has multiple active architecture/specification branches and needs a version-level framing

Agents should prefer creating or updating a release definition before a version accumulates a large amount of disconnected planning work.

---

## Constraints

Release definitions should:

* stay holistic and version-oriented
* link to source documents rather than duplicating all of their detail
* avoid turning into a second progression checklist
* avoid implying a feature is delivered merely because enabling layers exist

Release definitions should not:

* replace specifications
* replace progression
* track implementation status at the leaf-by-leaf level

---

## Guiding Principle

A release should be understandable as a coherent project step, not just a pile of unrelated completed tasks.

For user-facing release goals, the release definition should stay anchored on surfaced outcomes.

Support or enabling leaves may land earlier, but a release feature should not be treated as delivered until the intended visible or audible outcome is observable in the running app.
