---
name: specifications-core
description: Refine architecture or stabilized UI definitions into recursive specification trees with clear parentage, final leaves, surfaced outcomes, and paired verification.
---

# Specifications Core

Specification documents define how architecture or stabilized UI definitions are refined into actionable implementation work.

## Parentage Rule

Every specification must have exactly one backlink to its parent:

* an architecture document
* a UI definition document
* or another specification

Specifications should not have multiple parents.

## Change History Rule

Specification documents should end with a `## Change History` section.

Each entry should include:

* date
* short description of the change
* reason or context for the change

When a specification is revised after its first draft, add a new entry instead of replacing the earlier history.

## Refinement Model

Specification refinement is recursive.

The expected result is a tree:

* architecture or UI definitions produce first-generation specifications
* those specifications may produce child specifications
* refinement continues until executable leaves are implementation-sized

If a specification still hides another clean round of work, it is not final.

## Work Unit Annotation Rule

Every specification should be scored with Implementation Work Units (IWU) near the top of the document.

Place a `## Work Units` section immediately after the title/date block when present.

The section should identify:

* the unit as `Implementation Work Unit (IWU)`
* the shared unit definition
* the standard measures used to count the unit
* the spec's `Count: N IWU`
* a short basis for the count

Use the `specification-sizing` skill for the current IWU definition, calculation measures, and reporting rules.

## Final Leaf Rule

A final specification is implementable directly.

It should be the largest cohesive unit that:

* can be implemented in one round
* scores 1 IWU
* does not hide major unresolved decisions
* does not imply another full specification pass during implementation

If an intended final leaf scores above 1 IWU, refine it into child specifications and score those children instead of marking the larger branch final.

## User-Facing Outcome Rule

For user-facing branches, refinement is not complete merely because internal support leaves are final.

At least one final leaf must terminate in a surfaced running-app outcome a user can:

* see
* hear
* or interact with

If a feature could remain invisible after all current leaves are implemented, refinement is not finished.

## Completion Contract

When a feature leaf is marked final:

* create or update the paired test specification in the same refinement pass
* make documentation expectations explicit enough that durable docs are not left implied

Final features should not ship with verification or documentation left implied.
