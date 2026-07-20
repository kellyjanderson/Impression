---
name: architecture
description: Define or refine system architecture documents that resolve cross-domain structure, data flow, responsibilities, and relationships before specification work begins.
---

# Architecture

Use this Skill when the work is primarily about system structure rather than implementation detail.

## Purpose

Architecture defines:

* what parts exist
* what responsibilities they own
* how they relate
* how cross-domain constraints are reconciled

Architecture should resolve enough of the system-level picture that specifications refine a coherent structure instead of inventing missing structure during implementation.

## When To Use It

Use architecture work for questions such as:

* data representation
* processing flow
* asynchronous coordination
* interface boundaries
* cross-domain tradeoffs

If the issue is mostly visible behavior, prefer `ui-definitions`.
If the issue is already implementation-sized, prefer `specifications-core`.

## Recommended Structure

Architecture documents should usually include:

* overview
* relationship to sibling or parent architecture documents, when they exist
* components
* relationships
* data flow
* cross-domain solutions
* linked downstream specifications
* change history at the bottom of the document

## Change History Rule

Architecture documents should end with a `## Change History` section.

Each entry should include:

* date
* short description of the change
* reason or context for the change

When an architecture document extends, revises, or depends on another architecture document, make that relationship explicit near the top of the document.

## Sequencing Rule

Architecture work is breadth first across the relevant system area.

Do not start specification refinement for an architectural branch until the branch is complete enough to cover:

* the major parts involved
* how those parts interact
* the high-level data flow
* the system-level decisions shaping implementation

## Relationship To Specifications

Architecture defines the system-level solution.

Specifications define how individual parts of that solution are implemented.

If important system relationships are still missing, the correct next step is more architecture work, not specification work.
