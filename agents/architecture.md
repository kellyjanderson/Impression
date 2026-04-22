# Architecture Guidance

Architecture defines the structure of the system and resolves cross-domain problems.

It describes:

* what parts exist
* what responsibilities they have
* how they relate to each other

Architecture is where problems that span multiple domains are resolved into a single coherent system solution.

These are problems where competing needs must be reconciled, such as:

* data representation
* processing flow
* resource constraints
* asynchronous behavior
* interface requirements

These cannot be solved correctly at the specification level alone.

Architecture should be developed breadth first across the relevant system area.

Before specification work begins for an architectural branch, the architecture should describe the full system-level picture for that area well enough that specifications are refining a coherent structure rather than inventing missing structure as they go.

---

## Recommended Structure

Architecture documents should generally include:

### Overview

A short description of the system or subsystem.

### Components

The major parts of the system and their responsibilities.

### Relationships

How components interact with each other.

### Data Flow

How data moves through the system at a high level.

### Cross-Domain Solutions

System-level decisions that reconcile multiple constraints or domains.

### Specifications

References to specification documents that implement parts of this architecture.

Each referenced specification should correspond to a specific component, behavior, or concern defined in this document.

---

## Relationship to Specifications

Architecture defines the system-level solution.

Specifications define how individual parts of that solution are implemented.

Architecture should resolve enough of the problem that specifications do not need to invent system behavior.

Specifications should be written only after the architecture documents for the relevant area are complete enough to cover:

* the major parts involved
* how those parts interact
* the relevant data flow
* the cross-domain decisions that shape implementation

If those things are still missing, the correct next step is more architecture work, not specification work.

---

## Stability

Architecture should change less frequently than specifications.

Changes to architecture affect multiple parts of the system.
