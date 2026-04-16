# Workflow Guidance

This document defines how agents move through the project from product definition to implementation.

---

## Overview

Work progresses through three phases:

1. Exploration (product and architecture refinement)
2. Stabilization (product and architecture become fixed enough to proceed)
3. Specification refinement and implementation

Research supports all phases and is used to store durable information.

Release definitions provide version-level cohesion across stabilized UI, architecture, and specification work.

Test specifications provide durable verification contracts for final feature leaf specifications.

Project-specific agent rules are stored separately in:

```text
project/agents/
```

Agents should consult that folder before substantial work begins and keep project-specific operating rules there rather than scattering them through transient discussion.

Repository-wide branching and pull request rules are defined in:

```text
agents/git-and-github.md
```

Agents should satisfy those branch and PR requirements before making code changes.

Implementation work must not begin without a durable planning anchor:

* an issue for bug-fix work, or
* a specification for feature work

UI definition guidance is defined in:

```text
agents/ui-definitions.md
```

Mandatory structural UI rules are defined in:

```text
agents/ui-ux-invariants.md
```

Project UI definition documents should live in:

```text
project/ui/
```

Release definition guidance is defined in:

```text
agents/release-definitions.md
```

Project release definition documents should live in:

```text
project/release-x.y.z/
```

Test specification guidance is defined in:

```text
agents/test-specifications.md
```

Project test specification documents should live in:

```text
project/test-specifications/
```

---

## Exploration Loops

### Product ↔ Research

Product definition is refined through research.

Primary product document:

```text
project/product-definition.md
```

* identify unknowns in the product definition
* perform research to answer those questions
* store findings in the research folder
* update the product definition as needed

This loop continues until the product definition is sufficiently clear.

---

### UI Definitions ↔ Research

UI definitions are refined through research and observation.

Primary UI definition documents live in:

```text
project/ui/
```

* identify interface unknowns, clutter, ambiguity, or interaction problems
* perform research or observation where needed
* store findings in the research folder when they are durable
* update UI definitions to reflect resolved interaction structure

This loop continues until the relevant UI surface is sufficiently coherent.

UI definition work is breadth first across the relevant interface area.

The goal is to define the visible structure and interaction semantics of the surface before narrower specifications or implementation details are finalized.

Agents should apply `agents/ui-ux-invariants.md` during this loop rather than waiting until implementation to discover layout-shift, missing-state, or scanning-flow problems.

For broad UI system work, breadth-first refinement should usually move through:

1. brand and voice rules
2. design guide and look-and-feel rules
3. component-system rules
4. surface-specific UI definitions

Surface-specific UI specifications should usually come after those layers are stable enough to guide implementation.

---

### Release Definitions ↔ Stabilized Project Branches

Release definitions are refined once the relevant product, UI, and architecture branches are stable enough to be grouped into a version intent.

Primary release documents live in:

```text
project/release-x.y.z/README.md
```

* identify the intended outcome of the version
* link the relevant UI definitions
* link the relevant architecture
* link the planned specifications for that version
* keep version planning coherent as work branches multiply

Release definitions are not a substitute for progression.

They exist to keep a version understandable as one coherent project step.

---

### Architecture ↔ Research

Architecture is refined through research.

* identify cross-domain problems or unknowns
* perform research where needed
* store findings in the research folder
* update architecture to reflect resolved solutions

This loop continues until architecture provides a coherent system-level solution.

Architecture work is breadth first.

Agents should expand the architectural picture across the whole relevant system before beginning specification refinement for that architectural branch.

The goal is to understand how the parts work together at system level, not to fully specify one isolated branch while other architecture gaps remain unresolved.

---

## Stabilization

At some point:

* product definition becomes stable enough to guide development
* UI definitions become stable enough to guide interface work
* architecture becomes stable enough to define system structure
* release definitions become stable enough to define version intent

For architecture, "stable enough" means the relevant architecture documents for the area of work have been completed breadth first.

That means:

* the major system parts for that area are identified
* their relationships are described
* the high-level data flow is described
* the cross-domain solutions are resolved

Specification work should not start while important architecture-level gaps still exist for the same system area.

These do not need to be perfect, but must be stable enough that implementation can proceed without constant structural change.

Once stabilized, the exploration loops should diminish and specification work becomes primary.

When the work is versioned, release definitions should also be updated so the version scope remains explicit.

---

## Specification Refinement Loop

After stabilization:

* break architecture into specifications
* evaluate each specification for implementability

Specifications are written only after the relevant architecture and UI definition documents are complete enough to describe the whole system area being refined.

When the work is part of a planned version, release definitions should point at the resulting specifications rather than leaving version scope implicit.

Architecture and UI definitions come first for their respective concerns.

Specification refinement is not a substitute for unfinished architecture work.
It is also not a substitute for unfinished UI definition work when the interface structure itself is still unclear.

For each specification:

* if implementable, mark it final
* if not, refine it into child specifications

When a feature leaf specification is marked final, create or update its paired test specification in the same refinement pass.

Store all specifications and maintain backlinks.

This process continues until all leaf specifications are marked final.

This refinement is recursive.

Agents should expect to:

* create first-generation specifications from architecture
* review those specifications
* create child specifications where needed
* review those children in later rounds
* continue refining downward until the tree consists of implementation-sized final leaves

The intended result is a specification tree, not a flat one-pass list.

The goal is not to atomize work into the smallest possible leaves.

Specification refinement should usually stop at the largest cohesive unit that can still be implemented cleanly in one round without hiding another full round of work.

For user-facing feature branches, specification refinement must continue until surfaced running-app outcomes are also represented by final leaves.

If a running-app surface such as a sheet, dialog, panel, or settings surface already exists, the tree should also include a named parent UI/specification branch for that surface rather than only isolated behavior leaves.

Internal support leaves are often necessary, but they are not enough to finish a user-facing branch on their own.

If the intended outcome is something the user should see, hear, or interact with, at least one final leaf must describe that observable behavior explicitly.

After each specification pass, report how many specifications still require another refinement round so progress toward completed specification work remains visible.

---

## Progression Initialization

The progression document does not exist during product definition, architecture, or specification refinement.

Once all leaf specifications are marked final:

* create the progression document:

```text
project/planning/progression.md
```

* populate it with references to the final leaf specifications
* add paired test specification references for feature leaves
* arrange those references according to dependency order
* apply secondary ordering where needed

Feature leaves should not enter progression without their paired test specifications already represented in `project/test-specifications/`.

If a release definition exists for the active version, the progression document should remain consistent with that release scope.

The progression document must contain only:

* final leaf specifications
* paired test specification references for final feature leaves

For user-facing feature branches, progression should include the surfaced running-app leaves as well as any supporting leaves they depend on.

Support leaves being checked off does not, by itself, mean the feature is shipped.

Parent or umbrella specifications do not belong in progression and must not be tracked there.

Test specification references in progression exist to represent verification work explicitly.

The progression document becomes the execution sequence for the project.

---

## Specification Execution

After the progression document is created:

* implement specifications using the progression document as the source of execution order
* update completion status as specifications are implemented

Implementation proceeds only from specifications represented in the progression document.

Implementation should also proceed only from the correct git branch context:

* bug-fix work on a bug-fix branch linked to an issue
* feature work on a feature branch linked to a specification

Code changes should not be made directly on `main`.

Issue-driven code changes must also be back-referenced into the architecture/specification tree.

That means a bug fix should update the durable project documents that define the affected behavior, rather than living only as:

* an issue
* a branch
* a pull request
* and code changes

If the relevant architecture or specification does not exist yet, the correct step is to add or refine it before or alongside implementation.

---

## Progression Audit

During execution:

* ensure that implemented specifications remain consistent with their specification documents
* ensure that the progression document remains aligned with the current set of specifications

If specifications change during execution:

* update the affected specifications
* update the progression document to reflect those changes before continuing

---

## Architecture and Product Changes

If implementation reveals issues that affect:

* product definition, or
* UI definition, or
* system structure
* release intent or release scope

return to the appropriate loop:

* product ↔ research
* UI definitions ↔ research
* architecture ↔ research
* release definitions ↔ stabilized project branches

Update documents before continuing.

---

## Research Use

When research is required:

* perform research intentionally
* store findings in the research folder
* use research to inform product, architecture, or specifications

Do not rely on transient context for important information.

---

## Completion

Work continues until all relevant specifications in the progression document are implemented.

Completion criteria are defined separately.

---

## Guiding Principle

Follow the structure of the system.

Use loops to resolve uncertainty, and only proceed to implementation once the system is sufficiently defined.
