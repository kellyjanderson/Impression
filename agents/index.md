# Agent Documentation Index

This folder defines how agents should operate within this repository.

Start here before taking action.

---

## Documents

* **architecture.md**
  Defines what architecture documents are and how they describe system structure and cross-domain solutions.

* **ui-definitions.md**
  Defines the interface-structure layer that captures durable visible behavior and interaction semantics.

* **ui-ux-invariants.md**
  Defines mandatory structural interface rules such as layout stability, spatial memory, state coverage, accessibility, and responsive clarity.

* **release-definitions.md**
  Defines holistic version-level planning documents that connect release intent to UI, architecture, and specifications.

* **specifications.md**
  Defines how architecture is refined into implementable specification units.

* **test-specifications.md**
  Defines how final feature leaf specifications gain paired manual and automated verification documents, including durable coverage artifact expectations.

* **planning.md**
  Defines how specifications are sequenced and tracked for implementation.

* **research.md**
  Defines how research is stored as durable project knowledge.

* **workflow.md**
  Defines how agents move through product definition, architecture, specification refinement, and implementation.

* **delegation.md**
  Defines how the main agent delegates bounded work, assigns ownership, waits when needed, and reviews sub-agent output.

* **documentation.md**
  Defines documentation quality expectations and the rule that durable docs are a required part of completion.

* **git-and-github.md**
  Defines branching and pull request rules for bug fixes and feature work.

---

## Project Documentation

Project-level documents live in:

```text
project/
  agents/
  ui/
  product-definition.md
  architecture/
  specifications/
  test-specifications/
  planning/
  release-x.y.z/
  research/
```

Agents should consult relevant project documents before making changes.

Project-specific agent rules live in:

```text
project/agents/
```

That folder is the canonical place for instructions that apply specifically to this project, while the root `agents/` folder remains the shared process guidance for the repository.

Project-specific testing and rendering rules, such as reference-image lifecycle policy, should be documented there rather than in the shared root agent docs.

---

## Starting Point

When beginning work:

1. review `project/product-definition.md` (if present)
2. review `project/agents/` for project-specific operating rules
3. review `git-and-github.md` before implementation work
4. review relevant release definitions
5. review relevant UI definitions
6. review `ui-ux-invariants.md` for interface work
7. review relevant architecture
8. follow the workflow defined in `workflow.md`
9. if using sub-agents, review `delegation.md` before assigning work

---

## Guiding Principle

The system is defined by its documents.

Code is an implementation of that system.
