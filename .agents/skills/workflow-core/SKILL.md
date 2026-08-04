---
name: workflow-core
description: Follow the shared workflow from exploration through stabilization into spec-first discovery and implementation, with research, release definitions, and test specifications as durable support layers.
---

# Workflow Core

This Skill defines the shared project workflow from product definition to implementation.

## Overview

Work progresses through three phases:

1. exploration
2. stabilization
3. spec-first discovery and implementation

Research supports all phases.
Release definitions provide version-level cohesion.
Test specifications provide durable verification contracts.
Architectural Change Documents (ACDs) provide temporary transition management
when desired architecture is not yet true in code.

## Exploration Loops

Use explicit feedback loops between:

* product and research
* UI definitions and research
* architecture and research
* release definitions and stabilized project branches

Exploration continues until the relevant branch is clear enough to guide durable downstream work.

## Stabilization

Before implementation-heavy work begins, the relevant product, UI, and architecture branches must be stable enough to guide execution without constant structural churn.

For architecture, stability means the relevant branch has been completed breadth first:

* major parts are identified
* relationships are described
* high-level data flow is described
* cross-domain solutions are resolved

If the desired architecture is a change from current reality, stabilize that
intent in an ACD first. Do not make canonical architecture docs aspirational.

## Spec-First Specification Discovery

After stabilization:

* add or update `## Specification Sources` in architecture documents that imply downstream work
* use ACD-local source notes when the work is an architectural transition that is not yet true in code
* use `do specs` to create draft implementation specs from architecture, ACDs, parent specs, issues, or notes
* use `review specs` as the independent refinement mechanism
* when spec creation or review discovers missing/stale architecture, create or
  update an ACD and link it from the affected specs; do not update canonical
  architecture again until implementation is complete and reconciliation is
  explicitly run
* adversarially score, split, and review specs until they are small enough and ready enough to become final implementation specs
* record implementation owner/module, routes, data ownership, UI field/control inventory, reuse/extraction decisions, performance/privacy constraints, and test strategy in the specs
* create paired test specifications for final feature specs as needed
* remove completed process scaffolding from active canonical architecture documents after final specs and paired test specs exist
* archive or close parent specs only after child specs cover 100% of parent responsibilities

Implementation should not be used as a substitute for unfinished architecture or unfinished UI definition work.

Do not combine spec creation and spec review in one action. Creation uses
`do specs`; critical review uses `review specs`.

If shared specification-sizing guidance conflicts with local process guidance,
the stricter split, readiness, and coverage rule wins. Review scoring must use
the current implementation-spec template selected through the process registry.

## Path Rule

Implementation work must not begin without a durable planning anchor.

Workspace overlays may define the allowed local paths and anchor variants.

For spec-first projects, a valid planning anchor is a final implementation spec
whose template-governed Review Score, split decision, and readiness blockers
permit implementation.

Source notes and parent specs are not planning anchors once canonical final
specs exist. Progression should point to canonical final specs and paired test
specs.
