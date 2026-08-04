---
name: planning
description: Create and maintain progression-style planning documents that sequence final leaf specifications and paired test specifications in dependency order.
---

# Planning

Planning documents define the implementation and verification sequence for specifications.

## Purpose

Planning answers:

* what must be completed before other work can proceed
* what order executable specification leaves should be implemented in
* what has been completed

Planning does not define version intent. That belongs to `release-definitions`.

## Core Rule

Only these belong in progression-style planning:

* final leaf specifications
* paired feature test specifications

Parent or umbrella specifications must not appear there.
Superseded parent specs and incomplete split scaffolding must not appear as
implementation leaves.

## Structure

Planning documents should:

* group work into implementation lanes when useful
* order items by dependency
* use checkboxes for completion state

Common lanes include:

* core functionality
* obligate specifications
* polish specifications
* prerequisite resolution

## Completion Tracking

Keep implementation and verification visible separately.

For feature leaves:

* check the feature leaf when implementation is complete
* leave the paired test-spec item unchecked until verification is complete
* leave dependent items unchecked with `Status: Missing prerequisite - <path>`
  when architecture, ACD, spec, or prerequisite implementation is missing

## Relationship To Other Documents

* architecture defines the system
* ACDs define temporary architectural transitions
* specifications define implementation work
* code improvement issues record bad code discovered during implementation or review
* release definitions define holistic version scope
* planning defines execution order

Specification source sections are temporary discovery scaffolding. Once final
canonical specs and paired test specs exist, progression should point to those
specs and completed scaffolding should be removed from active architecture
documents.

Accepted `codeimprovement` issues can become progression work after they are
promoted into the appropriate implementation form: a final specification, paired
test specification, ACD-backed conformance task, or explicit cleanup task with
clear validation. Keep links back to the originating code improvement issue.

## SkillsKeeper Directives

<!-- skillskeeper-directive: app-integration-planning -->
### App Integration Planning

## App Integration Planning

Plans for user-facing features must separate helper implementation from app integration. After implementation tasks, include first-class integration tasks when applicable:

- wire the service, controller, adapter, or worker into the app surface;
- connect the UI action, command, external call, event, or background trigger;
- add integration validation through the real route;
- update docs and progression only after reachability is proven.

Unchecked specs that only create records, controllers, adapters, registries, helpers, or isolated tests must not imply the product feature is complete.
<!-- /skillskeeper-directive: app-integration-planning -->

<!-- skillskeeper-directive: route-wiring-tasks -->
### Route Wiring Tasks

## Route Wiring Tasks

Progression documents for feature-bearing work must keep helper implementation, route wiring, route validation, and status/documentation updates as separate first-class tasks when they can complete independently.

Use this shape when applicable:

```md
- [ ] Implement helper/service behavior.
- [ ] Wire behavior into <GUI/console/API/workflow/library consumer> route.
- [ ] Validate integrated route through <test/smoke/manual proof>.
- [ ] Update docs/progression after route validation.
```

Do not collapse route wiring and route validation into a helper implementation checkbox. Use app-type-specific route language so GUI, console, API/service, workflow, mixed, and library-only features have the correct integration proof.
<!-- /skillskeeper-directive: route-wiring-tasks -->

<!-- skillskeeper-directive: code-improvement-planning -->
### Code Improvement Planning

When a `codeimprovement` issue is accepted for implementation, plan it as first-class work instead of leaving it as an unsequenced note. Link the issue from the progression item, split broad issues into final specs and paired test specs when needed, and keep the code improvement issue open until the cleanup is implemented, validated, and indexed as done or superseded.
<!-- /skillskeeper-directive: code-improvement-planning -->

<!-- skillskeeper-directive: specification-canonicalization-planning -->
### Specification Canonicalization Planning

## Specification Canonicalization Planning

When planning work includes broad parent specs, split cleanup, or draft specs
that still need final canonical children, keep specification canonicalization
as explicit work:

- split parent or umbrella specs into children;
- verify 100% parent responsibility coverage;
- move uncovered responsibilities into children and re-verify;
- mark children canonical after coverage reaches 100%;
- archive superseded parent specs;
- update indexes and progression links to canonical children;
- remove completed process scaffolding from active architecture documents.

Do not list a parent spec as an implementation leaf once child specs exist or
split coverage is underway.
<!-- /skillskeeper-directive: specification-canonicalization-planning -->

<!-- skillskeeper-directive: progression-template-registry -->
### Progression Template Registry

## Progression Template Registry

When creating or substantially revising progression documents, load the `progression` template from the selected process registry before drafting the document.

Selection order:

1. `project/process/skills-templates-manifest.md` key `progression`, when present.
2. `.agents/process/skills-templates-manifest.md` key `progression`, when present.
3. `.agents/process/templates/progression-template.md` from the nearest shared ancestor.

The selected template is authoritative for separating helper implementation, route wiring, route validation, and documentation/status update tasks.
<!-- /skillskeeper-directive: progression-template-registry -->

<!-- skillskeeper-directive: prerequisite-planning -->
### Prerequisite Planning

## Prerequisite Planning

When planning or revising progression, determine prerequisites for each final
leaf spec before sequencing implementation. Use the `Prerequisites` field from
the linked spec when present, and add it when absent.

Represent prerequisite gaps explicitly:

- unarchitected prerequisite work links to an ACD and keeps dependent items
  unchecked as `Missing prerequisite`;
- architected prerequisite work without a final spec links to the new or
  updated spec and keeps dependent items unchecked as `Missing prerequisite`;
- existing but unimplemented prerequisite specs are sequenced before dependent
  items and implemented first.

Do not use a blocked status when the correct next action is creating/linking a
prerequisite artifact or implementing an existing prerequisite spec.
<!-- /skillskeeper-directive: prerequisite-planning -->
