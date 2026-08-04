---
name: architecture
description: Define or refine canonical system architecture documents that resolve cross-domain structure, data flow, responsibilities, reusable boundaries, specification source responsibilities, and Architectural Change Document (ACD) routing before specification work begins.
---

# Architecture

Use this Skill when the work is primarily about system structure rather than implementation detail.

## Purpose

Architecture defines:

* what parts exist
* what responsibilities they own
* how they relate
* how cross-domain constraints are reconciled
* which downstream specification responsibilities are discovered

Architecture should resolve enough of the system-level picture that specifications refine a coherent structure instead of inventing missing structure during implementation.

Canonical architecture documents describe the architecture that is true or is
accepted as the current system contract. If a requested edit would make a
canonical architecture document aspirational before code conforms, create or
update an Architectural Change Document (ACD) instead.

## When To Use It

Use architecture work for questions such as:

* data representation
* processing flow
* asynchronous coordination
* interface boundaries
* cross-domain tradeoffs
* reusable module boundaries
* architecture-level specification source responsibilities
* routing proposed architecture changes into ACDs when code does not yet conform

If the issue is mostly visible behavior, prefer `ui-definitions`.
If the issue is already implementation-sized, prefer `do-specs` or the `coding` skill.

## Recommended Structure

Architecture documents should usually include:

* overview
* relationship to sibling or parent architecture documents, when they exist
* components
* relationships
* data flow
* cross-domain solutions
* `## Specification Sources`, when the architecture implies downstream work
* change history at the bottom of the document

## Architectural Change Documents

Use an ACD for temporary transition work:

* desired architecture that is not yet true in code
* architectural migration or conformance work spanning specs, tests, and implementation
* changes that need temporary draft specs while canonical architecture must remain honest
* closure work that updates canonical architecture after implementation conforms

When an ACD is active, reference it from the affected architecture document.
When the ACD closes, update canonical architecture to the conformed state and
ensure active specs/plans no longer depend on the ACD as live authority.

## Change History Rule

Architecture documents should end with a `## Change History` section.

Each entry should include:

* date
* short description of the change
* reason or context for the change

When an architecture document extends, revises, or depends on another architecture document, make that relationship explicit near the top of the document.

## Sequencing Rule

Architecture work is breadth first across the relevant system area.

Do not start implementation-spec creation for an architectural branch until the branch is complete enough to cover:

* the major parts involved
* how those parts interact
* the high-level data flow
* the system-level decisions shaping implementation
* reusable module/component boundaries
* data ownership and read/write boundaries
* lifecycle, concurrency, privacy, and performance constraints where applicable
* source responsibilities ready for `do specs` and independent `review specs`

## Specification Creation Requirement

Architecture documents that imply downstream work should include:

```md
## Specification Sources
```

Specification sources are the architecture-to-specification handoff. They are
used to expose implementation responsibilities, hidden complexity,
reuse/extraction decisions, readiness blockers, and likely split boundaries
before draft implementation specs are written.

Use `do specs` to create draft implementation specs from architecture, ACDs,
parent specs, issues, or notes. Then use `review specs` as the independent
critical check. Creation and review must remain separate actions.

Readiness blockers discovered during architecture or spec work are not passive
notes. The next review action must resolve each blocker by defining the missing
architecture, creating/updating an ACD, splitting the spec, creating/updating a
prerequisite spec, or filling the missing readiness field. Do not mark specs
ready while unresolved blockers remain.

Spec creation and spec review may create or update architecture feedback
artifacts when they discover missing architecture. Once work has moved past the
architecting phase into `do specs`, `review specs`, planning, or implementation,
all architecture changes must be captured through an ACD. Do not update
canonical architecture again until implementation is complete and an explicit
architecture reconciliation process is run.

Architecture files are sacrosanct after specs start. This prevents architecture
changes from being lost or becoming impossible to distinguish from architecture
that was already speced, implemented, or still pending. ACDs are the durable
record for every architecture change discovered after the initial architecting
phase.

When a project provides a process/template registry, use it. Prefer:

1. `project/process/skills-templates-manifest.md`
2. `.agents/process/skills-templates-manifest.md`

Load the `implementation-spec` template named by the selected registry when
creating draft specs. Preserve the complete required Review Score, readiness,
UI, route, and reuse fields. A Review Score is valid only when it uses the
current template's front-matter total and complete final calculation section.

Specs must preserve source responsibilities explicitly. Score every applicable
category during `review specs` and do not tune scores downward by omitting UI
fields, reusable code work, concurrency, performance, privacy, write behavior,
or readiness blockers.

Shared split policy:

* `25+`: split required before implementation
* `16-24`: explicit split review and cohesion explanation required
* `0-15`: may remain small/cohesive if readiness fields are present

## Reuse And Boundary Rule

Architecture should prevent future siloed code. Make reusable boundaries explicit:

* existing code/components to reuse as-is
* existing libraries/modules that should be extended
* genuinely new reusable modules/components when there is a clear domain boundary and plausible repeated use
* one-off UI/service/query patterns only when intentionally justified

## Relationship To Specifications

Architecture defines the system-level solution.

`do specs` creates draft implementation specs from architecture, ACDs, parent
specs, issues, or notes.

`review specs` independently reviews, adversarially scores, splits, and verifies
draft specs until final leaf specifications are ready.

If important system relationships are still missing, the correct next step is more architecture work, not specification work.

When missing relationships are found during `do specs` or `review specs`, record
them immediately in an ACD before marking the spec ready. Do not edit canonical
architecture from spec work.

## Relationship To Code Improvement Issues

Code improvement issues live in `codeimprovement/`, sibling to the relevant
`architecture/` and `specs/` folders. They record discovered bad code with
line-number locations when the cleanup is too large for the current coding
task.

When a code improvement issue is architectural, use architecture or ACD work to
define the target structure before implementation begins. Do not make canonical
architecture aspirational before code conforms; use an ACD for temporary
transition work.

## SkillsKeeper Directives

<!-- skillskeeper-directive: application-integration-contract -->
### Application Integration Contract

## Application Integration Contract

Before handing architecture to `do specs`, classify each feature-bearing branch by app type: `GUI`, `console`, `API/service`, `workflow`, `library-only`, or `mixed`.

For each branch, name:

- `User/caller surface:` where the behavior is accessed;
- `Invocation route:` control, command, event, API call, background trigger, workflow step, or consuming module;
- `Wiring owner/module:` component responsible for connecting the route;
- `Observable result:` what the user or caller can see, receive, inspect, or depend on;
- `Integration validation:` expected test, smoke, or manual proof through the real route.

If any answer is unknown, record it as an architecture readiness blocker instead of letting final implementation specs invent wiring later.

After recording an architecture readiness blocker, continue the architecture or
spec review by resolving it. Use an ACD when the blocker represents an
architectural transition that is not yet true in code.

Apply app-type-specific proof expectations:

- GUI branches need visible entrypoint/state coverage, UI-thread handoff where applicable, and GUI route validation.
- Console branches need command/subcommand, flags/args/stdin/config, stdout/stderr/exit-code behavior, side effects, and CLI validation.
- API/service branches need endpoint/caller contract, auth/permission/error behavior, side effects, observability, and route validation.
- Mixed branches must name each independently failing surface.
- Library-only branches must name the consuming module or downstream caller and must not be presented as user-facing features.
<!-- /skillskeeper-directive: application-integration-contract -->

<!-- skillskeeper-directive: architecture-template-registry -->
### Architecture Template Registry

## Architecture Template Registry

When creating or reviewing architecture documents, prefer the selected process registry template before writing freehand structure.

Selection order:

1. `project/process/skills-templates-manifest.md` key `architecture`, when present.
2. `.agents/process/skills-templates-manifest.md` key `architecture`, when present.
3. `.agents/process/templates/architecture-template.md` from the nearest shared ancestor.

The architecture template is authoritative for required document sections, including the `Application Integration Contract`.

When creating an ACD from architecture work, use the `architectural-change-document`
template from the selected process registry.
<!-- /skillskeeper-directive: architecture-template-registry -->

<!-- skillskeeper-directive: code-improvement-architecture-routing -->
### Code Improvement Architecture Routing

When a `codeimprovement` issue changes component ownership, data flow, routing, reusable boundaries, storage, concurrency, or other architecture-level structure, route it through architecture or an ACD before implementation specs. Keep the code improvement issue as the discovery record, and link it from the architecture, ACD, source notes, or final specs that resolve it.
<!-- /skillskeeper-directive: code-improvement-architecture-routing -->
