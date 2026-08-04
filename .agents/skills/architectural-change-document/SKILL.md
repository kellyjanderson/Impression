---
name: architectural-change-document
description: Create, review, update, and close Architectural Change Documents (ACDs) for temporary architecture-transition work. Use when a proposed architectural change is not yet true in code, when architecture docs would otherwise become aspirational, or when ACD-backed specs, conformance checks, post-implementation architecture reconciliation, and closure/cleanup need to be managed.
---

# Architectural Change Document

Use this skill when architectural intent needs a temporary project-management
document before post-implementation reconciliation can integrate it into
canonical architecture.

## Core Rule

An Architectural Change Document, or ACD, is temporary transition scaffolding.
It describes an intended architectural change while the code, specs, tests, and
canonical architecture documents are moving into conformance.

Canonical architecture documents should describe what is true. If a requested
architecture edit would make a canonical architecture document aspirational,
create or update an ACD instead.

An ACD may also resolve a spec or architecture readiness blocker. When a
blocker is architectural after specs have started, create or update an ACD that
defines the target architecture, conformance path, and follow-on specs needed
to eliminate the blocker.

Spec creation and spec review may create or update ACDs before implementation.
This is the preferred feedback artifact when spec work discovers architecture
that is needed but not yet true in code. Do not wait for implementation to
preserve that architectural work.

After work has moved past the architecting phase, ACDs are the only allowed
path for new architecture changes until implementation is complete. Do not
update canonical architecture from spec creation, spec review, planning, or
implementation work; preserve the change in an ACD and leave canonical
architecture reconciliation for the explicit post-implementation process.

Architecture files are sacrosanct after specs start. ACDs exist to prevent
post-architecture-phase changes from being lost, merged into canonical docs too
early, or confused with work that has already been speced and implemented.

## Template Authority

When creating or reviewing an ACD, load the selected process template before
drafting.

Selection order:

1. `project/process/skills-templates-manifest.md` key `architectural-change-document`, when present.
2. `.agents/process/skills-templates-manifest.md` key `architectural-change-document`, when present.
3. `.agents/process/templates/acd-template.md` from the nearest shared ancestor.

The selected template is authoritative for ACD sections, status values,
spec source placement, conformance checklist, and closure criteria.

## Lifecycle

Use these statuses:

- `Proposed`: the desired change is being described.
- `Accepted`: the direction is approved enough to plan.
- `Drafting Specs`: spec creation is underway.
- `In Progress`: final specs, tests, progression, or implementation are underway.
- `Conformance Review`: implementation is complete enough to reconcile docs.
- `Closed`: canonical architecture is updated and active plans no longer depend on the ACD.

## Spec Handling

An active ACD may contain `## Specification Sources` while the change is being
planned. Treat that source section as ACD-local process scaffolding.

When final canonical specs and paired test specs exist:

- verify spec split coverage and canonical lineage;
- update progression/index links to canonical specs;
- remove completed process scaffolding from canonical architecture documents;
- keep or remove the ACD-local source notes at judgment, because the ACD itself is ephemeral;
- ensure active architecture and planning do not depend on completed scaffolding.

## Conformance And Closure

Do not close an ACD until:

- implementation conforms to the accepted target architecture;
- readiness blockers assigned to the ACD are resolved or moved into a new active ACD;
- final canonical specs use canonical architecture as their primary ancestor, or the ACD only while transition is still active;
- parent specs created during decomposition are 100% covered by children and archived;
- paired test specs and progression links point to canonical specs;
- canonical architecture documents have been updated to the conformed state;
- completed process scaffolding has been removed from active canonical architecture docs;
- unresolved deviations are recorded in a new ACD or explicit follow-up.

When an ACD closes, it may be archived whole. The important invariant is that
canonical architecture, active specs, and active plans no longer require the ACD
or its source notes as live authority.

## Code Improvement Sources

A `codeimprovement` issue may be the discovery record for an architectural
transition. When cleanup affects ownership, boundaries, data flow, routing,
concurrency, storage, or other architecture-level structure, create or update an
ACD instead of rewriting canonical architecture aspirationally.

Link the originating `codeimprovement` issue from the ACD and update the issue
status when the ACD is accepted, superseded, or closed.
