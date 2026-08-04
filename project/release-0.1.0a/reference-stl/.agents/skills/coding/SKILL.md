---
name: coding
description: Use when implementing or modifying production code, tests, scripts, UI components, service modules, database/query code, or reusable libraries. Apply before coding and during review to enforce reuse, extraction, validation, and project workflow discipline.
---

# Coding

Use this skill whenever code is added, changed, or reviewed.

## Core Rule

Code should leave the system more coherent than it found it. Prefer reuse and clear ownership over local one-off implementations. Do not create duplicated UI controls, query patterns, model behavior, service routing, or helper logic when the behavior belongs in an existing shared module or a clearly named reusable module.

## Reusable Code Requirements

When writing code:

- Reuse existing code when it already provides the needed behavior.
- Prefer adding to an existing library/module over creating a new module when the concept belongs there.
- Create a new reusable module only when there is a clear domain boundary, a stable public API, and at least one real current caller plus plausible near-term reuse.
- Do not create `misc`, `utils`, `helpers`, or vague shared modules without a named responsibility.
- Keep reusable module APIs narrow, explicit, typed where practical, and documented by tests.
- Keep implementation details private unless another module has a real need to call them.
- Do not duplicate existing UI controls, components, model/list behavior, query services, message routes, DTO shapes, or storage helpers.
- If code starts as one-off but reveals reusable behavior, extract the reusable part before considering the work complete.
- Tests must exercise the reusable boundary, not only the first caller.
- One-off code is allowed only when the spec or implementation note identifies why reuse/extraction would be premature or misleading.

## Reuse Classification

Every implementation should be classifiable as one of:

- **Reused existing code as-is**: imported/called existing behavior without changing its public boundary.
- **Added to an existing library/module**: extended a named shared module/component with new public behavior.
- **Created a new reusable module**: introduced a new named module/component with a stable public API.
- **Intentionally one-off**: kept local because reuse would be premature, misleading, or contrary to the spec.

Do not treat this as a request to print a questionnaire. Apply the classification in the code structure, tests, and concise implementation notes.

## Code Improvement Issues

When coding work exposes bad code that should be cleaned up but cannot be
reasonably fixed inside the current task, document it as a first-class code
improvement issue instead of leaving it only in chat, a final note, or an
untracked TODO.

Use a `codeimprovement` folder beside the project's architecture and specs
folders:

```text
<project-root>/
  architecture/
  specs/
  codeimprovement/
    index.md
    <issue-slug>.md
```

Choose the nearest project root that owns the affected architecture/spec tree.
For nested project areas, use the `codeimprovement` folder sibling to that
area's `architecture` and `specs` folders. If no such structure exists, create
the issue in the nearest project planning area and state the placement in the
implementation note.

Each issue document must include:

````md
# <Short Code Improvement Title>

Status: proposed
Discovered during: <task/spec/progression/PR>
Severity: low | medium | high
Scope: local | cross-module | architectural

## Summary

<One or two sentences describing the improvement needed.>

## Locations

```code-location
file: path/to/file.ext
lines: 10-24
symbol: OptionalSymbolOrFunction
```

## Problem

<What is duplicated, brittle, misleading, bypassing an abstraction, hard to
test, unsafe, or inconsistent?>

## Why Not Fixed Now

<Why the fix exceeds the current task scope, risk budget, or dependency state.>

## Proposed Improvement

<What future work should centralize, rename, extract, delete, reroute, split,
migrate, or test.>

## Validation Needed

<Tests, smoke checks, integration route, migration proof, or review needed to
close the issue.>
````

Use one or more `code-location` fenced blocks for every issue. Each block must
name the file and current line or line range where the problem is visible.

Maintain `codeimprovement/index.md` with links to active and closed issues.
When creating or updating a code improvement issue, add or update the index in
the same change.

If the issue is architectural, cross-cutting, or requires a migration plan,
promote it into an ACD, specification, progression item, or tracked issue as
appropriate, but keep the `codeimprovement` document as the discovery record
until the improvement is closed or superseded.

## Module Boundary Requirements

For any new or extended reusable module:

- Name the domain responsibility in the module/component name.
- Keep public symbols minimal and cohesive.
- Keep side effects explicit.
- Avoid importing UI layers into domain/service modules.
- Avoid importing operational/domain code into UI components except through controller/model/service boundaries.
- Provide tests or smoke coverage for the public boundary.

## Specification Readiness Alignment

Before implementation, prefer a reviewed final spec with:

- implementation routing;
- reuse/extraction plan;
- data ownership;
- UI field/control inventory, when applicable;
- concurrency route, when applicable;
- performance and privacy constraints;
- test strategy.

If the selected spec lacks reuse/extraction fields, update the spec from the
source architecture, ACD, parent spec, issue, or notes before implementing, or
state the reuse classification in the implementation note.

Implementation should proceed from active canonical specifications. Do not code
from parent specs, umbrella specs, superseded specs, archived specs, incomplete
split children, or source notes. If a parent split is incomplete,
finish split coverage and canonicalization first.

## Implementation Workflow

- Follow the selected spec as source of truth.
- Keep generated/runtime artifacts out of source control.
- Run relevant validation before committing.
- When completing a spec, move it to `implemented/`, update indexes, commit, push, and open/update the PR per project workflow.

## SkillsKeeper Directives

<!-- skillskeeper-directive: completion-gate -->
### Completion Gate

## Completion Gate

Before marking a user-facing app, CLI, API, or workflow spec complete, prove the behavior is reachable through the product surface named by the spec. Name the entrypoint, UI control, command, route, event handler, external call, or background trigger that invokes it.

Code and focused tests are not completion when the app does not call the behavior. Report unreachable work as `Implemented in isolation; not complete.` or `Wired; awaiting integration validation.` instead of complete.

For GUI work, assume blocking is forbidden. Before coding, identify which work stays on the UI thread, which work uses a task lane or process, how results return to the UI thread, and how stale results are rejected. Route detailed GUI architecture questions to `gui-async-application-architecture`.

For user-facing app specs, run or document validation that exercises the integrated route, such as launch smoke, UI event to behavior, command invocation, service route, controller/view integration, or manual smoke where automation is impractical.
<!-- /skillskeeper-directive: completion-gate -->

<!-- skillskeeper-directive: canonical-spec-coding-anchor -->
### Canonical Spec Coding Anchor

## Canonical Spec Coding Anchor

Before coding from a specification, confirm:

- `Canonical status` is canonical or the project-local equivalent;
- primary ancestor is architecture or an active ACD, not an incomplete parent;
- split provenance is retained only for history;
- paired test specs and progression point to the same canonical child;
- completed process scaffolding has been removed from active architecture docs
  or is not being used as implementation authority.

If these are false, update the specification/process artifacts before coding.
<!-- /skillskeeper-directive: canonical-spec-coding-anchor -->
