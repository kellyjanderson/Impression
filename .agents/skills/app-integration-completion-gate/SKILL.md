---
name: app-integration-completion-gate
description: Use when implementing, reviewing, or marking complete any specification, progression item, feature leaf, UI feature, service route, controller, adapter, background task, or app behavior. Enforces that work is not complete unless it is wired into the intended app surface, reachable by the user or caller, validated through the integrated route, and honestly reflected in docs/progression.
---

# App Integration Completion Gate

Use this skill when an agent is about to claim implementation completion, review a feature, update progression state, or decide whether user-facing behavior is done.

## Core Rule

Implemented but not wired is incomplete.

A spec, progression item, or feature is not complete unless the user or intended caller can reach it through the app surface, command, API, event, or workflow it belongs to. Passing unit tests is insufficient for user-facing app behavior when the real route does not exercise the code.

Controllers, adapters, registries, services, records, and tests become part of completion only when a real product route uses them.

## Completion Proof

Before marking work complete, name:

- user surface or intended caller;
- route, event, command, external call, or automatic trigger;
- module that wires the route into the behavior;
- focused validation for the behavior;
- integrated validation or manual smoke through the real route;
- docs or progression files updated to match reality;
- any hidden, unused, or unwired code that remains.

## Allowed Status Language

Use honest partial states instead of `complete`:

- `Implemented in isolation; not complete.`
- `Wired; awaiting integration validation.`
- `Complete; reachable and validated.`

Only use `Complete` when the route is reachable, validated, documented, and reflected in progression/spec state.

## Failure Examples

- Preview async controller exists, but the shell bypasses it.
- Dependency refresher exists, but dirty editor events do not call it.
- Editor adapter exists, but the app still uses a placeholder.
- Agent lifecycle exists, but UI callbacks can run on a worker thread.
- Feature is in a registry, but no UI or command exposes the registry.

## Deferred Integration Improvements

When completion review finds hidden, unused, unwired, or bypassed code that is
too broad to repair in the current task, document it as a `codeimprovement`
issue using the `coding` skill's Code Improvement Issues process. Include
`code-location` blocks for both the unused implementation and the route or
surface that should wire it.

## SkillsKeeper Directives

<!-- skillskeeper-directive: upstream-integration-contract-check -->
### Upstream Integration Contract Check

## Upstream Integration Contract Check

When architecture, source notes, final specs, or test specs define an application integration contract, completion review must verify that implementation preserved it.

Before marking complete, compare the implementation against the upstream:

- app type;
- user/caller surface;
- invocation route;
- wiring owner/module;
- observable result;
- integration validation;
- incomplete status risk.

If the upstream contract is missing, treat that as a readiness gap unless the work is genuinely library-only and names its consuming module or downstream caller. If implementation changed the route, update the durable architecture/spec/test/progression documents before claiming completion.
<!-- /skillskeeper-directive: upstream-integration-contract-check -->

<!-- skillskeeper-directive: document-deferred-integration-debt -->
### Document Deferred Integration Debt

If integration debt cannot be fixed in the current task, create or update a `codeimprovement` issue using the `coding` skill's Code Improvement Issues process. Link the issue from the final review or implementation note and do not mark the affected work complete when the missing integration is required for completion.
<!-- /skillskeeper-directive: document-deferred-integration-debt -->
