---
name: system-skills
description: Use for immutable home-entry system-skill authority rules that define how agents compose rules from `~` down to the working folder and what belongs in the system skill layer.
---

# System Skills

This skill defines the topmost skill layer for this machine.

The agent entry point is `~`.

## Composition Rule

When determining which rules apply, the agent should:

1. start at `~`
2. discover applicable skills and policies there
3. continue downward through each directory on the path to the current working folder
4. compose the rules from that path in order
5. follow the composed result

This means the active rule set is built from:

* home-entry system skills first
* then any lower shared or local layers discovered between `~` and the working folder

## Purpose

System skills hold rules that should apply before home-local app, project, workspace, or task-local rules.

They are for:

* global safety rules
* host-wide filesystem policies
* machine-level authority boundaries
* other durable system-wide invariants

They are not for:

* project workflow
* repository conventions
* app-specific UI or implementation guidance
* temporary local preferences

## Layering Rules

Lower layers may narrow behavior inside the space permitted by system skills.

Lower layers must not replace, weaken, or redefine a system skill.

## Authoring Rules

Create a new system skill only when the rule is genuinely machine-wide.

System skills should be:

* minimal
* durable
* narrowly scoped
* non-project-specific

Prefer a small number of focused immutable system skills over one large catch-all bundle.
