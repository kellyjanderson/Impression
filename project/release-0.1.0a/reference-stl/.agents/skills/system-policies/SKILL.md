---
name: system-policies
description: Use for immutable machine-wide system policies, especially filesystem access rules that define which root-level folders are writable and which are read-only by default.
---

# System Policies

This skill defines machine-wide filesystem policy.

## Root-Level Filesystem Policy

Treat root-level folders as read-only by default unless they are explicitly listed as writable.

Writable root-level paths:

* `/Users`
* `/usr`
* `/var`
* `/etc`

All other root-level paths should be treated as read-only unless a future system policy explicitly changes that decision.

## Lower-Layer Rule

Lower layers may add narrower restrictions inside writable roots.

Lower layers must not widen access for root-level paths that this policy marks as read-only.

## Scope

This policy governs machine-wide agent behavior around filesystem writes.

It is not a project-specific repository rule.
