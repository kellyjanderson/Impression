---
name: ui-definitions
description: Define durable visible-behavior structure, interaction semantics, component guidance, and state expectations before UI implementation details are finalized.
---

# UI Definitions

UI definitions are the visible-behavior analog to system architecture.

## Purpose

Use UI definitions to describe:

* interface surfaces
* primary and secondary actions
* stable state semantics
* appearance and accessibility rules
* reusable component families
* navigation and interaction structure

They should capture durable design intent, not transient implementation detail.

## Recommended Structure

UI definition documents should usually include:

* overview
* primary interface elements
* secondary access
* states and semantics
* appearance and accessibility
* component guidance
* related specifications

## Core State Rule

UI definitions must intentionally cover more than the happy path, including:

* loading
* empty
* partial
* full
* error
* disabled
* overflow
* responsive or narrow layouts

## Relationship To Architecture

Architecture defines invisible system structure.

UI definitions define visible interaction structure.

Both should stabilize before detailed implementation work begins in their area.

## SkillsKeeper Directives

<!-- skillskeeper-directive: ui-reachability-inventory -->
### UI Reachability Inventory

## UI Reachability Inventory

UI definitions for user-facing features must name how the feature is reached and what the user sees around the route:

- where the user accesses it, such as panel, toolbar, menu, button, tab, file activation, keyboard command, or automatic event;
- what event, command, selection, watcher, or background trigger invokes it;
- visible idle, loading, stale, error, disabled, success, and cancellation states;
- whether it can run while other panels are active.

For multi-panel workbenches, define concurrent visible states, not only one happy path: file scan while preview builds, agent task while code is edited, stale preview while last good render remains visible, or save conflict while context is assembled.
<!-- /skillskeeper-directive: ui-reachability-inventory -->
