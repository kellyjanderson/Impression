# UI Definition Guidance

UI definitions are the visible-behavior analog to system architecture.

They define the durable structure of the interface and interaction model without dropping immediately into implementation detail.

They must also satisfy the mandatory structural rules in:

```text
agents/ui-ux-invariants.md
```

---

## Purpose

UI definitions exist to describe:

* what interface surfaces exist
* what the product should sound like in UI language
* what brand and visual system choices govern the app
* what reusable component families exist
* which controls are primary
* how complexity is layered
* what interaction states mean
* how status is communicated
* how appearance modes and navigation patterns should work
* how the interface preserves layout stability and spatial memory across states

They should capture durable design intent, not transient implementation detail.

---

## When To Use UI Definitions

Use UI definitions when the question is primarily about visible behavior, interaction structure, or interface semantics.

Examples:

* top-level vs secondary controls
* brand voice for product and GitHub-facing surfaces
* typography and color-system choices
* component families such as transport, dialogs, rows, or toasts
* where advanced features should live
* what a disabled-but-communicating control should look like
* how themes and appearance modes should work
* how a live mode should behave from the user's perspective
* what persistent affordances should exist across screens

If the problem is about system structure, use architecture.

If the problem is about exact implementation rules, thresholds, or acceptance criteria, use specifications.

---

## Recommended Structure

UI definition documents should generally include:

### Overview

A short description of the interface area or UI concept.

### Primary Interface Elements

What should be visible and dominant on the primary surface.

### Secondary Access

How advanced or less frequent actions are accessed without cluttering the main interface.

### States And Semantics

The meaning of visible states, control conditions, and status communication.

This must cover more than the happy path.

UI definitions should intentionally describe stable behavior for:

- loading
- empty
- partial
- full
- error
- disabled
- overflow
- responsive or narrow layouts

### Appearance And Accessibility

Theme behavior, contrast requirements, and other durable visual rules.

### Component Guidance

Reusable UI objects, their roles, and their structural rules.

### Related Specifications

References to narrower specifications that implement parts of the UI definition.

---

## Relationship To Architecture

Architecture defines invisible system structure.

UI definitions define visible interaction structure.

Both sit above specifications and should stabilize before detailed implementation work begins in their area.

UI definitions should not allow structural instability to remain implicit.

If a surface can shift, collapse, disappear, or reorder as data changes, the UI definition should describe how stability is preserved.

Specifications should refine a coherent UI definition rather than inventing interface behavior ad hoc during implementation.

For broad UI work, the preferred order is:

1. brand and voice rules
2. design guide and look-and-feel rules
3. component system
4. surface-specific UI definitions
5. UI specifications

At each layer, agents should preserve the invariants in `agents/ui-ux-invariants.md`.

---

## Stability

UI definitions should change less often than implementation-level specifications.

They are intended to preserve coherent interface behavior across multiple implementation rounds.
