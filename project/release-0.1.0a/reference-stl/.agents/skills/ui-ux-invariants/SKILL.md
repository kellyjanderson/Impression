---
name: ui-ux-invariants
description: Enforce mandatory structural UI rules for stability, spatial memory, state coverage, clarity, accessibility, and resilient interaction behavior.
---

# UI/UX Invariants

These are mandatory structural rules for UI work.

## Core Invariants

Protect all of the following:

* layout stability across states
* spatial memory for repeated actions and controls
* intentional design for loading, empty, partial, full, error, disabled, overflow, and responsive states
* stable containers instead of disappearing structure
* explicit empty states
* consistent action hierarchy
* clear and stable control semantics
* standard interaction patterns
* visible system state
* accessible interaction
* responsive behavior that preserves meaning
* resilience under messy real-world content
* timely local feedback

## Local Promise Rule

If a control is presented as available, invoking it should usually perform the expected action immediately.

Do not expose controls that appear usable but fail because of ordinary background state the system could resolve automatically.

## Conflict Resolution Rule

When a user-triggered action conflicts with an existing reversible state, prefer automatic resolution over rejection when safe to do so.

Use errors only when the system cannot safely infer intent or automatic handling would create serious risk.

## Implementation Rule

When generating UI code:

* do not remove structural containers solely because data is absent
* reserve space where later content would otherwise shift layout
* prefer in-place placeholders and empty states
* keep core control regions in stable positions
* do not optimize for compactness at the expense of stability
