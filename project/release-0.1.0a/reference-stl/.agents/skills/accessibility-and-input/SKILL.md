---
name: accessibility-and-input
description: Use when work touches keyboard access, focus indication, contrast, touch targets, zoom behavior, pointer alternatives, or perceivability. Apply whenever an interface must remain operable beyond the ideal mouse-and-vision case.
---

# Accessibility And Input

Use this skill when accessibility needs to be treated as a first-order interface requirement.

## Use This For

* keyboard navigation and focus visibility
* contrast and perceivability checks
* non-mouse interaction support
* touch target and gesture considerations
* zoom, scaling, and adaptable UI behavior

## Hard Rules

* keyboard focus must be visible
* text contrast should meet at least WCAG AA
* meaningful non-text cues should meet at least `3:1`
* color should not be the only carrier of essential meaning
* important tasks should not require one precise input mode

## Workflow

1. Identify the supported input modes.
2. Check keyboard paths and focus visibility.
3. Check contrast and non-color fallbacks.
4. Check pointer, touch, and zoom behavior.
5. Use `../ui-ux-invariants/SKILL.md` as the final structural gate.

## References

* Read `references/accessibility-basics.md` for the concrete checklist.
