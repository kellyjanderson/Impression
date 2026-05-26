---
name: ui-review
description: Use when reviewing a UI, interaction change, mockup, or implementation for bugs, regressions, weak interaction semantics, accessibility gaps, or broken design invariants. Findings should come first and be grounded in visible behavior.
---

# UI Review

Use this skill for interface critique and pre-ship review.

## Primary Goal

Find structural, behavioral, accessibility, and hierarchy problems before they ship.

## Review Order

1. layout stability and spatial memory
2. control semantics and action hierarchy
3. state visibility and feedback
4. accessibility and input support
5. visual hierarchy and scanability
6. real-world content robustness
7. overflow, clipping, and narrow-width behavior

## Output Shape

Findings first, ordered by severity, with file or surface references when possible.

Then include:

* open questions
* assumptions
* brief summary only after findings

## Pairings

* Use `../ui-ux-invariants/SKILL.md` as the structural baseline.
* Use `../visual-design-foundations/SKILL.md` for hierarchy and composition critique.
* Use `../accessibility-and-input/SKILL.md` when interaction or accessibility is involved.

## References

* Read `references/review-checklist.md` for the detailed review rubric.
