---
name: color-system-design
description: Use when choosing or revising palettes, semantic color roles, state colors, theme tokens, or contrast behavior. Apply whenever color carries meaning, status, emphasis, or interaction state.
---

# Color System Design

Use this skill when color needs to be treated as a semantic system instead of decoration.

## Use This For

* defining palette roles
* assigning colors to UI semantics
* checking contrast and mode adaptability
* clarifying selected, disabled, success, warning, and destructive states
* reducing inconsistent or overloaded color use

## Core Rules

* map color to role before hue
* keep color meaning consistent
* do not rely on color alone for important meaning
* maintain accessible contrast
* ensure the system survives light, dark, and higher-contrast conditions

## Workflow

1. List the required roles: background, surface, text, accent, success, warning, destructive, selected, disabled.
2. Assign colors by meaning, not preference.
3. Check contrast and backup cues.
4. Reduce duplicate or conflicting color semantics.
5. Verify the palette works across states and themes.

## References

* Read `references/color-rules.md` for the expanded checklist.
* Pair with `../accessibility-and-input/SKILL.md` when contrast or perceivability is critical.
