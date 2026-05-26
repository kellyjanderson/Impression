---
name: interaction-state-design
description: Use when defining or revising hover, focus, pressed, selected, disabled, loading, empty, success, warning, or error behavior. Apply whenever a component or surface changes over time or in response to input.
---

# Interaction State Design

Use this skill when UI quality depends on clear dynamic behavior.

## Use This For

* component state modeling
* feedback timing and strength
* empty, loading, success, and error behavior
* interaction cues that must remain legible
* keeping state changes understandable without guesswork

## State Expectations

* components should expose their key states clearly
* state changes should use more than one cue when important
* routine feedback should stay lightweight
* risky or destructive feedback should be harder to miss
* recovery paths should be visible when failure occurs

## Workflow

1. List the states that matter for the component or surface.
2. Define how each state looks and how it is entered.
3. Define what feedback is local versus global.
4. Check that important states do not depend on color alone.
5. Verify continuity with `../ui-ux-invariants/SKILL.md`.

## References

* Read `references/state-model.md` for the expanded checklist.
