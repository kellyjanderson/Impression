# Modeling - Extrusions

Public extrusion modeling is being retired and should not be presented as an
active modeling path for Impression.

This page remains only as a tombstone so older links fail honestly instead of
silently presenting stale guidance.

Current direction:

- use surface-first primitives and loft for active 3D modeling guidance
- treat legacy public `linear_extrude()` and `rotate_extrude()` usage as
  removal work, not recommended workflow
- avoid adding new public examples, tutorials, or tests that depend on
  extrusion as a supported modeling capability

Scoped follow-up outside this file:

- remove or redirect remaining references from general docs indexes and
  tutorials that still point here
- remove the public extrusion exports once the implementation removal track is
  ready
