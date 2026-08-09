# Tutorial - Serious Modeling Workflow

This tutorial is retired.

Its previous walkthrough depended on public extrusion helpers, and those helpers
should no longer be presented as an active modeling path for Impression.

Use these lanes instead:

- loft for profile-to-profile transitions
- surface-first primitives for bounded solid construction
- surface-only public booleans for supported exact CSG routes
- explicit `impression.modeling.mesh_tools` utilities only for downstream mesh
  analysis, repair, and debugging

This page remains in place only so existing links fail honestly instead of
silently teaching a workflow the project is removing.
