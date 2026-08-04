---
name: impression
description: Use when asked to build, edit, or explain models with the Impression parametric modeling framework. First apply workspace-stewardship for 3D/CAD undo-stack commit discipline, then parametric-modeling for composition, encapsulation, local geometry, and exact parametric structure; then prefer repo docs/examples, keep modeling inside `impression.modeling`, follow the project's surface-first posture, and treat loft as a surfaced feature centered on `Loft(...)`, planner inspection, and explicit consumer-boundary tessellation.
---

# Impression

Use this skill when working in a repo or docs bundle that uses the Impression
modeling framework.

## Skill Dependency

Before making or reviewing Impression model changes, apply
`workspace-stewardship`, then `parametric-modeling`.

For 3D modeling, CAD, and other edit-heavy model work, `workspace-stewardship`
owns local git durability and commit-as-undo-stack discipline. Do not begin
substantive Impression edits until the workspace git state and commit policy
have been checked. If the project is a working-document modeling project,
checkpoint with local commits before risky edits and after each meaningful
geometry or generated-artifact checkpoint.

This Impression skill adds framework-specific API, docs, preview, and
surface-first rules on top of that workspace and parametric modeling discipline.

## Quick Start

1. Apply `workspace-stewardship` for git state and local checkpoint policy.
2. Apply `parametric-modeling` for composition and encapsulation.
3. If the workspace contains `docs/index.md`, read it first.
4. Open the relevant page under `docs/modeling/`.
5. If the task is loft-related, read `docs/agents/loft.md` or this skill's
   `references/loft.md` before coding.
6. Open the closest runnable example under `docs/examples/`.
7. Build with `impression.modeling`, not PyVista primitives.
8. Keep results app-owned until preview/export or another explicit consumer
   needs tessellated mesh output.

## Core Rules

- Return Impression modeling outputs from `build()`, not raw PyVista datasets.
- Prefer existing Impression capabilities over inventing custom geometry logic.
- Treat PyVista as a viewer only.
- Keep surface-first features surfaced until a consumer boundary.
- Keep an existing `impression preview` process running when one is already
  open. Let watched files hot reload; when a change in a linked/imported file
  is not picked up automatically, trigger the preview's model reload/switch
  mechanism instead of stopping and restarting the preview.
- Treat loft as the canonical path-driven body-construction lane in the current
  product, not as a mesh-first helper and not as a separate sweep/pipe family.
- Do not resurrect retired public modeling surfaces such as public `morph` or
  public `extrude` APIs unless the repo docs explicitly bring them back.
- If repo docs and this skill disagree, trust the repo docs.

## Feature Selection

Read [references/feature-map.md](references/feature-map.md) when you need help
choosing between primitives, drawing2d, topology, loft, text, threading,
hinges, drafting, heightmaps, `Path3D`, or CSG.

## Loft

For loft work, always read [references/loft.md](references/loft.md).

The short version:

- prefer `Loft(...)`
- use `loft(...)` and `loft_sections(...)` as convenience APIs over the same planner
- inspect ambiguity with `loft_plan_ambiguities(...)`
- feed explicit `candidate_id` choices back through interactive ambiguity controls
- tessellate loft output only at an explicit consumer boundary
- use the real-world hourglass example when you need a strong surfaced loft
  reference model

## SkillsKeeper Directives

<!-- skillskeeper-directive: aim-camera-at-the-object-under-inspection -->
### Aim Camera At The Object Under Inspection

When visually inspecting a specific modeled object or subassembly, compute that object's global bounds or center after all placement transforms and aim the camera at that point. Do not aim only at the scene origin or rely on a generic view when the question is about a named feature. Camera path selection and finding an unobstructed viewpoint may require additional collision or visibility analysis; this directive only requires that the camera target be the inspected object's global position.
<!-- /skillskeeper-directive: aim-camera-at-the-object-under-inspection -->

<!-- skillskeeper-directive: verify-movement-with-global-center-delta -->
### Verify Movement With Global Center Delta

When changing or debugging the position of a modeled object or subassembly, compute the object's global bounds or center before and after the change and verify the center changed in the intended direction and approximate amount. Do this for the placed object in the consuming assembly, not only for the local module preview. Use the center delta as the primary truth for whether the object moved; visual preview then confirms orientation, visibility, and surrounding fit.
<!-- /skillskeeper-directive: verify-movement-with-global-center-delta -->
