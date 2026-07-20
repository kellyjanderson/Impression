---
name: impression
description: Use when asked to build, edit, or explain models with the Impression parametric modeling framework. Prefer repo docs/examples when present, keep modeling inside `impression.modeling`, follow the project's surface-first posture, and treat loft as a surfaced feature centered on `Loft(...)`, planner inspection, and explicit consumer-boundary tessellation.
---

# Impression

Use this skill when working in a repo or docs bundle that uses the Impression
modeling framework.

## Quick Start

1. If the workspace contains `docs/index.md`, read it first.
2. Open the relevant page under `docs/modeling/`.
3. Open the closest runnable example under `docs/examples/`.
4. Build with `impression.modeling`, not PyVista primitives.
5. Keep results app-owned until preview/export or another explicit consumer
   needs tessellated mesh output.

## Core Rules

- Return Impression modeling outputs from `build()`, not raw PyVista datasets.
- Prefer existing Impression capabilities over inventing custom geometry logic.
- Treat PyVista as a viewer only.
- Keep surface-first features surfaced until a consumer boundary.
- If repo docs and this skill disagree, trust the repo docs.

## Feature Selection

Read [references/feature-map.md](references/feature-map.md) when you need help
choosing between primitives, drawing2d, topology, loft, text, threading,
hinges, drafting, heightmaps, or CSG.

## Loft

For loft work, always read [references/loft.md](references/loft.md).

The short version:

- prefer `Loft(...)`
- use `loft(...)` and `loft_sections(...)` as convenience APIs over the same planner
- inspect ambiguity with `loft_plan_ambiguities(...)`
- feed explicit `candidate_id` choices back through interactive ambiguity controls
- tessellate loft output only at an explicit consumer boundary
