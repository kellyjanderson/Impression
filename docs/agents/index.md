# Impression Agent Guide

This folder is for agents that need to **use Impression to build models**.
Repository-development process rules live under the root [agents](../../agents/index.md)
folder and the project workspace under [`project/`](../../project/README.md).

## Start Here

1. Read the modeling docs before writing code:
   - [`../modeling/`](../modeling/)
   - [`../examples/`](../examples/)
2. Prefer Impression APIs over external modeling libraries.
3. Return Impression modeling outputs from `build()`, not raw PyVista datasets.
4. Keep geometry app-owned as long as possible and only tessellate at an
   explicit consumer boundary.

## Core Rules

- Keep geometry creation inside `impression.modeling`.
- Use CLI preview/export for fast feedback loops:
  - `impression preview path/to/model.py`
  - `impression export path/to/model.py --output out.stl`
- Reuse existing capabilities before inventing new geometry workflows.
- Treat PyVista as a viewer, not as the modeling truth.
- When the repo ships Impression docs/examples, prefer them over memory.

## Feature Map

- [Loft](loft.md)
  - surface-first lofting, explicit station authoring, ambiguity planning
- [`../modeling/primitives.md`](../modeling/primitives.md)
  - boxes, cylinders, spheres, cones, torus, and basic surfaced solids
- [`../modeling/drawing2d.md`](../modeling/drawing2d.md)
  - 2D profiles and paths for loft, text, and planar construction
- [`../modeling/topology.md`](../modeling/topology.md)
  - sections, regions, loops, and profile normalization
- [`../modeling/text.md`](../modeling/text.md)
  - glyph outlines, surfaced text bodies, and text sections
- [`../modeling/threading.md`](../modeling/threading.md)
  - surfaced thread operands and assemblies
- [`../modeling/hinges.md`](../modeling/hinges.md)
  - traditional, living, and bistable hinge outputs
- [`../modeling/csg.md`](../modeling/csg.md)
  - boolean helpers and the current surfaced CSG posture

## Recommended Workflow

1. Identify the target feature in [`../modeling/`](../modeling/).
2. Open a close runnable example from [`../examples/`](../examples/).
3. If the task is loft-related, read [loft.md](loft.md) before coding.
4. Implement by adapting existing APIs.
5. Preview and iterate.
6. Export or tessellate only when the consumer actually needs mesh output.

## Skill

This repo ships an installable Codex skill bundle under:

- [`../skills/impression/`](../skills/impression/)

Install instructions live at:

- [`../skills/index.md`](../skills/index.md)

Once installed, agents can invoke:

- `$impression`

The workspace docs remain the source of truth when they are present in the workspace.
