# Impression Agent Usage Guide

This guide is for agents that need to **use Impression to build models**.
Repository-development instructions are maintained in `agents/` and the
project-level planning workspace under `project/`.

## First Steps

1. Read the modeling docs before writing code:
   - [`docs/modeling/`](modeling/)
   - [`docs/examples/`](examples/)
2. Prefer Impression APIs over external modeling libraries.
3. Return Impression mesh objects from `build()` (not raw PyVista datasets).

## Modeling Rules

- Keep geometry creation in `impression.modeling`.
- Use CLI preview/export for fast feedback loops:
  - `impression preview path/to/model.py`
  - `impression export path/to/model.py --output out.stl`
- Reuse existing capabilities (primitives, CSG, drawing2d, loft, threading,
  hinges, transforms, groups, drafting, text) before inventing new approaches.

## Documentation Navigation

- Overview: [`docs/index.md`](index.md)
- Core APIs: [`docs/modeling/`](modeling/)
- Runnable references: [`docs/examples/`](examples/)
- CLI behavior: [`docs/cli.md`](cli.md)

## Recommended Workflow For Agent Sessions

1. Identify the target feature in `docs/modeling`.
2. Open a close example from `docs/examples`.
3. Implement by adapting existing APIs.
4. Preview and iterate.
5. Export and validate mesh quality when required.

## Prompt Bootstrap (optional)

If you use a bootstrap prompt for external agents, you can source it from
`.vscode/impression.code-snippets` (`intro`) or maintain one in your project docs.
