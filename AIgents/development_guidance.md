# Impression Development Guidance for AI Agents

This file contains repository-development guidance for agents working **on Impression itself**.
For usage guidance (building models with Impression), see `docs/agents.md`.

## Pre-Action Rule

Review `AIgents/` before taking action and apply any relevant templates/instructions.

## Core Development Rules

- Use Impression internal meshes and helpers. Do not return PyVista datasets from `build()`.
- PyVista is a view layer only. Modeling, boolean ops, and mesh generation stay in Impression.
- Reuse existing utilities before adding dependencies or new geometry code.
- Optimize for real-world utility and polish. Outputs should not only work; they should clearly
  demonstrate why Impression is powerful.

Project quality bar: `docs/project-dna.md`

## Repository Feature Baseline

Before implementing a feature, verify existing support in:

- `docs/modeling/`
- `docs/examples/`
- `src/impression/modeling/`

Current major capabilities include primitives, CSG, 2D drawing, extrusions, paths, morph,
loft, threading, hinges, transforms, groups, drafting, preview, and STL export.

## Development Workflow

```bash
git clone https://github.com/kellyjanderson/Impression.git
cd Impression
python3 -m venv .venv
source .venv/bin/activate
scripts/dev/install_impression.sh --local
impression --help
```

## Additions and Dependencies

Before writing new code:

1. Check docs and examples for existing helpers.
2. Search for similar utilities in source.
3. Prefer extending internal mesh APIs over adding dependencies.

If a new dependency is unavoidable:

- document it in `docs/index.md`
- update both `requirements.txt` and `pyproject.toml` in the same change
