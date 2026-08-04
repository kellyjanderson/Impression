---
name: workspace-python
description: Use whenever running Python commands, scripts, tests, validators, model exports, or tooling inside a local workspace or repository. Default to the nearest workspace or repo `.venv` Python interpreter instead of system Python unless the user or project explicitly requires another interpreter.
---

# Workspace Python

Use this skill whenever you need to run Python in a local workspace or
repository.

## Core Rule

Default to the local workspace/repo virtual environment.

Before running `python`, `python3`, `pip`, test commands, validation scripts,
model exports, or project tooling, check for a nearby `.venv` and prefer its
interpreter:

```bash
.venv/bin/python ...
```

If the current directory does not contain `.venv`, look upward through the
workspace/repo parents before falling back to system Python.

## Why

System Python often lacks project dependencies or has the wrong versions. The
workspace `.venv` is usually the only interpreter that matches the repo's
installed packages, modeling libraries, CLIs, test tools, and generated
artifacts.

## Practical Defaults

- Use `.venv/bin/python -m ...` for module commands.
- Use `.venv/bin/python script.py` for repo scripts.
- Use `.venv/bin/python -m pip ...` instead of bare `pip`.
- Use project-specific wrappers only when the repo clearly establishes them.
- If no `.venv` exists or the `.venv` is unusable, say so and then choose the
  next best interpreter.

## Do Not

- Do not use system `python3` first inside a repo when `.venv` exists.
- Do not install packages into system Python to satisfy a repo command.
- Do not assume the shell alias `python` exists.
- Do not switch interpreters mid-task without a reason.
