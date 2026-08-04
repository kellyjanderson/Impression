# Hygiene And Updates

Use this checklist when entering an existing workspace or growing a small project.

## Orientation

Run fast checks:

```sh
pwd
git status --short --branch
git remote -v
find . -maxdepth 2 \( -name README.md -o -name AGENTS.md -o -name Agents -o -name agents -o -name .agents -o -name project -o -name pyproject.toml -o -name package.json -o -name requirements.txt \)
```

Read only the files needed to understand the current layer:

- `README.md`
- `AGENTS.md`, `Agents/`, `agents/`, `.agents/`
- `project/README.md`
- dependency manifests
- existing `.gitignore`

## Hygiene Checks

Look for:

- missing git repo;
- git status that is only coming from an ancestor mass-backup repo instead of the actual workspace root;
- missing or weak `.gitignore`;
- generated artifacts mixed with source;
- untracked useful source files;
- committed `.venv`, `node_modules`, caches, logs, build directories, or local runtime state;
- no README or no setup notes;
- source files at root that should be grouped once the project grows;
- 3D exports with unclear policy: primary artifact, checkpoint, or disposable output.

## Update Rules

- Add the smallest structure that solves the immediate problem.
- Keep current project naming unless it is actively misleading.
- Do not reorganize large trees during unrelated work.
- Commit hygiene/scaffold changes separately.
- Preserve user-created untracked files unless the user explicitly asks for cleanup.
- If a cleanup would delete or move data, archive or commit first.

## Common Upgrades

Add `.gitignore` when:

- a repo is new;
- generated artifacts are present;
- dependencies or virtualenvs exist.

Add `README.md` when:

- setup or usage is non-obvious;
- a future agent needs project purpose and commands.

Add `scripts/` when:

- commands are repeated;
- setup, build, export, or validation needs automation.

Add `tests/` when:

- project behavior can regress;
- a reusable library or CLI appears.

Add `project/` when:

- research, architecture, specs, planning, or release notes exist.

Add `.agents/` or `agents/` when:

- project-local agent instructions are needed;
- reusable project skills exist;
- the workspace has rules that differ from home-level rules.

## Existing Local Patterns To Preserve

- Full software/spec projects often use `project/{research,architecture,specifications,planning}` plus `docs`, `scripts`, `src`, and `tests`.
- 3D/modeling work often starts as a folder of source models and exports; protect it first with git, then sort sources and generated outputs.
- `~/.local/apps` projects often have runtime output, logs, models, captures, and vendor code. Treat these as local-machine app workspaces, not general source repos.
- `~/.local/bin`, `~/.local/log`, `~/.local/run`, `~/.local/share`, and `~/.local/state` are support locations; do not turn them into project roots unless the user asks.
