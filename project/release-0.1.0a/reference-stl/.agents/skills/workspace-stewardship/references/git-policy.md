# Git Policy

## Mandatory Local Git

Before editing a project workspace:

```sh
git status --short --branch
```

This check must apply to the actual project workspace root, not merely to an ancestor directory.

If `git status` succeeds only because a parent folder is a mass backup repo, that parent repo does not count as valid local project history. Treat the workspace as missing git and initialize or repair a repo at the real workspace root before editing.

If the command fails because no repo exists at the real workspace root:

```sh
git init
```

Then create or update `.gitignore`, add the initial durable files, and make an initial local commit.

## Remote Policy

Default to no remote for new local projects.

Create a private GitHub remote only when:

- the user explicitly asks;
- the work needs cross-machine backup now;
- PR review, issue tracking, CI, or sharing is expected;
- the project is a reusable tool/library rather than a disposable local workspace.

When creating a GitHub repo, default to private unless the user explicitly asks for public.

## Commit Policies

### Working-Document Policy

Use for 3D modeling, CAD, graphics, media, notebooks, measurement logs, exploratory artifacts, and similar work.

Rules:

- Commit before edits if there are existing changes.
- Commit after every meaningful edit, export, or checkpoint.
- Use local commits as the undo stack.
- Do not squash unless the user asks.
- Keep generated files ignored by default, but track important exported checkpoints when they are the artifact the user cares about.

Suggested messages:

```text
Initial workspace scaffold
Checkpoint model bearing mount
Update socket dimensions
Export print checkpoint
Add reference measurements
```

### Software Project Policy

Use for codebases, libraries, apps, services, and docs-heavy repos.

Rules:

- Commit coherent changes, not every keystroke.
- Keep structure/hygiene commits separate from feature commits.
- Run relevant tests or smoke checks before commit when practical.
- Do not commit generated build output, virtualenvs, dependency folders, logs, caches, or secrets.

Suggested messages:

```text
Initialize project scaffold
Add workspace hygiene docs
Implement <feature>
Fix <bug>
Update tests for <behavior>
```

## .gitignore Baseline

Start with macOS and local editor/runtime noise:

```gitignore
.DS_Store
._*
.AppleDouble
.LSOverride
.Spotlight-V100
.Trashes
.fseventsd
.DocumentRevisions-V100
.TemporaryItems

.history/
.vscode/
.idea/
*.log
tmp/
temp/
```

Python:

```gitignore
__pycache__/
*.py[cod]
.venv/
venv/
.pytest_cache/
.mypy_cache/
.ruff_cache/
*.egg-info/
build/
dist/
```

Node:

```gitignore
node_modules/
.next/
coverage/
npm-debug.log*
yarn-debug.log*
yarn-error.log*
```

Local app/service:

```gitignore
logs/
run/
state/
out/
captures/
*.pid
*.sock
.env
.env.*
!.env.example
```

3D/modeling:

```gitignore
gcode/
*.gcode
*.bgcode
*.tmp
*.bak
```

Do not blindly ignore all `.stl`, `.3mf`, `.step`, or `.obj` files. For 3D projects, decide whether exports are primary artifacts, review checkpoints, or disposable build products.

## Pre-Edit Checklist

1. Confirm repo root. Ancestor mass-backup repos are not valid project roots.
2. Run `git status --short --branch`.
3. If git only resolves through an ancestor backup repo, initialize or repair git in the workspace itself.
4. If dirty, understand whether changes are user work; commit, preserve, or work around them.
5. Check `.gitignore` before generating files.
6. For working-document projects, make a checkpoint commit before editing.
