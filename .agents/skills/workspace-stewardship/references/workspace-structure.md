# Workspace Structure

Use the lightest structure that preserves the work. Upgrade when current work creates a real need.

## Observed Local Roots

- `~/Documents/Projects`: primary project space, including full software repos, shared agent skills, 3D modeling folders, and mixed experiments.
- `~/Documents/Projects/3d printing`: edit-heavy model workspace. Many folders contain `.scad`, `.stl`, `.3mf`, Python model scripts, generated outputs, and partial git coverage.
- `~/.local/apps`: local machine apps, services, vendor clones, experiments, logs, state, and tool-specific sandboxes.
- `~/.local/{bin,docs,log,run,scripts,share,state,system}`: operational support, not normal project source roots.

## Variant 0: Scratch

Use for a one-off experiment that may be thrown away within the session.

Minimum:

```text
<name>/
  README.md
  .git/
  .gitignore
```

Rules:

- Still initialize git if an agent will edit files.
- Keep generated outputs ignored unless they are the primary artifact.
- Promote to Lightweight as soon as a second file type, dependency, or reusable result appears.

## Variant 1: Lightweight

Use for small tools, single-model 3D projects, scripts, and local experiments that need history but not a full planning layer.

Minimum:

```text
<name>/
  README.md
  .git/
  .gitignore
  src/ or models/ or scripts/
  dist/ or output/        # ignored unless final artifacts are intentionally tracked
```

3D lightweight projects may use:

```text
<name>/
  README.md
  .git/
  .gitignore
  models/                # source model files: .scad, .py, .blend, .FCStd, .step when editable
  stl/                   # exported print mesh checkpoints, tracked selectively
  gcode/                 # usually ignored
  dist/                  # generated previews/exports, usually ignored
  references/            # photos, measurements, datasheets
```

## Variant 2: Standard

Use for active software projects, reusable modeling libraries, local apps, and anything with tests or repeatable build steps.

Recommended:

```text
<name>/
  README.md
  .git/
  .gitignore
  src/ or app/
  tests/
  docs/
  scripts/
  project/
    research/
    architecture/
    specifications/
    planning/
  .agents/ or agents/
```

Rules:

- Add `project/` only when durable planning, research, architecture, or specs exist.
- Keep `agents/` or `.agents/` only when local project instructions or skills are needed.
- Keep generated test artifacts in an ignored folder with a tracked README if humans need to know the folder exists.

## Variant 3: Full

Use for projects that need durable architecture, specifications, test specifications, release definitions, agent handoff, or PR-style implementation.

Recommended:

```text
<name>/
  README.md
  .git/
  .gitignore
  .github/
  src/ or app/
  tests/
  docs/
  examples/
  scripts/
  assets/
  project/
    README.md
    research/
    architecture/
    releases/
    specifications/
    planning/
    documentation/
    issues/
    agents/
  .agents/
    skills/
```

Rules:

- Do not start here for small work unless the user asks.
- Upgrade to Full when multiple agents, specifications, releases, or long-lived docs become part of the work.
- Keep implementation artifacts and planning artifacts separated.

## Variant 4: Local App Or Service

Use under `~/.local/apps/<name>` for personal machine services, launchd-backed tools, local automation, or apps coupled to local paths and state.

Recommended:

```text
~/.local/apps/<name>/
  README.md
  .git/
  .gitignore
  src/ or app/
  tests/
  scripts/
  project/               # when research/specs exist
  docs/
```

Related runtime locations:

```text
~/.local/bin             # command shims/symlinks
~/.local/log/<name>      # logs
~/.local/run/<name>      # pid files, sockets, temp runtime
~/.local/state/<name>    # durable machine-local state
~/.local/system/services # service definitions and launchd notes
```

Rules:

- Keep logs, caches, model weights, captures, and runtime output out of git unless explicitly needed as fixtures.
- Avoid committing secrets or machine-specific credentials.

## Upgrade Triggers

Promote Scratch to Lightweight when:

- more than one editing session occurs;
- an agent will modify files;
- the work produces a useful artifact.

Promote Lightweight to Standard when:

- tests, dependencies, docs, or scripts appear;
- the project needs repeatable setup;
- generated outputs need policy instead of ad hoc naming.

Promote Standard to Full when:

- multiple features are planned;
- architecture/specification/release work exists;
- multiple agents need durable handoff context.
