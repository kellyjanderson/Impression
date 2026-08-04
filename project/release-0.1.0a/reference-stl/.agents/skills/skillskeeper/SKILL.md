---
name: skillskeeper
description: Use when creating, editing, installing, disabling, archiving, deleting, syncing, or inspecting agent/Codex skills on this machine. Routes skill work through SkillsKeeper so skills are backed up, mirrored into Codex, and preserved in watched .agents folders instead of being edited only in generated ~/.codex copies.
---

# SkillsKeeper

Use SkillsKeeper whenever work touches skills: new skills, skill edits, installation into Codex, disable/enable decisions, archives, restoration, or cleanup.

## Current Live Setup

SkillsKeeper is installed and expected to be running as a LaunchAgent:

```sh
skillskeeper service status
```

The current state file is:

```text
~/Library/Application Support/SkillsKeeper/state.json
```

The active private datastore is:

```text
~/Documents/Projects/skillsKeeper-datastore
```

The currently watched workspaces are expected to include:

```text
~/Documents/Projects
~/Documents/Projects/system-agent-skills-spec
```

Verify live truth before making non-trivial changes:

```sh
skillskeeper status
skillskeeper list
skillskeeper skill list
```

If the installed `skillskeeper` command does not expose a command described here, use the project module from `~/Documents/Projects/skillsKeeper` as the source of truth and reinstall/update the service runtime before relying on the command.

## API-First Skill Management

Manage skills through the `skillskeeper` API whenever the requested change fits
an available command. Do not create, copy, rename, alias, disable, enable,
archive, delete, or mirror skills by hand when SkillsKeeper has a command for
that operation.

Use these APIs as the default entry points:

```sh
skillskeeper skill validate /path/to/skill
skillskeeper skill add /path/to/source-skill --global
skillskeeper skill add /path/to/source-skill --workspace /path/to/workspace
skillskeeper skill add /path/to/source-skill --global --name <managed-alias>
skillskeeper skill update /path/to/source-skill --global
skillskeeper skill update /path/to/source-skill --workspace /path/to/workspace
skillskeeper skill update /path/to/source-skill --global --name <managed-alias>
skillskeeper skill directive add <skill-name> --global --title "Directive title" --body "Directive text"
skillskeeper skill directive remove <skill-name> --global --title "Directive title"
skillskeeper skill enable <skill-name>
skillskeeper skill disable <skill-name>
skillskeeper codex-sync
skillskeeper sync
```

Use `--name` with `skill add` or `skill update` when the user wants the same
skill installed under another managed identity or alias. This creates a
SkillsKeeper-managed source and mirror; do not hand-maintain alias folders under
`~/.codex/skills`.

Manual file edits are appropriate only for the content of an existing managed
source skill, such as changing `SKILL.md`, adding templates, or editing
`agents/openai.yaml`. After manual content edits, run validation and sync:

```sh
skillskeeper skill validate /path/to/managed-skill
skillskeeper sync
```

## Where To Edit Skills

Prefer durable watched source folders over generated mirrors.

- Project/shared skills: edit `~/Documents/Projects/.agents/skills/<skill-name>/`
  only for content changes that do not have a more specific SkillsKeeper API.
- Workspace-local skills: edit `<workspace>/.agents/skills/<skill-name>/` only
  for content changes when that workspace is watched by SkillsKeeper.
- Local agent-wide skills outside Projects: use `~/.agents/skills/<skill-name>/` only when the skill is intentionally home-local and not meant to be mirrored through Projects.
- Do not make durable edits only in `~/.codex/skills/keld-*`; those are generated mirrors from `~/Documents/Projects/.agents/skills`.

If a skill should be available to future Codex sessions as a shared Keld skill, install it under:

```text
~/Documents/Projects/.agents/skills/<skill-name>/
```

Prefer the managed add command when copying a new or external skill into place:

```sh
skillskeeper skill add /path/to/source-skill --global
```

Use update, not add, when the target skill already exists:

```sh
skillskeeper skill update /path/to/source-skill --global
skillskeeper skill update /path/to/source-skill --workspace /path/to/workspace
```

For an already-in-place edit, run:

```sh
skillskeeper sync
```

This archives watched skills into the datastore and mirrors Projects skills into `~/.codex/skills/keld-<skill-name>` with rewritten metadata names.

## Normal Workflows

Create or edit a shared skill:

1. Validate the source skill:

```sh
skillskeeper skill validate /path/to/source-skill
```

2. Install it into the shared Projects source with automatic validation, archive, and Codex mirror:

```sh
skillskeeper skill add /path/to/source-skill --global
```

3. To replace an existing shared skill from a revised source directory, use:

```sh
skillskeeper skill update /path/to/source-skill --global
```

4. For direct edits to an existing shared skill, edit `~/Documents/Projects/.agents/skills/<skill-name>/`, then run:

```sh
skillskeeper skill validate ~/Documents/Projects/.agents/skills/<skill-name>
skillskeeper sync
```

5. Confirm it was archived and mirrored:

```sh
test -f ~/Documents/Projects/skillsKeeper-datastore/registered/Documents-Projects/agents-skills/<skill-name>/SKILL.md
test -f ~/.codex/skills/keld-<skill-name>/SKILL.md
```

Create or edit a workspace-local skill:

1. Install a source skill into a workspace:

```sh
skillskeeper skill add /path/to/source-skill --workspace /path/to/workspace
```

If `--workspace` is omitted, `skill add` uses the current directory as the caller workspace. It auto-registers that workspace by default so the new skill is watched and preserved.

2. To replace an existing workspace skill from a revised source directory, use:

```sh
skillskeeper skill update /path/to/source-skill --workspace /path/to/workspace
```

3. For direct edits to an existing workspace skill, edit `<workspace>/.agents/skills/<skill-name>/`, then run:

```sh
skillskeeper skill validate <workspace>/.agents/skills/<skill-name>
skillskeeper sync
```

4. Confirm the datastore copy under `registered/<workspace-key>/agents-skills/<skill-name>/`.

Register a workspace before relying on preservation:

```sh
skillskeeper register /path/to/workspace
skillskeeper sync
```

Add or remove a targeted directive:

```sh
skillskeeper skill directive add <skill-name> --global --title "Directive title" --body "Directive text"
skillskeeper skill directive add <skill-name> --workspace /path/to/workspace --title "Directive title" --body-file directive.md
skillskeeper skill directive remove <skill-name> --global --title "Directive title"
skillskeeper skill directive remove <skill-name> --workspace /path/to/workspace --title "Directive title"
```

Directives are appended to `SKILL.md` under `## SkillsKeeper Directives` with marker comments. Titles are the removal handle and must be unique per skill.

## Disable, Enable, Archive, Delete

Do not move active skill folders by hand when the intent is disablement or archiving. Use SkillsKeeper so runtime copies, datastore copies, and Codex mirrors stay consistent.

Disable globally:

```sh
skillskeeper skill disable <skill-name>
```

Disable for one watched workspace:

```sh
skillskeeper skill disable <skill-name> --workspace /path/to/workspace
```

Enable again with the matching scope:

```sh
skillskeeper skill enable <skill-name>
skillskeeper skill enable <skill-name> --workspace /path/to/workspace
```

Intentionally archive an active datastore skill:

```sh
skillskeeper archive /path/to/workspace agents-skills <skill-name>
skillskeeper archive /path/to/workspace agents-local <skill-name>
```

Remove a skill from the current datastore set while preserving git history:

```sh
skillskeeper delete-current /path/to/workspace agents-skills <skill-name> --confirm delete-current
```

Use `--no-push` only when the user explicitly wants to keep datastore changes local.

## Safety Rules

- Check `skillskeeper status` and `skillskeeper list` before changing watched skill locations.
- Treat `~/.codex/skills/keld-*` as generated output; use SkillsKeeper APIs or edit the managed source in `~/Documents/Projects/.agents/skills`.
- Use `skillskeeper skill validate <path>` before and after manual skill edits.
- Use `skillskeeper skill add` only for new skills. Use `skillskeeper skill update` to replace existing skills.
- Use `skillskeeper skill add/update --name <managed-alias>` for managed aliases instead of hand-copying alias folders.
- Use `skillskeeper skill directive add/remove` for small titled append-only directives instead of hand-editing directive blocks.
- If a skill disappears unintentionally, inspect the datastore archive before recreating it:

```sh
find ~/Documents/Projects/skillsKeeper-datastore/archived -path '*/<skill-name>/SKILL.md'
```

- Do not bypass SkillsKeeper for disable/archive/delete operations.
- Do not hand-edit `~/Library/Application Support/SkillsKeeper/state.json` unless repairing a broken state file; prefer CLI commands.
- Runtime files such as `.system-skills-composer.json`, `.composer-state.json`, `.DS_Store`, and `__pycache__` are intentionally skipped.
