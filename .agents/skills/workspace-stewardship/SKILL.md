---
name: workspace-stewardship
description: "Initialize, inspect, clean, and evolve local project workspaces. Use when starting a new project, bringing an agent into an existing repo, deciding project layout, adding missing git or .gitignore coverage, codifying workspace hygiene, upgrading a small project into a fuller structure, or working with edit-heavy artifacts such as 3D models that need high-resolution local git history."
---

# Workspace Stewardship

Use this skill to make a workspace durable before doing substantive work.

## Required First Moves

1. Identify the workspace root and whether it lives under `~/Documents/Projects`, `~/.local/apps`, or another user-approved path.
2. Check git state before edits:
   - If the root has no `.git`, initialize git before modifying project files.
   - If git exists, inspect branch, status, remotes, and `.gitignore`.
   - Do not treat an ancestor mass-backup repo as valid project git. The actual workspace root must have its own usable repo.
   - Never overwrite user changes or remove untracked work just to make the tree clean.
3. Select a structure variant from `references/workspace-structure.md`.
4. Select a commit policy from `references/git-policy.md`.
5. Check `.gitignore` against the selected variant before the first commit.

## Git Default

Local git is mandatory for project work. A GitHub remote is opt-in unless the user explicitly asks for one or the project is clearly meant to be shared, deployed, backed up remotely, or reviewed by PR.

When creating a remote by default would be tempting, prefer this posture:

- initialize local git now;
- make the first local commit now;
- document that no remote exists;
- ask before creating a private GitHub repo.

## Commit Discipline

For working-document projects, especially 3D modeling, CAD, graphics, media, notebooks, exploratory research artifacts, and any file where edits are hard to reconstruct, use local commits as an undo stack:

- commit before edits if the tree has uncommitted work;
- commit after each meaningful edit or generated artifact checkpoint;
- use concise checkpoint messages such as `Checkpoint model bracket socket`;
- keep these commits local unless the user asks to publish or squash.

For larger software projects, do not commit every edit. Commit coherent implementation checkpoints after tests or smoke checks.

## Workspace Updates

When brought into an existing project:

1. Run the orientation checks in `references/hygiene-and-updates.md`.
2. Add only the missing structure needed for the current project stage.
3. Prefer migration in small steps: `scratch` -> `lightweight` -> `standard` -> `full`.
4. Commit structure and hygiene changes separately from feature/content work.

## Reference Files

- `references/workspace-structure.md`: structure variants and when to upgrade.
- `references/git-policy.md`: local commits, remotes, branches, and `.gitignore` rules.
- `references/hygiene-and-updates.md`: recurring checks for existing workspaces.
