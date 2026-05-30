# Release Workspace Lifecycle

## Overview

`project/` separates durable project knowledge from release-scoped work.

Durable project folders stay useful across releases. Active release folders are
temporary working containers. Completed release folders become historical
archives under `project/releases/`.

## Top-Level Shape

The intended steady-state shape is:

```text
project/
  future-features/
  meetings/
  research/
  releases/
  release-n.n.n/
```

Only active release work should live as a top-level `project/release-n.n.n/`
folder. Completed release folders should live under `project/releases/`.

## Durable Project Folders

These folders are durable by default:

- `project/future-features/`
- `project/meetings/`
- `project/research/`
- `project/releases/`

Durable folders hold information that remains useful after a release ships.

Examples:

- reusable research
- future feature ideas not committed to the active release
- meeting records
- completed release archives

## Active Release Folder

When a release begins, create a top-level release folder:

```text
project/release-n.n.n/
```

Example:

```text
project/release-0.1.0a/
```

All release-scoped work product belongs inside that folder:

```text
project/release-0.1.0a/
  README.md
  architecture/
  specifications/
  test-specifications/
  planning/
  research/
  reference-images/
  reference-stl/
  coverage/
  prs/
  adhoc/
```

Not every release needs every subfolder. Create only the folders that the
release actually uses.

## What Belongs In The Active Release Folder

Put work in the active release folder when it primarily exists to plan,
implement, verify, or document that release.

Examples:

- release-specific architecture
- release-specific feature specifications
- paired release test specifications
- implementation progression and sequencing
- release candidate lists
- release-specific reference artifacts
- release-specific coverage outputs
- PR notes tied to the release
- ad hoc release bookkeeping

## What Stays Outside The Active Release Folder

Keep work in durable top-level folders when it should outlive the active
release as reusable project knowledge.

Examples:

- durable research findings
- future feature architecture that is not part of the release
- meeting notes
- completed release archives

Use GitHub Issues for issue tracking instead of a parallel local issues folder.

When release work discovers durable information, promote or copy the durable
portion into the appropriate top-level folder before the release is archived.

## Release Completion

When a release is complete:

1. confirm release-scoped docs, specs, tests, reference artifacts, and coverage
   records are in the active release folder.
2. promote durable findings into `project/research/` or
   `project/future-features/` as appropriate, and file follow-up work in GitHub
   Issues when needed.
3. freeze the release folder.
4. move it under `project/releases/`.

Example:

```text
project/release-0.1.0a/
```

becomes:

```text
project/releases/release-0.1.0a/
```

After this move, the top level of `project/` should no longer contain that
release folder.

## Archive Rule

Archived release folders are historical records.

Do not use archived release folders as active planning surfaces. If follow-up
work is needed after a release:

- create or update a GitHub Issue
- record reusable findings under `project/research/`
- create a future feature entry under `project/future-features/`
- create a new top-level `project/release-n.n.n/` folder for the next active
  release

## Branch Workflow Relationship

The release folder lifecycle is separate from Git branch workflow.

The recommended branch pattern remains:

- plan the release
- create or update the top-level `project/release-n.n.n/` folder
- create a release working branch, typically `working/<release>`
- merge feature branches into the release working branch
- merge the working branch back to `main` when ready
- archive the release folder under `project/releases/`

## Naming Rule

Use hyphenated release folder names:

```text
release-0.1.0a
release-1.2.3
```

Avoid dotted folder prefixes such as `release.0.1.0a`.

Hyphenated names match the existing project convention and are easier to use in
paths, scripts, and links.

## Migration Note

Some existing project documents predate this lifecycle and may still contain
historical links to the old top-level release-scoped folders, including
`architecture/`, `specifications/`, `test-specifications/`, `planning/`,
`coverage/`, `prs/`, and reference artifact folders.

Those links and any remaining loose files should be migrated deliberately. Do
not bulk move or rewrite them without first classifying whether each item is:

- durable cross-release knowledge
- active release work
- completed release history
- obsolete release scaffolding
