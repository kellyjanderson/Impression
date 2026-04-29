# Ad Hoc Work

This folder is the durable planning space for small, named work that should not
be blocked on heavy feature-spec refinement.

It exists alongside:

- `project/specifications/`
- `project/test-specifications/`
- `project/planning/`

Use `project/adhoc/` when the work is real implementation work that still needs
developer/project-facing documentation, but does not justify a full
architecture-to-specification branch first.

Examples:

- a new example program
- a documentation-driven showcase addition
- a small developer-facing tool or helper
- a bounded cleanup or structural adjustment that is too large for chat-only
  context but too small for full feature-tree refinement

## Purpose

An ad hoc document should preserve:

1. what changed
2. why the change exists
3. the intended scope boundary
4. any important usage or maintenance notes
5. what tests or verification are expected

The goal is to avoid leaving meaningful work justified only by transient chat
history while still keeping lightweight work lightweight.

## What Ad Hoc Is Not

Ad hoc work is not:

- a replacement for architecture when architecture is actually needed
- a way to skip feature/test specification work for substantial product changes
- a place for vague notes with no implementation boundary

If the work changes durable system behavior in a way that needs branching
architecture or future refinement, it should stay on the feature path.

## Minimum Document Shape

Each ad hoc work item should be its own markdown file under `project/adhoc/`.

Recommended contents:

- title
- date
- status
- summary
- scope
- implementation notes
- verification notes
- related files or related documents

## Agent Rule

Before implementation begins, the work should be classified as exactly one of:

- feature-path work
- ad-hoc-path work

If the user has not specified which path to use, agents should ask a short
disambiguating question before proceeding with implementation.
