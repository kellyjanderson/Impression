# Project Workspace

This folder contains project-management and implementation-planning material for
Impression.

It is intentionally separate from `docs/`, which remains focused on end-user
and developer-facing product documentation.

## Structure

- [Project DNA](project-dna.md)
- [Future Features](future-features/README.md)
- [Active Release: 0.1.0a](release-0.1.0a/README.md)
- [Release Workspace Lifecycle](releases/README.md)
- [Research](research/)
- [Meetings](meetings/)

## Purpose

Use `project/` for:

- active release work under a top-level `project/release-n.n.n/` folder
- roadmap and planning material
- research and findings
- project records such as PR notes and meeting notes

Release-scoped architecture, specifications, test specifications, planning,
coverage, PR notes, and reference artifacts should live under the active release
folder while the release is underway.

When the release completes, move that folder under `project/releases/`.

Durable cross-release information should remain in:

- `project/future-features/`
- `project/meetings/`
- `project/research/`

Use GitHub Issues for issue tracking. Keep only durable conclusions, research,
release notes, or PR records in `project/`.

See [Release Workspace Lifecycle](releases/README.md) for the full process.

Planned releases may also use a version-specific working branch for integration.
In that workflow:

- the active release tree under `project/release-n.n.n/` is merged to `main`
- a working branch is created from `main`
- feature branches merge into that working branch
- the working branch merges back to `main` when the release is complete

Use `docs/` for:

- modeling guides
- tutorials
- examples
- CLI usage
- extension usage
- agent guidance for using Impression itself
