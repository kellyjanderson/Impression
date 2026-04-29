# Project Workspace

This folder contains project-management and implementation-planning material for
Impression.

It is intentionally separate from `docs/`, which remains focused on end-user
and developer-facing product documentation.

## Structure

- [Project DNA](project-dna.md)
- [Future Features](future-features.md)
- [Planning](planning/README.md)
- [0.1.0.a Planning](planning/0.1.0.a/README.md)
- [Architecture](architecture/README.md)
- [Ad Hoc Work](adhoc/README.md)
- [Specifications](specifications/README.md)
- [Research](research/)
- [Issues](issues/)
- [PR Notes](prs/)
- [Meetings](meetings/)

## Purpose

Use `project/` for:

- architecture and specification work
- roadmap and planning material
- research and findings
- project records such as issue and PR notes

Planned releases may also use a version-specific working branch for integration.
In that workflow:

- the version planning tree under `project/planning/<release>/` is merged to `main`
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
