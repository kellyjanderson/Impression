# Discussion Notes: Agentic Impression GUI Sibling Project

Date: 2026-07-09

## Context

- The proposed app is an agentic Impression GUI inspired by the Reference
  Review Workbench.
- The user described it as essentially a copy of the review app, but with:
  - a file browser instead of a fixture browser;
  - preview support;
  - a code view;
  - a Codex chat interface.
- The user is leaning toward creating a separate project and repository because
  Impression would be a dependency of the GUI rather than the core code.

## Decisions And Leanings

- Tentative leaning: make the agentic GUI a sibling project instead of placing
  it inside the Impression repository.
- Tentative leaning: create a neutral shared workbench/appkit package so the
  Reference Review Workbench and the agentic GUI can reuse base UI and
  application-shell code without either app depending on the other app's
  product namespace.
- Tentative boundary: keep Impression focused on modeling/runtime/library
  responsibilities, and let the GUI own workspace browsing, chat orchestration,
  code viewing, and agent-facing workflow state.
- Reuse target: mine the Reference Review Workbench for preview architecture,
  stale-result handling, render queue lessons, and Qt shell patterns, but avoid
  inheriting fixture-review concepts as core product language.
- Planning record: see
  `project/release-0.1.0a/architecture/agentic-gui-shared-workbench-code-architecture.md`.

## Open Questions

- Should the new repo start as a thin Qt/Python app that imports Impression, or
  as a broader agent-workbench platform with Impression as its first domain?
- Should Codex integration be local-process based, app-plugin based, or an
  explicit protocol boundary that can swap backends later?
- What is the first durable object of work: an Impression file, a workspace
  folder, a model module, or an agent task thread?
- Should the initial preview open only `.impress` files, Python model files that
  produce geometry, or both?

## Follow-Up

- Draft a short sibling-repo architecture note before implementation.
- Identify Reference Review Workbench components worth extracting or copying
  deliberately.
- Decide the new project name, repository path, and minimum first-screen
  workflow.
