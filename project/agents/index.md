# Project Agent Rules

This folder contains project-specific agent rules that extend the shared guidance in the repository root `agents/` folder.

Current project-specific rules:

- [testing.md](testing.md)
  - coverage expectations for finished work
  - required full coverage pass when progression reaches all complete
  - modern surface-first modules that must not retain stale geometry deprecation wiring

- [reference-images.md](reference-images.md)
  - reference image and STL generation
  - dirty vs clean reference lifecycle
  - first-run dirty bootstrap and subsequent comparison behavior
  - model-output completeness requirement for reference coverage
  - promotion rules
  - reference-artifact test expectations

- [session-handoff.md](session-handoff.md)
  - end-of-work handoff phrase
  - when the agent should explicitly signal that deeper re-engagement is required

Project planning spaces also include:

- [../adhoc/README.md](../adhoc/README.md)
  - lightweight durable planning for bounded ad hoc work
  - required choice between feature-path work and ad-hoc-path work before implementation
