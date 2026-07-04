# Reference Review Spec 31: Renderer Backend (v1.0)

## Overview

Implement the Markdown rendering backend for fixture context.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-qt-workbench-ui.md)

## Source Manifest

- This leaf is derived from the split manifest candidate `Markdown Context Renderer`.
- Manifest score: 13.5

## Scope

This specification covers:

- Markdown renderer backend selection
- render cache keyed by fixture/source digest
- blocked-link diagnostic creation

## Behavior

This leaf must define:

- prefer Qt-native Markdown rendering first
- keep optional markdown-it-py or WebEngine behind a boundary
- cache rendered context without bypassing link policy

## Constraints

- Async/concurrency behavior: expensive render may run through dispatcher
- Security/privacy-sensitive behavior: external links are blocked or
  explicitly confirmed
- Performance-sensitive behavior: rendered context is cached per
  fixture/source digest
- Cross-screen reusable behavior: renderer can be reused by notes preview
  and diagnostic panels

## Dependencies And Reuse

Dependencies/services:

- Qt text renderer
- optional markdown-it-py renderer

Reusable code plan:

- Existing code reused as-is: fixture context payload
- Additions to existing reusable library/module: Markdown context renderer
- New reusable library/module to create: none

Implementation owner/module:

- future `ui/markdown_context`

## Data Ownership And Routes

Data ownership:

- context payload owns content; renderer owns presentation

Routes:

- context payload to renderer to QML panel

## UI Contract

- Surface/component: Markdown context panel
- Field/element: rendered Markdown, blocked-link message, local-link action

## Test Strategy

- Markdown rendering, blocked link, long text, and cache invalidation tests

## Open Questions And Prerequisites

Open questions / nuance discovered:

- none

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Renderer Backend boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
