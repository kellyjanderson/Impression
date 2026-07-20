# Reference Review Test Spec 75a1: Preview Render Command Record Contract (v1.0)

## Paired Specification

- [Reference Review Spec 75a1](../specifications/reference-review-75a1-preview-render-command-record-contract-v1_0.md)

## Automated Tests

- command kind enum rejects unknown values
- command records are immutable
- identity fields are preserved
- neutral lifecycle commands may omit identity
- command result values are deterministic

## Acceptance

- command-record unit tests pass
- `git diff --check` passes
