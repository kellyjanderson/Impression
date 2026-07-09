# Reference Review Test Spec 75c2: Preview Command Application Efficiency (v1.0)

## Paired Specification

- [Reference Review Spec 75c2](../specifications/reference-review-75c2-preview-command-application-efficiency-v1_0.md)

## Automated Tests

- queued payload command applies datasets through fake renderer
- queued display command reuses current decoded datasets
- repeated display toggles produce one final renderer update
- display command does not reset camera
- payload command aligns camera for new payload
- renderer surface is not recreated by queued commands

## Acceptance

- widget command-application tests pass
- preview controller tests pass
