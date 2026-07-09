# Reference Review Test Spec 75c: Preview Widget Command Drain And Renderer Mutation Boundary (v1.0)

## Paired Specification

- [Reference Review Spec 75c](../specifications/reference-review-75c-preview-widget-command-drain-and-renderer-mutation-boundary-v1_0.md)

## Automated Tests

- queued payload command applies datasets through fake renderer
- queued display command reuses current decoded datasets
- repeated display toggles produce one final renderer update
- display command does not reset camera
- payload command aligns camera for new payload
- renderer surface is not recreated by queued commands

## Acceptance

- widget command-drain tests pass
- preview controller tests pass
