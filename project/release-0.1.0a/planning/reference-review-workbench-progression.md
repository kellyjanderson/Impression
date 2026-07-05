# Reference Review Workbench Progression

This progression orders the Reference Review Workbench final leaf
specifications in the recommended implementation sequence.

Only final leaf specifications appear here.

Paired test-spec work is implied for each leaf and should be completed together
with the implementation when the paired test specification exists. Reference
Review paired test specifications have not yet been created.

## Async Foundation

- [x] [Reference Review Spec 01: Review Workbench Message Envelope](../specifications/reference-review-01-review-workbench-message-envelope-v1_0.md)
- [x] [Reference Review Spec 02: Task Dispatcher And Worker Policy](../specifications/reference-review-02-task-dispatcher-and-worker-policy-v1_0.md)
- [x] [Reference Review Spec 03: Stale Completion And Cancellation Guards](../specifications/reference-review-03-stale-completion-and-cancellation-guards-v1_0.md)
- [x] [Reference Review Spec 04: Durable Write Serialization And File Locking](../specifications/reference-review-04-durable-write-serialization-and-file-locking-v1_0.md)
- [x] [Reference Review Spec 05: UI Thread Handoff And Sanitized Task Errors](../specifications/reference-review-05-ui-thread-handoff-and-sanitized-task-errors-v1_0.md)
- [x] [Reference Review Spec 06: Structured Task Audit Events](../specifications/reference-review-06-structured-task-audit-events-v1_0.md)

## Fixture Source Contract

- [x] [Reference Review Spec 13: Review Source Model Record Schema](../specifications/reference-review-13-review-source-model-record-schema-v1_0.md)
- [x] [Reference Review Spec 14: Source Record Validation And Diagnostics](../specifications/reference-review-14-source-record-validation-and-diagnostics-v1_0.md)
- [x] [Reference Review Spec 15: Fixture Discovery Integration](../specifications/reference-review-15-fixture-discovery-integration-v1_0.md)
- [x] [Reference Review Spec 16: Deterministic Review Context Payload](../specifications/reference-review-16-deterministic-review-context-payload-v1_0.md)
- [x] [Reference Review Spec 17: Generated Review Module Contract](../specifications/reference-review-17-generated-review-module-contract-v1_0.md)

## Promotion And Notes Lifecycle

- [x] [Reference Review Spec 18: Review Note Store](../specifications/reference-review-18-review-note-store-v1_0.md)
- [x] [Reference Review Spec 19: Review State Classifier](../specifications/reference-review-19-review-state-classifier-v1_0.md)
- [x] [Reference Review Spec 20: Promotion Validator](../specifications/reference-review-20-promotion-validator-v1_0.md)
- [x] [Reference Review Spec 21: Atomic Promotion Executor](../specifications/reference-review-21-atomic-promotion-executor-v1_0.md)
- [x] [Reference Review Spec 22: Promotion Provenance And Release Gate Report](../specifications/reference-review-22-promotion-provenance-and-release-gate-report-v1_0.md)

## Codex Sandbox

- [x] [Reference Review Spec 07: Codex Fixture Context Builder](../specifications/reference-review-07-codex-fixture-context-builder-v1_0.md)
- [x] [Reference Review Spec 08: Tool Policy Validator And Broker](../specifications/reference-review-08-tool-policy-validator-and-broker-v1_0.md)
- [x] [Reference Review Spec 09: Candidate Model Store](../specifications/reference-review-09-candidate-model-store-v1_0.md)
- [x] [Reference Review Spec 10: Candidate Note Patch Route](../specifications/reference-review-10-candidate-note-patch-route-v1_0.md)
- [x] [Reference Review Spec 11: Regeneration Request Route](../specifications/reference-review-11-regeneration-request-route-v1_0.md)
- [x] [Reference Review Spec 12: Sidecar Process Boundary And Audit](../specifications/reference-review-12-sidecar-process-boundary-and-audit-v1_0.md)

## UI Shell And Component Foundation

- [x] [Reference Review Spec 23: QML Launcher/Bootstrap](../specifications/reference-review-23-qml-launcher-bootstrap-v1_0.md)
- [x] [Reference Review Spec 24: Bridge Registration](../specifications/reference-review-24-bridge-registration-v1_0.md)
- [x] [Reference Review Spec 25: Qt Quick Controls Style And Component Framework](../specifications/reference-review-25-qt-quick-controls-style-and-component-framework-v1_0.md)
- [x] [Reference Review Spec 41: Workbench UI Dependency And Packaging Policy](../specifications/reference-review-41-workbench-ui-dependency-and-packaging-policy-v1_0.md)

## UI Review Workflow

- [x] [Reference Review Spec 26: Queue Navigation Panel](../specifications/reference-review-26-queue-navigation-panel-v1_0.md)
- [x] [Reference Review Spec 27: Selected Fixture Context Panel](../specifications/reference-review-27-selected-fixture-context-panel-v1_0.md)
- [x] [Reference Review Spec 28: Preview Adapter Decision Spike](../specifications/reference-review-28-preview-adapter-decision-spike-v1_0.md)
- [x] [Reference Review Spec 29: Preview Load Binding](../specifications/reference-review-29-preview-load-binding-v1_0.md)
- [x] [Reference Review Spec 30: Camera Controls](../specifications/reference-review-30-camera-controls-v1_0.md)
- [x] [Reference Review Spec 31: Renderer Backend](../specifications/reference-review-31-renderer-backend-v1_0.md)
- [ ] [Reference Review Spec 32: QML Panel/Link Policy](../specifications/reference-review-32-qml-panel-link-policy-v1_0.md)
- [ ] [Reference Review Spec 33: Artifact Panel](../specifications/reference-review-33-artifact-panel-v1_0.md)
- [ ] [Reference Review Spec 34: Notes Panel](../specifications/reference-review-34-notes-panel-v1_0.md)
- [ ] [Reference Review Spec 35: Chat Stream Panel](../specifications/reference-review-35-chat-stream-panel-v1_0.md)
- [ ] [Reference Review Spec 36: Candidate List/Adoption UI](../specifications/reference-review-36-candidate-list-adoption-ui-v1_0.md)
- [ ] [Reference Review Spec 37: Refusal Display](../specifications/reference-review-37-refusal-display-v1_0.md)

## UI Evidence And Review Quality

- [ ] [Reference Review Spec 38: Component Gallery](../specifications/reference-review-38-component-gallery-v1_0.md)
- [ ] [Reference Review Spec 39: Screenshot Runner](../specifications/reference-review-39-screenshot-runner-v1_0.md)
- [ ] [Reference Review Spec 40: Accessibility/Overflow State Matrix](../specifications/reference-review-40-accessibility-overflow-state-matrix-v1_0.md)

## Architecture Governance

- [ ] [Reference Review Spec 42: Reference Review Architecture Index And Domain Split](../specifications/reference-review-42-reference-review-architecture-index-and-domain-split-v1_0.md)

## Ordering Rationale

The order builds the reusable async substrate first, then fixture/source
contracts, durable review lifecycle services, and constrained Codex authority.
The UI starts only after those service boundaries are explicit enough for QML
panels to bind against stable protocols. Screenshot and state evidence come
after the component foundation and primary panels exist.

## Current Implementation Note

2026-07-04: implementation is complete through Spec 31. Spec 32 and later
remain unchecked because the next slice needs the QML Markdown panel surface,
artifact/notes/Codex panel view models, and screenshot/accessibility evidence
before those checkboxes can be honest truth markers.
