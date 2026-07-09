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
- [x] [Reference Review Spec 32: QML Panel/Link Policy](../specifications/reference-review-32-qml-panel-link-policy-v1_0.md)
- [x] [Reference Review Spec 33: Artifact Panel](../specifications/reference-review-33-artifact-panel-v1_0.md)
- [x] [Reference Review Spec 34: Notes Panel](../specifications/reference-review-34-notes-panel-v1_0.md)
- [x] [Reference Review Spec 35: Chat Stream Panel](../specifications/reference-review-35-chat-stream-panel-v1_0.md)
- [x] [Reference Review Spec 36: Candidate List/Adoption UI](../specifications/reference-review-36-candidate-list-adoption-ui-v1_0.md)
- [x] [Reference Review Spec 37: Refusal Display](../specifications/reference-review-37-refusal-display-v1_0.md)

## Preview Wrapper Remediation

These specs remediate the original preview path by extracting the real
`impression.preview` engine into a shared controller, embedding it through a
thin Qt wrapper, and moving fixture payload work behind explicit async
boundaries.

- [x] [Reference Review Spec 43: Shared Preview Controller API And Style Records](../specifications/reference-review-43-shared-preview-controller-api-and-style-records-v1_0.md)
- [x] [Reference Review Spec 44: Shared Scene Application And Camera Policy](../specifications/reference-review-44-shared-scene-application-and-camera-policy-v1_0.md)
- [x] [Reference Review Spec 45: CLI Preview Host Delegation](../specifications/reference-review-45-cli-preview-host-delegation-v1_0.md)
- [x] [Reference Review Spec 46: Preview Parity And Import-Boundary Guards](../specifications/reference-review-46-preview-parity-and-import-boundary-guards-v1_0.md)
- [x] [Reference Review Spec 47: Preview Payload Request And Result Records](../specifications/reference-review-47-preview-payload-request-and-result-records-v1_0.md)
- [x] [Reference Review Spec 48: Preview Source Load And Tessellation Adapter](../specifications/reference-review-48-preview-source-load-and-tessellation-adapter-v1_0.md)
- [x] [Reference Review Spec 49: Preview Payload Serialization Writer](../specifications/reference-review-49-preview-payload-serialization-writer-v1_0.md)
- [x] [Reference Review Spec 50: Preview Payload Builder Orchestration](../specifications/reference-review-50-preview-payload-builder-orchestration-v1_0.md)
- [x] [Reference Review Spec 51: Preview Payload Process Controller](../specifications/reference-review-51-preview-payload-process-controller-v1_0.md)
- [x] [Reference Review Spec 52: Preview Temporary Payload Cleanup](../specifications/reference-review-52-preview-temporary-payload-cleanup-v1_0.md)
- [x] [Reference Review Spec 53: Preview Current And Stale Payload Handoff](../specifications/reference-review-53-preview-current-and-stale-payload-handoff-v1_0.md)
- [x] [Reference Review Spec 54: Preview Payload Failure Diagnostic Handoff](../specifications/reference-review-54-preview-payload-failure-diagnostic-handoff-v1_0.md)
- [x] [Reference Review Spec 55: Preview Widget Renderer Lifecycle](../specifications/reference-review-55-preview-widget-renderer-lifecycle-v1_0.md)
- [x] [Reference Review Spec 56: Preview Widget Payload Application](../specifications/reference-review-56-preview-widget-payload-application-v1_0.md)
- [x] [Reference Review Spec 57: Preview Pane Visible State](../specifications/reference-review-57-preview-pane-visible-state-v1_0.md)
- [x] [Reference Review Spec 58: Preview Toolbar Command Routing](../specifications/reference-review-58-preview-toolbar-command-routing-v1_0.md)
- [x] [Reference Review Spec 59: Preview Wrapper Real-Render Smoke And Lifecycle Evidence](../specifications/reference-review-59-preview-wrapper-real-render-smoke-and-lifecycle-evidence-v1_0.md)
- [x] [Reference Review Spec 60: Preview Stale Success And Failure Rejection Tests](../specifications/reference-review-60-preview-stale-success-and-failure-rejection-tests-v1_0.md)
- [x] [Reference Review Spec 61: Preview Cancellation Ordering Tests](../specifications/reference-review-61-preview-cancellation-ordering-tests-v1_0.md)
- [x] [Reference Review Spec 62: Preview Cleanup Deletion Tests](../specifications/reference-review-62-preview-cleanup-deletion-tests-v1_0.md)

## Preview Display Controls

These specs define the reusable icon/control pieces and the composed preview
display-control button bar.

- [x] [Reference Review Spec 63: Preview Display Icon Asset Packaging](../specifications/reference-review-63-preview-display-icon-asset-packaging-v1_0.md)
- [x] [Reference Review Spec 63 Test: Preview Display Icon Asset Packaging](../test-specifications/reference-review-63-preview-display-icon-asset-packaging-v1_0.md)
- [x] [Reference Review Spec 64: Preview Display Icon Metadata Registry](../specifications/reference-review-64-preview-display-icon-metadata-registry-v1_0.md)
- [x] [Reference Review Spec 64 Test: Preview Display Icon Metadata Registry](../test-specifications/reference-review-64-preview-display-icon-metadata-registry-v1_0.md)
- [x] [Reference Review Spec 65: Icon Toggle Button Visual State Component](../specifications/reference-review-65-icon-toggle-button-visual-state-component-v1_0.md)
- [x] [Reference Review Spec 65 Test: Icon Toggle Button Visual State Component](../test-specifications/reference-review-65-icon-toggle-button-visual-state-component-v1_0.md)
- [x] [Reference Review Spec 66: Icon Toggle Button Interaction Contract](../specifications/reference-review-66-icon-toggle-button-interaction-contract-v1_0.md)
- [x] [Reference Review Spec 66 Test: Icon Toggle Button Interaction Contract](../test-specifications/reference-review-66-icon-toggle-button-interaction-contract-v1_0.md)
- [x] [Reference Review Spec 67: Exclusive Icon Group Selection Model](../specifications/reference-review-67-exclusive-icon-group-selection-model-v1_0.md)
- [x] [Reference Review Spec 67 Test: Exclusive Icon Group Selection Model](../test-specifications/reference-review-67-exclusive-icon-group-selection-model-v1_0.md)
- [x] [Reference Review Spec 68: Exclusive Icon Group Component Composition](../specifications/reference-review-68-exclusive-icon-group-component-composition-v1_0.md)
- [x] [Reference Review Spec 68 Test: Exclusive Icon Group Component Composition](../test-specifications/reference-review-68-exclusive-icon-group-component-composition-v1_0.md)
- [x] [Reference Review Spec 69: Preview Display Options State Record](../specifications/reference-review-69-preview-display-options-state-record-v1_0.md)
- [x] [Reference Review Spec 69 Test: Preview Display Options State Record](../test-specifications/reference-review-69-preview-display-options-state-record-v1_0.md)
- [x] [Reference Review Spec 70: Preview Display Command Routing](../specifications/reference-review-70-preview-display-command-routing-v1_0.md)
- [x] [Reference Review Spec 70 Test: Preview Display Command Routing](../test-specifications/reference-review-70-preview-display-command-routing-v1_0.md)
- [x] [Reference Review Spec 71: Preview Surface Layer Visibility Options](../specifications/reference-review-71-preview-surface-layer-visibility-options-v1_0.md)
- [x] [Reference Review Spec 71 Test: Preview Surface Layer Visibility Options](../test-specifications/reference-review-71-preview-surface-layer-visibility-options-v1_0.md)
- [x] [Reference Review Spec 72: Preview Surface Color And Lighting Options](../specifications/reference-review-72-preview-surface-color-and-lighting-options-v1_0.md)
- [x] [Reference Review Spec 72 Test: Preview Surface Color And Lighting Options](../test-specifications/reference-review-72-preview-surface-color-and-lighting-options-v1_0.md)
- [x] [Reference Review Spec 73: Preview Pane Display Control Slot](../specifications/reference-review-73-preview-pane-display-control-slot-v1_0.md)
- [x] [Reference Review Spec 73 Test: Preview Pane Display Control Slot](../test-specifications/reference-review-73-preview-pane-display-control-slot-v1_0.md)
- [x] [Reference Review Spec 74: Preview Display Control Row Composition](../specifications/reference-review-74-preview-display-control-row-composition-v1_0.md)
- [x] [Reference Review Spec 74 Test: Preview Display Control Row Composition](../test-specifications/reference-review-74-preview-display-control-row-composition-v1_0.md)

## UI Evidence And Review Quality

- [x] [Reference Review Spec 38: Component Gallery](../specifications/reference-review-38-component-gallery-v1_0.md)
- [x] [Reference Review Spec 39: Screenshot Runner](../specifications/reference-review-39-screenshot-runner-v1_0.md)
- [x] [Reference Review Spec 40: Accessibility/Overflow State Matrix](../specifications/reference-review-40-accessibility-overflow-state-matrix-v1_0.md)

## Architecture Governance

- [x] [Reference Review Spec 42: Reference Review Architecture Index And Domain Split](../specifications/reference-review-42-reference-review-architecture-index-and-domain-split-v1_0.md)

## Ordering Rationale

The order builds the reusable async substrate first, then fixture/source
contracts, durable review lifecycle services, and constrained Codex authority.
The UI starts only after those service boundaries are explicit enough for QML
panels to bind against stable protocols. Screenshot and state evidence come
after the component foundation and primary panels exist.

## Current Implementation Note

2026-07-04: implementation is complete through Spec 42. The original Reference Review
Workbench progression items are checked with focused service, UI, evidence, and
architecture-index verification.

2026-07-07: Specs 43-62 were added for preview-wrapper remediation and remain
unimplemented until the shared preview controller, Qt wrapper, and preview
payload boundary work is completed.

2026-07-08: Specs 63-74 and their paired test specifications were added for
the preview display-control icon assets, reusable icon controls, option-group
controls, display option state, renderer option application, and composed
button bar. They are intentionally unchecked until implemented and verified.
