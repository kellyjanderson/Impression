# Reference Review And Impression Bench GUI Delta Review

Status: review document
Created: 2026-07-16
Scope: current Reference Review workbench in this repository, sibling `impression-gui`, and sibling `impression-workbench-kit`.

## Purpose

This document reviews the significant differences between the current Reference Review app and the newer Impression Bench GUI project. The two apps still share visible ancestry, but their UI goals, async model, design documentation, and management code have diverged.

The goal is not to make the UIs identical. Reference Review remains a fixture and artifact review tool. Impression Bench is a file-centered modeling workbench. The useful work is to identify reusable infrastructure, decide what should move to or expand in `impression-workbench-kit`, and name which newer Bench refinements should be ported back into Reference Review.

## Reviewed Inputs

- Reference Review code:
  - `src/impression/devtools/reference_review/ui/shell.py`
  - `src/impression/devtools/reference_review/ui/preview_widget.py`
  - `src/impression/devtools/reference_review/preview_payload*.py`
  - `src/impression/devtools/reference_review/async_core/*.py`
  - `src/impression/devtools/reference_review/ui/{preview_controls,packaging,markdown_context,style}.py`
- Reference Review architecture:
  - `project/release-0.1.0a/architecture/reference-review-workbench-architecture.md`
  - `project/release-0.1.0a/architecture/reference-review-async-concurrency.md`
  - `project/release-0.1.0a/architecture/agentic-gui-shared-workbench-code-architecture.md`
- Impression Bench code:
  - `/Users/k/Documents/Projects/impression-gui/src/impression_gui/ui/shell.py`
  - `/Users/k/Documents/Projects/impression-gui/src/impression_gui/document_state.py`
  - `/Users/k/Documents/Projects/impression-gui/src/impression_gui/document_dependencies.py`
  - `/Users/k/Documents/Projects/impression-gui/src/impression_gui/workspace_files.py`
  - `/Users/k/Documents/Projects/impression-gui/src/impression_gui/preview_payload*.py`
  - `/Users/k/Documents/Projects/impression-gui/src/impression_gui/agent_*.py`
  - `/Users/k/Documents/Projects/impression-gui/src/impression_gui/codex_sidecar.py`
  - `/Users/k/Documents/Projects/impression-gui/src/impression_gui/prompt_injectors.py`
- Impression Bench docs:
  - `/Users/k/Documents/Projects/impression-gui/project/product/impression-bench.md`
  - `/Users/k/Documents/Projects/impression-gui/project/design/impression-workbench-design-guide.md`
  - `/Users/k/Documents/Projects/impression-gui/project/architecture/*.md`
  - `/Users/k/Documents/Projects/impression-gui/project/specifications/**/*.md`
- Shared kit:
  - `/Users/k/Documents/Projects/impression-workbench-kit/src/impression_workbench/async_core/*.py`
  - `/Users/k/Documents/Projects/impression-workbench-kit/src/impression_workbench/ui/*.py`

## Executive Findings

1. Impression Bench has a much more mature app architecture surface than Reference Review: product intent, design guide, architecture children, ACDs, and a large one-IWU spec tree.
2. The shared-kit direction exists and is active. `impression-workbench-kit` already owns generic async records, diagnostics, owner routing, lane policy, durable writes, callback handoff, preview controls, style records, markdown context rendering, QML packaging, and a small file browser.
3. Reference Review still carries local copies of helpers that now exist in the kit, especially async sanitization/staleness/durable writes, preview controls, packaging, markdown context rendering, and style tokens.
4. Bench has a stronger async architecture than Reference Review: explicit task envelopes, route registration, task lanes, dependency parse lanes, active-document preview routing, source-revision identities, stale completion records, last-good preview preservation, telemetry, and shutdown cleanup.
5. The UI differences are intentional. Bench should not be copied wholesale into Reference Review. The valuable ports are the management patterns and shared primitives, not the file-browser/code-editor/chat product shape.
6. Reference Review can benefit immediately by depending on the shared kit for neutral helpers and by adopting selected Bench async and diagnostics patterns around preview, notes, promotion, and durable writes.

## Intentional UI Differences

| Area | Reference Review | Impression Bench | Port guidance |
| --- | --- | --- | --- |
| Primary object | Fixture record and artifact review state | Workspace root, active code document, renderable document | Keep app-specific. Do not replace fixture queue with file browser. |
| Main navigation | Queue/list of fixtures with approve/decline state | File browser, open document tabs, preview/code panels | Keep app-specific. Reuse only file/list primitives if generalized. |
| Preview target | Selected fixture artifact or fixture source | Active renderable document or affected renderable dependency | Share preview surface and display controls; keep request source app-specific. |
| Context panel | Purpose, methodology, expected render, artifacts, notes | Chat, diagnostics, snapshots, editor, proposal/change review | Keep app-specific panels. Share markdown/diagnostic primitives. |
| Approval workflow | Approve/decline/unreviewed, dirty to gold artifact promotion | Agent proposal accept/reject/revise/apply/save | Do not merge workflows. Share durable write and authority/diagnostic patterns where neutral. |
| Editor | None, except notes text field | `qtmonaco` planned/adapter-backed code editor with open tabs and dirty buffers | Do not port editor into Reference Review unless review sources become editable. |
| Chat/agent | Earlier sidecar concepts mostly domain-bound | Codex chat bridge, prompt injectors, agent context, proposal records, direct edit authority | Port only if Reference Review reopens Codex-assisted fixture regeneration. |

## Documentation Differences

Reference Review has strong release architecture for fixture review, preview payload boundaries, and async concurrency. Its UI documentation is mostly tied to the review workflow and the preview remediation history.

Impression Bench has a more current and complete product-definition stack:

- Product document: records the app as a modeling workbench, not a review queue.
- Design guide: defines calm dense workbench styling, empty-state behavior, diagnostics placement, preview/editor/chat surfaces, and UI QA checklist.
- Architecture set: splits shell/layout, workspace file model, preview/execution, threading/concurrency, code editor, sidecar lifecycle, chat preview context, snapshots, and prompt injectors.
- ACDs: capture design-system conformance, tabbed-panel command routing, and agent diagnostics/direct edit reporting.
- Specifications: many active one-IWU leaves for threading, preview, workspace files, prompt injectors, editor, panel commands, and agent lifecycle.

The design guide is especially valuable for Reference Review. Even when the UI remains different, the guide's state placement rules apply well: routine absence should be quiet or action-oriented; blocking diagnostics should appear near the action; preview stale/failure status belongs in the preview header; input surfaces need output surfaces.

## Async And Threading Differences

### Reference Review Current Shape

Reference Review already has a useful async core:

- `ReviewWorkbenchMessage` and `WorkerResultEnvelope`.
- `TaskDispatcher` with `WorkerPolicy`, pending limits, and coalescing.
- `LatestRequestTracker` for stale completion decisions.
- `UICompletionBridge` and sanitized diagnostics.
- `DurableWriteLane` for serialized writes.
- `PreviewPayloadProcessController` for fixture preview builds, stale payload rejection, and temporary payload cleanup.
- UI-thread render command queue for preview surface mutation.

This is good enough for a single selected fixture workflow. The weak point is that several helpers are app-local copies and the shell still owns many route-specific policies directly.

### Impression Bench Current Shape

Bench extends the copied model into a workbench-scale async topology:

- Generic `WorkbenchTaskRequest` and `WorkbenchTaskCompletion` live in the shared kit.
- `WorkbenchDiagnostic`, `WorkbenchLanePolicy`, `WorkbenchQueueState`, `OwnerRouteRegistry`, and `WorkerCallbackHandoffAdapter` live in the shared kit.
- `DirtyDependencyParseLane` runs dependency parsing off the UI thread and routes stale completions.
- `ActiveDocumentPreviewRouter` maps active documents or support files to renderable subjects.
- `DependencyAwarePreviewRefresher` marks affected renderable subjects stale and debounces refresh.
- `PreviewBuildRequestIdentity` includes owner, request id, subject path, entrypoint, kind, and source revision.
- `BenchPreviewAsyncController` rejects stale preview task completions before UI mutation.
- Preview payload cleanup can mark active payload ownership and delete stale payloads.
- The shell uses timers for preview async polling and dependency refresh debounce.
- `OwnerRouteRegistry` gives completion routing an explicit owner/kind registry instead of broad shape-based dispatch.
- `WorkbenchTelemetryLane` records queue health and task duration events.

### Async Patterns Worth Porting

| Pattern | Bench implementation | Reference Review value |
| --- | --- | --- |
| Generic task records | shared kit `WorkbenchTaskRequest` / `WorkbenchTaskCompletion` | Allows review tasks to use neutral routing and diagnostics instead of review-specific copies. |
| Owner route registry | shared kit `OwnerRouteRegistry` | Reduces ad hoc completion routing as notes, promotion, preview, and future Codex tasks grow. |
| Lane policy records | shared kit `WorkbenchLanePolicy` / `WorkbenchQueueState` | Makes queue/backpressure state inspectable and testable. |
| Workbench diagnostics | shared kit `WorkbenchDiagnostic` | Gives preview, notes, promotion, and fixture load failures one display contract. |
| Source-revision identity | Bench `PreviewBuildRequestIdentity` | Useful for fixture/source preview if record generation alone is not enough, especially when source files or artifacts change. |
| Last-good preview preservation | Bench preview flow | Already partially ported; should become a formal Reference Review invariant. |
| Dependency parse lane | Bench `DirtyDependencyParseLane` | Not directly needed unless Reference Review starts editable source or generated modules. |
| Telemetry lane | Bench `WorkbenchTelemetryLane` | Useful after queue health and preview timing become visible review concerns. |

## Existing Shared Kit Surface

The shared kit is currently narrow but real.

### Async Core

| Kit module | Responsibility | Reference Review adoption status |
| --- | --- | --- |
| `diagnostics.py` | severity and diagnostic record creation | Not adopted; review has local diagnostics. |
| `task_records.py` | generic request/completion records | Not adopted; review still uses `ReviewWorkbenchMessage`. |
| `lane_policy.py` | lane policy and queue-state snapshots | Not adopted; review uses `WorkerPolicy`. |
| `owner_routes.py` | typed owner/kind completion routing | Not adopted. |
| `callback_handoff.py` | worker callback handoff result adapter | Not adopted. |
| `durable_writes.py` | serialized write lane | Review has local copy; should import from kit. |
| `qt_handoff.py` | sanitizer and UI completion bridge | Review has local copy; should import from kit. |
| `staleness.py` | latest-request tracker and cancellation | Review has local copy; should import from kit. |

### UI Core

| Kit module | Responsibility | Reference Review adoption status |
| --- | --- | --- |
| `preview_controls.py` | preview display options, icon toggles, grouped buttons | Review has local copy; should import from kit. |
| `packaging.py` | QML resource layout and preview icon registry | Review has local copy; should import from kit after packaging dependency is solved. |
| `markdown_context.py` | markdown rendering and blocked-link diagnostics by neutral `context_id` | Review has local fixture-specific copy; should import from kit and pass `context_id=fixture_id`. |
| `style.py` | style token records and component contracts | Review has local copy; should either import kit records or define app-specific token values on top. |
| `file_browser.py` | small workspace file browser widget | Bench-specific today; not useful to Reference Review unless generalized into a neutral tree/list primitive. |

## Significant Bench Management Code Not In Reference Review

These are mature enough to study, but not all should be ported.

| Bench module | What it adds | Port decision |
| --- | --- | --- |
| `document_state.py` | open documents, dirty buffer state, source snapshots, save/close results | Keep Bench-specific unless review sources become editable. |
| `document_dependencies.py` | dependency graph, parse lane, stale markers, workspace snapshot materialization | Keep Bench-specific for now; possible future shared library if Reference Review adds editable/generated source dependencies. |
| `workspace_files.py` | workspace root records, file records, renderability detection | Mostly Bench-specific. Renderability detection may become shared after another consumer. |
| `preview_payload.py` | source snapshot identities and Bench preview request records before legacy payload records | Keep Bench-specific until Reference Review needs source snapshots. |
| `preview_payload_controller.py` | active document routing, dependency-aware refresh, Bench async controller, source-revision identity | Partially portable: identity/stale/payload cleanup patterns are broadly useful. |
| `preview_snapshots.py` | named preview snapshot registry and name generator | Useful if Reference Review adds comparison snapshots or Codex attachments. |
| `telemetry.py` | queue health and task duration events | Useful future port; not first priority. |
| `agent_context.py` | active document, previewed renderable, dirty/dependency context payloads | Bench-specific. Review would need a fixture-context equivalent rather than direct port. |
| `agent_proposals.py` | proposal records, authority gates, conflict detection, buffer application | Bench-specific, but authority/diagnostic gates may inform future review-side Codex edits. |
| `prompt_injectors.py` | capability-aware prompt injector registry and diagnostics | Valuable conceptually; not a direct Reference Review port yet. |
| `codex_sidecar.py` | Codex CLI bridge, task lifecycle, tool policy broker, candidate model store | Review has older sidecar concepts; port only after Reference Review's Codex workflow is redefined. |

## Port And Extraction Recommendations

### Priority 1: Replace Local Copies With Kit Imports

These are the lowest-risk cleanup items because Bench already uses the kit and Reference Review still has local copies:

- `async_core/qt_handoff.py` -> `impression_workbench.async_core.qt_handoff`
- `async_core/staleness.py` -> `impression_workbench.async_core.staleness`
- `async_core/durable_writes.py` -> `impression_workbench.async_core.durable_writes`
- `ui/preview_controls.py` -> `impression_workbench.ui.preview_controls`
- `ui/packaging.py` preview display icon registry and QML resource checks -> `impression_workbench.ui.packaging`
- `ui/markdown_context.py` -> `impression_workbench.ui.markdown_context`
- shared style record classes -> `impression_workbench.ui.style`

This should be done before additional app behavior is ported, otherwise the fork keeps widening.

### Priority 2: Adopt Shared Diagnostics And Route Records

Reference Review should add or adapt to:

- `WorkbenchDiagnostic` for user-facing operation diagnostics.
- `WorkbenchLanePolicy` and `WorkbenchQueueState` for inspectable queue/backpressure state.
- `OwnerRouteRegistry` for routing preview, notes, promotion, and future sidecar completions.
- `WorkbenchTaskRequest` / `WorkbenchTaskCompletion` adapters around existing `ReviewWorkbenchMessage` and `WorkerResultEnvelope`.

This can be staged through adapters so existing tests do not require a full async rewrite.

### Priority 3: Formalize Preview State And Last-Good Render Behavior

Bench's preview model has better language for current/stale/failure. Reference Review should document and test:

- stale success cannot replace current preview;
- stale failure cannot clear a newer good preview;
- same-fixture failure after a good render preserves the last good render and marks it stale;
- preview payload cleanup never deletes the active payload;
- display-control commands mutate only the UI-thread render surface;
- renderable artifact fixtures render artifact payloads, while diagnostic fixtures remain non-renderable context rows.

Some of this behavior has been implemented recently, but it should become canonical architecture and regression coverage, not just a patch.

### Priority 4: Port Design-System And Diagnostics Placement Rules

Reference Review should adopt the design guide principles without copying the Bench layout:

- preview stale/current/failure status belongs near the preview, not buried in a generic status line;
- routine absence should be quiet or action-oriented;
- diagnostic/evidence fixtures should clearly say they are non-renderable;
- disabled controls should explain why or remain hidden until useful;
- approve/decline and notes state should have compact, visible feedback;
- design tokens should be centralized instead of scattered stylesheets.

### Priority 5: Defer Bench-Specific Workflows

Do not port these until Reference Review has a product requirement for them:

- file browser and workspace-root selection;
- open document tabs and `qtmonaco`;
- dirty buffer source snapshots;
- dependency parse lanes;
- prompt injector registry;
- named preview snapshots;
- agent proposal application and direct edits;
- Codex chat bridge changes.

They are valuable patterns, but direct porting would blur Reference Review's purpose.

## Shared Library Expansion Plan

### Current Package Boundary

`impression-workbench-kit` should remain a neutral application workbench kit. It should not import `impression.devtools.reference_review`, and it should not become the owner of fixture review, file editing, or Codex product workflows.

### Proposed Package Areas

| Package area | Move or expand | Notes |
| --- | --- | --- |
| `impression_workbench.async_core` | Expand | Add adapters for review envelopes, route registration helpers, lane telemetry hooks, and integration examples. |
| `impression_workbench.ui` | Expand | Keep preview controls, style tokens, markdown rendering, QML packaging, and add neutral status/diagnostic widgets. |
| `impression_workbench.preview` | Consider new package | Candidate home for render command records/queues, preview stale state records, and capture metadata if both apps use them. Avoid importing app payload builders. |
| `impression_workbench.diagnostics` | Consider subpackage if diagnostics grow | Could own diagnostic placement/view-model helpers shared by Bench and Review. |
| `impression_workbench.files` | Defer | File browser and workspace records are currently Bench-shaped. Extract only if another app needs a neutral file browser. |
| `impression_workbench.agent` | Defer | Agent lifecycle and prompt injectors are still product-specific. Extract only stable contracts, not workflows. |

### Extraction Rules

- Extract protocols and records before controllers.
- Keep app-specific nouns out of the kit: no fixture ids, dirty/gold status, active code tab assumptions, or proposal-specific labels.
- Prefer adapter modules in consuming apps while migrating.
- Shared code must have tests in the kit and at least one consuming-app integration smoke.
- Reference Review should not import from `impression_gui`; both apps should import from `impression_workbench`.
- Impression core should continue to own model/preview semantics such as `PreviewSceneController`, `QtPreviewSurface`, `.impress` IO, and tessellation.

## Recommended Implementation Sequence

1. Add `impression-workbench-kit` as a development dependency for the Reference Review UI environment.
2. Replace Reference Review local imports for sanitizer, staleness, durable writes, preview controls, markdown context, packaging, and style records with kit imports.
3. Delete or deprecate the app-local duplicate modules only after tests pass through the product entrypoint.
4. Add adapter tests proving Review messages can become `WorkbenchTaskRequest` / `WorkbenchTaskCompletion`.
5. Introduce `OwnerRouteRegistry` for one narrow route, likely preview completion, then expand to notes and promotion.
6. Convert queue/backpressure state to `WorkbenchLanePolicy` / `WorkbenchQueueState` while preserving existing `TaskDispatcher` behavior.
7. Formalize preview stale/current/failure state as shared records or a small `impression_workbench.preview` package.
8. Update Reference Review architecture/specs after each extraction so they describe imported shared ownership instead of local copies.

## Risks And Cautions

- The current branch has active Reference Review renderer changes. Shared-kit extraction should be done as a separate branch or separate commit series to avoid mixing renderer bug fixes with architecture cleanup.
- Bench code is not automatically better because it is newer. Some modules are product-shaped around active documents and should not be generalized prematurely.
- Reference Review's fixture status/promotion lifecycle is durable and should not be rewritten around Bench proposals or editor buffers.
- Any change touching preview or Qt lifecycle must preserve the rule that renderer creation, scene mutation, camera interaction, and disposal stay on the UI thread.
- Shared package adoption introduces packaging/dependency work. The review app entrypoint must still launch from the Impression `.venv`.

## Bottom Line

The best next step is not to copy the Bench UI into Reference Review. The best next step is to make Reference Review consume the existing shared kit for neutral infrastructure, then selectively port Bench's stronger async and preview-state patterns.

The most valuable immediate work is:

1. converge duplicate helper modules into `impression-workbench-kit`;
2. adopt shared diagnostics/task/lane records through adapters;
3. codify last-good preview and stale-failure behavior;
4. bring over the design guide's state-placement rules without changing the review workflow.
