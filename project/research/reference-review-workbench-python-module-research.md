# Reference Review Workbench Python Module Research

Topic: Python modules that may reduce custom implementation work for the
Reference Review Workbench.

Date: 2026-05-31

## Research Question

Which existing Python modules can solve parts of the Reference Review Workbench
architecture off the shelf, and where do we still need Impression-specific
code?

Relevant architecture:

- [Reference Review Workbench Architecture](../release-0.1.0a/architecture/reference-review-workbench-architecture.md)
- [Reference Review Fixture Source Contract](../release-0.1.0a/architecture/reference-review-fixture-source-contract.md)
- [Reference Review Qt Workbench UI](../release-0.1.0a/architecture/reference-review-qt-workbench-ui.md)
- [Reference Review Async Concurrency](../release-0.1.0a/architecture/reference-review-async-concurrency.md)
- [Reference Review Promotion And Notes Lifecycle](../release-0.1.0a/architecture/reference-review-promotion-and-notes-lifecycle.md)
- [Reference Review Codex Sandbox](../release-0.1.0a/architecture/reference-review-codex-sandbox.md)

## Short Answer

The workbench can reuse substantial libraries for UI, 3D preview integration,
file watching, settings, schema validation, atomic writes, and UI testing.

It should not outsource the core domain contracts:

- source model fixture contract
- dirty/gold promotion semantics
- notes-without-promotion failure semantics
- Codex tool authority
- release evidence gates

Those are Impression-specific.

## Recommended Module Set

### PySide6 / Qt Quick

Use PySide6 as the application shell and Qt Quick/QML for the visible workbench
UI.

Why:

- Qt Quick is the QML UI layer for Qt applications.
- Qt for Python exposes both QML API usage and Python extension points.
- This matches the ViewDown lessons and the existing architecture direction.

Architecture impact:

- Keep [reference-review-qt-workbench-ui.md](../release-0.1.0a/architecture/reference-review-qt-workbench-ui.md)
  centered on PySide6 + Qt Quick.
- Add PySide6 as an optional devtool dependency rather than a core modeling
  dependency.
- Keep QML away from direct filesystem, PyVista, and promotion writes.

Source:

- Qt for Python `PySide6.QtQuick` docs: https://doc.qt.io/qtforpython-6/PySide6/QtQuick/index.html

### pyvistaqt

Use `pyvistaqt` as the first candidate for embedded interactive preview.

Why:

- PyVista documents Qt embedding routes and points to `pyvistaqt`.
- `pyvistaqt.QtInteractor` is intended to provide Plotter-like functionality
  inside Qt applications.
- `BackgroundPlotter` can keep an interactive plotter alive without blocking
  the main Python thread.

Risks:

- The workbench is planned around Qt Quick/QML, while `pyvistaqt` is
  QWidget/Qt-interactor oriented. Embedding in a QML shell may require a bridge
  or hybrid widget window.
- If embedding becomes brittle, the fallback should be a supervised external
  `impression preview` process controlled by the workbench.

Architecture impact:

- Keep the preview bridge abstraction.
- Add an implementation spike: embedded `pyvistaqt.QtInteractor` versus
  supervised external preview process.
- Do not let the UI architecture assume direct PyVista ownership from QML.

Sources:

- PyVista Qt plotting docs: https://docs.pyvista.org/api/plotting/qt_plotting
- PyVistaQt usage docs: http://qt.pyvista.org/usage.html
- PyVistaQt API docs: http://qt.pyvista.org/api_reference.html

### PySide6.QtAsyncio or qasync

Consider Qt/asyncio integration only for I/O-shaped tasks and sidecar control.
Do not use it as a substitute for the explicit workbench task dispatcher.

Why:

- QtAsyncio provides a way to run asyncio with the Qt event loop, but Qt marks
  it as technical preview.
- `qasync` provides a PEP 3156 event-loop implementation for PyQt/PySide apps.

Risks:

- The workbench needs ownership, stale-result guards, cancellation, and
  serialized durable writes. A merged event loop does not provide those domain
  rules by itself.
- Model loading, tessellation, PyVista/VTK work, and artifact generation are
  often CPU or graphics-bound rather than cooperative async tasks.

Architecture impact:

- Keep [reference-review-async-concurrency.md](../release-0.1.0a/architecture/reference-review-async-concurrency.md)
  centered on typed envelopes and worker ownership.
- Allow QtAsyncio/qasync only behind the dispatcher for suitable tasks.
- Prefer Qt signals/worker completion envelopes for UI handoff.

Sources:

- PySide6 QtAsyncio docs: https://doc.qt.io/qtforpython-6.8/PySide6/QtAsyncio/index.html
- qasync PyPI: https://pypi.org/project/qasync/

### watchfiles

Keep using `watchfiles` for file watching and dirty artifact/source refresh.

Why:

- Impression already depends on `watchfiles`.
- The current package has recent wheels, including macOS ARM64 and current
  Python versions.
- It fits source-file reload and dirty reference root refresh needs.

Architecture impact:

- Use `watchfiles` for source file, candidate model, note, and artifact root
  change detection.
- Route all events through the workbench async dispatcher so file events cannot
  mutate QML state directly.

Source:

- watchfiles PyPI: https://pypi.org/project/watchfiles/

### pydantic and pydantic-settings

Use Pydantic for workbench records and `pydantic-settings` for configuration if
we are willing to add the dependency.

Good fit:

- `ReviewSourceModelRecord`
- `ReviewWorkbenchMessage`
- tool-policy records
- Codex context payloads
- note metadata records
- workbench config and allowed roots

Why:

- Pydantic settings supports typed settings loaded from environment/config
  inputs.
- It can reduce hand-written validation code for nested records and tool
  policies.

Risks:

- Adds a dependency to a currently small core package.
- For devtool-only use, keep it optional or isolate under an extra such as
  `impression[review-workbench]`.

Architecture impact:

- Prefer Pydantic for devtool schema validation if dependency policy allows.
- Do not use Pydantic as the source of domain semantics; it validates records,
  while Impression still owns what those records mean.

Sources:

- Pydantic settings docs: https://docs.pydantic.dev/latest/concepts/pydantic_settings/

### atomicwrites or standard-library atomic replace

Use an atomic write pattern for notes and promotion metadata. Consider the
`atomicwrites` package only after checking maintenance and platform needs.

Why:

- Promotion and review notes are durable evidence. Partial writes are not
  acceptable.
- `atomicwrites` documents a simple `atomic_write(..., overwrite=True)` API,
  but the package appears older than other candidates.

Architecture impact:

- Keep atomic promotion as an explicit lifecycle requirement.
- Prefer a tiny internal helper based on temp-file write + `Path.replace()` for
  note/metadata files unless `atomicwrites` offers clear value after a spike.
- Gold artifact promotion still needs checksum/provenance validation, not just
  atomic file replacement.

Source:

- atomicwrites PyPI: https://pypi.org/project/atomicwrites/

### pytest-qt and Qt Quick Test

Use `pytest-qt` for Python/PySide integration tests and consider Qt Quick Test
for QML-level component tests.

Why:

- `pytest-qt` supports PyQt and PySide applications, including PySide6.
- Qt provides Qt Quick Test for QML applications.

Architecture impact:

- Add test expectations to UI specs:
  - QML component smoke tests
  - state transition tests
  - signal/slot completion tests
  - screenshot/state matrix where practical
- Use pytest fixtures for Python-side service and bridge tests.

Sources:

- pytest-qt docs: https://pytest-qt.readthedocs.io/en/master/intro.html
- Qt Quick Test docs: https://doc.qt.io/qtforpython-6/PySide6/QtQuickTest/index.html

### Markdown Rendering

The workbench should include a Markdown context renderer because much of the
review context is already Markdown:

- architecture documents
- specifications
- test specifications
- review notes
- expected-output descriptions
- diagnostic explanations
- linked research

There are three credible implementation routes.

#### Qt-native QTextDocument Markdown

Qt's `QTextDocument` supports `setMarkdown()` and defaults to a GitHub-flavored
Markdown dialect. This is the lowest-friction route for local, trusted,
read-only Markdown context panels.

Good fit:

- review notes
- expected-output descriptions
- spec excerpts
- simple architecture/test-spec reading panes

Limitations:

- It is a rich-text document renderer, not a full documentation browser.
- Styling and extension behavior may be less predictable than a dedicated
  Markdown-to-HTML pipeline.
- QML integration still needs a wrapper component or Python bridge.

Architecture impact:

- Use this as the default first renderer for notes and short context documents.
- Keep Markdown rendering behind a `MarkdownContextRenderer` boundary so the
  renderer can be swapped.

Source:

- PySide6 `QTextDocument` docs: https://doc.qt.io/qtforpython-6.8/PySide6/QtGui/QTextDocument.html

#### markdown-it-py

`markdown-it-py` is a Python port of `markdown-it`, follows CommonMark as a
baseline, is configurable/pluggable, and can render Markdown to HTML.

Good fit:

- consistent spec/research rendering
- future extension support
- source transforms where the workbench wants to control links, headings, code
  blocks, and sanitized HTML

Architecture impact:

- Prefer `markdown-it-py` if Qt-native Markdown becomes visually or
  compatibility-limited.
- Pair it with a sanitizer/URL policy before displaying HTML.
- Use it for Markdown indexing/excerpt extraction even if display stays
  Qt-native.

Sources:

- markdown-it-py docs: https://markdown-it-py.readthedocs.io/
- markdown-it-py PyPI: https://pypi.org/project/markdown-it-py/

#### Qt WebEngine

Qt WebEngine can render HTML/CSS/SVG and could display Markdown rendered to
HTML with browser-quality styling.

Good fit:

- rich documentation panes
- complex tables/code blocks
- future documentation-browsing mode

Risks:

- Heavier dependency and packaging surface.
- It introduces a browser/security model inside a dev tool.
- It may complicate local-resource policy, link interception, and sandboxing.

Architecture impact:

- Do not make WebEngine the first default.
- Keep it as a future renderer option if QTextDocument or QML text surfaces are
  insufficient.

Source:

- Qt WebEngine overview: https://doc.qt.io/qtforpython-6.5/overviews/qtwebengine-overview.html

#### Recommendation

Add a `MarkdownContextRenderer` workbench service with this priority:

1. `QTextDocument.setMarkdown()` for v1 short-form local docs and notes.
2. `markdown-it-py` for normalized parsing, link policy, excerpting, and
   eventual HTML rendering.
3. Qt WebEngine only if the workbench needs browser-grade documentation display.

Security/link policy:

- Treat all displayed Markdown as local project context.
- Disable or intercept external links by default.
- Resolve local links relative to the source document.
- Do not execute embedded scripts or raw HTML in the default renderer.
- Keep rendered docs read-only.

### Qt Quick Controls 2 And Component Frameworks

The workbench should use Qt Quick Controls 2 as the base component framework
and build a small Impression-specific component library on top.

#### Qt Quick Controls 2

Qt Quick Controls supplies standard controls and built-in styles. Qt documents
styles including Basic, Fusion, Imagine, macOS, iOS, Material, Universal, and
Windows. Fusion is explicitly described as desktop-oriented and
platform-agnostic, while macOS is native-looking but macOS-only.

Good fit:

- buttons
- checkboxes/toggles
- split views
- scroll views
- text fields
- menus/popups
- tabs/segmented control equivalents
- standard focus, hover, pressed, disabled, and keyboard behavior

Recommendation:

- Use Qt Quick Controls 2 as the foundation.
- Start with Fusion or macOS style for development; decide final style after
  visual screenshot review.
- Wrap standard controls in workbench components rather than styling every use
  site directly.

Source:

- Qt Quick Controls styling docs: https://doc.qt.io/qt-6.8/qtquickcontrols-styles.html

#### Workbench Component Library

Create an Impression-owned QML component package, for example:

```text
src/impression/devtools/reference_review/ui/components/
```

This package should include tested components such as:

- `WorkbenchIconButton`
- `WorkbenchSplitDivider`
- `WorkbenchQueueRow`
- `WorkbenchStatusPill`
- `WorkbenchPanelHeader`
- `PathField`
- `MarkdownContextView`
- `ArtifactThumbnailTile`
- `DiagnosticBanner`
- `ReviewActionBar`
- `CodexMessageBubble`

Why:

- A prebuilt component library gives the speed benefits of a framework while
  keeping the visual and interaction contract project-owned.
- ViewDown showed that reusable components plus screenshot coverage are what
  turned UI work from patching into repeatable progress.

Rules:

- Compose from Qt Quick Controls 2 primitives.
- Do not create one-off controls in panel files when a component contract fits.
- Every reusable component needs state coverage: idle, hover, focus, pressed,
  disabled, selected, loading, warning, error where applicable.
- Every reusable component needs screenshot or QML smoke coverage before it is
  used as a workbench building block.

#### Kirigami

Kirigami is a larger Qt Quick component set built on top of Qt Quick Controls 2
for convergent KDE-style applications.

Good fit:

- broad app shells
- responsive/convergent applications
- KDE-aligned navigation patterns

Risks for this workbench:

- Adds KDE framework dependency and packaging surface.
- Its convergent/mobile-friendly patterns may be heavier than a dense
  dev-facing desktop workbench needs.
- Impression currently needs a focused internal tool, not a general KDE-style
  application shell.

Recommendation:

- Do not adopt Kirigami as v1 foundation.
- Mine it for ideas only if Qt Quick Controls 2 lacks a pattern we need.

Sources:

- Kirigami getting started docs: https://develop.kde.org/docs/getting-started/kirigami/
- Kirigami API overview: https://api.kde.org/kirigami-index.html

#### FluentPySide / Fluent Styles

`fluentpyside` packages a FluentWinUI3 style for Qt Quick Controls 2 in PySide6
applications. It is a visual theme, not a component library.

Risks for this workbench:

- The workbench is macOS-first and dev-tool-oriented, not Windows 11-branded.
- A visual theme does not solve our component contracts, state coverage,
  Markdown rendering, preview bridge, or async behavior.

Recommendation:

- Do not use FluentPySide for v1.
- If future Windows polish matters, consider it as a style experiment behind
  standard Qt Quick Controls 2 APIs.

Source:

- FluentPySide PyPI: https://pypi.org/project/fluentpyside/

#### Component Framework Bottom Line

Use this stack:

```text
Qt Quick Controls 2
-> Impression Workbench QML Components
-> Workbench Panels
```

Avoid this v1 stack:

```text
Third-party visual framework
-> product-specific overrides everywhere
-> unclear state and screenshot contracts
```

The speed win comes from building and testing the right ten to fifteen
Workbench components once, then composing panels from them.

### trame

Treat `trame` as a fallback or future alternative, not the main workbench UI
route.

Why:

- PyVista and Trame work well together for web-based interactive visualization.
- It can create reactive Python web applications around VTK/PyVista quickly.

Why not primary:

- Current architecture is headed toward PySide6 + Qt Quick, learning from
  ViewDown.
- Trame would shift the UI stack to a web/reactive model and re-open the app
  shell, native desktop, keyboard, and packaging decisions.

Architecture impact:

- Keep trame in reserve if Qt/PyVista embedding fails badly.
- Do not introduce it unless we decide the workbench should be browser-hosted.

Sources:

- PyVista Trame tutorial: https://tutorial.pyvista.org/tutorial/09_trame/index.html
- PyVista Trame backend docs: https://docs.pyvista.org/user-guide/jupyter/trame.html

### RestrictedPython

Do not rely on `RestrictedPython` as the Codex sandbox.

Why:

- Its own documentation says it is not a sandbox system or secured
  environment.
- The workbench requirement is capability sandboxing around tools, not running
  arbitrary Python in-process.

Architecture impact:

- Keep [reference-review-codex-sandbox.md](../release-0.1.0a/architecture/reference-review-codex-sandbox.md)
  based on an allowlisted tool broker.
- Candidate model files can be generated as text and reviewed/regenerated by
  controlled workbench commands.
- Do not execute untrusted Codex-written Python inside the UI process.

Source:

- RestrictedPython docs: https://restrictedpython.readthedocs.io/

## Dependency Shape Recommendation

Keep the core `impression` dependency set focused on modeling.

Add a devtool extra for the workbench:

```toml
[project.optional-dependencies]
review-workbench = [
    "PySide6>=6.8",
    "pyvistaqt>=0.11",
    "pytest-qt>=4.5",
    "pydantic>=2",
    "pydantic-settings>=2",
]
```

Notes:

- Version floors above are placeholders for a future spike, not final pins.
- `watchfiles` is already present in core today.
- `qasync` should be added only if we choose an asyncio-first sidecar route.
- `atomicwrites` should be added only if an internal atomic helper is not
  enough.
- `trame` should not be added unless the UI stack moves away from Qt.

## Architecture Updates Suggested

1. Add a dependency decision section to
   [reference-review-qt-workbench-ui.md](../release-0.1.0a/architecture/reference-review-qt-workbench-ui.md):
   PySide6 and pyvistaqt are preferred; trame is fallback only.
2. Add a preview bridge spike candidate:
   embedded `pyvistaqt.QtInteractor` versus supervised external preview.
3. Add Pydantic as the preferred schema implementation for source records,
   tool policies, and messages if optional dependency policy allows.
4. Add `pytest-qt` and Qt Quick Test to the UI verification plan.
5. Add an explicit rejection in the Codex sandbox architecture:
   `RestrictedPython` is not sufficient for the sidecar security model.
6. Keep the async dispatcher architecture even if QtAsyncio/qasync is used for
   particular tasks.

## Bottom Line

Useful off-the-shelf modules can reduce framework work:

- PySide6/Qt Quick for UI
- Qt Quick Controls 2 plus Impression-owned workbench components for UI speed
- pyvistaqt for preview embedding spike
- watchfiles for refresh
- pydantic for records and config
- QTextDocument/markdown-it-py for Markdown context rendering
- pytest-qt/Qt Quick Test for UI tests
- atomic write helpers for notes and metadata

But the workbench still needs custom Impression-owned architecture for source
fixtures, promotion semantics, lifecycle gates, and Codex authority. Those are
the value of the tool, not scaffolding to outsource.
