# Reference Review Workbench Broader Python Module Research

Topic: broader Python module consideration for the Reference Review Workbench,
including lurking concerns not fully captured in the first architecture pass.

Date: 2026-05-31

## Research Goal

The first workbench module research focused on direct architecture needs:
PySide6, Qt Quick, pyvistaqt, Markdown rendering, Pydantic, testing, and Codex
sandbox boundaries.

This second pass intentionally looks wider:

- modules developers recommend after experience
- underrated or "wish I knew earlier" Python modules
- modules that solve hidden operational concerns
- modules that could reduce future custom code
- modules we should explicitly reject or keep as fallback

Relevant prior research:

- [Reference Review Workbench Python Module Research](reference-review-workbench-python-module-research.md)

## Lurking Concerns

The current architecture should explicitly account for these concerns:

- app-local cache/config/log directories
- multi-process file coordination during promotion
- structured audit trails for promotion, notes, Codex sidecar, and tool refusals
- deterministic object diffing for fixture/source/context records
- perceptual similarity triage for rendered dirty/gold images
- long-running task cancellation and timeout semantics
- non-GUI fallback for remote/CI review triage
- screenshot/component state coverage
- packaging risk around optional Qt modules
- dependency boundaries between core modeling and devtools

## Candidate Modules By Concern

### Qt Quick Controls 2

Status: recommended foundation.

Qt Quick Controls 2 remains the right base component framework. It provides
standard controls, state machinery, and styles. The workbench should wrap those
controls in an Impression-owned component library rather than adopting a large
third-party visual framework.

Architectural use:

- base for buttons, rows, scroll panes, tabs, split views, fields, menus
- style foundation for the workbench QML component library

Source:

- Qt Quick Controls styling: https://doc.qt.io/qt-6.8/qtquickcontrols-styles.html

### Kirigami

Status: do not adopt for v1; keep as design reference.

Kirigami is a larger Qt Quick component set built on Qt Quick Controls 2 and
aimed at convergent KDE-style applications.

Why not v1:

- adds KDE framework dependency and packaging surface
- more general/convergent than this focused desktop dev tool needs
- may pull us toward app-shell patterns that are not review-workbench specific

Useful lesson:

- if Qt Quick Controls lacks a pattern, inspect Kirigami for component ideas
  before inventing blindly

Sources:

- Kirigami getting started: https://develop.kde.org/docs/getting-started/kirigami/
- Kirigami API: https://api.kde.org/kirigami-index.html

### FluentPySide

Status: do not adopt for v1.

FluentPySide packages a FluentWinUI3 visual style for Qt Quick Controls in
PySide6. It is a theme, not a component framework.

Why not v1:

- the workbench is macOS-first and dev-tool-oriented
- theme does not solve component contracts, model preview, Markdown, state
  coverage, or review lifecycle

Source:

- FluentPySide PyPI: https://pypi.org/project/fluentpyside/

### platformdirs

Status: strong candidate.

`platformdirs` finds platform-specific user cache/config/data/log directories.

Hidden need it solves:

- workbench caches should not assume cwd
- screenshot caches, preview thumbnails, transient generated modules, and logs
  need stable user-local locations when they are not release artifacts
- avoids hard-coding macOS paths while staying cross-platform

Architectural use:

- workbench cache root
- local settings root
- non-durable log root
- temp preview artifact cache

Source:

- platformdirs docs: https://platformdirs.readthedocs.io/

### filelock

Status: strong candidate for promotion safety.

`filelock` is a platform-independent file lock library.

Hidden need it solves:

- two workbench instances, a CLI promotion script, or a test run could touch the
  same fixture roots
- gold promotion and note writes need serialization across processes, not only
  threads

Architectural use:

- lock around promotion writes
- lock around note writes
- lock around fixture contract version writes

Source:

- filelock docs: https://py-filelock.readthedocs.io/

### structlog

Status: strong candidate for audit logging.

`structlog` provides structured logging built around event dictionaries,
context, and flexible rendering.

Hidden need it solves:

- promotion, rejection, Codex tool refusal, candidate generation, and artifact
  regeneration should be auditable
- plain log strings are hard to filter by fixture id, action, tool, or result

Architectural use:

- structured workbench event log
- fixture-scoped logging context
- Codex tool broker audit trail
- promotion provenance support

Source:

- structlog docs: https://www.structlog.org/en/stable/index.html

### DeepDiff

Status: candidate for diagnostics and review triage.

DeepDiff recursively compares Python objects and can search/hash object content.

Hidden need it solves:

- fixture source records, review context payloads, and promotion metadata will
  drift
- reviewers need to know what changed in the review context, not only that a
  dirty image changed

Architectural use:

- compare source records across regenerations
- compare context payloads before/after candidate adoption
- generate human-readable metadata diffs for the review panel

Source:

- DeepDiff PyPI: https://pypi.org/project/deepdiff/

### ImageHash

Status: candidate for visual triage, not authoritative proof.

ImageHash provides perceptual hashing algorithms such as average, perceptual,
difference, wavelet, color, and crop-resistant hashing.

Hidden need it solves:

- image artifacts may differ in tiny irrelevant ways
- the workbench could triage "probably visually same" versus "needs human
  attention" before a reviewer opens each fixture

Caution:

- perceptual hash is not correctness proof
- it should not replace existing reference image/slice contracts

Architectural use:

- queue sort/filter by visual distance
- highlight large visual changes
- speed up review triage

Source:

- ImageHash PyPI: https://pypi.org/project/ImageHash/

### msgspec

Status: candidate alternative to Pydantic for hot-path records.

`msgspec` is a fast serialization and validation library for JSON, MessagePack,
YAML, and TOML with type-annotation-based schemas.

Hidden need it solves:

- if fixture/context/message records become large or numerous, Pydantic may be
  heavier than needed
- fast serialization could matter for queue scans and cached review state

Recommendation:

- use Pydantic first for clarity and ecosystem familiarity
- reconsider `msgspec` if profiling shows record validation/serialization is a
  bottleneck

Source:

- msgspec PyPI: https://pypi.org/project/msgspec/

### orjson

Status: candidate for fast JSON output only.

`orjson` is a fast JSON library with native dataclass/datetime/UUID support.

Hidden need it solves:

- structured event logs or cached queue state may need fast JSON serialization

Recommendation:

- do not add just for v1
- consider only if structured logs/cache output are large enough to justify it

Source:

- orjson GitHub: https://github.com/ijl/orjson

### attrs

Status: candidate, but likely unnecessary if using dataclasses or Pydantic.

`attrs` reduces boilerplate for Python classes and offers validators,
converters, slots, and immutability.

Recommendation:

- use standard dataclasses for internal immutable envelopes unless Pydantic is
  chosen
- use attrs only if we need richer class behavior without Pydantic

Source:

- attrs docs: https://www.attrs.org/

### pyserde

Status: watchlist, not v1.

`pyserde` is a dataclass-oriented serialization library inspired by Rust Serde.

Recommendation:

- interesting if workbench records become dataclass-heavy and we want a
  serialization layer without Pydantic
- not needed if Pydantic or msgspec is selected

Source:

- pyserde PyPI: https://pypi.org/project/pyserde/

### AnyIO

Status: watchlist for structured async tasks.

AnyIO provides structured concurrency concepts, task groups, cancellation
scopes, and timeout handling.

Hidden need it solves:

- Codex sidecar and file-watching tasks may benefit from structured
  cancellation if the implementation becomes asyncio-oriented

Caution:

- Qt/PySide integration still has to be solved
- worker-thread tasks and VTK/PyVista work are not automatically cooperative

Recommendation:

- do not adopt as the primary concurrency foundation yet
- revisit if the sidecar or tool broker becomes asyncio-heavy

Sources:

- AnyIO cancellation docs: https://anyio.readthedocs.io/en/3.x/cancellation.html
- AnyIO task docs: https://anyio.readthedocs.io/en/stable/tasks.html

### Textual

Status: possible non-GUI fallback or triage companion, not primary UI.

Textual is a Python TUI framework with rich widgets, layouts, testing, and a
developer-friendly API. It builds on Rich.

Hidden need it solves:

- remote/SSH review triage
- CI-friendly dirty fixture browsing
- quick keyboard-heavy queue management without Qt packaging

Why not primary:

- the workbench needs interactive 3D model inspection
- Qt/PySide remains the better fit for embedding or coordinating 3D preview

Recommendation:

- keep Textual as a possible `reference-review --terminal` companion for
  listing, notes, diagnostics, and promotion status
- do not make it the main model-inspection UI

Sources:

- Textual GitHub: https://github.com/Textualize/textual
- Textualize site: https://www.textualize.io/

### Rich

Status: already useful; strong candidate for CLI/reporting surfaces.

Rich renders terminal tables, progress, Markdown, syntax, tracebacks, and
styled output.

Hidden need it solves:

- review-gate reports
- dirty fixture summaries
- structured diagnostics in CLI mode
- Markdown snippets in terminal fallback

Recommendation:

- Impression already uses Rich in CLI paths; keep using it for non-GUI
  workbench reporting rather than inventing plain text tables

Source:

- Rich PyPI: https://pypi.org/project/rich/

## Broad Search Notes

Community/breadth searches surfaced several patterns:

- General "top Python library" lists are increasingly AI-heavy; useful items
  still include tooling, validation, extraction, rate-limiting, and workflow
  modules.
- Reddit/library discussions repeatedly point to practical devtool modules:
  Pydantic, Rich, Textual, structlog, DeepDiff, filelock, platformdirs, and
  faster serialization libraries.
- PySide discussions often emphasize packaging, optional Qt module availability,
  and UI-thread freezes as real pain points.
- UI component framework searches reinforced that Qt Quick Controls 2 is the
  stable base; third-party QML frameworks mostly add style or broad app-shell
  behavior, not our domain-specific review workflow.

## Recommended Additions To Architecture

### Add To Async Concurrency

- Consider `filelock` for cross-process durable write coordination.
- Consider `structlog` for fixture-scoped structured task logs.
- Mention AnyIO as a sidecar-only option if asyncio becomes dominant.

### Add To Promotion And Notes Lifecycle

- Add `filelock` or equivalent lock protocol around note and promotion writes.
- Add structured event log/audit trail for every promotion and rejection.
- Add source/context metadata diff as review evidence when fixture context
  changes.

### Add To Qt Workbench UI

- Keep Qt Quick Controls 2 as base.
- Add a tested component gallery and state matrix as a required artifact.
- Consider ImageHash only for queue triage badges, not proof.
- Keep Rich/Textual as CLI/TUI companion options, not primary UI.

### Add To Codex Sandbox

- Use structured logs for every tool request, tool refusal, candidate write, and
  regeneration request.
- Include `fixture_id`, `tool_name`, `allowed_root`, `result`, and `reason` in
  every event.

### Add To Dependency Policy

- Keep these in a `review-workbench` optional extra:
  - PySide6
  - pyvistaqt
  - pytest-qt
  - pydantic / pydantic-settings
  - platformdirs
  - filelock
  - structlog
  - DeepDiff
  - ImageHash
- Watchlist, not v1:
  - msgspec
  - orjson
  - AnyIO
  - Textual
  - pyserde
  - Kirigami
  - FluentPySide

## Recommended Next Architecture Refinements

1. Add a "review-workbench optional dependency policy" document or section.
2. Add `platformdirs` to the UI/service architecture for caches and non-durable
   logs.
3. Add `filelock` to promotion lifecycle as the cross-process write guard.
4. Add `structlog` to async/concurrency and Codex sandbox as the audit event
   format.
5. Add `DeepDiff` as optional metadata/context diff support.
6. Add `ImageHash` as optional image triage support with a clear "not proof"
   boundary.
7. Add Rich/Textual CLI/TUI companion consideration, especially for CI or remote
   review.

## Bottom Line

The broader search does not change the core UI decision. The workbench should
still be PySide6 + Qt Quick Controls 2 + Impression-owned components.

It does broaden the supporting module picture:

- `platformdirs` for app-local roots
- `filelock` for promotion safety
- `structlog` for auditability
- `DeepDiff` for metadata/context drift
- `ImageHash` for visual triage
- `Rich` and maybe `Textual` for non-GUI reporting/triage

Those modules address operational problems that were lurking beneath the first
architecture pass.

## Expansion Pass: Architecture-Shaping Modules

Date: 2026-05-31

This pass looked for modules that could change the workbench architecture, not
only fill supporting utilities. The strongest new themes are plugin boundaries,
persistent local state, code-aware model editing, approval-style evidence, and
desktop packaging risk.

### Lurking Concern: Workbench Extension Points

The workbench will likely need extension points:

- new artifact kinds beyond PNG/STL
- new preview adapters
- new model-source fixture loaders
- new analysis badges
- new promotion validators
- new Codex tool broker commands

Hard-coding these as `if artifact_kind == ...` branches will make the
workbench difficult to grow and hard to test.

#### pluggy

Status: strong candidate for bounded extension points.

`pluggy` is the plugin and hook system extracted from pytest. It provides a
host/program model where the host declares hook specifications and plugins
provide hook implementations.

Architectural use:

- `reference_review_collect_fixtures`
- `reference_review_load_model_source`
- `reference_review_render_artifact`
- `reference_review_compare_artifacts`
- `reference_review_build_context`
- `reference_review_validate_promotion`
- `reference_review_codex_tools`

Recommendation:

- Use only for devtool extension seams, not for core geometry.
- Keep the hook surface narrow and typed.
- Treat plugins as repo-local or explicitly installed devtool extras.

Source:

- pluggy docs: https://pluggy.readthedocs.io/en/stable/

### Lurking Concern: Persistent Review State

The workbench has several kinds of state:

- authoritative fixture files
- generated dirty artifacts
- gold reference artifacts
- durable review notes
- local UI queue state
- transient thumbnails and preview caches
- audit events

Not all of these should be stored in Markdown or JSON next to the fixture. Some
are durable project evidence; some are local cache; some are queryable review
history.

#### diskcache

Status: candidate for transient local preview/cache state.

`diskcache` provides a persistent disk-backed cache using SQLite metadata and
filesystem storage.

Architectural use:

- preview thumbnails
- parsed Markdown/context cache
- expensive model load cache keyed by source digest
- generated diff thumbnails
- local queue-filter state that can be discarded

Recommendation:

- Use for local, non-authoritative cache only.
- Do not use for promotion state, gold references, or review notes.

Sources:

- diskcache PyPI: https://pypi.org/pypi/diskcache
- diskcache docs: https://www.grantjenks.com/docs/diskcache/

#### sqlite-utils

Status: candidate for queryable audit/history, not v1 default.

`sqlite-utils` is both a CLI and Python library for manipulating SQLite
databases. It is especially good when small local datasets need inspection,
transformation, and ad hoc queries.

Architectural use:

- optional review history database
- queryable audit trail mirror
- fixture review analytics
- "what changed since last release" reports

Recommendation:

- Do not replace file-backed review notes.
- Consider for a local `.impression-review/history.sqlite` mirror once audit
  event volume or review analytics become important.

Sources:

- sqlite-utils PyPI: https://pypi.org/project/sqlite-utils/
- sqlite-utils docs: https://sqlite-utils.datasette.io/

### Lurking Concern: Model Source Editing Must Preserve Human Code

The Codex sidecar may propose updates to model fixtures. If the workbench ever
offers structured assists, quick fixes, or automated fixture migrations, plain
string manipulation will damage comments, formatting, and intent blocks.

#### LibCST

Status: strong candidate for future source-aware edits.

LibCST parses Python as a concrete syntax tree that preserves formatting,
comments, whitespace, and parentheses. It is aimed at codemods and linters.

Architectural use:

- insert/update fixture metadata blocks
- migrate fixture entrypoint names
- add generated candidate functions without destroying comments
- enforce fixture-source contract shape
- support review-safe quick fixes

Recommendation:

- Do not require for first manual editing workflow.
- Add as a future dependency if Codex/tool-assisted fixture rewrites become
  structured operations rather than free-form patches.

Source:

- LibCST docs: https://libcst.readthedocs.io/

#### Griffe

Status: candidate for source introspection and context extraction.

Griffe extracts API structure from Python code and can help detect public API
changes. It is more documentation/API oriented than fixture-editing oriented.

Architectural use:

- inspect fixture modules for exported model functions
- build source context without importing unsafe or expensive modules
- detect fixture contract drift

Recommendation:

- Prefer direct fixture manifests for v1.
- Revisit if importing fixture modules becomes too risky or slow for context
  extraction.

Source:

- Griffe API docs: https://mkdocstrings.github.io/griffe/reference/api/

### Lurking Concern: Reference Review Is Approval Testing With Better Human UX

The workbench is a human approval system wrapped around generated artifacts.
Python already has several approval/snapshot/regression testing tools; the
workbench should learn from them even if it does not adopt them wholesale.

#### pytest-regressions

Status: useful reference; possible test-stack dependency.

`pytest-regressions` provides pytest fixtures for maintaining generated
regression data and image files.

Architectural use:

- existing automated regression checks
- fixture output file organization patterns
- update workflows for generated expected files

Recommendation:

- Compare current reference artifact lifecycle against its update model.
- Consider using or mirroring its image/data regression patterns where they
  align with Impression's reference-artifact lifecycle.

Source:

- pytest-regressions docs: https://pytest-regressions.readthedocs.io/en/stable/

#### Syrupy

Status: candidate for structured metadata snapshots, not image approval.

Syrupy is a zero-dependency pytest snapshot plugin focused on immutable computed
results.

Architectural use:

- snapshot fixture context payloads
- snapshot review manifests
- snapshot generated diagnostics

Recommendation:

- Consider for non-image structured snapshots if existing tests lack a
  first-class snapshot tool.
- Do not make it the workbench approval engine.

Source:

- Syrupy docs: https://syrupy-project.github.io/syrupy/

#### ApprovalTests.Python

Status: conceptually relevant; probably not direct v1 dependency.

ApprovalTests captures outputs as "golden master" approved files and compares
future results to them.

Architectural use:

- vocabulary and workflow reference for approved/received artifacts
- possible CLI-level reviewer integration ideas

Recommendation:

- Keep Impression's own promotion lifecycle because it needs model-source
  context, preview navigation, notes, and Codex candidate handling.

Source:

- ApprovalTests.Python: https://github.com/approvals/approvaltests.python

#### pixelmatch

Status: candidate for deterministic image diff views.

`pixelmatch` is a Python port of a pixel-level image comparison library built
for screenshot tests. It can produce diff images and account for antialiasing.

Architectural use:

- dirty/gold image diff overlays
- preview queue badges for pixel-level distance
- generated diff artifact next to dirty output

Recommendation:

- Pair with ImageHash: ImageHash triages perceptual similarity, pixelmatch
  explains exact image differences.
- Neither is a substitute for human approval of model correctness.

Source:

- pixelmatch PyPI: https://pypi.org/project/pixelmatch/

### Lurking Concern: Dependency Boundaries Need Enforcement

The workbench is a devtool. It must not leak heavy UI, preview, or review
dependencies into Impression's core modeling package.

#### import-linter

Status: strong candidate for protecting architecture boundaries.

Import Linter lets a Python project impose constraints on imports between
modules.

Architectural use:

- core modeling must not import review workbench
- reference review UI must not import release-only internals accidentally
- Codex sandbox broker must not import unrestricted filesystem helpers
- optional extras must stay optional

Recommendation:

- Add an import-linter contract once the workbench package layout exists.
- This is especially valuable if the review workbench becomes a sibling package
  or optional extra inside Impression.

Source:

- Import Linter docs: https://import-linter.readthedocs.io/en/stable/

### Lurking Concern: Packaging Qt Is Its Own Feature

Even if the first version remains a developer tool run from source, the
architecture should avoid choices that make packaging impossible later.

#### pyside6-deploy / Nuitka

Status: primary packaging route to evaluate for PySide/QML.

`pyside6-deploy` is the Qt for Python deployment tool. It wraps Nuitka and
produces platform executables, including `.app` on macOS. Its configuration can
explicitly include QML files and exclude unused QML plugins.

Architectural implications:

- QML files need a predictable package layout.
- Qt WebEngine should remain optional because it can increase package weight.
- Dependencies should be grouped into an optional `review-workbench` extra.
- Packaging tests need at least one smoke run that opens a fixture queue.

Source:

- pyside6-deploy docs: https://doc.qt.io/qtforpython-6/deployment/deployment-pyside6-deploy.html

#### PyInstaller

Status: fallback packaging route.

Qt for Python documents PyInstaller support, but notes that non-onefile cases
may require manually copying Qt plugins, QML imports, and translations.

Recommendation:

- Keep as fallback, not primary.
- If used, make QML/plugin inclusion explicit and tested.

Source:

- Qt for Python PyInstaller docs: https://doc.qt.io/qtforpython-6.10/deployment/deployment-pyinstaller.html

#### Briefcase

Status: watchlist only.

Briefcase packages Python projects as distributable native apps and is part of
BeeWare. It may be useful for generic Python desktop apps, but the workbench's
PySide/QML/PyVista stack is more naturally aligned with Qt's own deployment
tooling.

Recommendation:

- Do not target v1 around Briefcase.
- Revisit only if Qt deployment fails to meet distribution needs.

Source:

- Briefcase docs: https://briefcase.beeware.org/

### Lurking Concern: Sandboxing Cannot Be In-Process Trust

The workbench wants Codex to create candidate model code while constrained to
provided tools and specific folders. Python-level restrictions are useful for
linting and intent checks, but they are not enough to trust arbitrary generated
code in-process.

Architecture implication:

- Treat the Codex sidecar as a separate process.
- Run generated candidate models through the same fixture execution boundary as
  tests.
- Prefer filesystem and process-level policy over in-process "safe eval".
- Keep RestrictedPython-like tools as syntax/policy aids only, not security
  boundaries.

Possible future research:

- macOS sandbox-exec or containerized subprocess boundaries
- WASM/Pyodide limitations for geometry-heavy Python
- Linux-only seccomp approaches are not enough for macOS-first development

### Updated Recommended Module Tiers

#### Adopt Or Strongly Consider For V1

- PySide6
- Qt Quick Controls 2
- pyvistaqt
- pytest-qt
- Pydantic / pydantic-settings
- platformdirs
- filelock
- structlog
- pluggy
- import-linter

#### Consider For V1 If The Feature Exists

- DeepDiff for metadata/context drift
- ImageHash for visual triage
- pixelmatch for image diff overlays
- diskcache for transient preview/cache state
- pytest-regressions for image/data regression alignment

#### Watchlist For Later

- LibCST for structured fixture rewrites
- Griffe for source/API extraction without import
- sqlite-utils for queryable review history
- Syrupy for structured snapshots
- AnyIO if async sidecar complexity grows
- Textual for CLI/TUI review companion
- msgspec/orjson if serialization is hot
- Briefcase if Qt deployment routes fail

#### Avoid As Foundations For This Tool

- Kirigami as a required framework
- FluentPySide as a required theme
- in-process Python sandboxing as a security boundary

### Architecture Changes Suggested By This Pass

1. Add a workbench extension-point architecture section using `pluggy`-style
   hooks or an equivalent typed registry.
2. Add a dependency boundary policy and consider `import-linter` contracts.
3. Split local state into authoritative files, local cache, and optional
   queryable history.
4. Add image diff architecture that distinguishes perceptual triage from
   pixel-level explanation from human approval.
5. Add a source-aware fixture-editing future path using LibCST.
6. Add packaging constraints to the Qt workbench UI architecture so QML layout,
   optional WebEngine use, and extras are selected with deployment in mind.
7. Tighten the Codex sandbox architecture to say explicitly that in-process
   Python restrictions are not a security boundary.
