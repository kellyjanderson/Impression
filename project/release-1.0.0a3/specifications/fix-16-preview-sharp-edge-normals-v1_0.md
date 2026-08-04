# Fix 16: Preview Sharp-Edge Normals Specification

Date: 2026-08-04
Status: Final
Primary ancestor: `project/release-1.0.0a3/README.md`
Architecture ancestor: `not applicable - preview rendering correction`
Source artifact: test-modeling three-assembly release smoke
Split provenance: `none`
Canonical status: `Canonical`
Review Score: 10
Prerequisites:
- Fix 15 preview PNG export provides the repeatable visual qualification route.

## Source Field Carryover

- Source purpose:
  - Stop smooth preview normals from crossing sharp CAD seams and making valid flat geometry appear folded, spiked, or fragmented.
- Source responsibilities by category:
  - Functions/methods: `PreviewSceneController.apply_scene(...)`.
  - Data structures/models: not applicable.
  - Dependencies/services: existing PyVista sharp-edge vertex splitting.
  - Returns/outputs/signals: visually faithful interactive and PNG previews.
  - UI surfaces/components: the shared Impression preview renderer.
  - UI fields/elements: not applicable.
  - Reusable code plan: extend the existing shared preview mesh route.
  - Database queries/tables/migrations: not applicable.
  - Async/concurrency behavior: not applicable.
  - Destructive/write behavior: not applicable beyond user-requested PNG output already owned by Fix 15.
  - Security/privacy-sensitive behavior: not applicable.
  - Performance-sensitive behavior: sharp-edge splitting may duplicate render-only vertices but remains bounded by the input mesh.
  - Cross-screen reusable behavior: not applicable.
- Source open questions / nuance discovered:
  - Model bounds and topology QA pass; only interpolated renderer normals create the false geometry.
- Source split/provenance notes:
  - No parent split; this is one shared renderer-policy correction.

## Purpose

Preserve smooth shading within CAD faces while splitting render normals at the
same configured feature angle used for visible feature edges.

## Scope

Owns:

- Passing PyVista `split_sharp_edges=True` whenever mesh smooth shading is enabled.
- Reusing the preview style's feature-edge angle for the split threshold.
- Applying the rule to uniform-color and per-face-color mesh paths.
- Three-model visual smoke against the original, loft, and diagonal audio-cube assemblies.

Does not own:

- Modifying model vertices, faces, tessellation, CSG, colors, or authored geometry.
- Changing wireframe or polyline rendering.
- Treating intermediate model QA warnings as preview-normal failures.

## Split Coverage

- Parent spec: `none`
- Parent coverage status: not applicable
- Parent responsibilities owned by this child:
  - not applicable
- Parent responsibilities still missing from children:
  - none

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../planning/spec-review-ledger-20260804-preview-sharp-edges.md` | 1 | Fix 16 | none | reached |

## Implementation Routing

- Primary modules/files:
  - `src/impression/preview.py` - shared mesh-to-PyVista render arguments.
- Supporting modules/files:
  - `project/release-1.0.0a3/` - release scope and progression evidence.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/preview.py` - existing `PreviewSceneController` boundary.
- Tests:
  - `tests/test_preview_controller.py` - renderer argument contract.
  - test-modeling PNG smoke - real console-to-renderer visual proof.

## Chosen Defaults / Parameters

- Smooth shading remains enabled by default.
- Sharp-edge vertex splitting follows `PreviewStyle.feature_edge_angle`, currently 60 degrees.
- When smooth shading is disabled, sharp-edge splitting is disabled too.
- Render-only vertex duplication never mutates the Impression `Mesh`.

## Data Ownership

- Source of truth: the unchanged Impression mesh and preview style.
- Read ownership: `PreviewSceneController` reads mesh/color/style state.
- Write ownership: PyVista owns its derived render dataset and normals.
- Derived/cache data: split render vertices and normals are disposable.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - PyVista `Plotter.add_mesh(...)` smooth-shading and sharp-edge options.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - `impression preview` and embedded consumers call `PreviewSceneController.apply_scene(...)`.
- Background/concurrency route, if applicable:
  - not applicable; renderer setup remains on its existing calling thread.

## Prerequisite Handling

- Architecture feedback artifacts:
  - none; no architecture change.
- Architecture feedback status:
  - not applicable.
- Already implemented prerequisites:
  - Fix 15 PNG route and shared preview controller.
- Missing prerequisite architecture:
  - none.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - none.
- Progression handling:
  - current item may proceed before final artifact qualification.

## Application Integration

- App type: mixed GUI and console.
- User/caller surface: interactive `impression preview` and `preview --screenshot PATH`.
- Invocation route: model load -> dataset collection -> `PreviewSceneController.apply_scene(...)` -> PyVista mesh actor.
- Wiring owner/module: `src/impression/preview.py`.
- Observable result: flat CAD faces remain visually flat, sharp seams remain sharp, and the three audio-cube assemblies no longer show false folds/spikes.
- Integration validation: focused controller tests plus real PNG renders through the installed preview command.
- Incomplete status risk: a helper-only normal calculation would not affect the actual PyVista actors.

App-type-specific proof:

- GUI: the shared actor route is used by interactive preview; manual PNG evidence verifies equivalent render state.
- Console: the installed `preview --screenshot` command exits zero and writes the inspected images.
- Mixed: controller assertions and real command renders independently prove wiring and visible output.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `PreviewSceneController.apply_scene(...)` - shared mesh actor creation.
  - `PreviewStyle.feature_edge_angle` - existing sharp-feature threshold.
- Current reuse readiness:
  - add arguments to the existing shared module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/preview.py` - sharp-edge split actor arguments.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - not applicable.
- Functions/methods:
  - `PreviewSceneController.apply_scene(...) -> None` - configure faithful CAD actor normals.
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable; behavior corrects the default renderer.
- UI components, if applicable:
  - shared preview surface.

## Performance Contract

- Derived vertex growth is bounded by face-corner count and occurs only in the PyVista render dataset.
- No extra model tessellation or file-watch work is introduced.

## Error And State Behavior

- PyVista receives no sharp-edge split request when smooth shading is disabled.
- Unsupported renderer behavior remains visible as an ordinary preview backend failure; no geometry fallback mutates the model.

## Test Strategy

- Unit tests:
  - assert uniform and per-face actor calls use the configured split flag and angle; assert flat shading disables splitting.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - exercise the shared controller with fake PyVista actor capture.
- Integrated route tests:
  - render and inspect the original, loft, and direct-path diagonal audio-cube assemblies via `preview --screenshot`.
- Production-data rule:
  - tests use repository meshes and disposable PNGs only.

## Acceptance Criteria

- Smooth mesh actors split sharp render edges at the configured feature angle.
- Flat-shaded actors do not request sharp-edge splitting.
- Original, loft, and diagonal audio-cube assemblies render without false folds, spikes, or fragmented walls.
- Model mesh coordinates, topology analyses, and authored colors remain unchanged.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Review Score appears in front matter and matches the final calculation.
- [x] Current implementation-spec template and fresh adversarial score are recorded.
- [x] Deferral markers, missing prerequisites, and readiness blockers are absent.
- [x] Source fields, canonical status, split coverage, and review ledger are explicit.
- [x] Implementation owner, reuse decision, defaults, and data ownership are explicit.
- [x] UI fields are marked not applicable and mixed application routes are named.
- [x] Performance, privacy, test strategy, and acceptance criteria are explicit.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; independently scored from the current specification.
- Adversarial rescore basis: checked for hidden geometry mutation, separate GUI/console implementations, new controls, unbounded mesh work, missing real-route proof, and unresolved work; one shared actor-policy change owns the complete outcome.
- Functions/methods: 1 x 2 = 2
- Data structures/models: 0 x 1 = 0
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 10
- If total matches prior score, adversarial survival reason: not applicable because there was no prior score.
