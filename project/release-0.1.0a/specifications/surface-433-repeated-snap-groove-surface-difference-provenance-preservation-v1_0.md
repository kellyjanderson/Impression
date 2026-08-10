# Surface Spec 433: Repeated Snap-Groove Surface Difference Provenance Preservation (v1.0)

Date: 2026-08-09
Status: Complete
Primary ancestor: `../architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md`
Architecture ancestor: `../architecture/surfacebody-seam-adjacency-architecture.md`
Source artifact: GitHub issue #268 and `/Users/k/Documents/Projects/3d printing/testingImp/references/impression-fix-repeated-snap-groove-cuts.md`
Split provenance: none
Canonical status: Canonical
Review Score: 21
Prerequisites:
- `../architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md` - defines representation-specific field provenance re-entry.
- `surface-427c-loft-primitive-adjacency-rebuild-diagnostics-v1_0.md` - existing adjacency rebuild boundary remains implemented.
- `surface-428a-loft-primitive-runtime-validity-checker-v1_0.md` - existing accepted-result validity gate remains implemented.

## Source Field Carryover

- Source purpose:
  - Resolve GitHub issue #268 by allowing six copied snap-groove cutters to run as six public pairwise surfaced differences.
- Source responsibilities by category:
  - Functions/methods: recognize an accepted polygon-loft field root, compose the next difference, and normalize declarative field-change evidence.
  - Data structures/models: reuse `ImplicitFieldNode`, `ImplicitSurfacePatch`, and `SurfaceBooleanResult`; no new public DTO.
  - Dependencies/services: field provenance recognition, hard implicit difference, final validity, and the public difference success gate.
  - Returns/outputs/signals: a closed result body that is accepted by the next difference call, or a structured validity refusal.
  - UI surfaces/components: not applicable.
  - UI fields/elements: not applicable.
  - Reusable code plan: extend the existing polygon-loft field and difference-evidence boundaries in `src/impression/modeling/csg.py`.
  - Database queries/tables/migrations: not applicable.
  - Async/concurrency behavior: not applicable; the six calls execute synchronously in authored cutter order.
  - Destructive/write behavior: changes the topology and metadata attached to accepted CSG results; source bodies and cutters remain immutable.
  - Security/privacy-sensitive behavior: not applicable; diagnostics do not include full geometry payloads.
  - Performance-sensitive behavior: each call adds one bounded node to the existing declarative field graph and performs bounded evidence comparison.
  - Cross-screen reusable behavior: not applicable.
- Source open questions / nuance discovered:
  - Resolved by implementation feedback through the ACD: a one-patch implicit result re-enters through field provenance, not fabricated explicit seams; six explicit public calls remain the required proof even though adaptable batch difference may also succeed.
- Source split/provenance notes:
  - Not split. Seam reconstruction, validity, and eligibility form one inseparable re-entry invariant at the accepted-result boundary.

## Purpose

Ensure a successful polygon-loft field difference returns a validated,
provenance-bearing `SurfaceBody` whose declarative field root can immediately
serve as the base of the next copied snap-groove difference.

## Scope

Owns:

- field-root and Boolean provenance on successful polygon-loft difference results;
- representation-correct re-entry without explicit-shell seam requirements;
- declarative field-graph geometry-change evidence for the public success gate;
- public sequential completion of all six northwest snap-groove cutters;
- structured validity refusal when a result cannot meet the re-entry invariant.

Does not own:

- general public batch-difference lowering outside this field route;
- attached-feature union composition;
- changes to snap-groove cutter construction, transforms, or dimensions;
- mesh fallback, export tessellation, or unrelated cutter families.

## Split Coverage

- Parent spec: none.
- Parent coverage status: not applicable.
- Parent responsibilities owned by this child:
  - not applicable.
- Parent responsibilities still missing from children:
  - none.

## Refinement History

| Request ledger | Latest pass | Active specs reviewed | New leaves created this round | Fixed-point status |
|---|---:|---|---|---|
| `../spec-refinement-history/surface-432-433-open-csg-remediations-20260809-214523.md` | 1 | Surface Specs 432 and 433 | none | reached |

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - field-root adaptation, composition, accepted-result validity, metadata carryover, and eligibility handoff.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - existing implicit patch, shell, field, and body contracts.
  - `src/impression/modeling/loft.py` - existing loft shell-validity evidence consumed by CSG eligibility.
  - `project/release-0.1.0a/architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md` - result re-entry invariant.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - reusable accepted-result finalization boundary.
- Tests:
  - `tests/csg_reference_fixtures.py` - self-contained shell and six copied snap-groove cutters derived from issue #268.
  - `tests/test_surface_csg.py` - field re-entry, geometry-change, and six-step public difference integration.

## Chosen Defaults / Parameters

- Execute the six cutters as six explicit `boolean_difference(current_body, (cutter,))` calls in authored tuple order.
- Require every accepted non-empty intermediate body to have one connected closed shell containing one bounded `ImplicitSurfacePatch`.
- Preserve `boolean_provenance`, `polygon_loft_field_csg`, operand ids, and the declarative field root required for re-entry.
- Recognize that root before explicit loft eligibility so an implicit result is not rejected for missing patch seams.
- Treat a newly composed field graph as validated geometry-change evidence when cutter contact is present.
- If finalization cannot establish the invariant, return a structured non-success result with no body; never return a partial or mesh-backed result.
- A multi-cutter batch may succeed on the same field route, but is not the re-entry acceptance route for this leaf.

## Data Ownership

- Source of truth: the validated declarative field root on the accepted one-patch implicit `SurfaceBody`; route metadata authorizes reuse but does not replace the field graph.
- Read ownership: polygon-loft adaptation reads the result patch field and bounded route provenance.
- Write ownership: `src/impression/modeling/csg.py` creates the next field graph, result body metadata, and normalized change evidence.
- Derived/cache data: decomposition rows and evidence records are recomputable from operands and the field graph.
- Privacy/logging constraints: diagnostics may name missing field/boundary evidence and stable ids but must not serialize complete geometry.

## Dependencies And Routes

- Domain/service dependencies:
  - `_surface_body_polygon_loft_field_node` - original-loft adaptation and accepted-result re-entry.
  - `_compose_implicit_root` and `_surface_boolean_polygon_loft_field_result` - next difference composition and result construction.
  - `finalize_surface_csg_validity_gate` and `compare_surface_difference_geometry` - accepted-result and truthful-success gates.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; each call completes before its body is passed to the next call.

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md` - created during review to define result re-entry topology and metadata precedence.
- Architecture feedback status:
  - tracked in ACD; target structure, ownership, routes, and result invariants are defined.
- Already implemented prerequisites:
  - `src/impression/modeling/csg.py` - field adaptation/composition, result construction, validity gate, and loft eligibility boundaries.
  - `src/impression/modeling/surface.py` - canonical implicit patch, shell, field, and body records.
- Missing prerequisite architecture:
  - none.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - none.
- Progression handling:
  - current item may proceed from the active ACD and existing pairwise difference route.

## Application Integration

- App type: library-only
- User/caller surface: downstream model modules calling `boolean_difference`
- Invocation route: public pairwise difference, field adaptation/composition, validity and truthful-success gates, returned body, and next public pairwise difference
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: six consecutive succeeded results ending in one closed field body with six nested difference compositions; adaptable batch input may also succeed
- Integration validation: one test loops over the six fixture cutters through the public API and validates every intermediate result before continuing
- Incomplete status risk: copying route metadata without reusing the actual field root, or unit-testing a private adapter without feeding each public result into the next call, does not satisfy this spec

App-type-specific proof:

- Library-only: `tests/test_surface_csg.py` must consume each returned body in the next public `boolean_difference` call and inspect all six public results.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `_compose_implicit_root` - retain the existing hard implicit difference semantics.
  - `SurfaceBooleanResult` - retain the existing public success/refusal envelope.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none; repair the existing result assembly, rebuild, and validity boundaries.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - field-root re-entry, route metadata, and declarative geometry-change evidence.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `ImplicitFieldNode` - canonical reusable declarative geometry graph.
  - `ImplicitSurfacePatch` - bounded surface-native result representation.
  - `SurfaceBooleanResult` - public accepted or refused result.
- Functions/methods:
  - `_surface_body_polygon_loft_field_node(body) -> ImplicitFieldNode | None` - return an original polygon-loft node or the validated root of a prior accepted field result.
  - `_surface_boolean_polygon_loft_field_result(operands) -> SurfaceBooleanResult | None` - compose and finalize the next field difference.
  - `compare_surface_difference_geometry(...) -> NormalizedDifferenceEvidence` - recognize a changed declarative implicit graph as localized geometry change.
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Each call adapts one prior root and one cutter and adds one bounded difference node; graph safety limits remain authoritative.
- The six-cutter acceptance fixture performs exactly six pairwise public differences and no tessellation or unbounded topology search.

## Error And State Behavior

- A successful intermediate result is consumed only after asserting non-null body, closed classification, one connected shell, field-route provenance, and changed-geometry evidence.
- A failed intermediate result stops the caller loop and exposes no body for the next step.
- Batch difference may return `unsupported`, but its failure remains structured and no mesh fallback is attempted.
- Input shell and cutter objects remain unchanged.
- No retry, stale-result, loading, or empty-UI state applies to this synchronous library route.

## Test Strategy

- Unit tests:
  - original polygon lofts and prior accepted field results adapt deterministically while unrelated implicit bodies do not gain unauthorized re-entry.
  - validity and difference evidence agree on accepted field-composed bodies.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - apply all six copied northwest snap-groove cutters sequentially through `boolean_difference`.
  - assert every intermediate result's field provenance, validity, changed-geometry evidence, and no-hidden-mesh evidence before the next call.
  - assert the final field graph contains six deterministic nested difference compositions and geometry changes after every step.
  - assert a six-cutter batch either succeeds fully or returns structured unsupported with no body.
- Production-data rule:
  - tests use deterministic self-contained geometry fixtures and do not import the sibling audio-cube project or production data.

## Acceptance Criteria

- Six consecutive `boolean_difference(current_body, (cutter,))` calls return `SurfaceBooleanResult(status="succeeded", classification="closed")` with non-null bodies.
- Before each next call, the current body has `shell_count == 1`, one bounded implicit patch, polygon-loft field provenance, and a reusable field root.
- The final body carries six nested difference compositions and validated geometry-change evidence at every step; status alone is insufficient.
- A six-cutter batch either succeeds through the same mesh-free field route or returns a structured bodyless refusal.
- No input body/cutter is mutated and no accepted result is a `Mesh` or `MeshGroup`.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Review Score appears in the front matter and exactly matches the total in the final Review Score Calculation section.
- [x] The current implementation-spec template was loaded and its source path is recorded in the final Review Score Calculation section.
- [x] Review Score is adversarially recounted from the current spec text; the missing creation score was not trusted.
- [x] Unresolved deferral/gap markers are absent; batch difference is an explicit non-goal, not postponed work.
- [x] Source fields are carried into spec sections or preserved as explicit provenance/history.
- [x] Canonical status is explicit.
- [x] Prerequisites are linked and implemented, or represented by the active architecture-transition ACD.
- [x] Missing architecture discovered during review is linked through an ACD.
- [x] Missing prerequisite behavior is marked not applicable.
- [x] Split coverage is marked not applicable.
- [x] Per-request review ledger records the terminal new-leaf list.
- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing module additions are named; no new public module applies.
- [x] UI fields/elements are marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] GUI/concurrency routes are marked not applicable.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] Library-only proof matches the app type.
- [x] Performance bounds are explicit.
- [x] Privacy/logging constraints are explicit.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none; the obsolete IWU annotation was not a Review Score.
- Adversarial rescore basis: counted field adaptation, field composition/result construction, and validity/evidence finalization as separate method responsibilities; counted field graph, implicit result patch, and public result models; counted adaptation, composition, and validity/success dependencies; counted topology mutation at the result boundary and linear field-graph cost; verified batch difference is accepted but not the sequential re-entry proof and no UI, database, async, security, cross-screen, prerequisite, readiness, or deferral points apply.
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Missing prerequisites: 0 x 2 = 0
- Unresolved deferral/gap markers: 0 x 100 = 0
- Total: 21
- If total matches prior score, adversarial survival reason: not applicable; no prior Review Score existed.
