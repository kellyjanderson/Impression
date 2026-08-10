# Surface Spec 432: Attached Polygon-Loft Surface Union Completion (v1.0)

Date: 2026-08-09
Status: Complete
Primary ancestor: `../architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md`
Architecture ancestor: `../architecture/surfacebody-csg-architecture.md`
Source artifact: GitHub issue #267 and `/Users/k/Documents/Projects/3d printing/testingImp/references/impression-fix-attached-polygon-loft-union.md`
Split provenance: none
Canonical status: Canonical
Review Score: 18.5
Prerequisites:
- `../architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md` - defines deterministic polygon-loft field composition and structured route refusal.
- `../adhoc/2026-07-09-csg-reference-03-multi-operand-boolean-composition.md` - established input-order-independent union composition behavior that this loft route must preserve.
- `surface-395-loft-loft-pair-operation-routes-v1_0.md` - existing two-body loft/loft execution remains unchanged for non-field routes.

## Source Field Carryover

- Source purpose:
  - Resolve GitHub issue #267 by fusing already-authored attached polygon-loft features into one printable surface body.
- Source responsibilities by category:
  - Functions/methods: adapt polygon-loft bodies to declarative fields and execute canonical N-ary field union.
  - Data structures/models: reuse `SurfaceBooleanOperands` and `SurfaceBooleanResult`; no new public DTO.
  - Dependencies/services: polygon-loft field adaptation, hard implicit union composition, and the final surface CSG validity gate.
  - Returns/outputs/signals: one succeeded non-empty `SurfaceBooleanResult`, or a structured unsupported result with no partial body.
  - UI surfaces/components: not applicable.
  - UI fields/elements: not applicable.
  - Reusable code plan: extend the existing polygon-loft field route in `src/impression/modeling/csg.py` and reuse the no-hidden-mesh guard.
  - Database queries/tables/migrations: not applicable.
  - Async/concurrency behavior: not applicable; execution is synchronous and operand order is deterministic.
  - Destructive/write behavior: changes public union execution for eligible multi-operand `SurfaceBody` calls; input bodies remain immutable.
  - Security/privacy-sensitive behavior: not applicable; diagnostics contain stable ids and bounded failure text, not geometry payloads.
  - Performance-sensitive behavior: one bounded field adaptation per operand and one N-ary composition node.
  - Cross-screen reusable behavior: not applicable.
- Source open questions / nuance discovered:
  - Resolved by implementation feedback through the ACD: union operands use canonical stable-identity ordering, require a connected contact graph, and return no partial body on refusal.
- Source split/provenance notes:
  - Not split. This leaf owns one operation-specific composition route; successful-result re-entry is owned by Surface Spec 433.

## Purpose

Make the public surface-union contract able to fuse one enclosure shell with
its attached snap-tab or microphone-rail polygon-loft bodies while preserving
surface-native model truth.

## Scope

Owns:

- canonical N-ary declarative field composition for eligible connected polygon-loft unions;
- public-route success for the northwest snap-tab and microphone-rail fixtures;
- structured refusal with no partial body when adaptation or contact validation fails.

Does not own:

- successful-result seam/provenance re-entry for repeated differences;
- public batch-difference lowering;
- N-ary intersection, branching-loft decomposition, or mesh execution;
- export-boundary tessellation.

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
  - `src/impression/modeling/csg.py` - adapt, canonicalize, and compose polygon-loft field unions.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - existing immutable `SurfaceBody` and shell truth.
  - `project/release-0.1.0a/architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md` - composition and failure policy.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - existing public CSG and polygon-loft field boundary.
- Tests:
  - `tests/csg_reference_fixtures.py` - self-contained attached snap-tab and microphone-rail fixture builders derived from issue #267.
  - `tests/test_surface_csg.py` - public-route success, refusal, ordering, and regression assertions.

## Chosen Defaults / Parameters

- Require every operand to adapt to a bounded declarative polygon-loft field node.
- Require a connected AABB contact graph so disconnected component sets do not masquerade as one connected shell.
- Canonicalize equivalent union sets by stable body identity and compose one hard implicit union node.
- On failure, return a structured non-success result with no partial body.
- Return a final `SurfaceBooleanResult` whose `operands` remains the original prepared multi-operand request.
- Store canonical-order `boolean_operand_ids` and Boolean provenance on the final body so equivalent input permutations retain the same body identity.
- Do not expose a new public sequence API or DTO.

## Data Ownership

- Source of truth: the final `SurfaceBooleanResult`; its `operands` preserves request order while its surface-native body metadata preserves canonical execution order.
- Read ownership: `src/impression/modeling/csg.py` reads immutable prepared operands and derives canonical execution order without changing the retained request.
- Write ownership: `src/impression/modeling/csg.py` creates intermediate and final result bodies without mutating input bodies.
- Derived/cache data: adapted field roots and the contact graph are request-local and disposable.
- Privacy/logging constraints: failure text may include operation, operand index, and stable body ids; it must not serialize full geometry payloads.

## Dependencies And Routes

- Domain/service dependencies:
  - `surface_boolean_result` - public prepared-operation dispatch.
  - `_surface_body_polygon_loft_field_node` - bounded operand adaptation.
  - `_compose_implicit_root` - canonical hard field union.
  - `finalize_surface_csg_validity_gate` - final topology validation.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable; field adaptation and composition are synchronous.

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-surface-csg-pairwise-composition-and-result-reentry.md` - created during review and corrected during implementation to define field composition and re-entry semantics.
- Architecture feedback status:
  - tracked in ACD; target structure, ownership, routes, and data boundaries are defined.
- Already implemented prerequisites:
  - `../adhoc/2026-07-09-csg-reference-03-multi-operand-boolean-composition.md` - generic multi-operand ordering and failure contract.
  - `surface-395-loft-loft-pair-operation-routes-v1_0.md` - pairwise loft/loft route baseline.
  - `src/impression/modeling/csg.py` - pairwise loft selection/execution and public result envelope.
- Missing prerequisite architecture:
  - none.
- Missing prerequisite specifications:
  - none.
- Unimplemented prerequisite specifications:
  - none.
- Progression handling:
  - keep pairwise loft and orthogonal coplanar union regressions passing while validating the new field route.

## Application Integration

- App type: library-only
- User/caller surface: downstream model modules calling `boolean_union`
- Invocation route: `boolean_union` to prepared operands, polygon-loft field adaptation, canonical hard union, validity gate, and final `SurfaceBooleanResult`
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: one closed fused `SurfaceBody` for each issue fixture, or a structured refusal with no partial body
- Integration validation: public `boolean_union` tests using the complete attached-feature operand tuples
- Incomplete status risk: a helper-only fold that is not reached by `boolean_union`, or a fold that leaks a partial body, does not satisfy this spec

App-type-specific proof:

- Library-only: `tests/test_surface_csg.py` must call `boolean_union` with both complete fixture tuples and inspect the returned public result.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `surface_boolean_result` - retain the public prepared-operation dispatch boundary.
  - `_surface_body_polygon_loft_field_node` - extend the existing difference adapter.
  - `_compose_implicit_root` - reuse hard implicit composition.
  - `finalize_surface_csg_validity_gate` - validate the final result.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none; generalize the existing private polygon-loft field executor.
- Additions to existing library/modules:
  - `src/impression/modeling/csg.py` - canonical union ordering, contact validation, and field composition.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `SurfaceBooleanOperands` - retains the original ordered public request.
  - `SurfaceBooleanResult` - carries final success or structured refusal; no new fields.
- Functions/methods:
  - `_surface_body_polygon_loft_field_node(body) -> ImplicitFieldNode | None` - adapt an original polygon loft or re-enter a prior accepted polygon-loft field result.
  - `_surface_boolean_polygon_loft_field_result(operands) -> SurfaceBooleanResult | None` - validate contact, canonicalize union order, compose the field graph, and finalize one closed result.
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Perform at most one bounded adaptation per operand plus a quadratic AABB contact-graph scan and one N-ary composition.
- Do not build an exponential pair plan or tessellate.

## Error And State Behavior

- Polygon-loft union requests with three or more adaptable operands use the field route; established two-body dispatch remains unchanged.
- Empty, invalid, unsupported, disconnected, or mesh-backed results expose no partial body.
- The original orthogonal coplanar union remains a required regression.
- No retry, stale-result, loading, or empty-UI state applies to this synchronous library route.

## Test Strategy

- Unit tests:
  - canonical ordering across permutations, original-request retention, disconnected-contact refusal, and no partial body.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public union of the simplified northwest shell with every northwest snap-tab body.
  - public union of the simplified northwest shell with both microphone-rail bodies.
  - existing orthogonal coplanar union regression.
  - `assert_no_hidden_surface_csg_mesh_fallback` remains effective for every returned result.
- Production-data rule:
  - tests use deterministic self-contained geometry fixtures and do not require sibling-project imports or production data.

## Acceptance Criteria

- `boolean_union((shell, *snap_tabs))` returns `SurfaceBooleanResult(status="succeeded", classification="closed")` with a non-null body.
- `boolean_union((shell, *microphone_rails))` returns `SurfaceBooleanResult(status="succeeded", classification="closed")` with a non-null body.
- Permutations of each equivalent operand set produce the same stable result identity and classification.
- Each successful body has `shell_count == 1` and contains geometry contributed by every attached feature, asserted by deterministic patch/provenance evidence rather than status alone.
- A disconnected or non-adaptable request does not return a falsely connected partial body.
- No accepted route returns `Mesh` or `MeshGroup`, and the existing orthogonal coplanar union regression passes.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Review Score appears in the front matter and exactly matches the total in the final Review Score Calculation section.
- [x] The current implementation-spec template was loaded and its source path is recorded in the final Review Score Calculation section.
- [x] Review Score is adversarially recounted from the current spec text; the missing creation score was not trusted.
- [x] Unresolved deferral/gap markers are absent; explicit non-goals are resolved scope decisions.
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
- Adversarial rescore basis: counted field adaptation plus composition/result construction separately; retained the original request and final result as explicit models; counted adaptation, canonical field composition, and validity as dependencies; counted canonical ordering inside composition rather than inventing a third function; counted the public behavior change and linear operand bound; verified UI, database, async, security, cross-screen, prerequisite, readiness, and deferral categories do not apply.
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 3 x 0.5 = 1.5
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
- Total: 18.5
- If total matches prior score, adversarial survival reason: not applicable; no prior Review Score existed.
