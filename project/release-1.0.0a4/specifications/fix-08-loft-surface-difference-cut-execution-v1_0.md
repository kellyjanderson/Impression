# Fix 08: Loft Surface Difference Cut Execution

Date: 2026-08-04
Status: Proposed
Primary ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Architecture ancestor: [Active ACD](../architecture/acd-surface-boolean-correctness-and-api-boundary.md)
Source artifact: [GitHub issue #248](https://github.com/kellyjanderson/Impression/issues/248)
Split provenance: Issue #248 is split by `../planning/known-issue-intake.md`; this leaf owns cut construction and branch decomposition while Fix 09 owns shared no-op validation.
Canonical status: Draft
Review Score: pending independent review
Prerequisites:
- `fix-05-count-changing-region-identity-preservation-v1_0.md` - supplies reliable lineage for branched loft decomposition
- `fix-09-surface-difference-no-op-result-gate-v1_0.md` - blocks false success for every new executor result

## Source Field Carryover

- Source purpose: Construct real changed surface geometry for USB-C, acoustic, and snap-pocket loft cutters, including validated branch decomposition and recomposition where topology requires it.
- Source responsibilities by category:
  - Functions/methods: intersect, trim, fragment, classify, add reversed cutter patches, decompose/recompose branches, validate
  - Data structures/models: intersection curve, trim fragment, branch decomposition, provenance, and result-shell evidence
  - Dependencies/services: surface evaluators/intersections, seam graph, topology lineage, CSG result gate
  - Returns/outputs/signals: changed closed `SurfaceBody` with intended opening/pocket or precise unsupported/invalid result
  - UI surfaces/components: not applicable; downstream preview/export consumer
  - UI fields/elements: not applicable
  - Reusable code plan: extend existing loft surface CSG evidence and reconstruction
  - Database queries/tables/migrations: not applicable
  - Async/concurrency behavior: not applicable
  - Destructive/write behavior: no destructive runtime writes; preview/export smoke may write temporary artifacts
  - Security/privacy-sensitive behavior: not applicable
  - Performance-sensitive behavior: bounds-pruned patch candidates and bounded branch decomposition; no whole-body dense sampling
  - Cross-screen reusable behavior: not applicable
- Source open questions / nuance discovered: none hidden; independent review may refine split cohesion and exact symbol names.
- Source split/provenance notes: Issue #248 is split by `../planning/known-issue-intake.md`; this leaf owns cut construction and branch decomposition while Fix 09 owns shared no-op validation.

## Purpose

Construct real changed surface geometry for USB-C, acoustic, and snap-pocket loft cutters, including validated branch decomposition and recomposition where topology requires it.

## Scope

- Owns:
  - surface intersection evidence and patch-local trim curves
  - base/cutter fragmentation, retained-fragment classification, cutter-derived closure patches, and shell rebuild
  - bounded branch-graph decomposition/recomposition with provenance and seam validation
  - public difference, audio-cube, preview/export, closure, and no-mesh regressions

- Does not own:
  - the shared unchanged-result success gate, owned by Fix 09
  - universal support for every surface family or arbitrary underconstrained topology

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none
- Issue-level split disposition: Issue #248 is split by `../planning/known-issue-intake.md`; this leaf owns cut construction and branch decomposition while Fix 09 owns shared no-op validation.

## Refinement History

Not applicable before review. No request review ledger exists; this is a do-specs creation draft.

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/csg.py` - loft difference eligibility, intersection, trimming, reconstruction, branch handling, result evidence
- Supporting modules/files:
  - `src/impression/modeling/loft.py` - read-only branch/topology lineage contract
  - `src/impression/modeling/surface.py` - patch/seam/body invariants if extension is required
- GUI/QML files, if applicable:
  - none; no QML is involved
- Reusable library/module files:
  - `src/impression/modeling/csg.py` - loft difference eligibility, intersection, trimming, reconstruction, branch handling, result evidence
- Tests:
  - `tests/test_surface_csg.py` - public difference and evidence
  - `tests/csg_reference_fixtures.py` - USB-C, acoustic, snap-pocket, branch controls
  - test-model preview/export smoke - consumer proof without modeling mesh fallback

## Chosen Defaults / Parameters

- require closed trim loops and deterministic oriented inside/outside classification
- retain minuend fragments outside cutters and add correctly oriented cutter-derived boundary patches
- decompose branching lofts only when lineage/provenance is complete and recomposition validates
- route every result through Fix 09 and shared body validity gates

## Data Ownership

- source of truth: immutable operand surface topology and intersection evidence
- read ownership: difference executor and branch decomposer
- write ownership: result assembler creates new patch/seam topology; operands remain unchanged
- derived/cache data: candidates, trim fragments, classification, and branch records are recomputable
- privacy/logging: not applicable

## Dependencies And Routes

- Domain/service dependencies:
  - Fix 05 lineage, Fix 09 no-op gate, surface intersections/evaluators, seam rebuild, result envelope
  - library route: `boolean_difference` -> eligibility/decomposition -> intersection/fragment reconstruction -> result gates -> consumer
- Database dependencies:
  - none
- GUI route, if applicable:
  - not applicable
- Background/concurrency route, if applicable:
  - not applicable

## Prerequisite Handling

- Architecture feedback artifacts:
  - `../architecture/acd-surface-boolean-correctness-and-api-boundary.md` - owns trim reconstruction and branch handling
- Architecture feedback status:
  - tracked in active ACD
- Already implemented prerequisites:
  - surface evaluators/intersections, patch provenance, seam rebuild, and structured CSG results
- Missing prerequisite architecture:
  - none
- Missing prerequisite specifications:
  - none
- Unimplemented prerequisite specifications:
  - Fix 05 and Fix 09
- Progression handling:
  - implement after Fix 05 and Fix 09; retain precise refusal for unsupported families

## Application Integration

- App type: library-only
- User/caller surface: public `boolean_difference` consumed by the audio-cube model and preview/export
- Invocation route: surface base/cutters -> route selection -> decomposition/intersection/trims -> shell rebuild -> validators -> consumer
- Wiring owner/module: `src/impression/modeling/csg.py`
- Observable result: changed closed surfaced enclosure with intended opening/pocket
- Integration validation: public fixture suite plus real preview/export consumer with no workaround geometry
- Incomplete status risk: drafted and prerequisite-blocked

App-type-specific proof:

- GUI: not applicable
- Console: not applicable
- API/service: not applicable
- Mixed: not applicable
- Library-only: public `boolean_difference` consumed by the audio-cube model and preview/export is the consuming public route and public fixture suite plus real preview/export consumer with no workaround geometry

## Reuse And Extraction Plan

- Existing code to reuse:
  - existing code: CSG route/evidence records, surface evaluator/intersection helpers, trim fragments, seam/adjacency rebuild, result envelope
- Current reuse readiness:
  - readiness: complete existing loft trim/fragment executor
- Extraction/wrapping needed:
  - extraction: shared patch-fragment classifier and branch recomposition helpers inside CSG module
- Additions to existing library/modules:
  - readiness: complete existing loft trim/fragment executor
- New reusable modules to expose:
  - new reusable modules: none unless review identifies a stable surface-fragment boundary
- One-off code justification, if any:
  - one-off justification: none

## Required DTOs / Functions / Components

- DTOs/models:
  - intersection-curve evidence record
- Functions/methods:
  - patch-local trim-fragment record with provenance
  - inside/outside classification result
  - cutter-derived boundary patch builder
  - branch decomposition/recomposition record and result-shell assembler
- UI fields / visible data, if applicable:
  - not applicable
- UI elements / controls, if applicable:
  - not applicable
- UI components, if applicable:
  - none

## Performance Contract

- prune operand patch pairs by bounds
- branch decomposition obeys configured planner/CSG bounds
- fixture cuts complete within focused-test timeouts without dense tessellation

## Error And State Behavior

- missing closed trims, ambiguous classification, invalid branch graph, open seams, or failed body validity returns precise refusal
- no partial or unchanged body can report success
- operands remain immutable and no mesh fallback runs

## Test Strategy

- Unit tests:
  - intersection/trim evidence, fragment classification, cutter patch orientation, branch decomposition/recomposition, seam rebuild, invalid controls
- Service/DB tests:
  - not applicable
- GUI/controller tests, if applicable:
  - not applicable
- Integrated route tests:
  - USB-C, acoustic, and rotated snap-pocket fixtures through public difference, followed by preview and export consumer smoke
- Production-data rule:
  - tests use project fixtures and temporary directories; they do not require user production data

## Acceptance Criteria

- USB-C, acoustic, and rotated snap-pocket cutters produce geometry measurably different from the base and contain the intended opening or pocket.
- Every successful result is a closed `SurfaceBody` with valid caps, complete seams, operand witnesses, and geometry-change evidence.
- Validated branching loft topology is decomposed and recomposed without grouped-body, separated-rail, notch, or rim workarounds.
- Unsupported or invalid topology returns a precise refusal without partial or unchanged success.
- Preview/export consumers accept the surfaced result without mesh construction as a modeling fallback.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [ ] Review Score appears in front matter and matches a completed independent calculation.
- [x] Current implementation-spec template was loaded; its path is recorded below.
- [ ] Independent adversarial recount completed.
- [x] No unresolved placeholder is hidden as implementation-ready behavior.
- [x] Source responsibilities are carried into durable sections.
- [x] Canonical status is Draft.
- [x] Prerequisites are linked or marked not applicable.
- [x] Missing/stale architecture is tracked in the active ACD.
- [x] Missing prerequisite behavior is linked or marked not applicable.
- [x] Split coverage is recorded for issue-level splits.
- [x] Review ledger is marked not applicable before review.
- [x] Implementation owner/module and reuse/extraction decisions are named.
- [x] UI fields/elements and concurrency are explicit or not applicable.
- [x] Defaults, data ownership, app type, route, performance, privacy, and test strategy are explicit.
- [x] Acceptance criteria are observable and testable.
- [ ] Independent `review specs` confirms cohesion, scoring, canonical status, and final progression coverage.

## Review Score Calculation

- Template source: `/Users/k/Documents/Projects/.agents/process/templates/implementation-spec-template.md`
- Prior recorded score: none
- Adversarial rescore basis: pending independent `review specs`; this creation action does not count or certify categories.
- Functions/methods: pending independent review
- Data structures/models: pending independent review
- Dependencies/services: pending independent review
- Returns/outputs/signals: pending independent review
- UI surfaces/components: pending independent review
- UI fields/elements: pending independent review
- Existing reusable code reused as-is: pending independent review
- Adding code to an existing library/module: pending independent review
- Creating a new reusable library/module: pending independent review
- Database queries/tables/migrations: pending independent review
- Async/concurrency behavior: pending independent review
- Destructive/write behavior: pending independent review
- Security/privacy-sensitive behavior: pending independent review
- Performance-sensitive behavior: pending independent review
- Cross-screen reusable behavior: pending independent review
- Readiness blockers: pending independent review
- Missing prerequisites: pending independent review
- Unresolved deferral/gap markers: pending independent review
- Total: pending independent review
- If total matches prior score, adversarial survival reason: not applicable until independent review calculates a score.
