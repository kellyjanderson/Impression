# Surface Spec 403: Loft Boundary Graph And Seam Coverage Evidence (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-loft-shell-connectivity-and-closure-evidence.md`
Architecture ancestor: `../architecture/acd-loft-shell-connectivity-and-closure-evidence.md`
Manifest source: `Loft Boundary Graph And Seam Coverage Evidence`
Split provenance: none
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one loft boundary graph and seam coverage evidence route.

## Purpose

Build the loft-owned boundary graph and seam coverage evidence that downstream
shell validity and CSG eligibility consume.

## Scope

Owns:

- `LoftBoundaryGraph` and `LoftSeamCoverageRecord`.
- Loft boundary graph building and seam coverage classification.
- Complete, missing, duplicate, and dangling seam diagnostics.

Does not own:

- Closure or cap validity; see Surface Spec 404.
- CSG route selection or execution.

## Split Coverage

- Parent spec: none
- Parent coverage status: not applicable
- Parent responsibilities owned by this child: not applicable
- Parent responsibilities still missing from children: none

## Implementation Routing

- Primary modules/files:
  - `src/impression/modeling/loft.py` - graph builder and seam coverage classifier.
- Supporting modules/files:
  - `src/impression/modeling/surface.py` - reuse `SurfaceBoundaryRef` and `SurfaceSeam`.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - `src/impression/modeling/loft.py` - reusable loft boundary evidence helpers.
- Tests:
  - `tests/test_loft_surface_body.py` - loft executor graph and seam evidence tests.
  - `tests/test_surface_csg.py` - public CSG eligibility probe.

## Chosen Defaults / Parameters

- Every non-cap side boundary is accounted for by a station seam, transition
  seam, cap seam, or explicit open-boundary diagnostic.
- Missing or duplicate seam coverage is diagnostic evidence, not guessed
  topology.
- Evidence records are deterministic and stable across identical loft inputs.

## Data Ownership

- Source of truth: `src/impression/modeling/loft.py`.
- Read ownership: shell validity and CSG eligibility read evidence through the
  returned loft `SurfaceBody` metadata.
- Write ownership: only the loft executor writes boundary graph evidence.
- Derived/cache data: graph evidence can be recomputed from loft patch role
  metadata and seam refs.
- Privacy/logging constraints: not applicable.

## Dependencies And Routes

- Domain/service dependencies:
  - `SurfaceBoundaryRef`
  - `SurfaceSeam`
  - loft executor patch role metadata
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: library-only
- User/caller surface: public modeling API consumers creating loft-authored
  `SurfaceBody` values for CSG operands
- Invocation route: loft executor call during public modeling API construction
- Wiring owner/module: `src/impression/modeling/loft.py`
- Observable result: loft outputs include boundary graph and seam coverage
  evidence consumed by shell validity and CSG eligibility checks
- Integration validation: focused loft executor tests plus public CSG
  eligibility probes
- Incomplete status risk: implemented in isolation if graph evidence is not
  exposed through the shell validity summary consumed by CSG

App-type-specific proof:

- Library-only: public loft construction and boolean eligibility probes validate
  the consuming modules.

## Reuse And Extraction Plan

- Existing code to reuse:
  - `SurfaceSeam` - seam identity and adjacency.
  - `SurfaceBoundaryRef` - boundary identity.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - `src/impression/modeling/loft.py` - graph builder and classifier helpers.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - `LoftBoundaryGraph` - patch boundary refs, seam refs, and component edges.
  - `LoftSeamCoverageRecord` - complete/missing/duplicate/dangling coverage.
- Functions/methods:
  - `build_loft_boundary_graph(...) -> LoftBoundaryGraph`
  - `classify_loft_seam_coverage(...) -> LoftSeamCoverageRecord`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by loft patch and seam count.
- No global geometry scan or tessellation is allowed.

## Error And State Behavior

- Missing boundary refs produce deterministic diagnostics.
- Duplicate or dangling seams are recorded without mutating source geometry.
- Empty or malformed loft inputs continue through existing loft validation
  before graph construction.

## Test Strategy

- Unit tests:
  - complete, missing, duplicate, and dangling seam coverage.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - public loft construction followed by CSG eligibility evidence inspection.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Loft outputs expose deterministic boundary graph evidence.
- Missing and invalid seam coverage produce structured diagnostics.
- CSG eligibility can consume graph evidence without deriving loft topology.

## Rescore And Split Review

- Manifest score: 15.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Small; remains one loft evidence contract.
- Review update: checked after promotion from ACD manifest; no child spec is required before implementation.

## Readiness Checklist

- [x] Primary ancestor and architecture ancestor are explicit.
- [x] Canonical status is explicit.
- [x] Split coverage is complete, or marked not applicable.
- [x] Implementation owner/module is named.
- [x] Existing code reuse/extraction decision is explicit.
- [x] Existing library/module additions or new reusable module boundaries are named, or marked not applicable.
- [x] UI fields/elements are listed, or marked not applicable.
- [x] Chosen defaults are explicit.
- [x] Data source of truth and write owner are explicit.
- [x] GUI/concurrency route is explicit, or marked not applicable.
- [x] App type and application integration route are explicit.
- [x] Integrated route validation is named.
- [x] GUI/console/API-service/mixed/library-only proof matches the app type.
- [x] Performance bounds are explicit, or marked not applicable.
- [x] Privacy/logging constraints are explicit, or marked not applicable.
- [x] Test strategy does not depend on production data.
- [x] Acceptance criteria are testable.

## Review History

- Pass 1 - Template: Promoted from ACD manifest with all readiness fields.
- Pass 2 - Routing: Confirmed `loft.py` owns writes and CSG only reads evidence.
- Pass 3 - Rescore: Manifest score 15; no split required.
- Pass 4 - Blocker Review: Readiness blockers resolved; no predecessor needed.
- Pass 5 - Final: Ready as a final leaf specification.
