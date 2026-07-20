# Surface Spec 418: Loft CSG Section Artifact Generation (v1.0)

Date: 2026-07-15
Status: Proposed
Primary ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Architecture ancestor: `../architecture/acd-reference-fixture-multi-artifact-section-evidence-policy.md`
Manifest source: `Loft CSG Section Artifact Generation`
Split provenance: `Section Evidence Bundle Producer`
Canonical status: Canonical

## Work Units

Unit: Implementation Work Unit (IWU).
Definition: one independently deliverable, reviewable change set with its own verification surface. An IWU is intentionally abstract so the same unit can size software, documentation, tooling, service, design, and process projects.
Standard measures: count 1 IWU when the work has one primary outcome, one coherent responsibility boundary, one reviewable artifact or change set, one explicit verification method, declared inputs and outputs, and explicitly named unresolved assumptions or decisions. Split the work when any measure becomes plural, ambiguous, or unnamed.
Count: 1 IWU.
Basis: one section artifact generation route producing expected, actual, and diff artifacts.

## Purpose

Generate expected, actual, and diff section artifacts for loft CSG reference
fixtures.

## Scope

Owns:

- Section evidence generator.
- Bundle artifact writer.
- Dirty expected/actual/diff artifact output.

Does not own:

- Section evidence contract; see Surface Spec 417.
- Fixture registry integration; see Surface Spec 419.

## Split Coverage

- Parent spec: `Section Evidence Bundle Producer`
- Parent coverage status: 100% covered with Surface Specs 417-419.
- Parent responsibilities owned by this child:
  - loft CSG section artifact generation.
- Parent responsibilities still missing from children: none.

## Implementation Routing

- Primary modules/files:
  - `tests/reference_images.py` - section artifact writer.
  - `tests/reference_review_fixtures/stl_review_sources.py` - fixture source helper.
- Supporting modules/files:
  - computer-vision section utilities.
  - Surface Spec 407 loft CSG result geometry proof.
- GUI/QML files, if applicable:
  - not applicable.
- Reusable library/module files:
  - loft CSG section bundle helper in existing test/reference helpers.
- Tests:
  - `tests/test_reference_stl_generation.py` - section artifact generation tests.

## Chosen Defaults / Parameters

- Required output roles are `expected`, `actual`, and `diff`.
- Dirty artifacts are written at deterministic paths.
- Generated section evidence is tied to declared section planes.

## Data Ownership

- Source of truth: CSG result geometry.
- Read ownership: section generator reads public CSG result geometry.
- Write ownership: reference update workflow writes dirty artifacts.
- Derived/cache data: section images can be regenerated from CSG result and
  section plane metadata.
- Privacy/logging constraints: artifact paths stay under reference roots.

## Dependencies And Routes

- Domain/service dependencies:
  - Surface Spec 407 result geometry reference proof.
  - Surface Spec 417 section evidence contract records.
  - CV section utilities.
- Database dependencies:
  - not applicable.
- GUI route, if applicable:
  - not applicable.
- Background/concurrency route, if applicable:
  - not applicable.

## Application Integration

- App type: workflow
- User/caller surface: reference artifact generation tests
- Invocation route: successful loft CSG result to section artifact writer during
  reference update workflow
- Wiring owner/module: `tests/reference_images.py`,
  `tests/reference_review_fixtures/stl_review_sources.py`
- Observable result: dirty expected, actual, and diff section artifacts are
  written at deterministic paths
- Integration validation: reference update test producing all required section
  artifact roles
- Incomplete status risk: implemented in isolation if artifact files are written
  but not returned as fixture evidence bundles

App-type-specific proof:

- Workflow: reference update tests validate dirty section artifact side effects.

## Reuse And Extraction Plan

- Existing code to reuse:
  - CV section artifact helpers.
- Current reuse readiness:
  - add to existing library/module.
- Extraction/wrapping needed:
  - none.
- Additions to existing library/modules:
  - loft CSG section bundle helper.
- New reusable modules to expose:
  - none.
- One-off code justification, if any:
  - none.

## Required DTOs / Functions / Components

- DTOs/models:
  - section evidence bundle payload.
- Functions/methods:
  - `generate_loft_csg_section_evidence(...) -> SectionEvidencePayload`
  - `write_section_bundle_artifacts(...) -> list[Path]`
- UI fields / visible data, if applicable:
  - not applicable.
- UI elements / controls, if applicable:
  - not applicable.
- UI components, if applicable:
  - not applicable.

## Performance Contract

- Bounded by section sample count and image size.
- Generation must not repeat CSG execution if result geometry is already provided.

## Error And State Behavior

- Missing result geometry refuses generation with diagnostics.
- Missing required roles refuses bundle output.
- Partial artifact writes are diagnosed and not treated as complete evidence.

## Test Strategy

- Unit tests:
  - section payload construction and artifact writer behavior.
- Service/DB tests:
  - not applicable.
- GUI/controller tests, if applicable:
  - not applicable.
- Integrated route tests:
  - reference update test producing all section roles.
- Production-data rule:
  - Tests must not require production data.

## Acceptance Criteria

- Expected, actual, and diff dirty artifacts are generated.
- Artifacts are deterministic and tied to section plane metadata.
- Surface Spec 419 can attach generated artifacts to fixture records.

## Rescore And Split Review

- Manifest score: 16.5.
- IWU count: 1.
- Readiness blockers: none.
- Split decision: Review for split; the three required section roles are generated and verified atomically.
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

- Pass 1 - Template: Promoted from split manifest parent.
- Pass 2 - Routing: Confirmed artifact generation is separate from registry integration.
- Pass 3 - Rescore: Manifest score 16.5; review-for-split band retained.
- Pass 4 - Split Review: Three required section roles remain one generation route.
- Pass 5 - Final: Ready as a final leaf specification after Surface Specs 407 and 417.
