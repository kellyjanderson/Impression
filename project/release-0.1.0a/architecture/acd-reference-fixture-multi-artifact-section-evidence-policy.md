# Reference Fixture Multi-Artifact Section Evidence Policy Architectural Change Document

Date: 2026-07-14
Status: Proposed
Canonical architecture targets:

- `project/release-0.1.0a/architecture/lofted-body-csg-reference-architecture.md`
- `project/release-0.1.0a/architecture/reference-review-fixture-source-contract.md`
- `project/release-0.1.0a/architecture/reference-artifact-promotion-architecture.md`
- `project/release-0.1.0a/architecture/computer-vision-verification-architecture.md`

Related:

- Release / plan / issue: `project/release-0.1.0a/planning/progression.md`
- Release / plan / issue: `project/release-0.1.0a/planning/reference-test-expansion-plan.md`
- Parent ACD, if any: none

## Change Intent

Define how reference fixtures represent multi-artifact section evidence such as
expected/actual/diff outputs for loft CSG cases.

This change is necessary because `Surface Spec 400` requires section evidence
for `RT-LOFT-CSG-013`, but the current fixture schema treats artifacts mostly as
a flat path list. That is not enough to distinguish required evidence roles,
promotion state, diagnostic-only evidence, and review-app display semantics.

## Current Architecture

Current reference fixture records can list multiple artifact paths and can carry
diagnostic-only records with empty artifact paths. Existing CV verification
architecture describes expected/actual/diff artifacts at a high level, but the
reference fixture contract does not define a typed multi-artifact evidence
policy for section evidence.

Missing architecture:

- artifact role names and required role sets
- whether expected/actual/diff are promoted independently or as one bundle
- how dirty/gold paths are represented per role
- how review app context distinguishes render STL from section evidence
- how diagnostic section evidence is represented when a success STL is not
  available

## Target Architecture

Reference fixture records support typed evidence bundles.

The target architecture has these components:

- `ReferenceEvidenceBundleRecord`: fixture-local bundle with bundle id, evidence
  kind, role policy, and artifact records.
- `ReferenceEvidenceArtifactRecord`: typed artifact with role, kind, path,
  stage, required/optional status, and promotion behavior.
- `SectionEvidenceContractRecord`: declares expected, actual, and diff roles for
  section evidence.
- `ReferenceEvidencePromotionPolicy`: promotes or declines evidence bundles
  atomically unless a bundle explicitly allows independent role promotion.
- `ReferenceReviewEvidenceDisplayContract`: review-app contract for presenting
  non-STL evidence without making it look like the primary 3D artifact.

For `RT-LOFT-CSG-013`, the section bundle must include:

- `expected` section artifact
- `actual` section artifact
- `diff` section artifact
- declared section plane metadata
- model/fixture id and source result provenance
- deterministic dirty/gold storage paths per artifact role

## Non-Goals

- Implementing successful loft CSG routes.
- Replacing STL review artifacts.
- Defining a full generic media asset manager.
- Changing diagnostic-only refusal fixture behavior except where it shares
  bundle metadata structures.

## Canonical Document Impact

- Architecture docs to update on closure:
  - `reference-review-fixture-source-contract.md` - add typed evidence bundle
    schema.
  - `reference-artifact-promotion-architecture.md` - define bundle-level
    promotion behavior.
  - `computer-vision-verification-architecture.md` - align CV section bundles
    with fixture evidence roles.
  - `lofted-body-csg-reference-architecture.md` - replace current multi-artifact
    blocker with conformed section evidence policy.
- Specs or plans affected:
  - `surface-400-loft-csg-section-evidence-artifacts-v1_0.md` - superseded by
    child specs derived from this policy and the single-shell loft CSG route
    ACD.
  - `RT-LOFT-CSG-013` - remains unchecked until bundle policy and route result
    exist.

## Compatibility And Migration Strategy

- Existing `artifact_paths` remains valid for simple one-artifact fixtures.
- Typed `evidence_bundles` is additive.
- Review app can initially display bundle metadata in context/artifacts panels
  before specialized side-by-side visual comparison is implemented.
- Promotion must refuse partial required bundles unless an explicit independent
  role policy exists.

## Application Integration Contract

- App type: mixed
- User/caller surface: reference review fixture loader, artifact promotion, and
  review app artifacts/context panels
- Invocation route: fixture file load, review selection, approval/decline
  workflow, artifact preview workflow
- Wiring owner/module: `src/impression/devtools/reference_review`,
  `tests/reference_review_fixtures/stl_review_sources.py`
- Observable result: fixtures can carry typed section expected/actual/diff
  artifacts with deterministic review and promotion behavior
- Integration validation: fixture loader tests, promotion tests, and review-app
  context/artifact display smoke tests

## Specification Manifest for Discovery

### Five-Pass Manifest Review Notes

- Pass 1: Rechecked the two original parent candidates against the active
  manifest-entry template; both scored 25+ and therefore required splitting.
- Pass 2: Split `Reference Evidence Bundle Schema` into file fixture schema,
  database parity, and review UI display leaves so storage, persistence, and UI
  can fail independently.
- Pass 3: Split `Section Evidence Bundle Producer` into section contract
  records, artifact generation, and fixture registry integration leaves so the
  section schema can be implemented before CSG output geometry exists.
- Pass 4: Added app-type, invocation-route, observable-result, integration
  validation, and cleanup fields to every child candidate.
- Pass 5: Final rescore confirmed no child candidate is 25+; 16-24 candidates
  carry explicit cohesion reasons and all ordering dependencies are represented
  as predecessor candidates or ACDs, not readiness blockers.

### Five-Pass Manifest Review Notes - 2026-07-15

- Pass 1: Rechecked all seven leaves after the single-shell loft CSG route ACD
  was added; section artifact generation now references that ACD directly.
- Pass 2: Confirmed file fixture schema, database parity, context display,
  artifacts display, section contract, section generation, and registry
  integration remain independently failing leaves.
- Pass 3: Rescored all seven leaves; no candidate is 25+, and the highest score
  remains the artifacts tab display at 22.5.
- Pass 4: Added readiness blocker resolution records for every candidate.
- Pass 5: Final review found no split required; all sequencing is represented
  by predecessor candidates or the single-shell route ACD.

### Candidate Spec: File Fixture Evidence Bundle Schema

Discovery purpose:
- Define typed evidence bundles for JSON fixture files while preserving the
  existing simple `artifact_paths` route.

Responsibilities:
- Functions/methods:
  - file evidence bundle parser
  - evidence artifact path validator
- Data structures/models:
  - `ReferenceEvidenceBundleRecord`
  - `ReferenceEvidenceArtifactRecord`
- Dependencies/services:
  - fixture source registry
  - existing artifact path validation
- Returns/outputs/signals:
  - parsed evidence bundles
  - invalid bundle diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: fixture loading and artifact path validation
  - Additions to existing reusable library/module: file fixture evidence bundle
    records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - validates artifact paths inside allowed roots
- Performance-sensitive behavior:
  - bounded by fixture artifact count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - workflow
- User/caller surface:
  - reference fixture file loader
- Invocation route:
  - fixture JSON file load to source registry record construction
- Wiring owner/module:
  - `src/impression/devtools/reference_review/source_registry.py`
- Observable result:
  - file-backed fixtures expose typed evidence bundles without breaking
    existing one-artifact fixtures
- Integration validation:
  - fixture loader tests for valid bundles, missing roles, bad paths, and
    simple `artifact_paths` compatibility
- Incomplete status risk:
  - implemented in isolation if parsed bundles are not exposed on the fixture
    record consumed by review and promotion routes
- Implementation owner/module:
  - `src/impression/devtools/reference_review/source_registry.py`
- Chosen defaults / parameters:
  - simple `artifact_paths` remains the default for one-artifact fixtures
- Test strategy:
  - loader validation and compatibility tests
- Data ownership:
  - fixture file owns bundle declarations; source registry owns parsed records
- Routes:
  - fixture file to source record
- Reuse/extraction decision:
  - extend current fixture source contract
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Database parity is intentionally split out so file fixtures can land first.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none
- Required next action:
  - none
- Resolution artifact:
  - not applicable
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: file parsing, path validation, and parsed
  records are the same fixture load contract; database and UI responsibilities
  have been split away.

Manifest cleanup:
- Parent manifest candidate, if split: Reference Evidence Bundle Schema
- Child manifest candidates:
  - File Fixture Evidence Bundle Schema
  - Database Fixture Evidence Bundle Parity
  - Review UI Evidence Display Contract
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Database Fixture Evidence Bundle Parity

Discovery purpose:
- Add database-backed fixture support for the same evidence bundle shape used by
  file fixtures.

Responsibilities:
- Functions/methods:
  - database evidence bundle loader
  - database evidence bundle serializer
- Data structures/models:
  - database evidence bundle row payload
- Dependencies/services:
  - fixture database adapter
  - file fixture evidence bundle schema
- Returns/outputs/signals:
  - hydrated evidence bundles
  - schema parity diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: file fixture evidence bundle records
  - Additions to existing reusable library/module: database fixture mapper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - evidence bundle persistence field or table
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - fixture database migration or write-path update
- Security/privacy-sensitive behavior:
  - validates stored artifact paths inside allowed roots
- Performance-sensitive behavior:
  - bounded by fixture artifact count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - workflow
- User/caller surface:
  - database-backed reference fixture loader
- Invocation route:
  - fixture database read/write to source registry record construction
- Wiring owner/module:
  - `src/impression/devtools/reference_review/source_registry.py`
- Observable result:
  - database-backed fixtures expose the same evidence bundles as file-backed
    fixtures
- Integration validation:
  - database fixture load/save parity tests
- Incomplete status risk:
  - implemented in isolation if file and database fixtures produce divergent
    source records
- Implementation owner/module:
  - `src/impression/devtools/reference_review/source_registry.py`
- Chosen defaults / parameters:
  - database fields mirror file bundle semantics and preserve existing records
- Test strategy:
  - database migration/parity tests when database fixtures are active
- Data ownership:
  - fixture database owns persisted rows; source registry owns hydrated records
- Routes:
  - fixture database to source record
- Reuse/extraction decision:
  - reuse file fixture evidence bundle records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- If database fixtures remain inactive, this candidate can remain unselected
  while file fixture work proceeds independently.

Predecessor candidates:
- `File Fixture Evidence Bundle Schema`

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 1 x 2 = 2
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; file fixture schema ordering is represented by predecessor candidate
    `File Fixture Evidence Bundle Schema`
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `File Fixture Evidence Bundle Schema`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: database load, save, and migration must be
  validated together to prove parity with file fixtures.

Manifest cleanup:
- Parent manifest candidate, if split: Reference Evidence Bundle Schema
- Child manifest candidates:
  - File Fixture Evidence Bundle Schema
  - Database Fixture Evidence Bundle Parity
  - Review UI Evidence Display Contract
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Review UI Evidence Context Tab Display

Discovery purpose:
- Show evidence bundle summary metadata in the selected fixture context tab.

Responsibilities:
- Functions/methods:
  - context evidence summary mapper
- Data structures/models:
  - context evidence summary display model
- Dependencies/services:
  - fixture source registry
  - file fixture evidence bundle schema
- Returns/outputs/signals:
  - context tab evidence summary
- UI surfaces/components:
  - reference review app context tab
- UI fields/elements:
  - evidence bundle label
  - artifact role summary
- Reusable code plan:
  - Existing code reused as-is: selected fixture context state
  - Additions to existing reusable library/module: context evidence summary
    helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - displays sanitized artifact labels
- Performance-sensitive behavior:
  - bounded by selected fixture evidence bundle count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - GUI
- User/caller surface:
  - reference review app context tab
- Invocation route:
  - fixture selection change to context tab render
- Wiring owner/module:
  - `src/impression/devtools/reference_review/ui/shell.py`
- Observable result:
  - selected fixture context includes evidence bundle labels and role summary
- Integration validation:
  - review-app smoke test or view-model test for a bundled fixture
- Incomplete status risk:
  - implemented in isolation if helper output is not wired to fixture selection
- Implementation owner/module:
  - `src/impression/devtools/reference_review/ui/shell.py`
- Chosen defaults / parameters:
  - context tab shows summary only, not full artifact paths
- Test strategy:
  - view-model test plus manual smoke in review app
- Data ownership:
  - source registry owns bundle data; UI owns context display
- Routes:
  - selected fixture to context tab
- Reuse/extraction decision:
  - add to existing review UI module
- UI field/control inventory:
  - evidence bundle label, artifact role summary

Open questions / nuance discovered:
- Full artifact details belong to the artifacts tab child.

Predecessor candidates:
- `File Fixture Evidence Bundle Schema`

Score:
- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 1 x 1 = 1
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; file fixture schema ordering is represented by predecessor candidate
    `File Fixture Evidence Bundle Schema`
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `File Fixture Evidence Bundle Schema`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: the context tab summary is one display
  route with a small field inventory.

Manifest cleanup:
- Parent manifest candidate, if split: Review UI Evidence Display Contract
- Child manifest candidates:
  - Review UI Evidence Context Tab Display
  - Review UI Evidence Artifacts Tab Display
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Review UI Evidence Artifacts Tab Display

Discovery purpose:
- Show evidence artifact details in the selected fixture artifacts tab.

Responsibilities:
- Functions/methods:
  - artifact evidence list mapper
  - missing artifact status formatter
- Data structures/models:
  - artifact evidence row display model
- Dependencies/services:
  - fixture source registry
  - file fixture evidence bundle schema
- Returns/outputs/signals:
  - artifacts tab evidence rows
  - missing-artifact visible state
- UI surfaces/components:
  - reference review app artifacts tab
- UI fields/elements:
  - artifact role
  - artifact kind
  - artifact path/status
- Reusable code plan:
  - Existing code reused as-is: selected fixture artifacts state
  - Additions to existing reusable library/module: artifact evidence list helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - displays sanitized artifact paths
- Performance-sensitive behavior:
  - bounded by selected fixture artifact count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - GUI
- User/caller surface:
  - reference review app artifacts tab
- Invocation route:
  - fixture selection change to artifacts tab render
- Wiring owner/module:
  - `src/impression/devtools/reference_review/ui/shell.py`
- Observable result:
  - selected fixture artifacts tab lists evidence artifacts with role, kind, and
    path/status
- Integration validation:
  - review-app smoke test or view-model test for a bundled fixture
- Incomplete status risk:
  - implemented in isolation if helper output is not wired to fixture selection
- Implementation owner/module:
  - `src/impression/devtools/reference_review/ui/shell.py`
- Chosen defaults / parameters:
  - artifacts tab lists metadata only until preview behavior is specified
- Test strategy:
  - view-model test plus manual smoke in review app
- Data ownership:
  - source registry owns bundle data; UI owns artifacts display
- Routes:
  - selected fixture to artifacts tab
- Reuse/extraction decision:
  - add to existing review UI module
- UI field/control inventory:
  - artifact role, artifact kind, artifact path/status

Open questions / nuance discovered:
- Opening non-STL artifact previews is out of scope for this leaf.

Predecessor candidates:
- `File Fixture Evidence Bundle Schema`

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 1 x 2 = 2
- UI fields/elements: 3 x 1 = 3
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; file fixture schema ordering is represented by predecessor candidate
    `File Fixture Evidence Bundle Schema`
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `File Fixture Evidence Bundle Schema`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: the artifacts tab list is one display route;
  path opening and visual comparison are intentionally out of scope.

Manifest cleanup:
- Parent manifest candidate, if split: Review UI Evidence Display Contract
- Child manifest candidates:
  - Review UI Evidence Context Tab Display
  - Review UI Evidence Artifacts Tab Display
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Section Evidence Contract Records

Discovery purpose:
- Define the section evidence role contract used by loft CSG reference
  fixtures.

Responsibilities:
- Functions/methods:
  - section evidence role validator
- Data structures/models:
  - `SectionEvidenceContractRecord`
- Dependencies/services:
  - reference evidence bundle schema
- Returns/outputs/signals:
  - required role set
  - invalid role diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: file fixture evidence bundle schema
  - Additions to existing reusable library/module: section evidence contract
    records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by bundle artifact count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - workflow
- User/caller surface:
  - reference fixture bundle validation
- Invocation route:
  - evidence bundle validation for section evidence fixtures
- Wiring owner/module:
  - `src/impression/devtools/reference_review/source_registry.py`
- Observable result:
  - section evidence bundles require expected, actual, and diff roles with
    section plane metadata
- Integration validation:
  - fixture loader tests for complete and incomplete section bundles
- Incomplete status risk:
  - implemented in isolation if role validation is not called by fixture load
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
  - `src/impression/devtools/reference_review/source_registry.py`
- Chosen defaults / parameters:
  - required roles are expected, actual, diff
- Test strategy:
  - fixture validation tests for complete and missing-role bundles
- Data ownership:
  - reference fixture owns evidence role declarations
- Routes:
  - evidence bundle schema to section role validation
- Reuse/extraction decision:
  - add section-specific records to existing fixture evidence contract
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Section artifact generation is split out because it depends on successful
  loft CSG result geometry.

Predecessor candidates:
- `File Fixture Evidence Bundle Schema`

Score:
- Functions/methods: 1 x 2 = 2
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 1 x 1 = 1
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 9.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; file fixture schema ordering is represented by predecessor candidate
    `File Fixture Evidence Bundle Schema`
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `File Fixture Evidence Bundle Schema`
- Resolution status:
  - resolved

Split decision:
- Small.

Manifest cleanup:
- Parent manifest candidate, if split: Section Evidence Bundle Producer
- Child manifest candidates:
  - Section Evidence Contract Records
  - Loft CSG Section Artifact Generation
  - Fixture Registry Integration for Section Bundles
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Loft CSG Section Artifact Generation

Discovery purpose:
- Generate expected, actual, and diff section artifacts for loft CSG reference
  fixtures.

Responsibilities:
- Functions/methods:
  - section evidence generator
  - bundle artifact writer
- Data structures/models:
  - section evidence bundle payload
- Dependencies/services:
  - CSG result geometry
  - CV section utilities
  - section evidence contract records
- Returns/outputs/signals:
  - expected/actual/diff artifacts
  - section diagnostic payload
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: CV section artifact helpers
  - Additions to existing reusable library/module: loft CSG section bundle
    helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty section artifacts during reference update tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by section sample count and image size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - workflow
- User/caller surface:
  - reference artifact generation tests
- Invocation route:
  - successful loft CSG result to section artifact writer during reference
    update workflow
- Wiring owner/module:
  - `tests/reference_images.py`
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Observable result:
  - dirty expected, actual, and diff section artifacts are written at
    deterministic paths
- Integration validation:
  - reference update test producing all required section artifact roles
- Incomplete status risk:
  - implemented in isolation if artifact files are written but not returned as
    fixture evidence bundles
- Implementation owner/module:
  - `tests/reference_images.py`
  - `tests/reference_review_fixtures/stl_review_sources.py`
- Chosen defaults / parameters:
  - required output roles are expected, actual, and diff
- Test strategy:
  - artifact generation tests and dirty output existence checks
- Data ownership:
  - reference update workflow owns dirty artifacts; CSG result owns source
    geometry
- Routes:
  - CSG result to section artifact writer
- Reuse/extraction decision:
  - reuse CV section helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Loft CSG result geometry is supplied by the single-shell loft CSG operation
  route ACD.

Predecessor candidates:
- `Section Evidence Contract Records`

Predecessor ACDs:
- `acd-single-shell-loft-csg-operation-route.md`

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; section contract and route geometry ordering are represented by
    predecessor candidate/ACD references
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `Section Evidence Contract Records`
  - predecessor ACD `acd-single-shell-loft-csg-operation-route.md`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: generating the three required section
  roles is one artifact-production route and must be validated atomically.

Manifest cleanup:
- Parent manifest candidate, if split: Section Evidence Bundle Producer
- Child manifest candidates:
  - Section Evidence Contract Records
  - Loft CSG Section Artifact Generation
  - Fixture Registry Integration for Section Bundles
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

### Candidate Spec: Fixture Registry Integration for Section Bundles

Discovery purpose:
- Attach generated section evidence artifacts to fixture records so review and
  promotion workflows consume them through the standard bundle schema.

Responsibilities:
- Functions/methods:
  - section bundle fixture record builder
  - dirty/gold evidence path resolver
- Data structures/models:
  - fixture section evidence source record
- Dependencies/services:
  - section evidence contract records
  - loft CSG section artifact generation
  - fixture source registry
- Returns/outputs/signals:
  - fixture evidence bundle record
  - missing generated artifact diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: file fixture evidence bundle schema
  - Additions to existing reusable library/module: section fixture record
    builder
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - updates dirty fixture records during reference generation
- Security/privacy-sensitive behavior:
  - validates generated paths remain under dirty/gold roots
- Performance-sensitive behavior:
  - bounded by generated artifact count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- App type:
  - workflow
- User/caller surface:
  - reference fixture generation and review app fixture loading
- Invocation route:
  - generated section artifacts to fixture record to source registry
- Wiring owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
  - `src/impression/devtools/reference_review/source_registry.py`
- Observable result:
  - generated section artifacts appear as typed evidence bundles on the
    corresponding fixture
- Integration validation:
  - fixture generation test followed by fixture loader validation
- Incomplete status risk:
  - implemented in isolation if generated files exist but fixture records do not
    reference them
- Implementation owner/module:
  - `tests/reference_review_fixtures/stl_review_sources.py`
  - `src/impression/devtools/reference_review/source_registry.py`
- Chosen defaults / parameters:
  - dirty and gold paths mirror the section evidence role names
- Test strategy:
  - fixture generation/load integration test
- Data ownership:
  - fixture record owns evidence references; artifact generation owns files
- Routes:
  - section artifact generation to fixture registry
- Reuse/extraction decision:
  - add to existing fixture generation helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Promotion semantics are covered by the parent ACD; this leaf wires generated
  evidence into fixture records.

Predecessor candidates:
- `Section Evidence Contract Records`
- `Loft CSG Section Artifact Generation`

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 1 x 1 = 1
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- None.

Readiness blocker resolution:
- Blocking notes found this pass:
  - none; section contract and artifact generation ordering are represented by
    predecessor candidates
- Required next action:
  - none
- Resolution artifact:
  - predecessor candidate `Section Evidence Contract Records`
  - predecessor candidate `Loft CSG Section Artifact Generation`
- Resolution status:
  - resolved

Split decision:
- Review for split. Cohesion reason: fixture record construction and path
  resolution are one integration route from generated artifacts into source
  records.

Manifest cleanup:
- Parent manifest candidate, if split: Section Evidence Bundle Producer
- Child manifest candidates:
  - Section Evidence Contract Records
  - Loft CSG Section Artifact Generation
  - Fixture Registry Integration for Section Bundles
- Parent candidate responsibilities still missing from children:
  - none
- Removal readiness: ready after spec promotion

## Specification Conformance

- Parent specs created or affected:
  - `surface-400-loft-csg-section-evidence-artifacts-v1_0.md` - superseded by
    child specs derived from this ACD and the single-shell loft CSG route ACD.
- Canonical child specs:
  - `../specifications/surface-413-file-fixture-evidence-bundle-schema-v1_0.md` - canonical child from `File Fixture Evidence Bundle Schema`.
  - `../specifications/surface-414-database-fixture-evidence-bundle-parity-v1_0.md` - canonical child from `Database Fixture Evidence Bundle Parity`.
  - `../specifications/surface-415-review-ui-evidence-context-tab-display-v1_0.md` - canonical child from `Review UI Evidence Context Tab Display`.
  - `../specifications/surface-416-review-ui-evidence-artifacts-tab-display-v1_0.md` - canonical child from `Review UI Evidence Artifacts Tab Display`.
  - `../specifications/surface-417-section-evidence-contract-records-v1_0.md` - canonical child from `Section Evidence Contract Records`.
  - `../specifications/surface-418-loft-csg-section-artifact-generation-v1_0.md` - canonical child from `Loft CSG Section Artifact Generation`.
  - `../specifications/surface-419-fixture-registry-integration-for-section-bundles-v1_0.md` - canonical child from `Fixture Registry Integration for Section Bundles`.
- Paired test specs:
  - none yet

## Conformance Checklist

- [ ] Implementation conforms to the target architecture.
- [ ] Parent specs are 100% represented by canonical child specs.
- [ ] Superseded parent specs are archived.
- [ ] Canonical child specs point to architecture or active ACD as primary ancestor.
- [ ] Paired test specs point to canonical child specs.
- [ ] Progression and indexes point to canonical child specs.
- [ ] Completed manifests are removed from active canonical architecture docs.
- [ ] Canonical architecture docs describe the conformed architecture.

## Closure Criteria

- Fixture records can represent typed evidence bundles.
- Required section evidence roles validate cleanly.
- Bundle promotion semantics are defined for dirty/gold artifacts.
- `RT-LOFT-CSG-013` has a clear path to expected/actual/diff evidence once
  successful loft CSG geometry exists.

## Closure Notes

- Canonical architecture updated:
  - none yet
- Archived or removed scaffolding:
  - none yet
- Follow-up ACDs:
  - none

## Change History

- 2026-07-14 - Initial draft. Reason: loft CSG section evidence requires a
  typed multi-artifact fixture policy before implementation specs can be clean.
