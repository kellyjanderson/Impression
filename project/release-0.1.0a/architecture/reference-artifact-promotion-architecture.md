# Reference Artifact Promotion Architecture

## Overview

This document defines broader reference artifact promotion for the surface-body
system.

The current project can generate and compare reference images and STL artifacts,
but surface-body completion needs a stronger promotion model:

- every model-outputting surface capability has named reference coverage
- dirty generated artifacts are useful but not promoted evidence
- clean promoted baselines are tracked separately
- negative diagnostic fixtures are part of the evidence matrix
- `.impress` round-trip fixtures are promoted alongside rendered/STL artifacts

Reference evidence is not just visual polish. It is part of the completion
definition for modeling features.

## Related Architecture

This document extends:

- [Model Output Reference Verification](model-output-reference-verification.md)
- [Computer Vision Verification Architecture](computer-vision-verification-architecture.md)
- [Surface Body Completion Architecture](surface-body-completion-architecture.md)
- [Testing Architecture](testing-architecture.md)

## Artifact Classes

The promoted evidence model recognizes these artifact classes:

- rendered reference image
- exported STL or tessellation artifact
- `.impress` round-trip fixture
- canonical slice or contour expectation
- negative diagnostic snapshot
- capability matrix record

Not every capability needs every class, but each capability must state which
classes are required and why omitted classes are not applicable.

## Components

### Reference Requirement Matrix

The matrix owns:

- capability name
- owning spec
- required artifact classes
- fixture names
- positive versus negative evidence
- promotion status

This matrix is the durable answer to "what evidence proves this capability?"

### Dirty Artifact Store

Dirty artifacts are generated outputs waiting for review. They are useful for
bootstrap and regression detection, but they do not prove completion.

Dirty artifacts must be:

- deterministic
- named
- reproducible
- easy to compare with promoted baselines

### Promoted Baseline Store

Promoted baselines are reviewed reference artifacts.

They must:

- live under durable reference paths
- be tied to fixture contract versions
- be invalidated when fixture contracts change
- be protected from accidental generation churn

### Negative Diagnostic Snapshot Store

Unsupported states also need evidence.

A negative fixture should prove that the system refuses:

- unsupported family pairs
- unsafe `.impress` payloads
- unresolved loft ambiguity
- invalid seam continuity requests
- legacy mesh fallback routes

The diagnostic snapshot should be stable enough to catch accidental weakening
or vague errors.

### Promotion Gate

The gate determines whether a capability has enough promoted evidence.

It owns:

- missing artifact diagnostics
- dirty-versus-clean distinction
- fixture contract mismatch diagnostics
- source spec and test spec references
- release completion integration

## Data Flow

```text
Capability matrix
-> reference requirement matrix
-> fixture generation
-> dirty artifact comparison
-> human or scripted promotion
-> promoted baseline gate
-> release completion evidence
```

## Cross-Domain Decisions

### Dirty Is Not Promoted

Dirty artifacts may be used for bootstrap and change detection. They must not
be counted as final completion evidence.

### Negative Evidence Is Required

For every explicit refusal state that protects surface truth, at least one
negative diagnostic fixture should exist.

### `.impress` Fixtures Are Peer Artifacts

Surface-native persistence is core model output. `.impress` fixtures sit beside
rendered and tessellated artifacts rather than under a separate lower-priority
test category.

### Fixture Contracts Need Versioning

When the fixture shape, camera, slice, artifact set, or expected diagnostic
changes, the fixture contract changes. The promotion gate must notice contract
drift.

## Specification Manifest for Discovery

### Candidate Spec: Surface Reference Requirement Matrix

Discovery purpose:
- Define the durable matrix connecting model-outputting capabilities to
  required promoted evidence.

Responsibilities:
- Functions/methods:
  - requirement matrix loader
  - capability coverage assertion
- Data structures/models:
  - reference requirement record
  - artifact class record
  - fixture contract record
- Dependencies/services:
  - surface completion evidence gate
  - reference artifact lifecycle
- Returns/outputs/signals:
  - missing requirement diagnostic
  - matrix coverage report
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current completion evidence matrix
  - Additions to existing reusable library/module: reference requirement records
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
  - bounded repository scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py` and reference test helpers
- Chosen defaults / parameters:
  - promoted model-outputting capabilities require at least one positive and
    applicable negative evidence path
- Test strategy:
  - matrix completeness and missing requirement tests
- Data ownership:
  - release verification owns evidence requirements
- Routes:
  - capability matrix to reference requirement matrix to progression
- Open questions / nuance discovered:
  - fixture contract version may live in metadata or sidecar manifest
- Readiness blockers:
  - none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 14.5

Split decision:
- No split needed. The candidate is one evidence-matrix record contract.

### Candidate Spec: Dirty Versus Promoted Reference Artifact Gate

Discovery purpose:
- Ensure dirty generated artifacts cannot satisfy promoted completion evidence.

Responsibilities:
- Functions/methods:
  - artifact state classifier
  - promotion gate
  - contract invalidation checker
- Data structures/models:
  - artifact state record
  - fixture contract version record
  - promotion diagnostic
- Dependencies/services:
  - reference artifact lifecycle tooling
  - filesystem paths
- Returns/outputs/signals:
  - promoted/missing/dirty report
  - invalidation diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reference artifact lifecycle skill/process
  - Additions to existing reusable library/module: promotion gate helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - may create dirty artifacts during verification; must not promote
    automatically
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded file existence and checksum checks
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - reference test helpers under `tests/`
- Chosen defaults / parameters:
  - dirty artifacts fail promotion evidence
- Test strategy:
  - dirty-only, clean-present, partial-missing, and invalidated-contract tests
- Data ownership:
  - reference artifact lifecycle owns file state
- Routes:
  - test output to dirty artifact store to promotion gate
- Open questions / nuance discovered:
  - checksum versus metadata versioning should be decided by implementation
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Split decision:
- Review for split. Cohesion reason: dirty/promoted classification and
  promotion refusal are one artifact lifecycle gate.

### Candidate Spec: Diagnostic Snapshot Normalization

Discovery purpose:
- Normalize refusal diagnostics into stable snapshot payloads that ignore
  incidental stack traces, temporary paths, and machine-specific details.

Responsibilities:
- Functions/methods:
  - diagnostic snapshot normalizer
  - snapshot comparator
- Data structures/models:
  - diagnostic snapshot record
  - diagnostic key policy record
- Dependencies/services:
  - current refusal diagnostics
  - reference artifact lifecycle
- Returns/outputs/signals:
  - stable diagnostic snapshot
  - diagnostic drift failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current refusal diagnostics
  - Additions to existing reusable library/module: snapshot normalization
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty diagnostic snapshots during bootstrap
- Security/privacy-sensitive behavior:
  - snapshots must not include sensitive local paths beyond normalized fixture
    identifiers
- Performance-sensitive behavior:
  - bounded fixture count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - reference test helpers under `tests/`
- Chosen defaults / parameters:
  - snapshots compare stable diagnostic keys, not incidental stack traces
- Test strategy:
  - normalization tests for path stripping, ordering, and stable keys
- Data ownership:
  - snapshot normalizer owns portable diagnostic representation
- Routes:
  - refusal exception/result to normalized snapshot to comparator
- Open questions / nuance discovered:
  - diagnostic snapshots should avoid machine-specific path fragments
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Split decision:
- Review for split. Cohesion reason: normalization is one reusable snapshot
  layer; domain-specific negative fixtures are split separately.

### Candidate Spec: Negative Diagnostic Fixture Matrix Core

Discovery purpose:
- Define the matrix schema and coverage checker for negative diagnostic
  fixtures without owning domain-specific fixture construction.

Responsibilities:
- Functions/methods:
  - fixture matrix coverage checker
  - snapshot comparator integration
- Data structures/models:
  - negative fixture record
  - domain coverage record
  - expected diagnostic key record
- Dependencies/services:
  - diagnostic snapshot normalizer
  - reference artifact lifecycle
- Returns/outputs/signals:
  - negative fixture coverage report
  - missing domain diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current refusal diagnostics
  - Additions to existing reusable library/module: negative fixture matrix
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
  - bounded fixture count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - reference test helpers under `tests/`
- Chosen defaults / parameters:
  - every explicit refusal boundary has at least one negative fixture entry
- Test strategy:
  - matrix coverage tests with accepted and missing domains
- Data ownership:
  - negative fixture matrix owns expected refusal coverage
- Routes:
  - domain fixture records to matrix coverage report
- Open questions / nuance discovered:
  - each domain owns fixture construction while the matrix owns coverage
- Readiness blockers:
  - diagnostic snapshot normalizer must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 16.5

Split decision:
- Review for split. Cohesion reason: the core owns only fixture matrix schema
  and coverage; domain fixtures are split separately.

### Candidate Spec: .impress Unsafe Payload Negative Fixtures

Discovery purpose:
- Create negative diagnostic fixtures for unsafe or malformed `.impress`
  payloads.

Responsibilities:
- Functions/methods:
  - persistence negative fixture runner
- Data structures/models:
  - negative fixture record
  - expected diagnostic key record
- Dependencies/services:
  - `.impress` refusal
  - diagnostic snapshot normalizer
- Returns/outputs/signals:
  - diagnostic snapshot
  - diagnostic drift failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: `.impress` refusal tests
  - Additions to existing reusable library/module: fixture records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty diagnostic snapshots during bootstrap
- Security/privacy-sensitive behavior:
  - unsafe payload fixtures must not execute
- Performance-sensitive behavior:
  - bounded fixture count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - reference test helpers under `tests/`
- Chosen defaults / parameters:
  - unsafe persistence states require stable negative fixtures
- Test strategy:
  - unsafe payload, malformed payload, unsupported family, and metadata
    snapshot tests
- Data ownership:
  - domain fixture owns expected refusal contract
- Routes:
  - failing persistence operation to snapshot to matrix
- Open questions / nuance discovered:
  - unsafe payload fixtures must avoid executable side effects
- Readiness blockers:
  - diagnostic snapshot normalizer must exist

Score:
- Functions/methods: 1 x 2 = 2
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 19.5

Split decision:
- Review for split. Cohesion reason: `.impress` unsafe payloads are one
  persistence-domain fixture family.

### Candidate Spec: Mesh Boundary Negative Fixtures

Discovery purpose:
- Create negative diagnostic fixtures for hidden mesh fallback and legacy mesh
  assumption violations.

Responsibilities:
- Functions/methods:
  - mesh-boundary negative fixture runner
  - legacy call-site fixture builder
- Data structures/models:
  - negative fixture record
  - expected diagnostic key record
- Dependencies/services:
  - mesh fallback refusal
  - tessellation boundary policy
  - diagnostic snapshot normalizer
- Returns/outputs/signals:
  - diagnostic snapshot
  - diagnostic drift failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: mesh-boundary refusal tests
  - Additions to existing reusable library/module: fixture records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty diagnostic snapshots during bootstrap
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded fixture count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - reference test helpers under `tests/`
- Chosen defaults / parameters:
  - hidden mesh fallback requires stable negative fixture coverage
- Test strategy:
  - hidden mesh fallback and stale primitive mesh assumption snapshot tests
- Data ownership:
  - domain fixture owns expected refusal contract
- Routes:
  - failing mesh-boundary operation to snapshot to matrix
- Open questions / nuance discovered:
  - legacy mesh-specific APIs remain accepted only when explicitly named
- Readiness blockers:
  - diagnostic snapshot normalizer must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 19.5

Split decision:
- Review for split. Cohesion reason: mesh-boundary fixtures protect one
  compatibility/refusal domain.

### Candidate Spec: CSG Negative Diagnostic Fixtures

Discovery purpose:
- Create negative diagnostic fixtures for unsupported or non-executable CSG
  solver states.

Responsibilities:
- Functions/methods:
  - CSG negative fixture runner
  - CSG unsupported fixture builder
- Data structures/models:
  - CSG negative fixture record
  - expected diagnostic key record
- Dependencies/services:
  - CSG refusal
  - diagnostic snapshot normalizer
- Returns/outputs/signals:
  - diagnostic snapshot
  - diagnostic drift failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG refusal diagnostics
  - Additions to existing reusable library/module: fixture records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty diagnostic snapshots during bootstrap
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded fixture count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - reference test helpers under `tests/`
- Chosen defaults / parameters:
  - unsupported CSG family pairs and invalid operands require negative fixtures
- Test strategy:
  - unsupported pair, invalid shell, non-executable plan snapshot tests
- Data ownership:
  - CSG fixture owns expected refusal contract
- Routes:
  - failing CSG operation to snapshot to matrix
- Open questions / nuance discovered:
  - CSG fixtures should pin diagnostic family pairs and solver stage
- Readiness blockers:
  - diagnostic snapshot normalizer must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: CSG refusal fixtures are one geometry
  domain and no longer bundle loft or seam diagnostics.

### Candidate Spec: Loft Ambiguity Negative Diagnostic Fixtures

Discovery purpose:
- Create negative diagnostic fixtures for authored loft correspondence
  ambiguity and unresolved topology rails.

Responsibilities:
- Functions/methods:
  - loft ambiguity fixture runner
  - snapshot comparator integration
- Data structures/models:
  - loft negative fixture record
  - expected diagnostic key record
- Dependencies/services:
  - loft ambiguity refusal
  - diagnostic snapshot normalizer
- Returns/outputs/signals:
  - diagnostic snapshot
  - diagnostic drift failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current loft/seam refusal diagnostics
  - Additions to existing reusable library/module: fixture records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty diagnostic snapshots during bootstrap
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded fixture count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - reference test helpers under `tests/`
- Chosen defaults / parameters:
  - loft fixtures report all ambiguities, not first failure only
- Test strategy:
  - unresolved correspondence, point birth/death, anchor conflict, and missing
    rail snapshot tests
- Data ownership:
  - loft fixtures own expected refusal contracts
- Routes:
  - failing loft plan validation to snapshot to matrix
- Open questions / nuance discovered:
  - loft fixtures should report all ambiguities, not first failure only
- Readiness blockers:
  - diagnostic snapshot normalizer must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 17.5

Split decision:
- Review for split. Cohesion reason: loft ambiguity fixtures are one authored
  topology diagnostic family; seam continuity fixtures are split separately.

### Candidate Spec: Seam Continuity Negative Diagnostic Fixtures

Discovery purpose:
- Create negative diagnostic fixtures for unsupported or failed higher-order
  seam continuity requests.

Responsibilities:
- Functions/methods:
  - seam continuity fixture runner
  - snapshot comparator integration
- Data structures/models:
  - seam negative fixture record
  - expected diagnostic key record
  - expected continuity residual record
- Dependencies/services:
  - seam continuity refusal
  - diagnostic snapshot normalizer
  - continuity violation locators
- Returns/outputs/signals:
  - diagnostic snapshot
  - diagnostic drift failure
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current seam refusal diagnostics
  - Additions to existing reusable library/module: seam fixture records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes dirty diagnostic snapshots during bootstrap
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded fixture count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - reference test helpers under `tests/`
- Chosen defaults / parameters:
  - seam fixtures pin requested continuity, observed residual, and locator path
- Test strategy:
  - unsupported continuity, failed C1/G1, and failed C2/G2 snapshot tests
- Data ownership:
  - seam fixtures own expected refusal contracts
- Routes:
  - failing seam validation to snapshot to matrix
- Open questions / nuance discovered:
  - fixture snapshots must distinguish unsupported class from supported class
    with failed residual
- Readiness blockers:
  - diagnostic snapshot normalizer and continuity locator specs must exist

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:
- Review for split. Cohesion reason: seam continuity negative fixtures are one
  diagnostic family and no longer bundle loft ambiguity.

## Change History

- 2026-05-27: Ran two additional critical manifest cycles and split loft and
  seam negative diagnostic fixtures. Context: authored loft ambiguity and
  higher-order seam continuity have different fixture contracts.
- 2026-05-27: Critically reviewed, rescored, and split the specification
  manifest. Context: negative diagnostic references were under-defined until
  snapshot normalization, unsafe payloads, mesh boundary, CSG, and loft/seam
  fixture families were separated.
- 2026-05-27: Added reference artifact promotion architecture and manifest.
  Context: surface-body completion requires promoted baselines and negative
  diagnostic fixtures, not dirty artifacts alone.
