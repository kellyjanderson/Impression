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

### Candidate Spec: Negative Diagnostic Reference Fixtures

Discovery purpose:
- Promote stable negative diagnostics as first-class reference evidence for
  unsupported or unsafe surface-body states.

Responsibilities:
- Functions/methods:
  - diagnostic snapshot normalizer
  - negative fixture runner
  - snapshot comparator
- Data structures/models:
  - negative fixture record
  - diagnostic snapshot record
- Dependencies/services:
  - `.impress` refusal
  - CSG refusal
  - loft ambiguity refusal
  - seam continuity refusal
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
  - unsafe payload fixtures must not execute
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
  - negative fixtures for `.impress`, CSG, loft, seams, and mesh fallback
- Data ownership:
  - negative fixture owns expected refusal contract
- Routes:
  - failing operation to diagnostic snapshot to promotion gate
- Open questions / nuance discovered:
  - diagnostic snapshots should avoid machine-specific path fragments
- Readiness blockers:
  - none

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 4 x 1 = 4
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
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: the snapshot normalizer and negative
  fixture runner are one diagnostic reference lane; domain-specific fixtures
  remain data entries.

## Change History

- 2026-05-27: Added reference artifact promotion architecture and manifest.
  Context: surface-body completion requires promoted baselines and negative
  diagnostic fixtures, not dirty artifacts alone.
