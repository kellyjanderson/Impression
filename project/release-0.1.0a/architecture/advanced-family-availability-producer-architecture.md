# Advanced Family Availability Producer Architecture

## Overview

This architecture fills the gap between an advanced patch family being
`implemented` and being genuinely `available`.

The implementation-completion architecture already defines what must be true
for B-spline, NURBS, sweep, subdivision, implicit, heightmap, and displacement
patches to stop being `planned`. This document defines the additional producer,
import, authoring, and operation-facing contracts needed so the families are
not merely loadable/evaluable internals, but first-class authored capabilities.

The immediate gap is strongest for:

- subdivision
- implicit
- heightmap
- displacement

B-spline, NURBS, and sweep already have loft/path/conic producer work in the
current specification set. They still participate in the shared availability
gate, but they do not need new producer-family architecture here unless later
review finds a missing authoring path.

## Relationship To Existing Architecture

This document extends:

- [Advanced Patch Family Implementation Completion Architecture](advanced-patch-family-implementation-completion-architecture.md)
- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [Surface Body Completion Architecture](surface-body-completion-architecture.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)
- [Patch Family Integration Architecture](patch-family-integration-architecture.md)
- [Reference Artifact Promotion Architecture](reference-artifact-promotion-architecture.md)

The key distinction:

- `implemented` means the kernel can store, evaluate, persist, tessellate, and
  diagnose the family without mesh-as-truth.
- `available` means users or upstream systems have a supported way to author,
  import, construct, or intentionally select the family, and operations report
  honest family-specific capability.

## Availability State Model

The capability matrix should track these fields separately:

- `implementation_state`: `specified`, `implemented`, `verified`, or `retired`
- `availability_state`: `unavailable`, `import-only`, `producer-available`,
  `authoring-available`, or `retired`
- `producer_paths`: named functions, builders, importers, or documented payload
  construction routes that intentionally create the family
- `operation_posture`: supported, declared-tolerance, adapter, non-CSG, or
  unsupported with diagnostics for each relevant operation
- `evidence_state`: missing, local tests, reference artifacts, or promoted
  completion evidence

A family must not be called `available` only because a class exists. It needs
at least one durable authoring/import/producer path and an operation posture
that is honest enough for callers.

## Producer Path Types

### Native Builder

A native builder is a public or semi-public Impression API that constructs the
family from authored parameters.

Examples:

- `make_subdivision_surface(...)`
- `make_implicit_surface(...)`
- `make_heightmap_surface(...)`
- `make_displacement_surface(...)`

Native builders own validation, diagnostics, stable identity, and construction
metadata. They must return surface truth, never a mesh fallback.

### Import Adapter

An import adapter creates a native patch from an external representation.

Examples:

- subdivision cage import
- heightmap array/image import
- declarative implicit field payload import
- displacement source/payload import

Import adapters must normalize into native surface payloads immediately. Mesh
imports may remain mesh objects, but they must not masquerade as subdivision or
heightmap surface truth unless the native payload is explicitly present.

### Documented Payload Authoring

Some families may be available first through `.impress` payload authoring.
That is acceptable only when:

- the payload schema is stable
- examples exist
- diagnostics are precise
- round-trip evidence is promoted
- the capability matrix labels the family `import-only` or equivalent until a
  native builder exists

## Family Availability Contracts

### Subdivision

Subdivision becomes available when users can intentionally create or import a
control cage with scheme, crease, boundary, and hole policy metadata.

Required producer paths:

- native subdivision patch builder or documented cage import adapter
- `.impress` payload authoring examples
- reference fixtures for at least one non-planar cage and one creased boundary

Operation posture:

- tessellation is supported through deterministic refinement
- seam participation carries approximation metadata when exact limit
  comparison is unavailable
- CSG posture may be adapter, declared-tolerance, unsupported, or non-CSG, but
  must be explicit per operation/family pair

### Implicit

Implicit becomes available when users can author safe declarative field graphs
without executable callbacks.

Required producer paths:

- native field-graph builder with allow-listed nodes
- `.impress` payload examples for sphere, box, union/difference/intersection,
  transform, and smooth-min-style composition if supported
- diagnostics for unsafe, unbounded, or budget-exhausting fields

Operation posture:

- field evaluation and extraction are supported within declared budgets
- tessellation carries extraction and residual metadata
- CSG posture may be native field composition, adapter, unsupported, or non-CSG
  by operation, but arbitrary executable code is never an operation path

### Heightmap

Heightmap becomes available when users can construct a sampled heightfield from
embedded arrays or approved import sources.

Required producer paths:

- native array/grid builder
- optional image/array import adapter if dependencies are already acceptable
- `.impress` payload examples for finite grid, mask/no-data, interpolation, and
  units

Operation posture:

- tessellation is supported with sampled-source identity metadata
- seams use sampled-boundary approximation metadata
- CSG posture may be adapter, unsupported, or non-CSG unless exact heightfield
  operations are intentionally implemented
- external references are refused unless the identity and lifecycle policy
  explicitly supports them

### Displacement

Displacement becomes available when users can author a source surface plus a
bounded displacement field without baking the result to a mesh.

Required producer paths:

- native displacement builder from source patch/body plus displacement function
  or sampled displacement payload
- `.impress` examples for embedded source payload and stable in-body source
  identity
- diagnostics for missing source, invalid parameter mapping, non-finite
  displacement, and unsupported cross-body reference

Operation posture:

- tessellation is supported with source identity and lossiness metadata
- seam support compares displaced boundaries or refuses with exact locators
- CSG posture may be adapter, unsupported, or non-CSG unless native displaced
  operations are intentionally implemented

## Operation Availability Contract

Every family promoted to `available` must have operation rows for:

- primitive/feature production
- `.impress` load/save
- tessellation
- seam/adjacency validation
- loft or explicit non-applicability
- CSG operation planning
- reference artifact promotion

Operation rows may say `unsupported` or `non-CSG`, but only with:

- family name
- operation name
- phase
- reason
- required future capability, if applicable
- no-hidden-mesh-fallback proof

The capability matrix is therefore allowed to expose a family as available for
authoring while still saying some operations are unsupported. What is not
allowed is a family marked available with unknown operation behavior.

## Data Flow

```text
User/API/import/.impress payload
-> family-specific producer or import adapter
-> native patch payload validator
-> SurfacePatch / SurfaceBodyStore
-> availability capability matrix
-> operation support matrix
-> supported operation, declared-tolerance adapter, non-CSG classification, or
   structured refusal
-> tessellation only for preview/export/reference evidence
```

## Reusable Boundaries

Shared infrastructure should include:

- advanced-family availability gate checker
- producer-path registry
- documented payload example fixtures
- operation-posture matrix rows
- no-hidden-mesh-fallback assertion helpers
- reference artifact promotion hooks

Family-specific modules should own:

- payload validation
- native builder arguments
- family-local diagnostics
- family-local examples

## Specification Manifest for Discovery

### Candidate Spec: Advanced Family Availability Gate And Matrix Fields

Responsibilities by category:
- Functions/methods:
  - availability gate checker
  - matrix update helper
  - evidence summarizer
- Data structures/models:
  - availability state fields
  - producer path records
  - operation posture records
- Dependencies/services:
  - existing patch-family capability matrix
  - reference artifact promotion evidence
- Returns/outputs/signals:
  - promoted availability state
  - refused promotion diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: current capability matrix records
  - Additions to existing reusable library/module: availability fields and gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix fixture updates in tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - `available` requires at least one producer/import/payload authoring path and
    complete operation posture rows
- Test strategy:
  - promotion success/failure tests and missing-producer diagnostics
- Data ownership:
  - capability matrix owns state truth; producers own construction metadata
- Routes:
  - producer registry to capability matrix to operation posture matrix
- Reuse/extraction decision:
  - extend matrix records rather than creating separate per-family flags
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- B-spline, NURBS, and sweep may already satisfy availability through existing
  producer specs once evidence is implemented.

Readiness blockers:
- advanced-family implementation-completion gate

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:
- Review for split. Cohesion reason: the availability gate and matrix fields
  are one reusable capability-matrix change.

### Split Parent: Subdivision Authoring And Import Producer

Responsibilities by category:
- Functions/methods:
  - subdivision native builder
  - cage import adapter
  - crease/boundary diagnostic builder
- Data structures/models:
  - subdivision authoring request
  - producer provenance record
  - import diagnostic
- Dependencies/services:
  - subdivision patch payload validation
  - `.impress` subdivision codec
- Returns/outputs/signals:
  - native subdivision patch
  - producer-path capability record
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: subdivision patch record/evaluator
  - Additions to existing reusable library/module: subdivision producer helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - imported cage payload validation
- Performance-sensitive behavior:
  - bounded cage size validation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - Catmull-Clark finite-level authoring first
- Test strategy:
  - native builder, import adapter, creased boundary, malformed cage, and
    no-hidden-mesh-fallback tests
- Data ownership:
  - producer owns authoring metadata; patch owns cage payload
- Routes:
  - producer/import adapter to subdivision patch to store/codec/tessellation
- Reuse/extraction decision:
  - reuse subdivision payload validation and evaluator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Additional subdivision schemes should be added as explicit future producer
  records, not silent options.

Readiness blockers:
- subdivision implemented-family runtime and tessellation specs

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Split completed in review Loop 1. Native cage construction and external cage
  import have different diagnostics and dependency posture.

### Candidate Spec: Subdivision Native Cage Builder

Responsibilities by category:
- Functions/methods:
  - subdivision native builder
  - crease/boundary diagnostic builder
  - producer provenance recorder
- Data structures/models:
  - subdivision authoring request
  - producer provenance record
- Dependencies/services:
  - subdivision patch payload validation
  - subdivision evaluator
- Returns/outputs/signals:
  - native subdivision patch
  - producer-path capability record
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: subdivision patch record/evaluator
  - Additions to existing reusable library/module: native subdivision builder
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded cage size validation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - Catmull-Clark finite-level authoring first
- Test strategy:
  - native builder, creased boundary, malformed cage, and
    no-hidden-mesh-fallback tests
- Data ownership:
  - producer owns authoring metadata; patch owns cage payload
- Routes:
  - native builder to subdivision patch to store/codec/tessellation
- Reuse/extraction decision:
  - reuse subdivision payload validation and evaluator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Additional subdivision schemes should be explicit producer records, not silent
  options.

Readiness blockers:
- subdivision implemented-family runtime and tessellation specs

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:
- Review for split. Cohesion reason: this candidate owns direct authored
  subdivision construction and does not need import dependency policy.

### Candidate Spec: Subdivision Cage Import Adapter

Responsibilities by category:
- Functions/methods:
  - cage import adapter
  - import diagnostic builder
  - import provenance normalizer
- Data structures/models:
  - subdivision import request
  - import diagnostic
  - normalized cage payload
- Dependencies/services:
  - subdivision native cage builder
  - `.impress` subdivision codec
- Returns/outputs/signals:
  - native subdivision patch
  - explicit unsupported import diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: native subdivision builder
  - Additions to existing reusable library/module: cage import adapter
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - import fixture writes in tests
- Security/privacy-sensitive behavior:
  - imported cage payload validation
- Performance-sensitive behavior:
  - bounded import size validation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - successful imports normalize through the native subdivision builder
- Test strategy:
  - valid import, malformed cage import, unsupported scheme import, and
    no-hidden-mesh-fallback tests
- Data ownership:
  - import adapter owns source metadata; patch owns normalized cage payload
- Routes:
  - cage import adapter to native builder to subdivision patch
- Reuse/extraction decision:
  - reuse native builder and payload validation
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Mesh imports remain mesh objects unless native cage topology is explicitly
  present.

Readiness blockers:
- subdivision native cage builder

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: this candidate owns external cage
  normalization and explicit import refusal.

### Split Parent: Implicit Declarative Field Authoring Producer

Responsibilities by category:
- Functions/methods:
  - implicit field builder API
  - allow-listed node helper constructors
  - unsafe-field diagnostic builder
- Data structures/models:
  - field authoring request
  - field-node provenance record
  - unsafe authoring diagnostic
- Dependencies/services:
  - implicit payload safety validator
  - implicit evaluator/extraction adapter
- Returns/outputs/signals:
  - native implicit patch
  - safe field graph payload
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: implicit field validator/evaluator
  - Additions to existing reusable library/module: builder/helper functions
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - rejects executable callbacks and external code
- Performance-sensitive behavior:
  - bounded field tree and evaluation budget defaults
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - safe declarative nodes only; no arbitrary Python callbacks
- Test strategy:
  - safe primitive fields, composed fields, unsafe callback refusal, budget
    refusal, `.impress` round-trip, and no-hidden-mesh-fallback tests
- Data ownership:
  - producer owns authoring metadata; patch owns field graph
- Routes:
  - field builder to implicit patch to evaluator/codec/extraction
- Reuse/extraction decision:
  - reuse safety validator for both builder and decoder
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Native field-composition CSG can be separate operation work; this producer
  only guarantees safe authoring.

Readiness blockers:
- implicit field safety and evaluation/extraction specs

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 2 x 2 = 4
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 26.5

Split decision:
- Split required. The builder/helper API and the unsafe/budget diagnostic
  authoring gate should be split before implementation.

### Candidate Spec: Implicit Field Builder And Helper API

Responsibilities by category:
- Functions/methods:
  - implicit field builder API
  - allow-listed node helper constructors
  - field graph provenance recorder
- Data structures/models:
  - field authoring request
  - field-node provenance record
- Dependencies/services:
  - implicit payload safety validator
  - implicit evaluator
- Returns/outputs/signals:
  - native implicit patch
  - safe field graph payload
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: implicit field validator/evaluator
  - Additions to existing reusable library/module: builder/helper functions
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - builder accepts only allow-listed declarative nodes
- Performance-sensitive behavior:
  - bounded field tree defaults
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - safe declarative nodes only; no arbitrary Python callbacks
- Test strategy:
  - primitive field, composed field, provenance, `.impress` round-trip, and
    no-hidden-mesh-fallback tests
- Data ownership:
  - producer owns authoring metadata; patch owns field graph
- Routes:
  - field builder to implicit patch to evaluator/codec/extraction
- Reuse/extraction decision:
  - reuse safety validator for both builder and decoder
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Native field-composition CSG remains operation work, not builder work.

Readiness blockers:
- implicit field safety and evaluation/extraction specs

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: this candidate owns the positive
  declarative authoring path and reuses the existing safety validator.

### Split Parent: Implicit Authoring Safety And Budget Diagnostics

Responsibilities by category:
- Functions/methods:
  - unsafe-field diagnostic builder
  - budget-refusal diagnostic builder
  - authoring-time validator wrapper
- Data structures/models:
  - unsafe authoring diagnostic
  - budget diagnostic
- Dependencies/services:
  - implicit payload safety validator
  - implicit extraction budget policy
- Returns/outputs/signals:
  - structured refusal diagnostics
  - non-executable producer result
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: implicit safety validator
  - Additions to existing reusable library/module: authoring diagnostic wrappers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - diagnostic fixture writes in tests
- Security/privacy-sensitive behavior:
  - rejects executable callbacks and external code
- Performance-sensitive behavior:
  - bounded tree and extraction budget validation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - unsafe and over-budget authoring returns diagnostics before execution
- Test strategy:
  - callback refusal, unknown node refusal, unbounded domain refusal, budget
    exhaustion, and diagnostic payload tests
- Data ownership:
  - producer diagnostics own authoring failure reasons
- Routes:
  - builder to safety gate to refusal diagnostic
- Reuse/extraction decision:
  - share diagnostic keys with `.impress` decoder refusals
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Diagnostic text must name the rejected node/path, not only the family.

Readiness blockers:
- implicit payload safety policy

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 2 x 2 = 4
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Split completed in review Loop 2. Unsafe payload diagnostics and budget
  diagnostics have different triggers and fixture sets.

### Candidate Spec: Implicit Unsafe Authoring Diagnostics

Responsibilities by category:
- Functions/methods:
  - unsafe-field diagnostic builder
  - authoring-time safety validator wrapper
  - diagnostic path locator
- Data structures/models:
  - unsafe authoring diagnostic
  - rejected node locator
- Dependencies/services:
  - implicit payload safety validator
  - `.impress` implicit decoder diagnostics
- Returns/outputs/signals:
  - structured unsafe-payload diagnostic
  - non-executable producer result
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: implicit safety validator
  - Additions to existing reusable library/module: authoring diagnostic wrappers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - diagnostic fixture writes in tests
- Security/privacy-sensitive behavior:
  - rejects executable callbacks and external code
- Performance-sensitive behavior:
  - bounded diagnostic traversal
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - unsafe authoring returns diagnostics before evaluation or persistence
- Test strategy:
  - callback refusal, unknown node refusal, executable payload refusal, and
    diagnostic path tests
- Data ownership:
  - producer diagnostics own unsafe authoring failure reasons
- Routes:
  - builder to safety gate to unsafe diagnostic
- Reuse/extraction decision:
  - share diagnostic keys with `.impress` decoder refusals
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Diagnostic text must name the rejected node/path, not only the family.

Readiness blockers:
- implicit payload safety policy

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 2 x 2 = 4
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: this candidate owns security refusal for
  unsafe implicit authoring and shares policy with file decoding.

### Candidate Spec: Implicit Budget And Bound Diagnostics

Responsibilities by category:
- Functions/methods:
  - budget-refusal diagnostic builder
  - bounded-domain validator wrapper
  - extraction-budget locator
- Data structures/models:
  - budget diagnostic
  - bound diagnostic
- Dependencies/services:
  - implicit extraction budget policy
  - implicit evaluation/extraction adapter
- Returns/outputs/signals:
  - structured budget/bounds diagnostic
  - non-executable producer result
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: implicit budget policy
  - Additions to existing reusable library/module: authoring budget diagnostics
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - diagnostic fixture writes in tests
- Security/privacy-sensitive behavior:
  - avoids executing over-budget payloads
- Performance-sensitive behavior:
  - bounded tree and extraction budget validation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - over-budget authoring returns diagnostics before extraction
- Test strategy:
  - unbounded domain refusal, budget exhaustion, depth limit, node-count limit,
    and diagnostic payload tests
- Data ownership:
  - producer diagnostics own budget and bound failure reasons
- Routes:
  - builder to budget gate to refusal diagnostic
- Reuse/extraction decision:
  - reuse extraction budget policy rather than inventing builder-only limits
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Budget diagnostics must name the first exceeded limit and the family.

Readiness blockers:
- implicit extraction budget policy

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns bounded evaluation
  refusal and shares limits with extraction.

### Split Parent: Heightmap Native Grid And Import Producer

Responsibilities by category:
- Functions/methods:
  - heightmap grid builder
  - finite sample validator
  - optional image/array import adapter
- Data structures/models:
  - heightmap authoring request
  - sample grid provenance record
  - import diagnostic
- Dependencies/services:
  - heightmap payload validation
  - `.impress` heightmap codec
- Returns/outputs/signals:
  - native heightmap patch
  - producer-path capability record
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: heightmap patch record and validator
  - Additions to existing reusable library/module: grid builder/import helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - external references refused unless identity policy supports them
- Performance-sensitive behavior:
  - bounded grid size and memory diagnostics
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py`
- Chosen defaults / parameters:
  - embedded finite arrays first; image import only if dependency boundary is
    already present or kept optional
- Test strategy:
  - finite grid builder, mask/no-data, malformed samples, optional import,
    `.impress` round-trip, and no-hidden-mesh-fallback tests
- Data ownership:
  - producer owns source/provenance metadata; patch owns sampled grid
- Routes:
  - grid/import producer to heightmap patch to codec/tessellation/seams
- Reuse/extraction decision:
  - reuse heightmap payload validator and sampled boundary helper
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Image import dependency must stay optional unless the project already carries
  an approved image stack.

Readiness blockers:
- heightmap payload validation and tessellation specs

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 25.5

Split decision:
- Split required. Native finite-grid builder and optional image/import adapter
  should be split so optional dependency policy does not block core
  availability.

### Candidate Spec: Heightmap Native Finite Grid Builder

Responsibilities by category:
- Functions/methods:
  - heightmap grid builder
  - finite sample validator
  - mask/no-data diagnostic builder
- Data structures/models:
  - heightmap authoring request
  - sample grid provenance record
- Dependencies/services:
  - heightmap payload validation
  - `.impress` heightmap codec
- Returns/outputs/signals:
  - native heightmap patch
  - producer-path capability record
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: heightmap patch record and validator
  - Additions to existing reusable library/module: finite-grid builder
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - embedded finite arrays only
- Performance-sensitive behavior:
  - bounded grid size and memory diagnostics
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py`
- Chosen defaults / parameters:
  - embedded finite arrays first
- Test strategy:
  - finite grid builder, mask/no-data, malformed samples, `.impress`
    round-trip, and no-hidden-mesh-fallback tests
- Data ownership:
  - producer owns grid provenance; patch owns sampled grid
- Routes:
  - grid producer to heightmap patch to codec/tessellation/seams
- Reuse/extraction decision:
  - reuse heightmap payload validator and sampled boundary helper
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This is the required availability path; image import is optional and split.

Readiness blockers:
- heightmap payload validation and tessellation specs

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate is the minimal native
  heightmap availability path.

### Candidate Spec: Heightmap Optional Import Adapter

Responsibilities by category:
- Functions/methods:
  - image/array import adapter
  - import dependency boundary checker
  - import diagnostic builder
- Data structures/models:
  - import request
  - import diagnostic
- Dependencies/services:
  - native finite-grid builder
  - optional image/array dependency boundary
- Returns/outputs/signals:
  - native heightmap patch
  - unsupported import diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: finite-grid builder
  - Additions to existing reusable library/module: optional import adapter
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - import fixture writes in tests
- Security/privacy-sensitive behavior:
  - external references refused unless explicitly loaded into embedded grid
- Performance-sensitive behavior:
  - import size and memory diagnostics
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py`
- Chosen defaults / parameters:
  - import adapter may be absent; absent adapter reports explicit unsupported
    import diagnostics without blocking native grid availability
- Test strategy:
  - supported import if dependency exists, unsupported import if absent, bad
    data diagnostics, and embedded-grid normalization tests
- Data ownership:
  - import adapter owns source metadata; patch owns embedded grid
- Routes:
  - import adapter to finite-grid builder to heightmap patch
- Reuse/extraction decision:
  - normalize all successful imports through the native grid builder
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This candidate should not add mandatory image dependencies by accident.

Readiness blockers:
- native finite-grid builder

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: optional import is one adapter boundary
  and normalizes through the native grid builder.

### Split Parent: Displacement Native Authoring Producer

Responsibilities by category:
- Functions/methods:
  - displacement patch builder
  - source identity resolver
  - displacement payload diagnostic builder
- Data structures/models:
  - displacement authoring request
  - source identity/provenance record
  - displacement producer diagnostic
- Dependencies/services:
  - source patch evaluator
  - displacement payload validation
- Returns/outputs/signals:
  - native displacement patch
  - source identity and lossiness metadata
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: displacement patch record/evaluator
  - Additions to existing reusable library/module: displacement producer helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - cross-body external references refused by default
- Performance-sensitive behavior:
  - bounded displacement sampling/evaluation defaults
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - embedded source payload or stable in-body source identity first
- Test strategy:
  - source identity, sampled displacement, callable refusal if applicable,
    missing source, `.impress` round-trip, and no-hidden-mesh-fallback tests
- Data ownership:
  - producer owns source relationship metadata; patch owns displacement payload
- Routes:
  - displacement builder to source resolver to displacement patch to
    codec/tessellation/seams
- Reuse/extraction decision:
  - reuse source patch evaluation and displacement validator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- General callable displacement functions need the same safety posture as
  implicit fields if they are ever persisted.

Readiness blockers:
- displacement source identity and evaluation/tessellation specs

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Split completed in review Loop 3. Source identity resolution and
  displacement payload authoring have different validation and refusal records.

### Candidate Spec: Displacement Source Identity Resolver

Responsibilities by category:
- Functions/methods:
  - source identity resolver
  - source relationship diagnostic builder
  - source provenance recorder
- Data structures/models:
  - source identity/provenance record
  - missing source diagnostic
- Dependencies/services:
  - source patch evaluator
  - `.impress` source identity records
- Returns/outputs/signals:
  - resolved source identity
  - explicit missing/cross-body diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: source patch evaluation
  - Additions to existing reusable library/module: displacement source resolver
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - cross-body external references refused by default
- Performance-sensitive behavior:
  - bounded source lookup
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - embedded source payload or stable in-body source identity first
- Test strategy:
  - stable source identity, embedded source, missing source, cross-body
    reference refusal, and round-trip tests
- Data ownership:
  - resolver owns source relationship metadata; patch owns resolved identity
- Routes:
  - displacement builder to source resolver to displacement patch
- Reuse/extraction decision:
  - reuse `.impress` source identity records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Cross-body references stay refused unless a later lifecycle policy promotes
  them.

Readiness blockers:
- displacement source identity spec

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: this candidate owns source lookup and
  identity refusal independent of displacement field details.

### Candidate Spec: Displacement Payload Authoring Builder

Responsibilities by category:
- Functions/methods:
  - displacement patch builder
  - displacement payload validator wrapper
  - displacement payload diagnostic builder
- Data structures/models:
  - displacement authoring request
  - displacement producer diagnostic
  - lossiness metadata record
- Dependencies/services:
  - displacement payload validation
  - displacement source identity resolver
- Returns/outputs/signals:
  - native displacement patch
  - source identity and lossiness metadata
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: displacement patch record/evaluator
  - Additions to existing reusable library/module: displacement builder
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference fixture writes in tests
- Security/privacy-sensitive behavior:
  - callable displacement functions are refused unless safe policy supports them
- Performance-sensitive behavior:
  - bounded displacement sampling/evaluation defaults
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - sampled displacement payload first; persisted callable displacement refused
    unless safety policy supports it
- Test strategy:
  - sampled displacement, invalid displacement data, callable refusal,
    `.impress` round-trip, and no-hidden-mesh-fallback tests
- Data ownership:
  - builder owns construction metadata; patch owns displacement payload
- Routes:
  - builder to source resolver to displacement patch to codec/tessellation/seams
- Reuse/extraction decision:
  - reuse source patch evaluation and displacement validator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- General callable displacement functions need the same safety posture as
  implicit fields if they are ever persisted.

Readiness blockers:
- displacement source identity resolver
- displacement evaluation/tessellation specs

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: this candidate owns authoring and payload
  validation after source identity exists.

### Split Parent: Available-Family Operation Posture Evidence

Responsibilities by category:
- Functions/methods:
  - operation posture evidence collector
  - no-hidden-mesh-fallback verifier
  - availability report builder
- Data structures/models:
  - operation evidence record
  - family availability report
- Dependencies/services:
  - CSG support matrix
  - seam support matrix
  - tessellation/reference artifact gates
- Returns/outputs/signals:
  - per-family availability report
  - missing-operation diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: support matrix records and reference gates
  - Additions to existing reusable library/module: availability evidence report
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference/evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves unsafe/refused operation reasons without executing payloads
- Performance-sensitive behavior:
  - bounded report generation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - every available family must have rows for production, `.impress`,
    tessellation, seams, loft/non-applicable, CSG, and reference evidence
- Test strategy:
  - missing row, unsupported row, non-CSG row, supported row, and no mesh
    fallback report tests
- Data ownership:
  - operation matrices own support truth; report owns evidence summary
- Routes:
  - capability matrix to operation matrices to reference gates to report
- Reuse/extraction decision:
  - extend existing support-matrix diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This report should become the durable proof used when answering whether the
  surface-body system is complete.

Readiness blockers:
- availability gate fields
- operation matrices for advanced families

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 2 x 2 = 4
- Total: 27

Split decision:
- Split required. Evidence collection and report formatting should be split
  from no-hidden-mesh-fallback operation verification before implementation.

### Split Parent: Available-Family Operation Matrix Completeness Verifier

Responsibilities by category:
- Functions/methods:
  - operation posture matrix verifier
  - no-hidden-mesh-fallback verifier
  - missing-row diagnostic builder
- Data structures/models:
  - operation evidence record
  - missing-operation diagnostic
- Dependencies/services:
  - CSG support matrix
  - seam support matrix
  - tessellation gate
- Returns/outputs/signals:
  - pass/fail operation completeness result
  - missing/unsafe operation diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: support matrix records
  - Additions to existing reusable library/module: operation completeness verifier
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves unsafe/refused operation reasons without executing payloads
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - every available family must have rows for production, `.impress`,
    tessellation, seams, loft/non-applicable, CSG, and reference evidence
- Test strategy:
  - missing row, unsupported row, non-CSG row, supported row, and no mesh
    fallback tests
- Data ownership:
  - operation matrices own support truth; verifier owns completeness result
- Routes:
  - capability matrix to operation matrices to verifier
- Reuse/extraction decision:
  - extend existing support-matrix diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Unsupported/non-CSG is acceptable only when explicit and tested.

Readiness blockers:
- availability gate fields
- operation matrices for advanced families

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 25.5

Split decision:
- Split required. CSG, seam, and producer operation rows may need separate
  verifiers if this remains above threshold during final spec promotion.

### Split Parent: Available-Family Producer Storage And Tessellation Operation Rows

Responsibilities by category:
- Functions/methods:
  - producer-path row verifier
  - `.impress` row verifier
  - tessellation row verifier
- Data structures/models:
  - operation evidence record
  - missing-operation diagnostic
- Dependencies/services:
  - capability matrix
  - `.impress` codec registry
  - tessellation adapters
- Returns/outputs/signals:
  - producer/storage/tessellation completeness result
  - missing-row diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: capability and codec records
  - Additions to existing reusable library/module: row verifier helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves refused producer/import reasons
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - every available family must have at least one producer/import/payload path,
    `.impress` rows, and tessellation rows
- Test strategy:
  - missing producer, missing codec, missing tessellation, and explicit
    unsupported import diagnostics
- Data ownership:
  - operation matrices own support truth; verifier owns completeness result
- Routes:
  - capability matrix to codec/tessellation registries to verifier
- Reuse/extraction decision:
  - extend existing support-matrix diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Documented `.impress` payload authoring may satisfy availability before a
  native builder only when labeled `import-only`.

Readiness blockers:
- availability gate fields

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- Split completed in critical rescore pass. Producer-path verification and
  storage/tessellation verification are different operation surfaces with
  different registries.

### Candidate Spec: Available-Family Producer Path Operation Rows

Responsibilities by category:
- Functions/methods:
  - producer-path row verifier
  - producer/import missing-row diagnostic builder
  - producer availability row summarizer
- Data structures/models:
  - operation evidence record
  - missing-producer diagnostic
- Dependencies/services:
  - capability matrix
  - producer-path registry
- Returns/outputs/signals:
  - producer-path completeness result
  - missing producer/import diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: capability records
  - Additions to existing reusable library/module: producer row verifier
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves refused producer/import reasons
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - every available family must have at least one producer, import, or
    documented payload-authoring path
- Test strategy:
  - missing producer, import-only producer, native producer, and unsupported
    import diagnostics
- Data ownership:
  - producer registry owns producer truth; verifier owns completeness result
- Routes:
  - capability matrix to producer registry to verifier
- Reuse/extraction decision:
  - extend existing support-matrix diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Documented `.impress` payload authoring may satisfy availability before a
  native builder only when labeled `import-only`.

Readiness blockers:
- availability gate fields

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: this candidate owns whether the family has
  an intentional authoring path.

### Candidate Spec: Available-Family Storage And Tessellation Operation Rows

Responsibilities by category:
- Functions/methods:
  - `.impress` row verifier
  - tessellation row verifier
  - storage/tessellation missing-row diagnostic builder
- Data structures/models:
  - operation evidence record
  - missing storage/tessellation diagnostic
- Dependencies/services:
  - `.impress` codec registry
  - tessellation adapters
- Returns/outputs/signals:
  - storage/tessellation completeness result
  - missing-row diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: codec and tessellation records
  - Additions to existing reusable library/module: storage/tessellation row
    verifier
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves refused payload/import reasons
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - available families must have native `.impress` rows and tessellation rows
- Test strategy:
  - missing codec, missing tessellation, supported codec, supported tessellation,
    and explicit tessellation-boundary diagnostics
- Data ownership:
  - codec/tessellation registries own operation truth; verifier owns
    completeness result
- Routes:
  - capability matrix to codec/tessellation registries to verifier
- Reuse/extraction decision:
  - extend existing support-matrix diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Tessellation rows prove preview/export availability, not authored surface
  truth.

Readiness blockers:
- availability gate fields

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: storage and tessellation are the two
  infrastructure rows every available family must expose after production.

### Split Parent: Available-Family Seam CSG And Loft Operation Rows

Responsibilities by category:
- Functions/methods:
  - seam row verifier
  - CSG row verifier
  - loft/non-applicable row verifier
- Data structures/models:
  - operation evidence record
  - no-hidden-mesh-fallback diagnostic
- Dependencies/services:
  - seam support matrix
  - CSG support matrix
  - loft family selection records
- Returns/outputs/signals:
  - operation completeness result
  - missing/unsafe operation diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: seam, CSG, and loft support records
  - Additions to existing reusable library/module: row verifier helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves unsafe/refused operation reasons without executing payloads
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported and non-CSG rows are acceptable only when explicit and tested
- Test strategy:
  - missing seam row, missing CSG row, non-CSG row, supported row, and no mesh
    fallback tests
- Data ownership:
  - operation matrices own support truth; verifier owns completeness result
- Routes:
  - capability matrix to seam/CSG/loft matrices to verifier
- Reuse/extraction decision:
  - extend existing support-matrix diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Loft may be explicitly non-applicable for some families, but that must be a
  row, not an omission.

Readiness blockers:
- availability gate fields
- advanced-family operation matrices

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 25.5

Split decision:
- Split required if promoted directly. CSG and seam/loft rows may need separate
  final specs; keep visible as a split target for the next manifest pass.

### Candidate Spec: Available-Family Seam And Loft Operation Rows

Responsibilities by category:
- Functions/methods:
  - seam row verifier
  - loft/non-applicable row verifier
  - no-hidden-mesh-fallback diagnostic builder
- Data structures/models:
  - operation evidence record
  - no-hidden-mesh-fallback diagnostic
- Dependencies/services:
  - seam support matrix
  - loft family selection records
- Returns/outputs/signals:
  - seam/loft completeness result
  - missing/unsafe operation diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: seam and loft support records
  - Additions to existing reusable library/module: row verifier helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - loft may be non-applicable only when the matrix says so explicitly
- Test strategy:
  - missing seam row, missing loft row, explicit non-applicable loft row, and
    no-hidden-mesh-fallback tests
- Data ownership:
  - seam/loft matrices own support truth; verifier owns completeness result
- Routes:
  - capability matrix to seam/loft matrices to verifier
- Reuse/extraction decision:
  - extend existing support-matrix diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Loft non-applicability must name the reason and family.

Readiness blockers:
- availability gate fields

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:
- Review for split. Cohesion reason: seam and loft rows are the non-CSG
  operation availability surface and share no-hidden-mesh-fallback assertions.

### Split Parent: Available-Family CSG Operation Rows

Responsibilities by category:
- Functions/methods:
  - CSG row verifier
  - non-CSG classification verifier
  - CSG no-hidden-mesh-fallback diagnostic builder
- Data structures/models:
  - operation evidence record
  - CSG missing-row diagnostic
- Dependencies/services:
  - CSG support matrix
  - operation planner refusal diagnostics
- Returns/outputs/signals:
  - CSG completeness result
  - missing/unsafe CSG operation diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: CSG support records
  - Additions to existing reusable library/module: CSG row verifier helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves unsafe/refused operation reasons without executing payloads
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported and non-CSG rows are acceptable only when explicit and tested
- Test strategy:
  - missing CSG row, unsupported row, non-CSG row, supported row, unsafe
    implicit row, and no mesh fallback tests
- Data ownership:
  - CSG matrix owns support truth; verifier owns completeness result
- Routes:
  - capability matrix to CSG support matrix to verifier
- Reuse/extraction decision:
  - extend existing CSG support-matrix diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This verifier must distinguish unsupported from non-CSG, since both are valid
  but mean different things.

Readiness blockers:
- availability gate fields
- advanced-family CSG support matrix

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 24.5

Split decision:
- Split completed in review Loop 4. CSG classification row completeness and
  no-hidden-mesh-fallback proof are separate verification responsibilities.

### Candidate Spec: Available-Family CSG Classification Row Verifier

Responsibilities by category:
- Functions/methods:
  - CSG row verifier
  - non-CSG classification verifier
  - missing-row diagnostic builder
- Data structures/models:
  - CSG operation evidence record
  - CSG missing-row diagnostic
- Dependencies/services:
  - CSG support matrix
  - operation planner refusal diagnostics
- Returns/outputs/signals:
  - CSG completeness result
  - missing/unsafe CSG operation diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: CSG support records
  - Additions to existing reusable library/module: CSG row verifier helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves unsafe/refused operation reasons without executing payloads
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported and non-CSG rows are acceptable only when explicit and tested
- Test strategy:
  - missing CSG row, unsupported row, non-CSG row, supported row, and unsafe
    implicit row tests
- Data ownership:
  - CSG matrix owns support truth; verifier owns completeness result
- Routes:
  - capability matrix to CSG support matrix to verifier
- Reuse/extraction decision:
  - extend existing CSG support-matrix diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This verifier must distinguish unsupported from non-CSG, since both are valid
  but mean different things.

Readiness blockers:
- availability gate fields
- advanced-family CSG support matrix

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: this candidate owns CSG support truth and
  non-CSG classification completeness.

### Candidate Spec: Available-Family CSG No-Mesh-Fallback Evidence

Responsibilities by category:
- Functions/methods:
  - CSG no-hidden-mesh-fallback verifier
  - mesh-boundary diagnostic assertion helper
  - operation-regression evidence collector
- Data structures/models:
  - no-hidden-mesh-fallback diagnostic
  - CSG regression evidence record
- Dependencies/services:
  - CSG operation planner
  - tessellation boundary guard
- Returns/outputs/signals:
  - no-mesh-fallback pass/fail result
  - regression diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: CSG operation diagnostics
  - Additions to existing reusable library/module: no-mesh-fallback verifier
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - regression fixture writes in tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded test/evidence scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - CSG unsupported/non-CSG rows must return diagnostics, never mesh fallback
- Test strategy:
  - unsupported pair, non-CSG pair, adapter pair, supported pair, and explicit
    tessellation-boundary tests
- Data ownership:
  - verifier owns regression evidence; CSG matrix owns support truth
- Routes:
  - CSG row verifier to operation planner to no-mesh-fallback evidence
- Reuse/extraction decision:
  - reuse tessellation boundary guardrails
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This verifier should reuse broader mesh-boundary checks rather than inventing
  a CSG-only scanner.

Readiness blockers:
- available-family CSG classification row verifier
- tessellation boundary guardrails

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns CSG-specific
  no-mesh-fallback evidence after CSG classification rows exist.

### Split Parent: Available-Family Evidence Report And Reference Gate

Responsibilities by category:
- Functions/methods:
  - availability report builder
  - reference evidence collector
  - promoted evidence gate checker
- Data structures/models:
  - family availability report
  - reference evidence summary
- Dependencies/services:
  - operation completeness verifier
  - reference artifact promotion gates
- Returns/outputs/signals:
  - per-family availability report
  - missing-reference diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: reference gates
  - Additions to existing reusable library/module: availability report builder
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference/evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves refusal reasons without executing unsafe payloads
- Performance-sensitive behavior:
  - bounded report generation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - dirty artifacts never count as availability evidence
- Test strategy:
  - missing reference, dirty reference, promoted reference, diagnostic reference,
    and report snapshot tests
- Data ownership:
  - report owns evidence summary; reference system owns artifacts
- Routes:
  - operation verifier to reference gates to availability report
- Reuse/extraction decision:
  - reuse reference artifact promotion records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This report should become the durable proof used when answering whether the
  surface-body system is complete.

Readiness blockers:
- operation matrix completeness verifier
- reference artifact promotion gates

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 2 x 2 = 4
- Total: 25.5

Split decision:
- Split required. Report generation and reference promotion checks may need
  separate specs if this remains above threshold during final spec promotion.

### Candidate Spec: Available-Family Reference Evidence Gate

Responsibilities by category:
- Functions/methods:
  - reference evidence collector
  - promoted evidence gate checker
  - dirty-artifact refusal diagnostic builder
- Data structures/models:
  - reference evidence summary
  - missing-reference diagnostic
- Dependencies/services:
  - reference artifact promotion gates
  - operation completeness verifier
- Returns/outputs/signals:
  - reference evidence pass/fail result
  - missing/dirty artifact diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: reference gates
  - Additions to existing reusable library/module: availability reference gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference/evidence fixture writes in tests
- Security/privacy-sensitive behavior:
  - preserves refusal reasons without executing unsafe payloads
- Performance-sensitive behavior:
  - bounded evidence scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - dirty artifacts never count as availability evidence
- Test strategy:
  - missing reference, dirty reference, promoted reference, and diagnostic
    reference tests
- Data ownership:
  - reference system owns artifacts; gate owns evidence result
- Routes:
  - operation verifier to reference gates
- Reuse/extraction decision:
  - reuse reference artifact promotion records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Negative diagnostic references should count only for refusal-path evidence,
  not positive model-output evidence.

Readiness blockers:
- operation matrix completeness verifier
- reference artifact promotion gates

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns evidence acceptance
  rules and reference artifact lifecycle integration.

### Candidate Spec: Available-Family Completion Report Builder

Responsibilities by category:
- Functions/methods:
  - availability report builder
  - report snapshot serializer
  - missing-evidence summary builder
- Data structures/models:
  - family availability report
  - report snapshot record
- Dependencies/services:
  - reference evidence gate
  - operation completeness verifiers
- Returns/outputs/signals:
  - per-family availability report
  - missing-evidence diagnostics
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: operation/reference evidence records
  - Additions to existing reusable library/module: availability report builder
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - report snapshot fixture writes in tests
- Security/privacy-sensitive behavior:
  - reports refusal reasons without executing unsafe payloads
- Performance-sensitive behavior:
  - bounded report generation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - report is generated from evidence records; it does not infer availability
    from class existence
- Test strategy:
  - complete family report, missing producer report, missing operation report,
    dirty reference report, and snapshot determinism tests
- Data ownership:
  - report owns evidence summary, not source support truth
- Routes:
  - operation/reference evidence to availability report
- Reuse/extraction decision:
  - reuse matrix and reference evidence records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This report should become the durable proof used when answering whether the
  surface-body system is complete.

Readiness blockers:
- operation matrix completeness verifier
- reference evidence gate

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: report building is a deterministic
  summarization layer over already-collected operation and reference evidence.

## Manifest Review Notes

Initial review found three candidates above the required split threshold:

- `Implicit Declarative Field Authoring Producer`: 26.5
- `Heightmap Native Grid And Import Producer`: 25.5
- `Available-Family Operation Posture Evidence`: 27

Loop 1 review found that subdivision native construction and external cage
import should not stay bundled even though the parent was below `25`.

Loop 1 split:

- `Subdivision Authoring And Import Producer`: 24.5, split into native cage
  builder and cage import adapter.

Loop 2 review found that implicit negative authoring diagnostics were hiding two
different refusal surfaces.

Loop 2 split:

- `Implicit Authoring Safety And Budget Diagnostics`: 24.5, split into unsafe
  authoring diagnostics and budget/bound diagnostics.

Loop 3 review found that displacement authoring bundled source identity
resolution with payload construction.

Loop 3 split:

- `Displacement Native Authoring Producer`: 24.5, split into source identity
  resolver and payload authoring builder.

Loop 4 review found that CSG availability rows bundled support-truth
classification with regression evidence that mesh fallback never reappears.

Loop 4 split:

- `Available-Family CSG Operation Rows`: 24.5, split into classification row
  verifier and no-mesh-fallback evidence.

Critical rescore pass:

- `Available-Family Producer Storage And Tessellation Operation Rows` was
  rescored as underdefined because it mixed producer-path, `.impress`, and
  tessellation registries. It is now split into producer-path operation rows
  and storage/tessellation operation rows.

Active candidates after critical rescore:

- Advanced Family Availability Gate And Matrix Fields: 21.5
- Subdivision Native Cage Builder: 21.5
- Subdivision Cage Import Adapter: 22.5
- Implicit Field Builder And Helper API: 22.5
- Implicit Unsafe Authoring Diagnostics: 24.5
- Implicit Budget And Bound Diagnostics: 23.5
- Heightmap Native Finite Grid Builder: 23.5
- Heightmap Optional Import Adapter: 22.5
- Displacement Source Identity Resolver: 22.5
- Displacement Payload Authoring Builder: 24.5
- Available-Family Producer Path Operation Rows: 22.5
- Available-Family Storage And Tessellation Operation Rows: 22.5
- Available-Family Seam And Loft Operation Rows: 20.5
- Available-Family CSG Classification Row Verifier: 24.5
- Available-Family CSG No-Mesh-Fallback Evidence: 23.5
- Available-Family Reference Evidence Gate: 23.5
- Available-Family Completion Report Builder: 23.5

No active candidate remains at `25+`. Every active candidate in the `16-24`
range carries a split-review cohesion reason.

## Change History

- 2026-05-27: Critically rescored the active manifest and split
  producer-path operation rows from storage/tessellation operation rows.
- 2026-05-27: Ran four review/rescore/split loops on the specification
  manifest, splitting subdivision native/import work, implicit safety/budget
  diagnostics, displacement source/payload authoring, and CSG
  classification/no-mesh-fallback evidence.
- 2026-05-27: Added architecture for promoting subdivision, implicit,
  heightmap, and displacement from implemented internals to available authored
  surface-body families through producer/import/authoring paths and operation
  posture evidence.
