# Advanced Patch Family Implementation Completion Architecture

## Overview

This architecture defines what must be true before the following patch
families can move from `planned` to `implemented` and, where appropriate,
`available` in the surface-body capability matrix:

- B-spline
- NURBS
- sweep
- subdivision
- implicit
- heightmap
- displacement

The target is not to rename a phase flag. The target is to make each family a
real authored surface-body capability across runtime storage, evaluation,
seams, CSG policy, loft and producer integration, diagnostics, `.impress`
persistence, tessellation, and completion evidence.

## Relationship To Existing Architecture

This document extends:

- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [Patch Family Integration Architecture](patch-family-integration-architecture.md)
- [Surface Body Completion Architecture](surface-body-completion-architecture.md)
- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [SurfaceBody CSG Architecture](surfacebody-csg-architecture.md)
- [Higher-Order Surface CSG Solver Architecture](higher-order-surface-csg-solver-architecture.md)
- [Exact Surface Intersection Kernel Architecture](exact-surface-intersection-kernel-architecture.md)
- [Higher-Order Seam Continuity Architecture](higher-order-seam-continuity-architecture.md)
- [Loft Evolution System Architecture](loft-evolution-system.md)
- [Loft Topology Point Correspondence Architecture](loft-topology-point-correspondence-architecture.md)
- [Reference Artifact Promotion Architecture](reference-artifact-promotion-architecture.md)

It specializes the broad patch-family integration contract into a release
completion program for the families that are still `planned`.

## Implementation State Model

The patch-family matrix needs three distinct concepts. They must not be
collapsed into one boolean.

`specified`:

- architecture and final specs exist
- runtime payload shape may be named
- no claim is made that users can rely on the family

`implemented`:

- runtime patch class exists and validates its payload
- evaluation, derivatives, tessellation, transforms, canonical payloads, stable
  identity, `.impress` round-trip, and diagnostics have tests
- every unsupported consumer has a structured refusal path
- mesh is not used as authored truth

`available`:

- the family is implemented
- at least one authored producer or documented external authoring path creates
  the family intentionally
- seams, trims, `.impress`, tessellation, reference evidence, and completion
  gate evidence are promoted
- CSG/loft/support gaps are either implemented for the required release scope
  or explicitly classified as supported, declared-tolerance, adapter, or
  unsupported with durable diagnostics

For this program, the user-facing goal is `implemented`. A family may become
`available` only when its operation support matrix is honest enough that users
can rely on it without reading source code.

## Shared Completion Gates

Every listed family must pass these gates before it leaves `planned`.

### Runtime Gate

Required:

- concrete patch record or class
- finite payload validation
- parameter-domain ownership
- deterministic `canonical_payload`
- deterministic `stable_identity`
- transform preservation
- metadata preservation
- trim-loop policy
- boundary descriptor extraction
- family-local malformed payload diagnostics

### Evaluation Gate

Required:

- `point_at(u, v)` or an explicitly named family-local evaluation operation
- first derivatives where mathematically defined
- normals or a structured normal-refusal diagnostic
- higher-order derivative support where C1/C2/G1/G2 seams require it
- finite numerical result validation
- deterministic approximation metadata for sampled families

### Seam And Adjacency Gate

Required:

- boundary-use records for each seam-capable boundary
- family-local boundary descriptor extraction
- cross-family seam comparison inputs
- C0/G0 seam participation
- C1/C2/G1/G2 residual evaluation when the family can provide derivatives
- exact "what and where" diagnostics when a seam cannot be evaluated
- no conversion to mesh vertices as the primary seam truth

### CSG Gate

Required:

- family-pair entries in the surface CSG support matrix
- exact, declared-tolerance, adapter, unsupported, or non-CSG classification
  for every operation/family pair
- intersection solver selection or refusal diagnostics
- trim-graph and fragment-graph support for supported pairs
- cap family selection for generated boundaries
- shell assembly, seam rebuild, validity, healing, and provenance handoff for
  supported pairs
- explicit non-executable operation plans for unsupported pairs
- no hidden mesh fallback

Implicit, heightmap, and displacement may be classified as bounded adapter or
non-CSG families for some operations, but that classification must be explicit
and tested.

### Loft And Producer Gate

Required:

- at least one producer or external authoring path for each family
- loft family selection rules where loft can reasonably produce the family
- authored topology rails and ambiguity diagnostics honored before execution
- point birth/death and split/merge behavior either supported or refused with
  exact locators
- no mesh executor as canonical surface output

### `.impress` Gate

Required:

- payload version
- encoder
- decoder
- malformed-payload diagnostics
- unsafe-payload diagnostics where applicable
- whole-store round-trip tests
- stable identity preservation
- metadata, seams, trims, and provenance preservation
- no mesh-wrapper serialization as surface truth

Heightmap and displacement cannot leave `planned` until their `.impress`
payloads round-trip as native authored payloads.

### Tessellation Boundary Gate

Required:

- family tessellation adapter
- request normalization behavior
- preview/export quality behavior
- source patch identity in mesh metadata
- lossiness and approximation metadata
- deterministic output under fixed request
- no mutation of authored surface truth

### Diagnostics And Evidence Gate

Required:

- negative diagnostic fixtures for malformed, unsupported, unsafe, ambiguous,
  and non-executable states
- reference artifact requirements where the family produces visible model
  output
- dirty artifacts excluded from completion evidence
- completion-gate evidence records for implementation and verification
- promotion readiness audit has no family-local gaps for the claimed phase

## Family Completion Contracts

### B-spline

B-spline patches become implemented when Impression can author, store,
evaluate, persist, seam-check, tessellate, and diagnose non-rational
tensor-product spline surfaces.

Required runtime payload:

- degree in `u` and `v`
- knot vectors in `u` and `v`
- rectangular control net
- parameter domain and knot normalization policy
- trim loops
- metadata and transform

Required algorithms:

- knot vector validation
- basis function evaluation
- basis derivative evaluation
- de Boor or equivalent tensor-product evaluation
- boundary curve extraction
- adaptive tessellation by curvature or declared tolerance

Required integrations:

- `.impress` codec and whole-store round trip
- seam C0/G0 and derivative residual support
- loft output for smooth non-rational multi-station surfaces
- CSG intersection participation through analytic/spline and spline/spline
  solver registry entries
- malformed knot/control-net diagnostics

### NURBS

NURBS patches become implemented when rational tensor-product surfaces reuse the
B-spline basis infrastructure while preserving weights and conic-exact intent.

Required runtime payload:

- B-spline degree and knot data
- rectangular control net
- rectangular weight net or weighted control net
- rational normalization policy
- trim loops
- metadata and transform

Required algorithms:

- all B-spline basis and derivative algorithms
- homogeneous coordinate evaluation
- rational derivative correction
- weight validation
- conic-preservation diagnostics

Required integrations:

- `.impress` codec and whole-store round trip
- seam C0/G0 and higher-order residual support
- loft output only when rational intent is authored
- CSG participation through analytic/NURBS and NURBS/NURBS solver entries
- diagnostics for zero, negative, non-finite, or mismatched weights

### Sweep

Sweep patches become implemented when a profile transported along a path can be
stored as authored truth instead of being expanded into mesh or unrelated ruled
patches.

Required runtime payload:

- profile curve or profile section
- path curve
- frame transport policy
- twist, scale, and orientation policy
- start/end closure policy
- metadata and transform

Required algorithms:

- path evaluation
- profile evaluation
- frame transport
- twist/scale interpolation
- sweep point and derivative evaluation
- self-intersection and degeneracy diagnostics

Required integrations:

- `.impress` codec and whole-store round trip
- seam boundaries for start, end, and profile closure
- loft or path-extrude producer selection when path/profile intent is explicit
- tessellation by path curvature and frame-change error
- CSG support matrix entries and explicit refusal records where unsupported

### Subdivision

Subdivision patches become implemented when a cage-and-crease authored surface
can stay as a native patch and produce deterministic evaluation and tessellation
evidence.

Required runtime payload:

- control cage vertices
- control cage faces
- subdivision scheme
- crease/sharpness records
- boundary and hole policy
- metadata and transform

Required algorithms:

- deterministic finite-level subdivision
- limit-position or approved deterministic approximation
- limit-normal or normal approximation diagnostics
- crease and boundary handling
- boundary chain extraction

Required integrations:

- `.impress` codec and whole-store round trip
- seam boundary descriptors with approximation metadata
- tessellation adapter with deterministic refinement level or error target
- CSG matrix entries with adapter/unsupported diagnostics unless exact support
  is implemented
- reference artifacts proving non-planar organic output

### Implicit

Implicit patches become implemented when declarative field-defined surfaces are
safe, bounded, evaluable, persistable, and consumable at explicit adapter
boundaries.

Required runtime payload:

- allow-listed field node graph
- iso value
- bounded evaluation domain
- sampling and safety policy
- optional gradient policy
- metadata and transform

Required algorithms:

- field-node validation
- scalar field evaluation
- gradient evaluation or finite-difference gradient
- bounded isosurface extraction
- budget and safety refusal
- residual classification

Required integrations:

- `.impress` codec that refuses executable callbacks and unsafe payloads
- seam participation only through declared approximation records unless exact
  boundary is available
- CSG matrix entries as exact, adapter, unsupported, or non-CSG by operation
- diagnostic fixtures for unsafe fields, budget exhaustion, residual failure,
  and unsupported family pairs
- no executable code persistence

### Heightmap

Heightmap patches become implemented when sampled heightfield data is a native
surface payload with persistence and explicit sampling limits.

Required runtime payload:

- height sample array or external-data reference policy
- grid origin and axes
- sample spacing or parameter domain
- mask/no-data policy
- interpolation policy
- metadata and transform

Required algorithms:

- height evaluation
- derivative/normal estimation
- boundary extraction
- mask-aware tessellation
- sample-grid validation

Required integrations:

- `.impress` codec and whole-store round trip
- seam support with sampled-boundary approximation metadata
- tessellation adapter preserving sampled source identity
- CSG classification as adapter or unsupported unless native heightfield CSG is
  implemented
- diagnostics for mismatched grid shape, non-finite samples, invalid masks, and
  unsupported seam/CSG requests

### Displacement

Displacement patches become implemented when a source surface plus displacement
field stays as authored truth instead of being baked to a mesh.

Required runtime payload:

- source patch reference or embedded source patch payload
- displacement field or sampled displacement map
- displacement direction policy
- amplitude and units
- parameter-domain mapping
- metadata and transform

Required algorithms:

- source patch evaluation
- displacement evaluation
- displaced point evaluation
- derivative and normal approximation
- bounded tessellation with lossiness metadata

Required integrations:

- `.impress` codec and whole-store round trip
- seam support that can compare displaced boundaries or refuse with locators
- CSG classification as adapter or unsupported unless native displaced-surface
  CSG is implemented
- diagnostics for missing source patch, invalid displacement data, domain
  mismatch, and non-finite output

## Shared Data Flow

```text
Authoring input or .impress payload
-> family payload validator
-> family-native SurfacePatch
-> SurfaceShell / SurfaceBody / SurfaceBodyStore
-> seam and adjacency validation
-> operation-specific support matrix
-> exact solver, declared-tolerance adapter, or structured refusal
-> explicit tessellation only for preview/export/analysis/reference evidence
```

No path in this flow may replace the authored patch with a mesh-derived wrapper
before the tessellation boundary.

## Capability Matrix Update Rule

The capability matrix must move the seven families out of `planned` only after
the corresponding promotion audit proves:

- required runtime gates pass
- required `.impress` gates pass
- required tessellation gates pass
- seam participation is implemented or explicitly scoped with diagnostics
- CSG support matrix is complete for that family, including unsupported pairs
- loft/producer role is implemented or explicitly not applicable
- diagnostics and reference evidence are present

The update must be atomic with tests. A phase change without evidence is a bug.

## Specification Manifest for Discovery

The following manifest follows the project specification-manifest entry shape
used by sibling architecture documents. Scores use the shared policy:

- `25+`: split required before implementation
- `16-24`: explicit split review required
- `0-15`: may remain cohesive when readiness fields are complete

### Candidate Spec: Advanced Family Promotion Gate And Capability Matrix Update

Discovery purpose:
- Define the code gate that moves B-spline, NURBS, sweep, subdivision,
  implicit, heightmap, and displacement from `planned` to `implemented` only
  when evidence exists.

Responsibilities:
- Functions/methods:
  - promotion gate evaluator
  - capability matrix updater/assertion helper
- Data structures/models:
  - advanced family promotion gate record
  - promotion evidence record
- Dependencies/services:
  - `PATCH_FAMILY_CAPABILITY_MATRIX`
  - promotion readiness audit
  - completion evidence gate
- Returns/outputs/signals:
  - promotion report
  - blocking diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current capability and readiness records
  - Additions to existing reusable library/module: surface completion gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - updates release capability truth
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded matrix evaluation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - no family phase changes without evidence
- Test strategy:
  - promotion gate tests for passing and missing-evidence states
- Data ownership:
  - capability matrix owns family phase truth
- Routes:
  - readiness audit to capability matrix to completion gate
- Reuse/extraction decision:
  - extend existing gate records rather than adding a second matrix
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- `implemented` and `available` may need separate matrix fields.

Readiness blockers:
- none

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
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
- Total: 17.5

Split decision:
- Review for split. Cohesion reason: this spec owns only the cross-family gate,
  not the family implementations.

### Split Parent: B-spline Implemented Family Completion

Discovery purpose:
- Complete B-spline runtime, evaluation, seams, tessellation, `.impress`, CSG
  matrix, loft output, diagnostics, and evidence.

Responsibilities:
- Functions/methods:
  - B-spline evaluator
  - B-spline derivative evaluator
  - B-spline boundary extractor
  - B-spline tessellation adapter
- Data structures/models:
  - B-spline patch payload
  - knot validation diagnostic
  - promotion evidence record
- Dependencies/services:
  - spline basis utilities
  - `.impress` codec
  - CSG and intersection registries
- Returns/outputs/signals:
  - implemented family readiness
  - malformed payload diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `BSplineSurfacePatch`
  - Additions to existing reusable library/module: spline basis/evaluation helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - adaptive evaluation and tessellation bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - non-rational tensor-product B-spline with finite clamped/open knots first
- Test strategy:
  - evaluation, derivative, boundary, tessellation, `.impress`, seam, CSG matrix,
    and loft producer tests
- Data ownership:
  - B-spline patch owns spline payload; shared spline utilities own basis math
- Routes:
  - authoring/loft to B-spline patch to store to `.impress`/tessellation/CSG
- Reuse/extraction decision:
  - extract shared spline basis utilities for B-spline and NURBS
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Periodic and non-uniform knots can be staged behind explicit support flags.

Readiness blockers:
- shared spline basis utilities

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 3 x 1 = 3
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
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Split completed in Loop 2. Shared spline basis infrastructure is hidden work
  used by both B-spline and NURBS, so it is split from B-spline family
  completion.

### Candidate Spec: Shared Spline Basis And Knot Infrastructure

Discovery purpose:
- Build reusable spline basis, knot validation, derivative, and control-net
  canonicalization utilities shared by B-spline and NURBS surfaces.

Responsibilities:
- Functions/methods:
  - knot vector validator
  - basis evaluator
  - basis derivative evaluator
  - control-net canonicalizer
- Data structures/models:
  - spline basis diagnostic
  - knot policy record
- Dependencies/services:
  - B-spline patch record
  - NURBS patch record
- Returns/outputs/signals:
  - evaluated basis values
  - validation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current spline patch records
  - Additions to existing reusable library/module: shared spline utility layer
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
  - bounded basis evaluation and finite validation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - finite clamped/open knots first; periodic support behind explicit flags
- Test strategy:
  - knot validation, basis values, derivative values, malformed control nets
- Data ownership:
  - shared spline utilities own basis math; patch classes own geometry payloads
- Routes:
  - B-spline/NURBS patch evaluators to shared spline utility layer
- Reuse/extraction decision:
  - required shared utility, not family-local duplication
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- periodic knot support can be separate future scope if not needed for promotion.

Readiness blockers:
- none

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
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
- Total: 17.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only shared spline
  math and validation utilities.

### Candidate Spec: B-spline Runtime Persistence And Tessellation Completion

Discovery purpose:
- Complete B-spline patch evaluation, boundary extraction, tessellation,
  `.impress`, seams, CSG matrix participation, loft output, diagnostics, and
  evidence using shared spline utilities.

Responsibilities:
- Functions/methods:
  - B-spline evaluator
  - B-spline boundary extractor
  - B-spline tessellation adapter
- Data structures/models:
  - B-spline patch payload
  - B-spline promotion evidence record
- Dependencies/services:
  - shared spline basis utilities
  - `.impress` codec
  - CSG and intersection registries
- Returns/outputs/signals:
  - implemented B-spline readiness
  - malformed payload diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `BSplineSurfacePatch`
  - Additions to existing reusable library/module: B-spline family adapters
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - adaptive tessellation bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - non-rational tensor-product B-spline using shared finite knot policy
- Test strategy:
  - evaluation, boundary, tessellation, `.impress`, seam, CSG matrix, loft producer
- Data ownership:
  - B-spline patch owns geometry payload
- Routes:
  - loft/authoring to B-spline patch to store/codec/tessellation/CSG
- Reuse/extraction decision:
  - consume shared spline utilities
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- smooth loft producer may be its own spec if it affects loft planning broadly.

Readiness blockers:
- shared spline basis utilities

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:
- Review for split. Cohesion reason: this candidate is one family adapter layer
  after shared spline utilities exist.

### Split Parent: NURBS Implemented Family Completion

Discovery purpose:
- Complete rational surface support by extending shared B-spline infrastructure
  with weights, rational evaluation, persistence, seams, CSG policy, diagnostics,
  and evidence.

Responsibilities:
- Functions/methods:
  - rational evaluator
  - rational derivative evaluator
  - NURBS boundary extractor
- Data structures/models:
  - NURBS patch payload
  - weight validation diagnostic
  - conic-preservation metadata
- Dependencies/services:
  - shared spline basis utilities
  - `.impress` codec
  - CSG/intersection registries
- Returns/outputs/signals:
  - implemented family readiness
  - malformed weight diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `NURBSSurfacePatch`
  - Additions to existing reusable library/module: rational layer over spline helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - rational evaluation and tessellation bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - finite positive weights; rational correction required for derivatives
- Test strategy:
  - rational evaluation, malformed weights, `.impress`, tessellation, seam, and
    CSG matrix tests
- Data ownership:
  - NURBS patch owns weights; spline utilities own basis functions
- Routes:
  - rational authoring/import to NURBS patch to store to consumers
- Reuse/extraction decision:
  - share spline basis with B-spline
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Exact conic construction helpers may be separate producer work.

Readiness blockers:
- B-spline basis helper completion

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Split completed in Loop 4. Rational evaluation and weight validation are
  reusable enough to separate from NURBS family integration.

### Candidate Spec: NURBS Rational Evaluation And Weight Validation

Discovery purpose:
- Implement rational homogeneous evaluation, derivative correction, and weight
  validation on top of shared spline basis utilities.

Responsibilities:
- Functions/methods:
  - rational evaluator
  - rational derivative evaluator
  - weight validator
- Data structures/models:
  - weight validation diagnostic
  - rational evaluation metadata
- Dependencies/services:
  - shared spline basis utilities
  - NURBS patch payload
- Returns/outputs/signals:
  - rational point/derivative values
  - malformed weight diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `NURBSSurfacePatch`
  - Additions to existing reusable library/module: rational evaluation helper
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
  - bounded rational evaluation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - finite positive weights; reject zero, negative, non-finite, and shape-mismatched weights
- Test strategy:
  - rational point, derivative, malformed weights, and conic metadata tests
- Data ownership:
  - NURBS patch owns weights; rational helper owns homogeneous evaluation
- Routes:
  - NURBS patch evaluator to shared spline basis to rational helper
- Reuse/extraction decision:
  - keep rational layer separate from non-rational B-spline evaluator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Exact conic producer helpers are split into their own manifest candidate so
  rational evaluation does not hide construction-helper work.

Readiness blockers:
- shared spline basis utilities

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 17.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only rational math and
  validation.

### Candidate Spec: NURBS Exact Conic Producer Helpers

Responsibilities by category:
- Functions/methods:
  - exact circular arc rational control-net builder
  - exact ellipse/conic profile helper for rational section construction
  - helper for producing NURBS patch/profile payloads with conic provenance
- Data structures/models:
  - conic construction request
  - conic construction diagnostic
- Dependencies/services:
  - shared spline basis utilities
  - NURBS rational evaluation helper
- Returns/outputs/signals:
  - NURBS-compatible control points, weights, knots, and metadata
  - validation diagnostics for unsupported conic requests
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: rational evaluator and knot validation helpers
  - Additions to existing reusable library/module: conic helper functions in
    surface modeling
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
  - constant-size conic helper construction
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - support exact circular arcs and ellipses first; unsupported conic kinds
    return structured diagnostics rather than approximate silently
- Test strategy:
  - exact circle/ellipse helper tests, malformed request tests, and NURBS
    round-trip metadata tests
- Data ownership:
  - helper owns construction metadata; resulting patch/profile owns the
    generated rational payload
- Routes:
  - conic helper to NURBS rational evaluator to `.impress` codec
- Reuse/extraction decision:
  - keep helpers on top of the rational evaluator; do not duplicate basis math
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Hyperbola and parabola helpers are explicit unsupported diagnostics unless a
  later manifest candidate promotes them; circle and ellipse cover the current
  exact-conic completion gap.

Readiness blockers:
- shared spline basis utilities
- NURBS rational evaluation helper

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 0 x 2 = 0
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 2 x 2 = 4
- Total: 21

Split decision:
- Review for split. Cohesion reason: circle and ellipse exact-conic helpers
  share one rational construction pipeline and depend on the same NURBS
  evaluator; unsupported conic kinds are explicit diagnostics, not hidden
  approximation work.

### Candidate Spec: NURBS Runtime Persistence And Tessellation Completion

Discovery purpose:
- Complete NURBS patch boundary extraction, tessellation, `.impress`, seams,
  CSG matrix participation, diagnostics, and evidence using shared spline and
  rational helpers.

Responsibilities:
- Functions/methods:
  - NURBS boundary extractor
  - NURBS tessellation adapter
  - NURBS payload codec hook
- Data structures/models:
  - NURBS patch payload
  - NURBS promotion evidence record
- Dependencies/services:
  - shared spline basis utilities
  - rational evaluation helper
  - `.impress` codec
- Returns/outputs/signals:
  - implemented NURBS readiness
  - malformed payload diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `NURBSSurfacePatch`
  - Additions to existing reusable library/module: NURBS family adapters
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - rational tessellation bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - NURBS uses shared spline and rational evaluation helpers
- Test strategy:
  - boundary, tessellation, `.impress`, seam, CSG matrix, and diagnostics tests
- Data ownership:
  - NURBS patch owns rational surface payload
- Routes:
  - rational authoring/import to NURBS patch to store/codec/tessellation/CSG
- Reuse/extraction decision:
  - consume shared spline and rational helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- conic authoring helpers can be separate producer specs.

Readiness blockers:
- shared spline basis utilities; rational evaluation helper

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 2 x 2 = 4
- Total: 24.5

Split decision:
- Review for split. Cohesion reason: this candidate is one NURBS family
  integration layer after shared math exists.

### Split Parent: Sweep Implemented Family Completion

Discovery purpose:
- Complete sweep surfaces as authored profile-along-path truth with frame
  transport, persistence, seams, tessellation, producer integration, CSG policy,
  diagnostics, and evidence.

Responsibilities:
- Functions/methods:
  - sweep evaluator
  - frame transport evaluator
  - sweep boundary extractor
  - sweep tessellation adapter
- Data structures/models:
  - sweep patch payload
  - frame policy record
  - sweep degeneracy diagnostic
- Dependencies/services:
  - path evaluation
  - profile evaluation
  - `.impress` codec
- Returns/outputs/signals:
  - implemented family readiness
  - path/profile diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `SweepSurfacePatch`
  - Additions to existing reusable library/module: frame transport helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - path/profile sampling bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`, loft/path producers
- Chosen defaults / parameters:
  - rotation-minimizing frame first; explicit twist/scale metadata
- Test strategy:
  - evaluator, frame continuity, tessellation, `.impress`, seam, producer, and
    diagnostic tests
- Data ownership:
  - sweep patch owns profile/path relationship and frame policy
- Routes:
  - path/profile authoring to sweep patch to store to consumers
- Reuse/extraction decision:
  - share path/frame helpers with loft and future path extrusion
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Self-intersection detection may be diagnostic-only in this release.

Readiness blockers:
- frame transport policy must be finalized

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 3 x 1 = 3
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
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Split completed in Loop 2. Frame transport is reusable path infrastructure
  and must not be buried inside sweep family completion.

### Candidate Spec: Shared Path Frame Transport Policy

Discovery purpose:
- Define deterministic path frame transport, twist, scale, and orientation
  helpers reusable by sweep, loft rails, and future path extrusion.

Responsibilities:
- Functions/methods:
  - path frame evaluator
  - twist/scale interpolator
  - frame degeneracy diagnostic builder
- Data structures/models:
  - frame transport policy record
  - path frame sample record
- Dependencies/services:
  - path evaluation
  - profile placement
- Returns/outputs/signals:
  - deterministic frame samples
  - degeneracy diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current path records
  - Additions to existing reusable library/module: frame transport helper
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
  - bounded path sampling
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py` or a shared path helper module
- Chosen defaults / parameters:
  - rotation-minimizing frame first
- Test strategy:
  - straight path, curved path, inflection, twist, and degeneracy tests
- Data ownership:
  - frame helper owns frame samples; sweep patch owns authored sweep payload
- Routes:
  - path/profile producers to frame helper to sweep evaluation
- Reuse/extraction decision:
  - shared helper used by sweep and future path extrusion
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Frenet frame can be future scope if rotation-minimizing frame is sufficient.

Readiness blockers:
- none

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Split decision:
- Review for split. Cohesion reason: bounded shared frame transport helper.

### Candidate Spec: Sweep Runtime Persistence And Tessellation Completion

Discovery purpose:
- Complete sweep patch evaluation, boundary extraction, `.impress`,
  tessellation, seams, producer integration, CSG matrix policy, diagnostics, and
  evidence using shared frame transport.

Responsibilities:
- Functions/methods:
  - sweep evaluator
  - sweep boundary extractor
  - sweep tessellation adapter
- Data structures/models:
  - sweep patch payload
  - sweep degeneracy diagnostic
- Dependencies/services:
  - shared path frame transport
  - profile evaluation
  - `.impress` codec
- Returns/outputs/signals:
  - implemented sweep readiness
  - path/profile diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `SweepSurfacePatch`
  - Additions to existing reusable library/module: sweep family adapters
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - path/profile tessellation bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - profile transported by shared rotation-minimizing frame
- Test strategy:
  - evaluator, boundary, tessellation, `.impress`, seam, producer, diagnostics
- Data ownership:
  - sweep patch owns profile/path relationship and authored sweep truth
- Routes:
  - path/profile authoring to sweep patch to store/codec/tessellation/CSG
- Reuse/extraction decision:
  - consume shared frame transport helper
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- full self-intersection solving can remain diagnostic-only.

Readiness blockers:
- shared path frame transport policy

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 21.5

Split decision:
- Review for split. Cohesion reason: sweep family completion is one adapter
  layer after shared frame transport exists.

### Split Parent: Subdivision Implemented Family Completion

Discovery purpose:
- Complete subdivision surfaces as native cage-and-crease authored truth with
  deterministic evaluation, `.impress`, seams, tessellation, CSG policy,
  diagnostics, and evidence.

Responsibilities:
- Functions/methods:
  - subdivision evaluator
  - boundary chain extractor
  - subdivision tessellation adapter
- Data structures/models:
  - subdivision payload
  - crease/sharpness record
  - approximation diagnostic
- Dependencies/services:
  - subdivision scheme evaluator
  - `.impress` codec
  - tessellation request policy
- Returns/outputs/signals:
  - implemented family readiness
  - approximation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `SubdivisionSurfacePatch`
  - Additions to existing reusable library/module: Catmull-Clark refinement helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - deterministic subdivision budget
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - Catmull-Clark finite-level deterministic evaluation first
- Test strategy:
  - cage validation, crease validation, tessellation, `.impress`, seam
    approximation, CSG matrix, and reference tests
- Data ownership:
  - subdivision patch owns cage and crease truth
- Routes:
  - cage authoring/import to subdivision patch to store to consumers
- Reuse/extraction decision:
  - keep refinement helper reusable for tessellation and evaluation
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Limit evaluation can be later if finite-level approximation is explicit.

Readiness blockers:
- deterministic subdivision scheme selection

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Split completed in Loop 4. Subdivision scheme/cage evaluation is reusable
  infrastructure and should be separate from persistence, seams, tessellation,
  and evidence.

### Candidate Spec: Subdivision Scheme And Cage Evaluation

Discovery purpose:
- Implement deterministic subdivision scheme evaluation for cage-and-crease
  payloads, with validation and approximation diagnostics.

Responsibilities:
- Functions/methods:
  - cage validator
  - crease validator
  - subdivision evaluator
- Data structures/models:
  - subdivision scheme record
  - crease/sharpness record
  - approximation diagnostic
- Dependencies/services:
  - subdivision patch payload
  - finite numeric validation
- Returns/outputs/signals:
  - refined cage/evaluation samples
  - cage/crease diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `SubdivisionSurfacePatch`
  - Additions to existing reusable library/module: Catmull-Clark refinement helper
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
  - deterministic subdivision budget
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - finite-level Catmull-Clark first
- Test strategy:
  - cage validation, crease validation, finite refinement, approximation diagnostics
- Data ownership:
  - subdivision patch owns cage and crease truth
- Routes:
  - subdivision patch payload to scheme evaluator
- Reuse/extraction decision:
  - reusable subdivision helper feeds tessellation and boundary extraction
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- true limit evaluation is future scope unless explicitly required.

Readiness blockers:
- none

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only scheme/cage
  evaluation infrastructure.

### Candidate Spec: Subdivision Runtime Persistence Tessellation And Evidence

Discovery purpose:
- Complete subdivision boundary extraction, `.impress`, seam approximation,
  tessellation, CSG matrix policy, diagnostics, and reference evidence using
  the subdivision scheme evaluator.

Responsibilities:
- Functions/methods:
  - boundary chain extractor
  - subdivision tessellation adapter
  - subdivision payload codec hook
- Data structures/models:
  - boundary approximation record
  - subdivision promotion evidence record
- Dependencies/services:
  - subdivision scheme evaluator
  - `.impress` codec
  - tessellation request policy
- Returns/outputs/signals:
  - implemented subdivision readiness
  - approximation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `SubdivisionSurfacePatch`
  - Additions to existing reusable library/module: subdivision family adapters
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - deterministic tessellation budget
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - approximate seams must carry subdivision-level metadata
- Test strategy:
  - boundary extraction, tessellation, `.impress`, seam approximation, CSG matrix,
    and reference tests
- Data ownership:
  - subdivision patch owns cage truth; tessellation owns mesh output only
- Routes:
  - cage authoring/import to subdivision patch to store/codec/tessellation/CSG
- Reuse/extraction decision:
  - consume subdivision scheme evaluator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- none

Readiness blockers:
- subdivision scheme and cage evaluation

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: this candidate is one subdivision family
  integration layer after scheme evaluation exists.

### Split Parent: Implicit Implemented Family Completion

Discovery purpose:
- Complete implicit surfaces as declarative, safe, bounded field-defined
  authored truth with evaluation, extraction, persistence, CSG policy,
  diagnostics, and evidence.

Responsibilities:
- Functions/methods:
  - field evaluator
  - gradient evaluator
  - bounded extraction adapter
  - safety validator
- Data structures/models:
  - implicit field payload
  - safety policy record
  - residual diagnostic
- Dependencies/services:
  - implicit field node validator
  - `.impress` codec
  - tessellation/extraction adapter
- Returns/outputs/signals:
  - implemented family readiness
  - unsafe/budget diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `ImplicitSurfacePatch`
  - Additions to existing reusable library/module: field safety and extraction helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - declarative field allow-list and no executable payloads
- Performance-sensitive behavior:
  - extraction budget and residual bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`, `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - allow-listed field nodes only; bounded extraction required
- Test strategy:
  - safe field round trips, unsafe field refusals, extraction, residual, CSG
    policy, and diagnostics tests
- Data ownership:
  - implicit patch owns declarative field graph and safety policy
- Routes:
  - field authoring to implicit patch to bounded extraction or refusal
- Reuse/extraction decision:
  - extend existing implicit validation and intersection safety helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Some CSG operations may remain explicit adapter or non-CSG classifications.

Readiness blockers:
- safety/budget policies must be stable

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 3 x 1 = 3
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
- Readiness blockers: 1 x 2 = 2
- Total: 27.5

Split decision:
- Split required and completed in Loop 1. Minimum split:
  - implicit field safety and `.impress` payload gate
  - implicit evaluation/extraction adapter
  - implicit CSG/intersection policy and diagnostics

### Split Parent: Heightmap Implemented Family Completion

Discovery purpose:
- Complete heightmaps as native sampled heightfield surfaces with evaluation,
  seams, `.impress`, tessellation, CSG policy, diagnostics, and evidence.

Responsibilities:
- Functions/methods:
  - height evaluator
  - normal/derivative estimator
  - mask-aware tessellation adapter
- Data structures/models:
  - heightmap payload
  - mask/no-data policy
  - grid validation diagnostic
- Dependencies/services:
  - heightmap modeling helpers
  - `.impress` codec
  - tessellation adapter
- Returns/outputs/signals:
  - implemented family readiness
  - sampled-surface diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `HeightmapSurfacePatch`
  - Additions to existing reusable library/module: heightmap `.impress` codec and seam helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - external data references must be bounded and non-executable
- Performance-sensitive behavior:
  - sample-grid size limits
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py`, `src/impression/modeling/surface.py`,
    `src/impression/modeling/tessellation.py`, `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - embedded finite sampled grid first; external references require explicit policy
- Test strategy:
  - evaluation, mask, tessellation, `.impress`, seam approximation, CSG policy,
    and diagnostics tests
- Data ownership:
  - heightmap patch owns sampled grid and mask policy
- Routes:
  - heightfield authoring to heightmap patch to store/tessellate/persist
- Reuse/extraction decision:
  - extend existing heightmap helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- External data storage should not be introduced unless needed by this release.

Readiness blockers:
- heightmap `.impress` payload policy

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Readiness blockers: 1 x 2 = 2
- Total: 25.5

Split decision:
- Split required and completed in Loop 1. Minimum split:
  - heightmap `.impress` payload and validation
  - heightmap seam/tessellation/evidence completion
  - heightmap CSG policy and diagnostics

### Split Parent: Displacement Implemented Family Completion

Discovery purpose:
- Complete displacement surfaces as source-surface plus displacement-field
  authored truth with evaluation, persistence, seams, tessellation, CSG policy,
  diagnostics, and evidence.

Responsibilities:
- Functions/methods:
  - displaced evaluator
  - displacement derivative estimator
  - displacement tessellation adapter
- Data structures/models:
  - displacement payload
  - source patch reference record
  - domain mismatch diagnostic
- Dependencies/services:
  - source patch evaluator
  - `.impress` codec
  - tessellation adapter
- Returns/outputs/signals:
  - implemented family readiness
  - source/displacement diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `DisplacementSurfacePatch`
  - Additions to existing reusable library/module: source patch embedding/reference codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - capability matrix update after evidence
- Security/privacy-sensitive behavior:
  - external displacement references must be bounded and non-executable
- Performance-sensitive behavior:
  - sampling and source evaluation bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - embedded source patch payload first; references only if identity policy is complete
- Test strategy:
  - source evaluation, displacement evaluation, `.impress`, seam, tessellation,
    CSG policy, and diagnostics tests
- Data ownership:
  - displacement patch owns displacement data and source-surface relationship
- Routes:
  - source patch plus displacement to displacement patch to consumers
- Reuse/extraction decision:
  - reuse source patch codec recursively rather than serializing mesh output
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Source patch identity policy must choose embedded payload versus store reference.

Readiness blockers:
- displacement `.impress` source-patch payload policy

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
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
- Readiness blockers: 1 x 2 = 2
- Total: 25.5

Split decision:
- Split required and completed in Loop 1. Minimum split:
  - displacement source payload and `.impress` identity policy
  - displacement evaluation/tessellation/seam completion
  - displacement CSG policy and diagnostics

### Candidate Spec: Implicit Field Safety And Impress Payload Gate

Discovery purpose:
- Make implicit payloads safe, declarative, and persistable without executable
  callbacks or unbounded field definitions.

Responsibilities:
- Functions/methods:
  - implicit field validator
  - implicit payload encoder
  - implicit payload decoder
- Data structures/models:
  - field safety policy record
  - unsafe payload diagnostic
- Dependencies/services:
  - implicit field node allow-list
  - `.impress` patch payload dispatch
- Returns/outputs/signals:
  - safe payload round-trip
  - unsafe payload refusal
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current implicit field node records
  - Additions to existing reusable library/module: implicit `.impress` safety gate
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - file-format fixture writes in tests
- Security/privacy-sensitive behavior:
  - executable field payload refusal
- Performance-sensitive behavior:
  - bounded payload size
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - allow-listed declarative field nodes only
- Test strategy:
  - safe round-trip, malformed payload, callback refusal, and determinism tests
- Data ownership:
  - implicit patch owns field graph; `.impress` owns serialized form
- Routes:
  - implicit patch to `.impress` writer to reader to implicit patch
- Reuse/extraction decision:
  - extend existing implicit payload validation
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- External field references are explicitly refused by this candidate. Embedded,
  allow-listed declarative field graphs are the canonical implemented payload.

Readiness blockers:
- none

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only safe implicit
  persistence, not extraction or CSG.

### Candidate Spec: Implicit Evaluation And Extraction Adapter

Discovery purpose:
- Implement bounded scalar evaluation, gradient estimation, and explicit
  extraction/tessellation for safe implicit patches.

Responsibilities:
- Functions/methods:
  - field evaluator
  - gradient evaluator
  - bounded extraction adapter
- Data structures/models:
  - extraction budget record
  - residual classification record
- Dependencies/services:
  - implicit field safety policy
  - tessellation request policy
- Returns/outputs/signals:
  - extracted mesh at tessellation boundary
  - budget/residual diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current implicit patch record
  - Additions to existing reusable library/module: bounded extraction helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - evaluates only allow-listed field nodes
- Performance-sensitive behavior:
  - extraction budget and sample count bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - bounded extraction required before tessellation
- Test strategy:
  - scalar evaluation, gradient, extraction, budget refusal, and residual tests
- Data ownership:
  - implicit evaluator owns sampled field behavior
- Routes:
  - implicit patch to evaluator to tessellation adapter
- Reuse/extraction decision:
  - extraction helper should also feed implicit intersection adapters
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- extraction algorithm can be changed later if output contract remains stable.

Readiness blockers:
- implicit field safety gate

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:
- Review for split. Cohesion reason: evaluation and extraction are one
  tessellation-boundary adapter path.

### Candidate Spec: Implicit CSG And Intersection Policy Diagnostics

Discovery purpose:
- Define exact, adapter, unsupported, or non-CSG behavior for implicit
  surfaces in CSG and intersection consumers.

Responsibilities:
- Functions/methods:
  - implicit CSG support classifier
  - implicit intersection refusal builder
- Data structures/models:
  - implicit CSG policy record
  - implicit unsupported diagnostic
- Dependencies/services:
  - CSG support matrix
  - implicit extraction adapter
  - intersection registry
- Returns/outputs/signals:
  - executable adapter plan or refusal
  - no-hidden-fallback diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG/intersection support records
  - Additions to existing reusable library/module: implicit matrix entries
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - respects implicit safety policy
- Performance-sensitive behavior:
  - solver/extraction budget classification
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`, `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - unsupported or adapter classifications are valid only when explicit
- Test strategy:
  - family-pair matrix, refusal diagnostics, and no-hidden-mesh-fallback tests
- Data ownership:
  - CSG owns operation support truth; implicit owns safety limits
- Routes:
  - implicit family pair to CSG/intersection support matrix
- Reuse/extraction decision:
  - extend existing support matrix records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- native exact implicit booleans are not required for implemented-family status.

Readiness blockers:
- implicit extraction adapter

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 19.5

Split decision:
- Review for split. Cohesion reason: policy and diagnostics are one consumer
  matrix layer; exact solvers must split separately.

### Candidate Spec: Heightmap Impress Payload And Grid Validation

Discovery purpose:
- Add native heightmap `.impress` payloads, sampled-grid validation, malformed
  payload refusals, and whole-store round-trip evidence.

Responsibilities:
- Functions/methods:
  - heightmap payload encoder
  - heightmap payload decoder
  - grid validator
- Data structures/models:
  - heightmap payload version record
  - grid validation diagnostic
- Dependencies/services:
  - heightmap patch record
  - `.impress` patch dispatch
- Returns/outputs/signals:
  - round-trip heightmap patch
  - malformed grid diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current heightmap patch and `.impress` dispatch
  - Additions to existing reusable library/module: heightmap codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - file-format fixture writes in tests
- Security/privacy-sensitive behavior:
  - external data references explicitly refused by default
- Performance-sensitive behavior:
  - bounded fixture/sample sizes
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py`, `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - embedded finite sampled grid first
- Test strategy:
  - encode/decode, malformed grid, mask, identity, and whole-store tests
- Data ownership:
  - heightmap patch owns sampled grid; `.impress` owns serialized payload
- Routes:
  - heightmap patch to `.impress` writer to reader
- Reuse/extraction decision:
  - extend existing patch payload codec
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- External data references are explicitly refused by this candidate. Embedded
  finite sampled grids are the canonical implemented payload.

Readiness blockers:
- none

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 21.5

Split decision:
- Review for split. Cohesion reason: payload validation and codec are one
  persistence boundary.

### Split Parent: Heightmap Evaluation Tessellation Seam Evidence

Discovery purpose:
- Complete sampled height evaluation, derivative/normal estimation, seam
  approximation, tessellation, and reference evidence.

Responsibilities:
- Functions/methods:
  - height evaluator
  - normal estimator
  - mask-aware tessellation adapter
  - sampled boundary extractor
- Data structures/models:
  - sampled-boundary approximation record
  - heightmap evidence record
- Dependencies/services:
  - heightmap payload validator
  - tessellation request policy
  - seam validator
- Returns/outputs/signals:
  - tessellation result
  - seam approximation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current heightmap modeling helpers
  - Additions to existing reusable library/module: sampled boundary helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference artifact fixture writes
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - sample-grid size and tessellation bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py`, `src/impression/modeling/tessellation.py`,
    `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - bilinear evaluation and explicit approximation metadata
- Test strategy:
  - evaluation, normals, mask tessellation, seam approximation, and reference tests
- Data ownership:
  - heightmap patch owns sampled surface truth
- Routes:
  - heightmap patch to evaluator/seam/tessellation/reference evidence
- Reuse/extraction decision:
  - sampled boundary helper may be reused by displacement
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- higher-order continuity can be approximate unless exact sample derivatives are
  explicitly added.

Readiness blockers:
- heightmap payload validation

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
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
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Split completed in Loop 4. Height evaluation/tessellation and seam/reference
  evidence are distinct verification boundaries.

### Candidate Spec: Heightmap Evaluation And Tessellation Adapter

Discovery purpose:
- Complete heightmap point evaluation, derivative/normal estimation, mask-aware
  tessellation, and bounded tessellation diagnostics.

Responsibilities:
- Functions/methods:
  - height evaluator
  - normal estimator
  - mask-aware tessellation adapter
- Data structures/models:
  - heightmap evaluation diagnostic
  - mask tessellation record
- Dependencies/services:
  - heightmap payload validator
  - tessellation request policy
- Returns/outputs/signals:
  - tessellation result
  - evaluation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current heightmap modeling helpers
  - Additions to existing reusable library/module: heightmap tessellation adapter
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
  - sample-grid size and tessellation bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/heightmap.py`, `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - bilinear evaluation and bounded sampled-grid tessellation
- Test strategy:
  - point evaluation, normals, masks, tessellation determinism, and bounds tests
- Data ownership:
  - heightmap patch owns sampled grid; tessellation owns mesh output
- Routes:
  - heightmap patch to evaluator to tessellation adapter
- Reuse/extraction decision:
  - keep sampled-grid evaluator reusable by seam evidence
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- higher-order derivatives can remain approximate metadata.

Readiness blockers:
- heightmap payload validation

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: this candidate owns heightmap evaluation
  and explicit tessellation only.

### Candidate Spec: Heightmap Seam Approximation And Reference Evidence

Discovery purpose:
- Complete heightmap sampled-boundary extraction, seam approximation metadata,
  negative seam diagnostics, and promoted reference evidence.

Responsibilities:
- Functions/methods:
  - sampled boundary extractor
  - seam approximation diagnostic builder
  - reference evidence gate
- Data structures/models:
  - sampled-boundary approximation record
  - heightmap evidence record
- Dependencies/services:
  - heightmap evaluator
  - seam validator
  - reference artifact promotion gate
- Returns/outputs/signals:
  - seam diagnostics
  - promoted reference evidence
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current seam and reference records
  - Additions to existing reusable library/module: sampled boundary helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference artifact fixture writes
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded seam sampling
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `tests/reference_images.py`
- Chosen defaults / parameters:
  - sampled boundaries carry approximation metadata
- Test strategy:
  - seam approximation, unsupported seam diagnostics, and reference evidence tests
- Data ownership:
  - seam validator owns seam diagnostics; reference gate owns evidence state
- Routes:
  - heightmap boundary to seam validator to reference evidence
- Reuse/extraction decision:
  - sampled boundary helper may be reused by displacement
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- none

Readiness blockers:
- heightmap evaluation and tessellation adapter

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: seam approximation and reference evidence
  are one verification boundary after heightmap evaluation exists.

### Candidate Spec: Heightmap CSG Policy Diagnostics

Discovery purpose:
- Classify heightmap CSG behavior as exact, adapter, unsupported, or non-CSG
  by operation/family pair with durable diagnostics.

Responsibilities:
- Functions/methods:
  - heightmap CSG classifier
  - non-executable plan diagnostic builder
- Data structures/models:
  - heightmap CSG policy record
  - unsupported operation diagnostic
- Dependencies/services:
  - CSG support matrix
  - heightmap evaluator
- Returns/outputs/signals:
  - CSG support matrix rows
  - refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG support records
  - Additions to existing reusable library/module: heightmap matrix entries
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
  - operation budget classification
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported/non-CSG is acceptable when exact and tested
- Test strategy:
  - matrix and no-hidden-mesh-fallback tests
- Data ownership:
  - CSG owns support truth
- Routes:
  - heightmap family pair to operation plan or refusal
- Reuse/extraction decision:
  - extend current CSG matrix
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- native heightfield boolean solving is not required for implemented status.

Readiness blockers:
- heightmap evaluator

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
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
- Readiness blockers: 1 x 2 = 2
- Total: 16.5

Split decision:
- Review for split. Cohesion reason: this is a bounded matrix/diagnostic leaf.

### Candidate Spec: Displacement Source Payload And Impress Identity Policy

Discovery purpose:
- Define displacement source-surface persistence and identity rules without
  serializing derived mesh output.

Responsibilities:
- Functions/methods:
  - displacement payload encoder
  - displacement payload decoder
  - source patch identity validator
- Data structures/models:
  - source patch reference record
  - displacement payload version record
  - identity diagnostic
- Dependencies/services:
  - source patch codec dispatch
  - SurfaceBodyStore identity policy
- Returns/outputs/signals:
  - round-trip displacement patch
  - identity/refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current patch payload dispatch
  - Additions to existing reusable library/module: recursive source patch codec
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - file-format fixture writes in tests
- Security/privacy-sensitive behavior:
  - external references explicitly refused unless identity policy supports them
- Performance-sensitive behavior:
  - bounded embedded source payload
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - embedded source patch payload first
- Test strategy:
  - embedded source round trip, missing source refusal, identity preservation
- Data ownership:
  - displacement patch owns source relationship; `.impress` owns payload identity
- Routes:
  - displacement patch to `.impress` writer to reader
- Reuse/extraction decision:
  - reuse patch codec recursively
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Cross-body source references are explicitly refused by this candidate.
  Embedded source payloads and stable in-body source identities are the
  canonical implemented payload.

Readiness blockers:
- none

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: source identity and serialization are one
  persistence boundary.

### Split Parent: Displacement Evaluation Tessellation Seam Evidence

Discovery purpose:
- Complete displaced-surface evaluation, derivative approximation, seam
  comparison, tessellation, and reference evidence.

Responsibilities:
- Functions/methods:
  - displaced point evaluator
  - displacement derivative estimator
  - displaced boundary extractor
  - displacement tessellation adapter
- Data structures/models:
  - domain mapping record
  - displacement approximation diagnostic
- Dependencies/services:
  - source patch evaluator
  - tessellation request policy
  - seam validator
- Returns/outputs/signals:
  - tessellation result
  - seam/domain diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current displacement patch record
  - Additions to existing reusable library/module: displaced boundary helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference artifact fixture writes
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - source evaluation and sampling bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - source patch domain is authoritative
- Test strategy:
  - point evaluation, normals, seam approximation, tessellation, and reference tests
- Data ownership:
  - displacement patch owns displaced authored truth
- Routes:
  - displacement patch to evaluator/seam/tessellation/reference evidence
- Reuse/extraction decision:
  - reuse source patch evaluator and sampled boundary helper
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- derivative exactness depends on source family and displacement function.

Readiness blockers:
- source payload identity policy

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 2 x 1 = 2
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
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- Split completed in Loop 4. Displaced evaluation/tessellation and
  seam/reference evidence are distinct verification boundaries.

### Candidate Spec: Displacement Evaluation And Tessellation Adapter

Discovery purpose:
- Complete displaced point evaluation, derivative approximation, and bounded
  tessellation against the source patch domain.

Responsibilities:
- Functions/methods:
  - displaced point evaluator
  - displacement derivative estimator
  - displacement tessellation adapter
- Data structures/models:
  - domain mapping record
  - displacement evaluation diagnostic
- Dependencies/services:
  - source patch evaluator
  - tessellation request policy
- Returns/outputs/signals:
  - tessellation result
  - domain/evaluation diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current displacement patch record
  - Additions to existing reusable library/module: displacement evaluator
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
  - source evaluation and sampling bounds
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `src/impression/modeling/tessellation.py`
- Chosen defaults / parameters:
  - source patch domain is authoritative
- Test strategy:
  - point evaluation, normals, domain mismatch, tessellation determinism, bounds
- Data ownership:
  - displacement patch owns displaced authored truth
- Routes:
  - displacement patch to source evaluator to tessellation adapter
- Reuse/extraction decision:
  - reuse source patch evaluator
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- derivative exactness depends on source family and displacement function.

Readiness blockers:
- source payload identity policy

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: this candidate owns displacement
  evaluation and explicit tessellation only.

### Candidate Spec: Displacement Seam Approximation And Reference Evidence

Discovery purpose:
- Complete displaced-boundary extraction, seam approximation metadata, negative
  seam diagnostics, and promoted reference evidence.

Responsibilities:
- Functions/methods:
  - displaced boundary extractor
  - seam approximation diagnostic builder
  - reference evidence gate
- Data structures/models:
  - displaced-boundary approximation record
  - displacement evidence record
- Dependencies/services:
  - displacement evaluator
  - seam validator
  - reference artifact promotion gate
- Returns/outputs/signals:
  - seam diagnostics
  - promoted reference evidence
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current seam and reference records
  - Additions to existing reusable library/module: displaced boundary helper
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reference artifact fixture writes
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded seam sampling
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`, `tests/reference_images.py`
- Chosen defaults / parameters:
  - displaced boundaries carry approximation metadata when exact comparison is unavailable
- Test strategy:
  - seam approximation, unsupported seam diagnostics, and reference evidence tests
- Data ownership:
  - seam validator owns seam diagnostics; reference gate owns evidence state
- Routes:
  - displacement boundary to seam validator to reference evidence
- Reuse/extraction decision:
  - reuse sampled/displaced boundary helper patterns
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- none

Readiness blockers:
- displacement evaluation and tessellation adapter

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 22.5

Split decision:
- Review for split. Cohesion reason: seam approximation and reference evidence
  are one verification boundary after displacement evaluation exists.

### Candidate Spec: Displacement CSG Policy Diagnostics

Discovery purpose:
- Classify displacement CSG behavior as exact, adapter, unsupported, or non-CSG
  by operation/family pair with durable diagnostics.

Responsibilities:
- Functions/methods:
  - displacement CSG classifier
  - non-executable plan diagnostic builder
- Data structures/models:
  - displacement CSG policy record
  - unsupported operation diagnostic
- Dependencies/services:
  - CSG support matrix
  - displacement evaluator
- Returns/outputs/signals:
  - CSG support matrix rows
  - refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG support records
  - Additions to existing reusable library/module: displacement matrix entries
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
  - operation budget classification
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported/non-CSG is acceptable when exact and tested
- Test strategy:
  - matrix and no-hidden-mesh-fallback tests
- Data ownership:
  - CSG owns support truth
- Routes:
  - displacement family pair to operation plan or refusal
- Reuse/extraction decision:
  - extend current CSG matrix
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- native displaced-surface boolean solving is not required for implemented status.

Readiness blockers:
- displacement evaluator

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
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
- Readiness blockers: 1 x 2 = 2
- Total: 16.5

Split decision:
- Review for split. Cohesion reason: this is a bounded matrix/diagnostic leaf.

### Candidate Spec: Advanced Family Seam And Boundary Participation Matrix

Discovery purpose:
- Make seam participation explicit for the seven advanced families across
  boundary extraction, C0/G0 comparison, higher-order residuals, approximation
  metadata, and refusal diagnostics.

Responsibilities:
- Functions/methods:
  - boundary descriptor extractor
  - derivative summary adapter
  - seam diagnostic builder
- Data structures/models:
  - family boundary support record
  - approximation metadata record
- Dependencies/services:
  - higher-order seam continuity architecture
  - family evaluators
- Returns/outputs/signals:
  - seam support matrix
  - unsupported seam diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current seam and derivative records
  - Additions to existing reusable library/module: family boundary adapters
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
  - bounded seam sampling
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - exact boundary first; sampled boundaries carry approximation metadata
- Test strategy:
  - per-family seam matrix and negative diagnostics
- Data ownership:
  - seam system owns support matrix; families own boundary extraction
- Routes:
  - patch boundary to seam validator to continuity diagnostics
- Reuse/extraction decision:
  - extend current seam derivative machinery
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Implicit and sampled families may only support approximate seams initially.

Readiness blockers:
- family evaluators must exist

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 17.5

Split decision:
- Review for split. Cohesion is acceptable because this is one matrix and one
  seam adapter layer, dependent on family evaluators.

### Split Parent: Advanced Family CSG Support Matrix And Refusal Policy

Discovery purpose:
- Define exact, declared-tolerance, adapter, unsupported, or non-CSG posture for
  every operation/family pair involving the seven advanced families.

Responsibilities:
- Functions/methods:
  - CSG support matrix builder
  - operation plan refusal builder
  - family-pair solver selector
- Data structures/models:
  - advanced CSG family pair support record
  - non-CSG classification record
- Dependencies/services:
  - CSG solver registry
  - intersection kernel
  - family evaluators
- Returns/outputs/signals:
  - support matrix
  - non-executable plan diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG family support records
  - Additions to existing reusable library/module: advanced family matrix rows
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - implicit/field operations must obey safety policy
- Performance-sensitive behavior:
  - solver budget classification
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`, `src/impression/modeling/surface_intersections.py`
- Chosen defaults / parameters:
  - unsupported is acceptable only when explicit and tested
- Test strategy:
  - operation/family matrix tests and no-hidden-mesh-fallback tests
- Data ownership:
  - CSG owns operation support truth
- Routes:
  - family pair to solver registry to executable plan or refusal
- Reuse/extraction decision:
  - extend current CSG support records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Implicit, heightmap, and displacement may be non-CSG for some operations.

Readiness blockers:
- family evaluators and intersection adapters

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- Split completed in Loop 3. Support classification and refusal/no-fallback
  diagnostics are separate enough to deserve independent leaves.

### Candidate Spec: Advanced Family CSG Support Classification Matrix

Discovery purpose:
- Define exact, declared-tolerance, adapter, unsupported, or non-CSG posture for
  every operation/family pair involving advanced patch families.

Responsibilities:
- Functions/methods:
  - CSG support matrix builder
  - family-pair support classifier
- Data structures/models:
  - advanced CSG family pair support record
  - non-CSG classification record
- Dependencies/services:
  - CSG solver registry
  - family evaluators
- Returns/outputs/signals:
  - support matrix
  - support-state diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG family support records
  - Additions to existing reusable library/module: advanced family matrix rows
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - implicit/field operations must carry safety classifications
- Performance-sensitive behavior:
  - solver budget classification
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported/non-CSG classifications are valid only when explicit and tested
- Test strategy:
  - operation/family matrix tests
- Data ownership:
  - CSG owns operation support truth
- Routes:
  - family pair to support classification matrix
- Reuse/extraction decision:
  - extend current CSG support records
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- exact solver leaves remain separate from support classification.

Readiness blockers:
- family evaluators and intersection adapters

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only classification
  matrix truth.

### Candidate Spec: Advanced Family CSG Refusal And No-Fallback Diagnostics

Discovery purpose:
- Convert unsupported advanced-family CSG classifications into deterministic
  non-executable operation plans and no-hidden-mesh-fallback diagnostics.

Responsibilities:
- Functions/methods:
  - operation plan refusal builder
  - no-fallback assertion helper
- Data structures/models:
  - non-executable CSG plan diagnostic
  - fallback violation diagnostic
- Dependencies/services:
  - advanced CSG support matrix
  - no-hidden-mesh-fallback tests
- Returns/outputs/signals:
  - non-executable plan diagnostics
  - fallback regression failures
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG diagnostic records
  - Additions to existing reusable library/module: advanced-family refusal rows
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - implicit operations preserve safety refusal reason
- Performance-sensitive behavior:
  - none
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`, `tests/test_no_hidden_mesh_fallback.py`
- Chosen defaults / parameters:
  - every unsupported advanced-family pair refuses before solver execution
- Test strategy:
  - refusal diagnostics and no-hidden-mesh-fallback tests
- Data ownership:
  - CSG operation planner owns refusal diagnostics
- Routes:
  - support matrix to operation planner to refusal diagnostic
- Reuse/extraction decision:
  - extend current operation planning diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- none

Readiness blockers:
- advanced CSG support classification matrix

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 0 x 2 = 0
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 16.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only refusal and
  fallback diagnostics.

### Split Parent: Advanced Family `.impress` Whole-Store Coverage

Discovery purpose:
- Ensure B-spline, NURBS, sweep, subdivision, implicit, heightmap, and
  displacement have native `.impress` payloads and whole-store round-trip
  evidence.

Responsibilities:
- Functions/methods:
  - family encoder coverage gate
  - family decoder coverage gate
  - whole-store round-trip assertion
- Data structures/models:
  - family payload version record
  - malformed payload diagnostic
- Dependencies/services:
  - `.impress` document root
  - patch payload dispatch
  - SurfaceBodyStore
- Returns/outputs/signals:
  - round-trip report
  - refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `.impress` codec dispatch
  - Additions to existing reusable library/module: missing heightmap/displacement codec coverage
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - file-format fixture writes in tests
- Security/privacy-sensitive behavior:
  - unsafe implicit and external sampled-data payload refusal
- Performance-sensitive behavior:
  - bounded fixture sizes
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - native payloads only; no mesh-wrapper serialization
- Test strategy:
  - per-family encode/decode, malformed payload, unsafe payload, and whole-store
    round-trip tests
- Data ownership:
  - `.impress` owns serialized family payload truth
- Routes:
  - SurfaceBodyStore to writer to reader to validated SurfaceBodyStore
- Reuse/extraction decision:
  - extend existing patch payload dispatch
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Displacement source patch can be embedded first; cross-body references can be
  later.

Readiness blockers:
- family runtime payload validation

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
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 24.5

Split decision:
- Split completed in Loop 3. Codec coverage and whole-store evidence/negative
  fixtures are separate verification boundaries.

### Candidate Spec: Advanced Family Impress Codec Coverage Gate

Discovery purpose:
- Ensure every advanced family has native `.impress` encoder and decoder
  dispatch coverage.

Responsibilities:
- Functions/methods:
  - family encoder coverage gate
  - family decoder coverage gate
- Data structures/models:
  - family payload version record
  - codec coverage diagnostic
- Dependencies/services:
  - `.impress` document root
  - patch payload dispatch
- Returns/outputs/signals:
  - codec coverage report
  - missing codec diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `.impress` codec dispatch
  - Additions to existing reusable library/module: advanced family codec rows
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - unsafe implicit and external sampled-data payload refusal
- Performance-sensitive behavior:
  - bounded fixture sizes
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/io/impress.py`
- Chosen defaults / parameters:
  - native payloads only; no mesh-wrapper serialization
- Test strategy:
  - per-family encoder/decoder coverage tests
- Data ownership:
  - `.impress` owns serialized family payload truth
- Routes:
  - patch payload to codec dispatch to family decoder
- Reuse/extraction decision:
  - extend existing patch payload dispatch
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- none

Readiness blockers:
- family runtime payload validation

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 1 x 1 = 1
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 0 x 3 = 0
- Security/privacy-sensitive behavior: 1 x 3 = 3
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 1 x 2 = 2
- Total: 18.5

Split decision:
- Review for split. Cohesion reason: this candidate owns codec coverage only.

### Candidate Spec: Advanced Family Impress Whole-Store Round-Trip Evidence

Discovery purpose:
- Prove whole-store `.impress` round trips and negative payload fixtures for
  all advanced families.

Responsibilities:
- Functions/methods:
  - whole-store round-trip assertion
  - negative fixture runner
- Data structures/models:
  - whole-store evidence record
  - malformed payload diagnostic
- Dependencies/services:
  - SurfaceBodyStore
  - advanced family codec coverage
- Returns/outputs/signals:
  - round-trip report
  - refusal diagnostics
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current `.impress` fixture helpers
  - Additions to existing reusable library/module: advanced family whole-store fixtures
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - file-format fixture writes in tests
- Security/privacy-sensitive behavior:
  - unsafe implicit and sampled-data payload refusal fixtures
- Performance-sensitive behavior:
  - bounded fixture sizes
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `tests/test_impress_io.py`
- Chosen defaults / parameters:
  - whole-store fixtures include seams, trims, metadata, identities, and provenance where applicable
- Test strategy:
  - whole-store round-trip and negative diagnostic fixture tests
- Data ownership:
  - `.impress` tests own round-trip evidence; codecs own serialization truth
- Routes:
  - SurfaceBodyStore to writer to reader to validated SurfaceBodyStore
- Reuse/extraction decision:
  - extend existing `.impress` fixture helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- dirty reference artifacts do not count as completion evidence.

Readiness blockers:
- advanced family codec coverage gate

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
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
- Readiness blockers: 1 x 2 = 2
- Total: 20.5

Split decision:
- Review for split. Cohesion reason: this candidate owns whole-store evidence
  and negative fixtures after codec coverage exists.

## Manifest Review Cycles

### Loop 1

Critical review found mandatory splits:

- Implicit Implemented Family Completion: 27.5, split into three active
  candidates.
- Heightmap Implemented Family Completion: 25.5, split into three active
  candidates.
- Displacement Implemented Family Completion: 25.5, split into three active
  candidates.

Active scores after Loop 1:

- Advanced Family Promotion Gate And Capability Matrix Update: 17.5
- B-spline Implemented Family Completion: 24.5
- NURBS Implemented Family Completion: 22.5
- Sweep Implemented Family Completion: 24.5
- Subdivision Implemented Family Completion: 22.5
- Implicit Field Safety And Impress Payload Gate: 21.5
- Implicit Evaluation And Extraction Adapter: 20.5
- Implicit CSG And Intersection Policy Diagnostics: 19.5
- Heightmap Impress Payload And Grid Validation: 21.5
- Heightmap Evaluation Tessellation Seam Evidence: 24.5
- Heightmap CSG Policy Diagnostics: 16.5
- Displacement Source Payload And Impress Identity Policy: 22.5
- Displacement Evaluation Tessellation Seam Evidence: 23.5
- Displacement CSG Policy Diagnostics: 16.5
- Advanced Family Seam And Boundary Participation Matrix: 17.5
- Advanced Family CSG Support Matrix And Refusal Policy: 23.5
- Advanced Family `.impress` Whole-Store Coverage: 24.5

### Loop 2

Critical review found hidden reusable infrastructure below the mandatory split
threshold:

- B-spline Implemented Family Completion: 24.5, split because shared spline
  basis and knot infrastructure is reused by NURBS.
- Sweep Implemented Family Completion: 24.5, split because path frame transport
  is reusable beyond sweep.

Active scores after Loop 2:

- Advanced Family Promotion Gate And Capability Matrix Update: 17.5
- Shared Spline Basis And Knot Infrastructure: 17.5
- B-spline Runtime Persistence And Tessellation Completion: 21.5
- NURBS Implemented Family Completion: 22.5
- Shared Path Frame Transport Policy: 17.5
- Sweep Runtime Persistence And Tessellation Completion: 21.5
- Subdivision Implemented Family Completion: 22.5
- Implicit Field Safety And Impress Payload Gate: 21.5
- Implicit Evaluation And Extraction Adapter: 20.5
- Implicit CSG And Intersection Policy Diagnostics: 19.5
- Heightmap Impress Payload And Grid Validation: 21.5
- Heightmap Evaluation Tessellation Seam Evidence: 24.5
- Heightmap CSG Policy Diagnostics: 16.5
- Displacement Source Payload And Impress Identity Policy: 22.5
- Displacement Evaluation Tessellation Seam Evidence: 23.5
- Displacement CSG Policy Diagnostics: 16.5
- Advanced Family Seam And Boundary Participation Matrix: 17.5
- Advanced Family CSG Support Matrix And Refusal Policy: 23.5
- Advanced Family `.impress` Whole-Store Coverage: 24.5

### Loop 3

Critical review found two high-coupling matrix candidates that benefit from
split regardless of score:

- Advanced Family CSG Support Matrix And Refusal Policy: 23.5, split into
  support classification and refusal/no-fallback diagnostics.
- Advanced Family `.impress` Whole-Store Coverage: 24.5, split into codec
  coverage and whole-store evidence/negative fixtures.

Active scores after Loop 3:

- Advanced Family Promotion Gate And Capability Matrix Update: 17.5
- Shared Spline Basis And Knot Infrastructure: 17.5
- B-spline Runtime Persistence And Tessellation Completion: 21.5
- NURBS Implemented Family Completion: 22.5
- Shared Path Frame Transport Policy: 17.5
- Sweep Runtime Persistence And Tessellation Completion: 21.5
- Subdivision Implemented Family Completion: 22.5
- Implicit Field Safety And Impress Payload Gate: 21.5
- Implicit Evaluation And Extraction Adapter: 20.5
- Implicit CSG And Intersection Policy Diagnostics: 19.5
- Heightmap Impress Payload And Grid Validation: 21.5
- Heightmap Evaluation Tessellation Seam Evidence: 24.5
- Heightmap CSG Policy Diagnostics: 16.5
- Displacement Source Payload And Impress Identity Policy: 22.5
- Displacement Evaluation Tessellation Seam Evidence: 23.5
- Displacement CSG Policy Diagnostics: 16.5
- Advanced Family Seam And Boundary Participation Matrix: 17.5
- Advanced Family CSG Support Classification Matrix: 18.5
- Advanced Family CSG Refusal And No-Fallback Diagnostics: 16.5
- Advanced Family Impress Codec Coverage Gate: 18.5
- Advanced Family Impress Whole-Store Round-Trip Evidence: 20.5

### Loop 4

Critical review found four candidates that were below the mandatory split
threshold but still hid work that would benefit from clearer ownership:

- NURBS Implemented Family Completion: 22.5, split into rational
  evaluation/weight validation and runtime persistence/tessellation.
- Subdivision Implemented Family Completion: 22.5, split into scheme/cage
  evaluation and runtime persistence/tessellation/evidence.
- Heightmap Evaluation Tessellation Seam Evidence: 24.5, split into
  evaluation/tessellation adapter and seam/reference evidence.
- Displacement Evaluation Tessellation Seam Evidence: 23.5, split into
  evaluation/tessellation adapter and seam/reference evidence.

Active scores after Loop 4:

- Advanced Family Promotion Gate And Capability Matrix Update: 17.5
- Shared Spline Basis And Knot Infrastructure: 17.5
- B-spline Runtime Persistence And Tessellation Completion: 21.5
- NURBS Rational Evaluation And Weight Validation: 17.5
- NURBS Exact Conic Producer Helpers: 21
- NURBS Runtime Persistence And Tessellation Completion: 24.5
- Shared Path Frame Transport Policy: 17.5
- Sweep Runtime Persistence And Tessellation Completion: 21.5
- Subdivision Scheme And Cage Evaluation: 18.5
- Subdivision Runtime Persistence Tessellation And Evidence: 22.5
- Implicit Field Safety And Impress Payload Gate: 21.5
- Implicit Evaluation And Extraction Adapter: 20.5
- Implicit CSG And Intersection Policy Diagnostics: 19.5
- Heightmap Impress Payload And Grid Validation: 21.5
- Heightmap Evaluation And Tessellation Adapter: 18.5
- Heightmap Seam Approximation And Reference Evidence: 22.5
- Heightmap CSG Policy Diagnostics: 16.5
- Displacement Source Payload And Impress Identity Policy: 22.5
- Displacement Evaluation And Tessellation Adapter: 18.5
- Displacement Seam Approximation And Reference Evidence: 22.5
- Displacement CSG Policy Diagnostics: 16.5
- Advanced Family Seam And Boundary Participation Matrix: 17.5
- Advanced Family CSG Support Classification Matrix: 18.5
- Advanced Family CSG Refusal And No-Fallback Diagnostics: 16.5
- Advanced Family Impress Codec Coverage Gate: 18.5
- Advanced Family Impress Whole-Store Round-Trip Evidence: 20.5

No active candidate remains at `25+` after Loop 4. Every active candidate in
the `16-24` range carries a split-review cohesion reason.

### Deferred/Not-Implemented Signal Check

Critical search terms checked:

- `deferred`
- `intentionally not implemented`
- `not implemented`
- `out of scope`
- `excluded`
- `not yet supported`
- `required future capability`
- `TODO` / `TBD` / `stub`

Disposition:

- The NURBS note about exact conic producer helpers was real hidden work and is
  now split into `NURBS Exact Conic Producer Helpers`.
- Implicit external field references are not deferred work; the implemented
  canonical payload is embedded allow-listed declarative fields, with external
  references explicitly refused.
- Heightmap external data references are not deferred work; the implemented
  canonical payload is an embedded finite sampled grid, with external
  references explicitly refused.
- Displacement cross-body source references are not deferred work; the
  implemented canonical payload is embedded source payloads or stable in-body
  source identities, with cross-body references explicitly refused.
- Legacy deferred-family specs remain in the release tree as superseded
  historical work and are tracked for retirement/replacement by the capability
  matrix.
- Existing code/test strings that say `not implemented yet`,
  `not-yet-implemented`, or `unsupported` for CSG and higher-order seam
  continuity are covered by the surface-body completion and depth-completion
  architecture/specification sets. They are not acceptable terminal states for
  completion.

## Change History

- 2026-05-27: Added architecture defining the implementation-complete bar for
  B-spline, NURBS, sweep, subdivision, implicit, heightmap, and displacement
  patch families so they can move from `planned` to implemented only with
  runtime, seam, CSG, loft, `.impress`, diagnostics, tessellation, and evidence
  coverage.
