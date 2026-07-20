# SurfaceBody CSG Architecture

## Overview

This document defines the architecture for true `SurfaceBody` boolean
operations.

The current surfaced CSG branch already defines:

- input eligibility
- surfaced result envelopes
- public migration posture

What it does **not** yet define is the kernel law for actually executing
boolean operations on surface-native operands.

This architecture fills that gap.

The governing rule is:

> SurfaceBody CSG must operate on shells, patches, seams, trims, and
> classification truth directly.
>
> Mesh may appear only at preview, export, analysis, or explicit repair
> boundaries.

## Backlink

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [Surface-Native Capability Replacement Architecture](surface-native-capability-replacement-architecture.md)

## Components

### Operand Preparation

Operand preparation is responsible for turning caller-provided `SurfaceBody`
operands into canonical boolean inputs.

It owns:

- operand eligibility checks
- transform baking required for geometric comparison
- deterministic patch/shell ordering
- trim and seam validation required before execution

It does **not** decide the boolean result.

### Surface Intersection and Classification

The boolean kernel requires a surface-native intersection and classification
stage.

It owns:

- computing surface/surface intersection curves between operand patches
- mapping those intersections into patch-local trim space
- splitting affected operand boundaries into classified fragments
- determining which fragments lie inside, outside, or on the other operand

This stage is where boolean truth is discovered.

It should operate on:

- patch families
- canonical seam geometry
- patch-local trim references
- shell containment/classification queries

It must not degrade the problem into mesh-primary triangle clipping.

### Operand Fragment Graph

After intersections are discovered, each operand becomes a set of classified
surface fragments rather than a single untouched shell.

The fragment graph owns:

- surviving patch fragments
- discarded patch fragments
- newly created cut boundaries
- per-fragment provenance back to the source operand

This stage is temporary and executor-internal.

It exists so the result shell can be reconstructed deterministically.

### Result Topology Reconstructor

The reconstructor owns the actual result `SurfaceBody`.

It is responsible for:

- assembling surviving fragments into result shells
- constructing new trim loops from cut boundaries
- constructing new shared seams and open boundaries
- preserving or rebuilding adjacency truth
- deciding whether the result is one shell, multiple shells, or empty

This is where temporary fragment truth becomes durable kernel truth.

### Validity and Healing Gate

The boolean result must pass through an explicit validity gate.

This gate owns:

- trim validity checks on reconstructed patches
- seam pairing and open/shared classification checks
- closed-shell eligibility checks
- bounded healing or canonical cleanup where permitted

Healing here must remain narrow.

Allowed healing should be limited to:

- topological cleanup
- seam/use normalization
- trim canonicalization
- deterministic removal of zero-measure artifacts

It must not silently invent materially different geometry in order to force a
successful boolean result.

### Metadata and Provenance Propagation

Boolean execution also needs a durable rule for non-geometric information.

This stage owns:

- source provenance
- consumer metadata carry-forward
- operation metadata such as `union`, `difference`, or `intersection`
- color/material inheritance rules where relevant

Kernel-native topology truth must remain primary.

Consumer metadata propagation should be explicit and deterministic.

### Public CSG Surface

The public `csg.py` surface remains the stable caller boundary.

It owns:

- surfaced boolean entrypoints
- surfaced operation selection
- surfaced success/failure reporting
- migration and documentation posture

It should not reach through private kernel helpers in other modules.

## Relationships

- operand preparation produces canonical boolean operands
- intersection/classification consumes those operands and emits classified cut
  fragments
- the fragment graph feeds the result topology reconstructor
- the reconstructor emits candidate result shells and seams
- the validity/healing gate either accepts, rejects, or classifies the result
- metadata propagation finalizes surfaced result records
- the public CSG surface returns surfaced results to callers

The key relationship is:

```text
SurfaceBody operands
-> classified fragments
-> reconstructed SurfaceBody result
```

not:

```text
SurfaceBody operands
-> temporary mesh boolean
-> rewrapped surface result
```

## Data Flow

### Nominal Boolean Flow

```text
SurfaceBody operands
-> operand preparation
-> patch/patch intersection discovery
-> fragment classification
-> fragment graph
-> shell / seam / trim reconstruction
-> validity and bounded healing
-> SurfaceBody boolean result
-> tessellation on demand
```

### Failure Flow

```text
SurfaceBody operands
-> operand preparation
-> unsupported or invalid execution condition
-> surfaced boolean result with explicit failure / unsupported status
```

Failure must happen before any compatibility mesh shortcut is chosen as a
hidden fallback.

## Cross-Domain Solutions

### Split First, Then Reconstruct

The architectural heart of surfaced CSG is:

1. intersect and classify operands
2. split operands into fragments
3. reconstruct the result shell from those fragments

This is the surface-native analogue of industrial B-rep boolean behavior.

It keeps:

- patch meaning
- trim meaning
- seam truth
- shell truth

intact long enough for the result to remain surface-native.

### Seams Are Rebuilt as Kernel Truth

Boolean execution cannot treat seams as a post-process convenience.

When a cut boundary becomes part of the result, the resulting body must own:

- new seam identity where the boundary is shared
- new open-boundary truth where the boundary remains exposed
- new boundary-use records for participating patches

That keeps the result compatible with seam-first tessellation and watertight
classification.

### Trims Are the Result Boundary Law

For surfaced booleans, trims become the practical result-boundary mechanism.

This means the boolean result architecture depends on:

- patch-local trim reconstruction
- loop role classification
- boundary orientation correctness

The reconstructor should prefer trim reconstruction over inventing new patch
families or flattening the result into tessellated approximations.

### Bounded Healing, Not Mesh-Style Repair

SurfaceBody CSG will inevitably face near-degenerate cases.

The architecture permits bounded healing only when it preserves the intended
surface result.

Examples of allowed healing:

- removing duplicate seam uses
- removing zero-area trim slivers
- canonical loop orientation repair

Examples of disallowed healing:

- substituting a mesh boolean as hidden truth
- warping boundaries to force closure
- collapsing materially distinct patches into a fake success

### Initial Executable Scope Must Be Explicit

The first executable surfaced boolean slice should be intentionally bounded.

The architecture expects the initial executable scope to be defined in specs in
terms of:

- supported operation families
- supported operand shell classes
- supported patch families
- supported trim complexity
- unsupported cases that remain explicit

This keeps boolean implementation incremental without pretending the entire
general problem is solved at once.

### Metadata Follows Result Ownership

Boolean results should not simply concatenate operand metadata.

Instead:

- kernel metadata follows the reconstructed body/shell ownership
- provenance tracks source operands and operation type
- consumer metadata uses explicit carry-forward rules

This avoids the mesh-era pattern of “whichever side happened to win the union”
becoming the accidental metadata source.

## Related Architecture

- [Surface-First Internal Model Architecture](surface-first-internal-model.md)
- [SurfaceBody Seam and Adjacency Architecture](surfacebody-seam-adjacency-architecture.md)
- [Surface-Native Capability Replacement Architecture](surface-native-capability-replacement-architecture.md)

## Specifications

This architecture extends the existing surfaced boolean specification branch:

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](../specifications/surface-102-surface-body-boolean-replacement-v1_0.md)
- [Surface Spec 108: Surface Boolean Input Eligibility and Canonicalization (v1.0)](../specifications/surface-108-surface-boolean-input-eligibility-and-canonicalization-v1_0.md)
- [Surface Spec 109: Surface Boolean Result Contract and Failure Modes (v1.0)](../specifications/surface-109-surface-boolean-result-contract-and-failure-modes-v1_0.md)
- [Surface Spec 110: Surface Boolean Public API Migration and Reference Verification (v1.0)](../specifications/surface-110-surface-boolean-public-api-migration-and-reference-verification-v1_0.md)

The execution gap introduced by those earlier leaves is closed by these new
boolean execution leaves:

- [Surface Spec 117: Surface Boolean Intersection, Classification, and Operand Splitting (v1.0)](../specifications/surface-117-surface-boolean-intersection-classification-and-operand-splitting-v1_0.md)
- [Surface Spec 118: Surface Boolean Result Topology Reconstruction (v1.0)](../specifications/surface-118-surface-boolean-result-topology-reconstruction-v1_0.md)
- [Surface Spec 119: Surface Boolean Validity, Healing Limits, and Metadata Propagation (v1.0)](../specifications/surface-119-surface-boolean-validity-healing-limits-and-metadata-propagation-v1_0.md)
- [Surface Spec 120: Surface Boolean Initial Executable Scope and Reference Fixture Matrix (v1.0)](../specifications/surface-120-surface-boolean-initial-executable-scope-and-reference-fixture-matrix-v1_0.md)

## CSG Completion Scope

The earlier boolean execution leaves created the initial executable surface CSG
slice. That slice is intentionally not the full CSG program. Full surface-body
modeling requires a second CSG completion pass that removes any remaining mesh
fallback temptation and makes unsupported combinations explicit until exact
surface execution exists.

The completion pass is governed by these rules:

- CSG truth is computed from surface patches, trims, seams, shells, and
  classification records.
- Tessellation may be used only after a `SurfaceBody` result exists, or for an
  explicitly named mesh analysis/compatibility tool.
- Unsupported family pairs return deterministic surface CSG diagnostics; they
  never invoke a mesh boolean as a fallback.
- Operation-specific primitive helpers that need CSG must either consume the
  surface CSG result or refuse with the same diagnostics.
- Reference fixtures must prove both positive execution and no-hidden-mesh
  behavior.

The CSG completion work is defined by these additional leaves:

- [Surface Spec 218: Surface CSG Family Capability Matrix And Refusal Gate (v1.0)](../specifications/surface-218-surface-csg-family-capability-matrix-and-refusal-gate-v1_0.md)
- [Surface Spec 219: Surface CSG Analytic Patch Intersection Library (v1.0)](../specifications/surface-219-surface-csg-analytic-patch-intersection-library-v1_0.md)
- [Surface Spec 220: Surface CSG Trim Fragment Graph For Non-Box Patch Families (v1.0)](../specifications/surface-220-surface-csg-trim-fragment-graph-for-non-box-patch-families-v1_0.md)
- [Surface Spec 221: Surface CSG Multi-Family Result Reconstruction (v1.0)](../specifications/surface-221-surface-csg-multi-family-result-reconstruction-v1_0.md)
- [Surface Spec 222: Surface CSG Primitive Integration And No-Mesh Fallback Gates (v1.0)](../specifications/surface-222-surface-csg-primitive-integration-and-no-mesh-fallback-gates-v1_0.md)
- [Surface Spec 223: Surface CSG Reference Fixture Matrix Expansion (v1.0)](../specifications/surface-223-surface-csg-reference-fixture-matrix-expansion-v1_0.md)

Critical review of those leaves exposed additional hidden work. The following
smaller leaves define that work explicitly:

- [Surface Spec 226: CSG Curve Primitive And Tolerance Policy (v1.0)](../specifications/surface-226-csg-curve-primitive-and-tolerance-policy-v1_0.md)
- [Surface Spec 227: CSG Patch-Local Curve Mapping (v1.0)](../specifications/surface-227-csg-patch-local-curve-mapping-v1_0.md)
- [Surface Spec 228: CSG Planar/Linear Analytic Intersections (v1.0)](../specifications/surface-228-csg-planar-linear-analytic-intersections-v1_0.md)
- [Surface Spec 229: CSG Revolution/Conic Analytic Intersections (v1.0)](../specifications/surface-229-csg-revolution-conic-analytic-intersections-v1_0.md)
- [Surface Spec 230: CSG Higher-Order Analytic Intersection Refusal Or Solver Boundary (v1.0)](../specifications/surface-230-csg-higher-order-analytic-intersection-refusal-or-solver-boundary-v1_0.md)
- [Surface Spec 231: CSG Curve Arrangement And Trim Loop Splitting (v1.0)](../specifications/surface-231-csg-curve-arrangement-and-trim-loop-splitting-v1_0.md)
- [Surface Spec 232: CSG Fragment Inside/Outside Classification Predicates (v1.0)](../specifications/surface-232-csg-fragment-inside-outside-classification-predicates-v1_0.md)
- [Surface Spec 233: CSG Operation Selection Rules (v1.0)](../specifications/surface-233-csg-operation-selection-rules-v1_0.md)
- [Surface Spec 234: CSG Shell Assembly From Fragments (v1.0)](../specifications/surface-234-csg-shell-assembly-from-fragments-v1_0.md)
- [Surface Spec 235: CSG Seam And Adjacency Rebuild (v1.0)](../specifications/surface-235-csg-seam-and-adjacency-rebuild-v1_0.md)
- [Surface Spec 236: CSG Validity, Healing, And Provenance Gate (v1.0)](../specifications/surface-236-csg-validity-healing-and-provenance-gate-v1_0.md)
- [Surface Spec 237: CSG Primitive Caller Inventory And Gate Helper (v1.0)](../specifications/surface-237-csg-primitive-caller-inventory-and-gate-helper-v1_0.md)
- [Surface Spec 238: CSG Primitive Migration (v1.0)](../specifications/surface-238-csg-primitive-migration-v1_0.md)
- [Surface Spec 239: CSG Feature Builder Boolean Migration (v1.0)](../specifications/surface-239-csg-feature-builder-boolean-migration-v1_0.md)

## Specification Manifest for Discovery

### Candidate Spec: Surface CSG Family Capability Matrix And Refusal Gate

Discovery purpose:
- Define the authoritative family-pair and operation matrix for CSG support,
  including explicit refusal behavior for unsupported pairs.

Responsibilities:
- Functions/methods:
  - CSG capability matrix lookup
  - operation/family eligibility gate
- Data structures/models:
  - family-pair support record
  - unsupported CSG diagnostic payload
- Dependencies/services:
  - `src/impression/modeling/csg.py`
  - patch family capability matrix
- Returns/outputs/signals:
  - executable support decision
  - explicit refusal result
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: current CSG diagnostics and patch family matrix
  - Additions to existing reusable library/module: CSG family support table
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - updates CSG behavior and tests
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded matrix lookup on CSG entry
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
- Chosen defaults / parameters:
  - unsupported pairs refuse before any mesh path is reachable
- Test strategy:
  - matrix coverage for supported and unsupported pairs
- Data ownership:
  - CSG owns CSG operation support; patch families own patch capabilities
- Routes:
  - public boolean API to CSG support gate
- Reuse/extraction decision:
  - extend existing CSG diagnostics
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Refusal is a valid surface CSG result until the exact intersection library
  supports the family pair.

Score:
- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 2 x 1 = 2
- Existing reusable code reused as-is: 2 x 0.5 = 1
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
- Total: 17

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: the matrix and refusal gate are one
  decision boundary.

### Candidate Spec: Surface CSG Analytic Patch Intersection Library

Discovery purpose:
- Add exact intersection primitives for analytic patch families so common CSG
  work does not stop at box/box or planar-only scope.

Responsibilities:
- Functions/methods:
  - plane/plane intersection
  - plane/cylinder and plane/cone intersection
  - plane/sphere and plane/torus intersection
  - cylinder/cylinder initial intersection
- Data structures/models:
  - surface intersection curve record
  - patch-local curve mapping record
- Dependencies/services:
  - analytic patch evaluators
  - CSG classifier
- Returns/outputs/signals:
  - exact curve segments
  - unsupported/degenerate diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: analytic patch evaluation
  - Additions to existing reusable library/module: CSG intersection helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG kernel behavior
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded pairwise patch intersection
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py` or a CSG-private helper module
- Chosen defaults / parameters:
  - exact analytic curves are preferred; ambiguous degeneracies refuse
- Test strategy:
  - unit fixtures for each named analytic pair and degeneracy
- Data ownership:
  - CSG owns intersection records; patch families own evaluation
- Routes:
  - prepared operands to intersection discovery
- Reuse/extraction decision:
  - keep helpers CSG-private until reused by another kernel operation
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Spline and implicit intersections are intentionally not bundled here.

Score:
- Functions/methods: 4 x 2 = 8
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
- Readiness blockers: 0 x 2 = 0
- Total: 20.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this is the analytic-only intersection
  set; spline, subdivision, sweep, and implicit pairs remain separate refusal
  or future exact-support work.

### Candidate Spec: Surface CSG Trim Fragment Graph For Non-Box Patch Families

Discovery purpose:
- Generalize boolean fragment records beyond the initial box slice so analytic
  and trimmed patch families can be split without tessellating first.

Responsibilities:
- Functions/methods:
  - intersection curve to trim-fragment mapper
  - fragment classification builder
- Data structures/models:
  - trim fragment graph
  - fragment provenance record
- Dependencies/services:
  - intersection curve records
  - seam and trim validation
- Returns/outputs/signals:
  - classified patch fragments
  - invalid fragment diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: initial boolean fragment concepts
  - Additions to existing reusable library/module: generalized fragment records
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG internal records
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded by patch/curve count
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - CSG internal fragment/classification code
- Chosen defaults / parameters:
  - trim fragments retain patch-local curve references and provenance
- Test strategy:
  - fragment fixtures for planar, cylinder, sphere, and trimmed cut cases
- Data ownership:
  - CSG owns temporary fragment graph
- Routes:
  - intersection discovery to result reconstruction
- Reuse/extraction decision:
  - reuse seam/trim validation helpers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Fragment graph is executor-internal and must not become persisted `.impress`
  truth.

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
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: mapping and classification share the same
  fragment graph contract.

### Candidate Spec: Surface CSG Multi-Family Result Reconstruction

Discovery purpose:
- Reconstruct boolean results from mixed analytic and trimmed fragments as
  `SurfaceBody` shells with durable seams, trims, and provenance.

Responsibilities:
- Functions/methods:
  - fragment-to-shell assembler
  - seam/use rebuilder
  - result validity gate
- Data structures/models:
  - reconstructed shell record
  - boolean provenance metadata
- Dependencies/services:
  - fragment graph
  - seam/adjacency model
  - `.impress` surface payload constraints
- Returns/outputs/signals:
  - `SurfaceBody` boolean result
  - invalid reconstruction diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: seam and adjacency primitives
  - Additions to existing reusable library/module: result assembly helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG result construction
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded reconstruction validation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - CSG result reconstruction code
- Chosen defaults / parameters:
  - reconstructed truth is surface-native; no mesh rewrap is permitted
- Test strategy:
  - mixed-family union, difference, and intersection reconstruction fixtures
- Data ownership:
  - result `SurfaceBody` owns durable topology; CSG owns provenance metadata
- Routes:
  - fragment graph to validity gate to public result
- Reuse/extraction decision:
  - reuse existing `SurfaceBody` and seam containers
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Empty and multi-shell results must remain valid outcomes, not failures.

Score:
- Functions/methods: 3 x 2 = 6
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
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: shell assembly, seam rebuilding, and
  validity are one result-boundary operation.

### Candidate Spec: Surface CSG Primitive Integration And No-Mesh Fallback Gates

Discovery purpose:
- Ensure primitives and modeled features that require booleans consume surface
  CSG or refuse explicitly rather than invoking mesh booleans.

Responsibilities:
- Functions/methods:
  - primitive CSG route audit
  - boolean-dependent feature gate
- Data structures/models:
  - primitive CSG dependency record
  - no-mesh-fallback assertion fixture
- Dependencies/services:
  - primitives
  - Impression-owned feature builders such as hinges
  - generic external feature interop boundaries
  - CSG support gate
- Returns/outputs/signals:
  - surface primitive result or diagnostic
  - test failure on mesh boolean route
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: primitive surface defaults
  - Additions to existing reusable library/module: feature CSG gates
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes primitive/feature routing where needed
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded per feature operation
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - primitive and feature builder modules that call CSG
- Chosen defaults / parameters:
  - boolean-dependent surface features refuse when surface CSG cannot execute
- Test strategy:
  - audit tests proving no primitive falls back to mesh boolean
- Data ownership:
  - feature modules own caller policy; CSG owns operation support truth
- Routes:
  - feature builder to CSG support gate to surface result/diagnostic
- Reuse/extraction decision:
  - share one gate helper instead of per-feature ad hoc checks
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- This spec is allowed to leave explicit mesh compatibility APIs intact only
  when names and docs make the mesh boundary obvious.

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
- Readiness blockers: 0 x 2 = 0
- Total: 17.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: every caller uses the same CSG support
  gate and no-fallback assertion.

### Candidate Spec: Surface CSG Reference Fixture Matrix Expansion

Discovery purpose:
- Add reference fixtures that prove the completed CSG scope works as
  surface-native modeling and refuses unsupported cases without mesh fallback.

Responsibilities:
- Functions/methods:
  - CSG fixture builder
  - no-mesh assertion helper
  - tessellation-boundary verifier
- Data structures/models:
  - CSG fixture matrix
  - expected result/diagnostic record
- Dependencies/services:
  - CSG public API
  - tessellation boundary
  - reference artifact harness
- Returns/outputs/signals:
  - passing reference fixtures
  - explicit failure reports
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reference artifact harness
  - Additions to existing reusable library/module: CSG fixture matrix
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes reference fixtures/artifacts as needed
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - fixtures must be bounded and deterministic
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - tests and reference fixture modules
- Chosen defaults / parameters:
  - every CSG fixture declares whether surface execution or refusal is expected
- Test strategy:
  - automated acceptance for exact results, diagnostics, and tessellation-only mesh
- Data ownership:
  - tests own fixture expectations; CSG owns operation behavior
- Routes:
  - fixture builder to public CSG API to tessellation verifier when needed
- Reuse/extraction decision:
  - reuse no-hidden-mesh helper from mesh boundary specs
- UI field/control inventory:
  - not applicable

Open questions / nuance discovered:
- Fixture outputs should not bless tessellated meshes as modeled truth.

Score:
- Functions/methods: 3 x 2 = 6
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
- Readiness blockers: 0 x 2 = 0
- Total: 19.5

Readiness blockers:
- [x] Missing implementation owner/module.
- [x] Missing reuse/extraction decision.
- [x] Missing library/module boundary where adding or creating reusable code.
- [x] Missing UI field/control inventory where applicable.
- [x] Missing chosen defaults / parameters.
- [x] Missing test strategy.
- [x] Unclear data ownership.
- [x] Missing GUI/concurrency route where applicable.
- [x] Missing performance bound/index plan where applicable.
- [x] Missing privacy/logging rule where applicable.

Split decision:
- Review for split. Cohesion reason: this is one fixture matrix that verifies
  the CSG completion pass end to end.

## Change History

- 2026-05-27: Limited feature-builder CSG language to Impression-owned
  feature builders.
- 2026-05-26: Added CSG completion scope and manifest entries for full
  surface-native CSG without mesh fallback.
