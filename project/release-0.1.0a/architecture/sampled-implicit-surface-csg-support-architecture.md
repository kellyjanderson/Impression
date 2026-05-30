# Sampled and Implicit Surface CSG Support Architecture

## Overview

This document defines what is required to turn the CSG rows involving implicit,
heightmap, and displacement families from `unsupported` into supported
surface-native behavior.

The word `unsupported` is not acceptable as a final state unless the row is
reclassified as intentionally non-CSG and paired with a supported authored
replacement workflow. In the current matrix, sampled/implicit rows are
unsupported because the kernel does not yet know how to represent, execute, and
persist boolean results for these families without falling back to mesh truth.

This document exists because sampled and implicit families are not solved by
the same machinery as B-spline/NURBS/sweep/subdivision CSG.

## Related Architecture

This document extends:

- [Surface CSG Executable Completion Architecture](surface-csg-executable-completion-architecture.md)
- [Higher-Order Surface CSG Solver Architecture](higher-order-surface-csg-solver-architecture.md)
- [Full Surface Patch Family Architecture](full-surface-patch-family-architecture.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)
- [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)

This document is refined by:

- [Sampled and Implicit CSG Unsupported Row Implementation Architecture](sampled-implicit-csg-unsupported-row-implementation-architecture.md)

## Unsupported Row Audit

For each boolean operation, 51 ordered rows are sampled or implicit:

- analytic with implicit, heightmap, or displacement
- B-spline/NURBS/sweep/subdivision with implicit, heightmap, or displacement
- implicit/heightmap/displacement with each other

These rows are currently `unsupported` because:

- implicit, heightmap, and displacement have different representation laws
- many boolean results are not representable as heightmap or displacement
  surfaces
- implicit field composition is not wired into SurfaceBody CSG result topology
- sampled contour extraction is a tessellation-adjacent operation unless the
  output remains a native surface record with lossiness metadata
- the result family policy is undefined for hybrid operations

## Family-Specific Requirements

### Implicit

Implicit CSG can be supported for many cases through field composition:

- union -> field min/composition
- intersection -> field max/composition
- difference -> base field intersected with negated cutter fields

To make implicit rows supported, the kernel needs:

- conversion policies from analytic and higher-order operands into implicit
  fields when accepted
- field-composition provenance records
- bounded evaluation domains
- safety and sample-budget validation
- result `ImplicitSurfacePatch` persistence in `.impress`
- diagnostics for unbounded or unsafe fields

Implicit support is not mesh fallback when the result remains an implicit
surface family record.

### Heightmap

Heightmap CSG is representable only for restricted 2.5D outcomes.

To make heightmap rows supported, the kernel needs:

- shared projection-plane policy
- XY-domain overlap and clipping rules
- sample-grid resampling policy
- min/max/difference height-composition semantics where mathematically valid
- no-overhang representability checks
- promotion policy when the result cannot remain a heightmap
- lossiness and resampling metadata

Rows that would produce multi-valued or overhanging results must promote to a
richer family or refuse with a representation diagnostic.

### Displacement

Displacement CSG is representable only when the result can remain a bounded
offset over a known source surface or can be promoted deliberately.

To make displacement rows supported, the kernel needs:

- source-surface identity agreement checks
- source-domain overlap and clipping rules
- displacement sample-grid resampling
- offset-composition semantics
- source-patch promotion policy for mixed-family cutters
- lossiness and source-provenance metadata
- diagnostics for cross-body or incompatible source references

Displacement rows must never silently detach from their source surface.

## Result Family Policy

Every sampled/implicit CSG row must choose exactly one result policy:

- preserve implicit
- preserve heightmap
- preserve displacement
- promote to implicit
- promote to subdivision
- promote to NURBS/B-spline
- intentionally non-CSG with replacement workflow
- refuse as unrepresentable

The result policy is part of the executable CSG matrix. It is not a local
decision inside a producer helper.

## Data Flow

```text
SurfaceBody operands
-> sampled/implicit row policy
-> representability check
-> family-specific operation route
-> result family policy
-> native patch payload with provenance/lossiness
-> SurfaceBody result
```

Failure flow:

```text
SurfaceBody operands
-> sampled/implicit row policy
-> representability failure or unsafe field/grid/source
-> structured diagnostic naming missing capability or impossible result
```

## Cross-Domain Decisions

### Unsupported Must Split Into Missing Solver Versus Impossible Representation

Some rows are unsupported because code is missing. Others may be impossible for
the requested source family. These must not share one vague state.

Required final states:

- `supported-native`
- `supported-promoted-family`
- `representation-refused`
- `unsafe-refused`
- `non-csg-replacement`

### Promotion Is Allowed When It Preserves Surface Truth

Promoting a heightmap or displacement result to implicit or subdivision is
acceptable only when:

- the promotion is explicit in the row policy
- the result is persisted as the promoted family
- lossiness/provenance is recorded
- no mesh is treated as source truth

### Sampled Extraction Must Stay Behind Surface Records

Sampling may be used to produce a native sampled surface payload. It may not
produce a mesh that is then treated as a boolean result.

## Specification Manifest for Discovery

### Candidate Spec: Sampled Implicit CSG Row Policy Matrix

Discovery purpose:
- Replace generic `unsupported` sampled/implicit CSG rows with explicit result
  policies and refusal classes.

Responsibilities:
- Functions/methods:
  - sampled/implicit row policy builder
  - representability policy lookup
  - unsupported-row classifier
- Data structures/models:
  - sampled/implicit CSG policy row
  - result family policy record
  - representation refusal diagnostic
- Dependencies/services:
  - CSG support matrix
  - patch family capability matrix
  - `.impress` family codec contracts
- Returns/outputs/signals:
  - policy row
  - support/refusal class
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: family capability and CSG support records
  - Additions to existing reusable library/module: sampled/implicit policy
    table
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - none
- Security/privacy-sensitive behavior:
  - validates unsafe implicit/source payload states without execution
- Performance-sensitive behavior:
  - bounded matrix scan
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - no generic `unsupported`; every row names missing solver, impossible
    representation, unsafe input, promotion route, or non-CSG replacement
- Test strategy:
  - row coverage for all sampled/implicit pairs across all operations
- Data ownership:
  - CSG owns operation policy; family modules own representability checks
- Routes:
  - CSG support matrix to sampled policy to operation planner
- Open questions / nuance discovered:
  - which heightmap/displacement cases should promote to implicit versus
    subdivision
- Readiness blockers:
  - result family promotion policy

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
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 0 x 2 = 0
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 3 x 1 = 3
- Readiness blockers: 1 x 2 = 2
- Total: 23.5

Split decision:
- Review for split. Cohesion reason: this candidate owns only the policy matrix
  and row classification; execution routes are split.

### Candidate Spec: Implicit Field Composition CSG Route

Discovery purpose:
- Make implicit-compatible boolean rows executable by composing native implicit
  fields and preserving result truth as `ImplicitSurfacePatch`.

Responsibilities:
- Functions/methods:
  - implicit boolean field composer
  - operand-to-field adapter
  - implicit CSG safety gate
- Data structures/models:
  - implicit CSG composition record
  - operand field adapter record
  - implicit CSG diagnostic
- Dependencies/services:
  - implicit field node builders
  - CSG operation planner
  - `.impress` implicit codec
- Returns/outputs/signals:
  - implicit result patch
  - unsafe/unbounded diagnostic
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: implicit field nodes and safety records
  - Additions to existing reusable library/module: implicit CSG composer
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes CSG result construction
- Security/privacy-sensitive behavior:
  - refuses unsafe executable payloads
- Performance-sensitive behavior:
  - bounded evaluation-domain and extraction budgets
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - implicit results require finite bounds before tessellation or analysis
- Test strategy:
  - implicit/implicit, analytic/implicit, unsafe field, unbounded field, and
    `.impress` round-trip tests
- Data ownership:
  - implicit family owns field payload; CSG owns operation provenance
- Routes:
  - operation planner to implicit composer to SurfaceBody result
- Open questions / nuance discovered:
  - conversion from arbitrary higher-order patches to implicit may require a
    declared-tolerance adapter
- Readiness blockers:
  - sampled/implicit row policy matrix

Score:
- Functions/methods: 3 x 2 = 6
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 2 x 1 = 2
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 30.5

Split decision:
- Split required. Split into implicit/implicit composition, analytic-to-implicit
  adapters, higher-order-to-implicit adapters, safety/bounds gate, and
  persistence/provenance fixtures.

### Candidate Spec: Heightmap And Displacement Representable CSG Routes

Discovery purpose:
- Support the subset of heightmap and displacement boolean rows whose results
  can remain native sampled surface payloads, and classify/promote/refuse the
  rest.

Responsibilities:
- Functions/methods:
  - heightmap CSG representability checker
  - displacement CSG representability checker
  - sampled result payload builder
  - promotion/refusal diagnostic builder
- Data structures/models:
  - sampled CSG representability record
  - sampled CSG result payload record
  - sampled promotion/lossiness record
- Dependencies/services:
  - heightmap grid validation
  - displacement source resolver
  - CSG operation planner
- Returns/outputs/signals:
  - sampled result patch
  - promotion or representation refusal
- UI surfaces/components:
  - none
- UI fields/elements:
  - none
- Reusable code plan:
  - Existing code reused as-is: heightmap and displacement authoring records
  - Additions to existing reusable library/module: sampled CSG
    representability and payload helpers
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - changes sampled CSG execution behavior
- Security/privacy-sensitive behavior:
  - refuses external/cross-body references outside policy
- Performance-sensitive behavior:
  - bounded resampling and grid-size budgets
- Cross-screen reusable behavior:
  - not applicable

Project readiness fields:
- Implementation owner/module:
  - `src/impression/modeling/csg.py`
  - `src/impression/modeling/heightmap.py`
  - `src/impression/modeling/surface.py`
- Chosen defaults / parameters:
  - non-representable sampled results do not execute as sampled outputs
- Test strategy:
  - representable same-domain grids, incompatible domains, overhang/multivalue
    refusal, source mismatch, promotion metadata, and `.impress` fixtures
- Data ownership:
  - sampled family owns payload validity; CSG owns operation selection
- Routes:
  - CSG policy row to sampled representability checker to result builder
- Open questions / nuance discovered:
  - promotion target should be implicit by default for volumetric sampled CSG,
    subdivision for bounded surface reconstruction
- Readiness blockers:
  - result family promotion policy

Score:
- Functions/methods: 4 x 2 = 8
- Data structures/models: 3 x 1 = 3
- Dependencies/services: 3 x 1 = 3
- Returns/outputs/signals: 2 x 1 = 2
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 1 x 0.5 = 0.5
- Adding code to an existing library/module: 3 x 1 = 3
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 1 x 2 = 2
- Security/privacy-sensitive behavior: 1 x 2 = 2
- Performance-sensitive behavior: 2 x 1 = 2
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 5 x 1 = 5
- Readiness blockers: 1 x 2 = 2
- Total: 34.5

Split decision:
- Split required. Split into heightmap representability, displacement
  representability, sampled result payloads, promotion policy, and reference
  fixtures.

## Change History

- 2026-05-30: Linked the row-level implementation architecture that takes the
  153 unsupported sampled/implicit CSG rows as in-progress work.
- 2026-05-28: Created sampled/implicit CSG support architecture after the CSG
  row audit showed 153 operation rows still marked unsupported across implicit,
  heightmap, and displacement families.
