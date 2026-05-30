# Sampled and Implicit CSG Unsupported Row Implementation Architecture

## Overview

This document takes the 153 currently unsupported SurfaceBody CSG rows as the
body of work for the next architecture branch.

These rows are unsupported because every row involves at least one sampled or
implicit family:

- `implicit`
- `heightmap`
- `displacement`

The target is not to hide those rows behind diagnostics. The target is to turn
each row into one of the following supported surface-native outcomes:

- executable native implicit composition
- executable native heightmap composition
- executable native displacement composition
- executable promoted-family result
- executable explicit representation refusal where the requested result is
  mathematically impossible and the refusal is itself the supported kernel
  behavior
- deliberately non-CSG authored replacement workflow

No row may use mesh as source truth. Sampling is allowed only as part of a
native surface payload, bounded contour extraction, diagnostics, or the explicit
tessellation boundary.

## Related Architecture

This document extends:

- [Sampled and Implicit Surface CSG Support Architecture](sampled-implicit-surface-csg-support-architecture.md)
- [Surface CSG Executable Completion Architecture](surface-csg-executable-completion-architecture.md)
- [Advanced Patch Family Implementation Completion Architecture](advanced-patch-family-implementation-completion-architecture.md)
- [.impress Surface-Native File Format Architecture](impress-surface-native-file-format-architecture.md)
- [Mesh Execution To Tessellation Boundary Architecture](mesh-execution-tessellation-boundary-architecture.md)

## Source Truth

The source inventory is the executable CSG support matrix in
`src/impression/modeling/csg.py`.

Current row count:

- total CSG rows: 300
- supported rows: 147
- unsupported rows: 153
- unsupported row state: `unsupported`
- unsupported row operations: `union`, `difference`, `intersection`
- unsupported row family-pair count per operation: 51

The 153 rows are grouped below as 51 ordered family pairs multiplied across the
three boolean operations. Every row in this document is marked `In Progress`
for architecture purposes. The row remains incomplete until its downstream
specification and implementation promote it out of `unsupported`.

## Unsupported Row Inventory

| # | Left family | Right family | Operations | Unsupported rows | Architecture status |
|---:|---|---|---|---:|---|
| 1 | `bspline` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 2 | `bspline` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 3 | `bspline` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 4 | `displacement` | `bspline` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 5 | `displacement` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 6 | `displacement` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 7 | `displacement` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 8 | `displacement` | `nurbs` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 9 | `displacement` | `planar` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 10 | `displacement` | `revolution` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 11 | `displacement` | `ruled` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 12 | `displacement` | `subdivision` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 13 | `displacement` | `sweep` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 14 | `heightmap` | `bspline` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 15 | `heightmap` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 16 | `heightmap` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 17 | `heightmap` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 18 | `heightmap` | `nurbs` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 19 | `heightmap` | `planar` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 20 | `heightmap` | `revolution` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 21 | `heightmap` | `ruled` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 22 | `heightmap` | `subdivision` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 23 | `heightmap` | `sweep` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 24 | `implicit` | `bspline` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 25 | `implicit` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 26 | `implicit` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 27 | `implicit` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 28 | `implicit` | `nurbs` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 29 | `implicit` | `planar` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 30 | `implicit` | `revolution` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 31 | `implicit` | `ruled` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 32 | `implicit` | `subdivision` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 33 | `implicit` | `sweep` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 34 | `nurbs` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 35 | `nurbs` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 36 | `nurbs` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 37 | `planar` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 38 | `planar` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 39 | `planar` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 40 | `revolution` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 41 | `revolution` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 42 | `revolution` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 43 | `ruled` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 44 | `ruled` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 45 | `ruled` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 46 | `subdivision` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 47 | `subdivision` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 48 | `subdivision` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 49 | `sweep` | `displacement` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 50 | `sweep` | `heightmap` | `union`, `difference`, `intersection` | 3 | `In Progress` |
| 51 | `sweep` | `implicit` | `union`, `difference`, `intersection` | 3 | `In Progress` |

## Implementation Strategy

### Route Families

The unsupported rows collapse into four route families:

- implicit-preserving routes
- heightmap-preserving routes
- displacement-preserving routes
- promotion/refusal routes for hybrid or unrepresentable results

Each route family must produce a surface-native result contract. If the result
cannot be represented in the requested family, the route must either promote to
a declared surface family or return a structured representation refusal.

### Implicit-Preserving Routes

Rows with an implicit result target should use signed or predicate field
composition:

- `union`: soft or hard minimum composition, according to field policy
- `intersection`: maximum composition
- `difference`: maximum of the base field and negated cutter field

Required components:

- implicit field expression graph
- bounded evaluation domain
- field safety validation
- operand-to-field adapter records
- composed-field provenance
- `.impress` implicit-composition payload

Parametric operands may participate through analytic field adapters when exact,
or through bounded evaluator adapters when declared-tolerance. The adapter is
surface-native only when the persisted result is an implicit field record with
bounded provenance, not a mesh.

### Heightmap-Preserving Routes

Heightmap-preserving CSG is allowed only for 2.5D representable results over a
declared projection domain.

Required components:

- projection-plane agreement
- XY-domain overlap and clipping
- grid alignment or resampling policy
- per-operation height composition
- no-overhang representability check
- lossiness and resampling metadata
- `.impress` heightmap CSG payload

If a row produces a multi-valued vertical result, creates overhangs, or cannot
be expressed over the output grid, it must promote to implicit or subdivision,
or refuse with a representation diagnostic.

### Displacement-Preserving Routes

Displacement-preserving CSG is allowed only when the output remains an offset
field over a known source surface.

Required components:

- source-surface identity resolution
- source-domain overlap and clipping
- displacement sample alignment or resampling
- offset composition semantics
- source normal and tangent-frame validation
- source-preserving provenance
- `.impress` displacement CSG payload

If the operation changes the source topology, crosses source identity
boundaries, or detaches the offset field from its source, it must promote or
refuse explicitly.

### Promotion Routes

Promotion is required when a supported row cannot preserve the sampled family
but can still produce surface truth.

Allowed promotion targets:

- implicit, for volumetric or predicate-preserving results
- subdivision, for bounded reconstructed surfaces with topology changes
- NURBS or B-spline, when exact or declared-tolerance spline reconstruction is
  valid

Promotion requires:

- explicit result family policy in the CSG matrix
- provenance linking source operands and route decision
- lossiness metadata when sampling or reconstruction changes representation
- `.impress` round-trip payload support
- reference evidence proving no mesh source truth

### Representation Refusal Routes

Representation refusal is a supported route only when the operation is
mathematically impossible or intentionally non-CSG for the selected families.

The refusal must include:

- operation
- left and right family
- row id
- reason category
- representability test that failed
- suggested promotion or authored replacement, when available
- no mesh fallback evidence

Refusal is not a substitute for missing solver code. Missing solver code must
remain an implementation blocker until the row has a route.

## Data Flow

```text
SurfaceBooleanOperands
-> sampled/implicit row policy lookup
-> route family selection
-> family-specific representability check
-> native operation, promotion, or representation refusal
-> result payload/provenance/lossiness record
-> SurfaceBody result or supported refusal diagnostic
-> .impress persistence and reference evidence
```

## Completion Gates

The architecture branch is complete only when:

- the 153 rows no longer report raw `unsupported`
- each row has a native route, promotion route, representation refusal route, or
  deliberate non-CSG replacement workflow
- no route attempts mesh fallback
- every promoted route has `.impress` round-trip coverage
- reference artifacts distinguish clean executable evidence from dirty or
  diagnostic-only evidence
- the CSG support matrix report shows zero unresolved unsupported rows

## Specification Manifest for Discovery

### Candidate Spec: Sampled Implicit Unsupported Row Tracker

Discovery purpose:
- Add a durable row tracker that enumerates the 153 unsupported rows and fails
  completion when a row lacks an in-progress or final route classification.

Responsibilities:
- Functions/methods: row inventory builder, unsupported-row status verifier
- Data structures/models: unsupported row tracking record, route status enum
- Dependencies/services: CSG support matrix, patch family capability matrix
- Returns/outputs/signals: unsupported row report, missing-route diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing CSG matrix reporting helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: none
- Performance-sensitive behavior: bounded matrix scan
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: all 153 rows begin as `in-progress`
- Test strategy: count assertions for 153 rows and per-operation 51-row coverage
- Data ownership: CSG owns row state; family modules own representability facts
- Routes: matrix to tracker to completion gate
- Open questions / nuance discovered: none
- Readiness blockers: none

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
- Async/concurrency behavior: 0 x 1 = 0
- Destructive/write behavior: 0 x 2 = 0
- Security/privacy-sensitive behavior: 0 x 2 = 0
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 2 x 1 = 2
- Readiness blockers: 0 x 2 = 0
- Total: 16.5

Split decision:
- Keep cohesive. This spec only creates the tracker and gate for the known
  unsupported rows.

### Candidate Spec: Implicit Composition Route

Discovery purpose:
- Implement implicit-preserving and implicit-promoted CSG routes for rows where
  an implicit result is the correct surface-native representation.

Responsibilities:
- Functions/methods: field adapter builder, field composition builder, safety
  validator
- Data structures/models: implicit expression graph, bounded field domain,
  field provenance record
- Dependencies/services: implicit payload builder, CSG route registry,
  `.impress` implicit codec
- Returns/outputs/signals: implicit result patch, unsafe-field diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend implicit family and CSG route modules
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates new CSG result payloads
- Security/privacy-sensitive behavior: validates untrusted field payloads before
  composition
- Performance-sensitive behavior: bounded evaluation domains and sample budgets
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`,
  `src/impression/modeling/surface.py`, `src/impression/io/impress.py`
- Chosen defaults / parameters: hard Boolean composition first; soft blending is
  an explicit authored option
- Test strategy: implicit/implicit, analytic/implicit, sampled/implicit,
  unsafe-field refusal, `.impress` round trip
- Data ownership: implicit payload owns field expression; CSG owns provenance
- Routes: CSG planner to implicit route to result body
- Open questions / nuance discovered: exact field adapters for arbitrary
  higher-order operands may require declared-tolerance evaluator records
- Readiness blockers: implicit payload safety verifier

Score:
- Functions/methods: 3 x 2 = 6
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
- Total: 32.5

Split decision:
- Split required into field expression graph, operand field adapters, safety
  validation, composition operation semantics, persistence, and fixtures.

### Candidate Spec: Heightmap CSG Route

Discovery purpose:
- Implement heightmap-preserving CSG for 2.5D representable rows and route
  non-representable rows to promotion or representation refusal.

Responsibilities:
- Functions/methods: projection-domain checker, grid resampler, height
  composition operator, overhang detector
- Data structures/models: heightmap CSG payload, grid alignment record,
  heightmap representability diagnostic
- Dependencies/services: heightmap payload builder, CSG route registry,
  `.impress` heightmap codec
- Returns/outputs/signals: heightmap result patch, promotion/refusal decision
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend heightmap family and CSG route modules
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates new CSG result payloads
- Security/privacy-sensitive behavior: validates external grid/source payloads
- Performance-sensitive behavior: bounded resampling and grid-size budgets
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`,
  `src/impression/modeling/surface.py`, `src/impression/io/impress.py`
- Chosen defaults / parameters: non-2.5D output refuses unless a promotion route
  is declared
- Test strategy: aligned grids, resampled grids, overhang refusal, promotion,
  `.impress` round trip
- Data ownership: heightmap payload owns grid data; CSG owns route decision
- Routes: planner to heightmap route to result body or promotion route
- Open questions / nuance discovered: default promotion target for overhangs
- Readiness blockers: result-family promotion policy

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
- Split required into projection/grid alignment, height composition,
  representability/refusal, promotion integration, persistence, and fixtures.

### Candidate Spec: Displacement CSG Route

Discovery purpose:
- Implement displacement-preserving CSG where source identity and domain rules
  allow the result to remain a displacement surface.

Responsibilities:
- Functions/methods: source identity resolver, source-domain overlap checker,
  displacement resampler, offset composition operator
- Data structures/models: displacement CSG payload, source identity record,
  source mismatch diagnostic
- Dependencies/services: displacement payload builder, CSG route registry,
  `.impress` displacement codec
- Returns/outputs/signals: displacement result patch, promotion/refusal decision
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend displacement family and CSG route modules
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates new CSG result payloads
- Security/privacy-sensitive behavior: rejects unsafe external or cross-body
  source references
- Performance-sensitive behavior: bounded resampling and source-domain checks
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`,
  `src/impression/modeling/surface.py`, `src/impression/io/impress.py`
- Chosen defaults / parameters: source mismatch refuses unless promotion is
  declared
- Test strategy: same-source composition, source mismatch refusal, cross-domain
  promotion, `.impress` round trip
- Data ownership: displacement payload owns source reference; CSG owns route
  decision
- Routes: planner to displacement route to result body or promotion route
- Open questions / nuance discovered: how much source-topology change can remain
  displacement
- Readiness blockers: source identity resolver and promotion policy

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
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 2 x 2 = 4
- Total: 35.5

Split decision:
- Split required into source identity, domain/resampling, offset composition,
  source mismatch refusal, promotion integration, persistence, and fixtures.

### Candidate Spec: Sampled Implicit Result Promotion Policy

Discovery purpose:
- Define and implement promotion from sampled or implicit-involving rows into
  implicit, subdivision, NURBS, or B-spline result families.

Responsibilities:
- Functions/methods: promotion selector, lossiness recorder, provenance builder
- Data structures/models: promotion policy row, promotion provenance record,
  lossiness metadata record
- Dependencies/services: CSG route registry, target family payload codecs,
  reference evidence gate
- Returns/outputs/signals: promoted result patch, promotion diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG policy matrix and `.impress` payload dispatch
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: changes result family for supported operations
- Security/privacy-sensitive behavior: refuses unsafe payload promotion
- Performance-sensitive behavior: bounded reconstruction and sample budgets
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`,
  `src/impression/io/impress.py`
- Chosen defaults / parameters: implicit for volumetric predicate preservation;
  subdivision for bounded reconstructed surfaces
- Test strategy: promotion matrix, lossiness metadata, source provenance,
  round-trip, no mesh fallback
- Data ownership: CSG owns promotion decision; target family owns payload
- Routes: sampled/implicit route to promotion selector to target payload
- Open questions / nuance discovered: exact NURBS/B-spline promotion criteria
- Readiness blockers: target family reconstruction criteria

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
- Test scenarios/fixtures: 5 x 1 = 5
- Readiness blockers: 1 x 2 = 2
- Total: 31.5

Split decision:
- Split required into promotion matrix, provenance/lossiness records,
  target-family reconstruction criteria, persistence, and fixture evidence.

### Candidate Spec: Representation Refusal And Non-CSG Replacement Contract

Discovery purpose:
- Make impossible sampled/implicit boolean outcomes supported explicit refusals
  or deliberate non-CSG replacement workflows instead of raw unsupported rows.

Responsibilities:
- Functions/methods: representability failure classifier, replacement workflow
  suggester, refusal diagnostic builder
- Data structures/models: representation refusal record, non-CSG replacement
  record, route diagnostic
- Dependencies/services: CSG planner, family representability checks,
  no-mesh-fallback evidence
- Returns/outputs/signals: supported refusal result, replacement workflow hint
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG diagnostics and operation planner records
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: refuses unsafe payloads before execution
- Performance-sensitive behavior: bounded diagnostic generation
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: missing solver code is not a representation
  refusal
- Test strategy: impossible heightmap overhang, displacement source mismatch,
  unsafe implicit field, deliberate non-CSG replacement
- Data ownership: CSG owns refusal result; family modules own failure facts
- Routes: representability check to supported refusal route
- Open questions / nuance discovered: replacement workflow vocabulary
- Readiness blockers: none

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
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 0 x 2 = 0
- Total: 22.5

Split decision:
- Review for split. Keep cohesive because this spec owns the refusal contract
  shared by all sampled/implicit routes.

### Candidate Spec: Sampled Implicit CSG Persistence And Reference Evidence

Discovery purpose:
- Add `.impress` and reference-evidence coverage for native, promoted, and
  refusal outcomes created by sampled/implicit CSG routes.

Responsibilities:
- Functions/methods: payload round-trip verifier, reference artifact promoter,
  dirty evidence detector
- Data structures/models: sampled/implicit CSG persistence record, reference
  evidence row, no-mesh-fallback proof record
- Dependencies/services: `.impress` codecs, reference artifact lifecycle, CSG
  result records
- Returns/outputs/signals: round-trip result, evidence gate report
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing `.impress` and reference evidence helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: writes reference evidence artifacts
- Security/privacy-sensitive behavior: refuses unsafe serialized payloads
- Performance-sensitive behavior: fixture generation budget
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/io/impress.py`,
  `src/impression/modeling/csg.py`
- Chosen defaults / parameters: dirty evidence never counts as completion
- Test strategy: native route round trips, promoted route round trips, refusal
  serialization, no mesh fallback evidence
- Data ownership: `.impress` owns serialization; CSG owns semantic payload
- Routes: CSG result to persistence codec to reference evidence gate
- Open questions / nuance discovered: fixture naming convention for 153 rows
- Readiness blockers: route payload records must exist first

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
- Performance-sensitive behavior: 1 x 1 = 1
- Cross-screen reusable behavior: 0 x 1 = 0
- Test scenarios/fixtures: 4 x 1 = 4
- Readiness blockers: 1 x 2 = 2
- Total: 27.5

Split decision:
- Split required into `.impress` codec coverage, reference fixture promotion,
  no-mesh-fallback evidence, and dirty-evidence rejection.

## Change History

- 2026-05-30: Created the unsupported-row implementation architecture and
  marked all 153 sampled/implicit CSG unsupported rows as architecture
  `In Progress`.
