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

## Manifest Review History

- 2026-05-30 loop 1: Critical review found that implicit composition, heightmap CSG, displacement CSG, promotion, and persistence/evidence were too broad. Split candidates scoring 25 or more.
- 2026-05-30 loop 2: Reviewed split leaves for hidden storage, safety, and route ownership. Split implicit work into expression graph, operand adapters, safety, operation semantics, persistence, and fixtures.
- 2026-05-30 loop 3: Reviewed sampled-family leaves for representability gaps. Split heightmap and displacement work into domain planning, operation semantics, refusal, promotion, persistence, and fixtures.
- 2026-05-30 loop 4: Reviewed promotion and evidence leaves for hidden work. Split promotion into matrix, provenance/lossiness, target reconstruction criteria, persistence, and no-mesh evidence.
- 2026-05-30 loop 5: Final review rescored every leaf and confirmed no manifest candidate remains at 25 or more; 16-24 candidates carry explicit cohesion rationale.

Final manifest result:

- Initial candidates reviewed: 7
- Final candidate specs: 29
- Split-trigger candidates resolved: 5
- Remaining candidates scoring 25 or more: 0
- Unsupported operation rows represented: 153

### Candidate Spec: Sampled Implicit Unsupported Row Tracker

Discovery purpose:
- Add a durable tracker for the 153 unsupported sampled/implicit CSG rows and fail completion when any row lacks a current route classification.

Responsibilities:
- Functions/methods: row inventory builder; unsupported-row status verifier
- Data structures/models: unsupported row tracking record; route status enum
- Dependencies/services: CSG support matrix; patch family capability matrix
- Returns/outputs/signals: unsupported row report; missing-route diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing CSG matrix reporting helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
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
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 0
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 16.5

Split decision:
- Keep cohesive. This leaf only creates the tracker and gate for the known unsupported rows.

### Candidate Spec: Implicit Field Expression Graph

Discovery purpose:
- Define the implicit field expression graph used by implicit-preserving and implicit-promoted sampled/implicit CSG routes.

Responsibilities:
- Functions/methods: field node constructor; expression normalization; domain binder
- Data structures/models: implicit field expression node; bounded field domain; composition provenance seed
- Dependencies/services: implicit payload builder; CSG route registry
- Returns/outputs/signals: normalized expression graph; invalid-expression diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend implicit family records and CSG route records
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/surface.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters: hard Boolean field expression nodes are canonical; soft blends are explicit authored options
- Test strategy: primitive expression nodes, nested composition, invalid domains, deterministic ids
- Data ownership: implicit payload owns field expression; CSG owns operation provenance
- Routes: CSG planner to expression graph builder
- Open questions / nuance discovered: blend-node parameter vocabulary
- Readiness blockers: implicit payload safety verifier

Score:
- Functions/methods: 4
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 20.5

Split decision:
- Review for split. Cohesive because this leaf only defines expression representation, not adapters or execution.

### Candidate Spec: Implicit Operand Field Adapters

Discovery purpose:
- Add operand-to-field adapters for analytic, spline, sweep, subdivision, heightmap, and displacement operands that can participate in implicit CSG routes.

Responsibilities:
- Functions/methods: adapter selector; analytic field adapter; sampled evaluator adapter
- Data structures/models: operand field adapter record; adapter residual record; adapter refusal diagnostic
- Dependencies/services: field expression graph; family evaluators; CSG route registry
- Returns/outputs/signals: field adapter payload; adapter refusal diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG route modules with reusable adapter records
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`, `src/impression/modeling/surface.py`
- Chosen defaults / parameters: exact analytic adapters are preferred; higher-order and sampled adapters are declared-tolerance with bounded domains
- Test strategy: analytic, spline, sweep, subdivision, heightmap, displacement adapter coverage plus unsupported adapter refusal
- Data ownership: source family owns evaluation facts; CSG owns adapter route selection
- Routes: operand family to adapter selector to implicit expression graph
- Open questions / nuance discovered: which higher-order operands can expose exact signed fields
- Readiness blockers: field expression graph

Score:
- Functions/methods: 4
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 2
- Total: 22.5

Split decision:
- Review for split. Cohesive because all adapters share one route contract and feed the same graph.

### Candidate Spec: Implicit Field Safety Validation

Discovery purpose:
- Validate implicit field payloads before sampled/implicit CSG composition so unsafe or unbounded fields refuse deterministically.

Responsibilities:
- Functions/methods: safety validator; bounded-domain checker; evaluation-budget checker
- Data structures/models: field safety report; unsafe-field diagnostic
- Dependencies/services: implicit payload builder; route planner
- Returns/outputs/signals: accepted safety report; unsafe refusal diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: reuse existing implicit safety records and add CSG route integration
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/surface.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters: unbounded or external executable fields refuse before composition
- Test strategy: safe field, missing bounds, unsafe external payload, budget overflow
- Data ownership: implicit payload owns safety facts; CSG consumes them before execution
- Routes: field graph to safety validator to composition route
- Open questions / nuance discovered: none
- Readiness blockers: none

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 1
- Total: 19.5

Split decision:
- Keep cohesive. This is the shared security/safety gate.

### Candidate Spec: Implicit Composition Operation Semantics

Discovery purpose:
- Implement hard Boolean field composition semantics for union, difference, and intersection implicit CSG results.

Responsibilities:
- Functions/methods: union composer; difference composer; intersection composer; residual annotator
- Data structures/models: composition operation record; operand sign policy; composition diagnostic
- Dependencies/services: field expression graph; safety validator; route registry
- Returns/outputs/signals: implicit result patch; composition diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG route modules with reusable composition helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: union uses min, intersection uses max, difference uses max(base, negated cutter)
- Test strategy: implicit/implicit and adapter/implicit operation fixtures, operand order checks, sign policy checks
- Data ownership: CSG owns operation semantics; implicit payload owns resulting graph
- Routes: planner to composer to SurfaceBody result
- Open questions / nuance discovered: soft Boolean composition is not default
- Readiness blockers: field graph and safety validator

Score:
- Functions/methods: 5
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 23.5

Split decision:
- Review for split. Cohesive because all three operations share one composition policy and result shape.

### Candidate Spec: Implicit CSG Impress Payload Persistence

Discovery purpose:
- Persist composed implicit CSG payloads and round-trip them through `.impress` without converting to mesh truth.

Responsibilities:
- Functions/methods: implicit CSG payload encoder; decoder; round-trip verifier
- Data structures/models: implicit composition payload record; operation provenance payload
- Dependencies/services: implicit field expression graph; `.impress` root codec
- Returns/outputs/signals: serialized payload; round-trip diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing `.impress` implicit codec dispatch
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/io/impress.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters: payload includes operation provenance and bounded domain
- Test strategy: round-trip composed fields, malformed payload refusal, version compatibility
- Data ownership: `.impress` owns serialization; CSG owns semantic payload
- Routes: CSG result to codec to restored SurfaceBody
- Open questions / nuance discovered: none
- Readiness blockers: composition payload record

Score:
- Functions/methods: 3
- Data structures/models: 2
- Dependencies/services: 2
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 2
- Readiness blockers: 0
- Total: 18.5

Split decision:
- Keep cohesive. This leaf only covers implicit persistence.

### Candidate Spec: Implicit CSG Fixture And Evidence Matrix

Discovery purpose:
- Add reference and diagnostic evidence for implicit-preserving and implicit-promoted CSG routes.

Responsibilities:
- Functions/methods: fixture enumerator; evidence collector; no-mesh assertion
- Data structures/models: implicit CSG fixture row; evidence report; diagnostic snapshot
- Dependencies/services: reference artifact lifecycle; implicit route records
- Returns/outputs/signals: clean evidence report; dirty evidence diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing reference evidence helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `tests/test_surface_csg.py`, `project/release-0.1.0a/reference-artifacts/`
- Chosen defaults / parameters: dirty artifacts never count as completion
- Test strategy: success, unsafe refusal, adapter refusal, persistence, no-mesh fallback fixtures
- Data ownership: tests own evidence; CSG owns route behavior
- Routes: fixture builder to route execution to evidence promotion
- Open questions / nuance discovered: fixture matrix size
- Readiness blockers: implicit route payloads

Score:
- Functions/methods: 3
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 2
- Total: 21.5

Split decision:
- Review for split. Cohesive because this leaf only proves implicit route evidence.

### Candidate Spec: Heightmap Projection And Grid Alignment

Discovery purpose:
- Define projection-plane agreement, XY-domain overlap, clipping, and grid alignment for heightmap-preserving CSG.

Responsibilities:
- Functions/methods: projection checker; domain overlap checker; grid alignment planner
- Data structures/models: projection domain record; grid alignment record; clipping record
- Dependencies/services: heightmap payload builder; CSG route registry
- Returns/outputs/signals: aligned grid plan; projection refusal diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend heightmap family helpers and CSG route modules
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`, `src/impression/modeling/surface.py`
- Chosen defaults / parameters: nonmatching projection frames refuse unless a promotion route is declared
- Test strategy: aligned grids, resampled grids, disjoint domains, projection mismatch
- Data ownership: heightmap owns grid facts; CSG owns route decision
- Routes: planner to projection checker to grid plan
- Open questions / nuance discovered: default resampling kernel
- Readiness blockers: none

Score:
- Functions/methods: 6
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 22.5

Split decision:
- Review for split. Cohesive because this leaf produces the grid plan consumed by height composition.

### Candidate Spec: Heightmap Composition Operators

Discovery purpose:
- Implement heightmap-preserving union, difference, and intersection operators over an aligned output grid.

Responsibilities:
- Functions/methods: height union operator; height intersection operator; height difference operator; operation diagnostic builder
- Data structures/models: height composition record; sampled operation metadata
- Dependencies/services: grid alignment plan; heightmap payload builder
- Returns/outputs/signals: heightmap result patch; operation diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG route module with reusable height operators
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: operators only run after projection/grid alignment succeeds
- Test strategy: same-grid, resampled-grid, operand-order, boundary clipping fixtures
- Data ownership: CSG owns operation semantics; heightmap owns payload values
- Routes: grid plan to height operator to result patch
- Open questions / nuance discovered: difference semantics for non-solid height layers
- Readiness blockers: grid alignment leaf

Score:
- Functions/methods: 5
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 23.5

Split decision:
- Review for split. Cohesive because all operators share the same aligned-grid execution shape.

### Candidate Spec: Heightmap Representability And Refusal

Discovery purpose:
- Detect overhangs, multi-valued results, invalid projection states, and other non-2.5D outcomes before heightmap CSG executes as heightmap.

Responsibilities:
- Functions/methods: representability checker; overhang detector; refusal diagnostic builder
- Data structures/models: heightmap representability report; overhang diagnostic
- Dependencies/services: grid alignment plan; route planner
- Returns/outputs/signals: representability verdict; supported refusal result
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG diagnostics and heightmap helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`, `src/impression/modeling/surface.py`
- Chosen defaults / parameters: non-2.5D output refuses unless promotion is declared
- Test strategy: overhang, multivalue, invalid projection, unsafe grid fixtures
- Data ownership: heightmap owns representability facts; CSG owns refusal route
- Routes: grid plan to representability check to operator or refusal
- Open questions / nuance discovered: none
- Readiness blockers: none

Score:
- Functions/methods: 4
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 20.5

Split decision:
- Keep cohesive. This is the heightmap refusal boundary.

### Candidate Spec: Heightmap Promotion Integration

Discovery purpose:
- Route heightmap CSG results that cannot remain heightmaps into declared promotion targets.

Responsibilities:
- Functions/methods: promotion trigger; promotion target selector; promotion diagnostic builder
- Data structures/models: heightmap promotion decision; promotion trigger record
- Dependencies/services: heightmap representability report; sampled implicit promotion policy
- Returns/outputs/signals: promotion decision; promoted route request
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG promotion policy helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: implicit is default for volumetric outcomes; subdivision is default for bounded reconstructed surfaces
- Test strategy: overhang promotion, topology-change promotion, no-route refusal
- Data ownership: CSG owns promotion; target family owns payload
- Routes: heightmap refusal boundary to promotion selector
- Open questions / nuance discovered: exact spline promotion criteria
- Readiness blockers: promotion policy leaf

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 20.5

Split decision:
- Review for split. Cohesive because this leaf only bridges heightmap refusal into shared promotion.

### Candidate Spec: Heightmap CSG Impress Payload Persistence

Discovery purpose:
- Persist heightmap CSG payloads with grid alignment, resampling, operation provenance, and lossiness metadata.

Responsibilities:
- Functions/methods: heightmap CSG payload encoder; decoder; round-trip verifier
- Data structures/models: heightmap CSG payload record; resampling metadata record
- Dependencies/services: heightmap composition record; `.impress` codec
- Returns/outputs/signals: serialized payload; round-trip diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing heightmap `.impress` codec dispatch
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/io/impress.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters: payload records projection frame and resampling metadata
- Test strategy: native heightmap round-trip, malformed payload refusal, version compatibility
- Data ownership: `.impress` owns serialization; CSG owns semantic payload
- Routes: heightmap result to codec to restored SurfaceBody
- Open questions / nuance discovered: none
- Readiness blockers: heightmap result payload

Score:
- Functions/methods: 3
- Data structures/models: 2
- Dependencies/services: 2
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 2
- Readiness blockers: 0
- Total: 18.5

Split decision:
- Keep cohesive. This leaf only covers heightmap persistence.

### Candidate Spec: Heightmap CSG Fixture And Evidence Matrix

Discovery purpose:
- Add reference and diagnostic evidence for heightmap-preserving, promoted, and refused CSG routes.

Responsibilities:
- Functions/methods: fixture enumerator; evidence collector; no-mesh assertion
- Data structures/models: heightmap CSG fixture row; evidence report
- Dependencies/services: reference artifact lifecycle; heightmap route records
- Returns/outputs/signals: clean evidence report; dirty evidence diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing reference evidence helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `tests/test_surface_csg.py`, `project/release-0.1.0a/reference-artifacts/`
- Chosen defaults / parameters: dirty artifacts never count as completion
- Test strategy: aligned-grid success, overhang refusal, promotion, persistence, no-mesh fixtures
- Data ownership: tests own evidence; CSG owns route behavior
- Routes: fixture builder to route execution to evidence promotion
- Open questions / nuance discovered: fixture naming for 51-row set
- Readiness blockers: heightmap route payloads

Score:
- Functions/methods: 3
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 2
- Total: 21.5

Split decision:
- Review for split. Cohesive because this leaf only proves heightmap evidence.

### Candidate Spec: Displacement Source Identity Resolution

Discovery purpose:
- Resolve and validate displacement source-surface identity for displacement-preserving CSG routes.

Responsibilities:
- Functions/methods: source resolver; source compatibility checker; source provenance normalizer
- Data structures/models: source identity record; source compatibility report
- Dependencies/services: displacement payload builder; surface identity records
- Returns/outputs/signals: source identity verdict; source mismatch diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: reuse displacement source resolver and extend CSG route integration
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`, `src/impression/modeling/surface.py`
- Chosen defaults / parameters: source mismatch refuses unless promotion is declared
- Test strategy: same-source, missing-source, cross-body source, transformed-source fixtures
- Data ownership: displacement owns source reference; CSG owns route decision
- Routes: planner to source resolver to domain checker
- Open questions / nuance discovered: source topology equivalence threshold
- Readiness blockers: none

Score:
- Functions/methods: 5
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 21.5

Split decision:
- Review for split. Cohesive because this leaf only establishes source identity.

### Candidate Spec: Displacement Domain And Sample Resampling

Discovery purpose:
- Compute source-domain overlap, clipping, sample alignment, and resampling for displacement-preserving CSG.

Responsibilities:
- Functions/methods: source-domain overlap checker; displacement resampler; frame validator
- Data structures/models: source-domain overlap record; displacement resampling record; tangent-frame diagnostic
- Dependencies/services: source identity record; displacement payload builder
- Returns/outputs/signals: aligned displacement sample plan; domain refusal diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend displacement helpers and CSG route modules
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`, `src/impression/modeling/surface.py`
- Chosen defaults / parameters: resampling is bounded and records lossiness
- Test strategy: same-domain, partial overlap, incompatible frame, resampling budget fixtures
- Data ownership: displacement owns sample values; CSG owns route plan
- Routes: source identity to domain plan to offset composition
- Open questions / nuance discovered: normal-frame mismatch policy
- Readiness blockers: source identity leaf

Score:
- Functions/methods: 6
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 22.5

Split decision:
- Review for split. Cohesive because this leaf produces the domain/sample plan.

### Candidate Spec: Displacement Offset Composition Operators

Discovery purpose:
- Implement displacement-preserving union, difference, and intersection offset composition over a compatible source domain.

Responsibilities:
- Functions/methods: offset union operator; offset difference operator; offset intersection operator; composition diagnostic builder
- Data structures/models: offset composition record; source-frame operation metadata
- Dependencies/services: domain/sample plan; displacement payload builder
- Returns/outputs/signals: displacement result patch; operation diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG route module with reusable displacement operators
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: operators run only after source/domain compatibility passes
- Test strategy: same-source composition, operand order, clipped-domain, budget fixtures
- Data ownership: CSG owns operation semantics; displacement owns payload values
- Routes: domain plan to offset operator to result patch
- Open questions / nuance discovered: offset semantics for non-solid layers
- Readiness blockers: domain/sample leaf

Score:
- Functions/methods: 5
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 23.5

Split decision:
- Review for split. Cohesive because all operators share the same source-domain execution shape.

### Candidate Spec: Displacement Source Mismatch Refusal

Discovery purpose:
- Return supported refusal diagnostics for displacement CSG cases that cannot preserve source identity or source-domain compatibility.

Responsibilities:
- Functions/methods: source mismatch classifier; refusal diagnostic builder
- Data structures/models: source mismatch refusal record; incompatible-domain diagnostic
- Dependencies/services: source identity resolver; domain resampling plan
- Returns/outputs/signals: supported refusal result; replacement hint
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG diagnostics and displacement helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: missing solver code is not source mismatch refusal
- Test strategy: cross-body, missing source, topology change, incompatible frame fixtures
- Data ownership: family modules own mismatch facts; CSG owns refusal result
- Routes: source/domain checks to supported refusal route
- Open questions / nuance discovered: none
- Readiness blockers: none

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 18.5

Split decision:
- Keep cohesive. This is the displacement refusal boundary.

### Candidate Spec: Displacement Promotion Integration

Discovery purpose:
- Route displacement CSG results that cannot preserve source identity into declared promotion targets.

Responsibilities:
- Functions/methods: promotion trigger; promotion target selector; promotion diagnostic builder
- Data structures/models: displacement promotion decision; source-detach trigger record
- Dependencies/services: source mismatch refusal; sampled implicit promotion policy
- Returns/outputs/signals: promotion decision; promoted route request
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG promotion policy helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: source-detaching results promote or refuse; they never silently detach
- Test strategy: source mismatch promotion, topology-change promotion, no-route refusal
- Data ownership: CSG owns promotion; target family owns payload
- Routes: displacement refusal boundary to promotion selector
- Open questions / nuance discovered: exact spline promotion criteria
- Readiness blockers: promotion policy leaf

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 20.5

Split decision:
- Review for split. Cohesive because this leaf only bridges displacement refusal into shared promotion.

### Candidate Spec: Displacement CSG Persistence And Fixtures

Discovery purpose:
- Persist displacement CSG payloads and add reference evidence for displacement-preserving, promoted, and refused routes.

Responsibilities:
- Functions/methods: payload encoder; decoder; fixture enumerator; no-mesh assertion
- Data structures/models: displacement CSG payload record; source provenance payload; evidence report
- Dependencies/services: displacement composition records; reference artifact lifecycle
- Returns/outputs/signals: round-trip result; evidence gate report
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend displacement `.impress` codec and reference evidence helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/io/impress.py`, `tests/test_surface_csg.py`, `project/release-0.1.0a/reference-artifacts/`
- Chosen defaults / parameters: dirty artifacts never count as completion
- Test strategy: same-source round-trip, source mismatch refusal, promotion, no-mesh fixtures
- Data ownership: `.impress` owns serialization; tests own evidence; CSG owns route behavior
- Routes: CSG result to codec and fixture evidence gate
- Open questions / nuance discovered: fixture naming for source variants
- Readiness blockers: route payload records

Score:
- Functions/methods: 4
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 2
- Total: 24.5

Split decision:
- Review for split. Kept as one leaf because displacement persistence and evidence share the same source-provenance payload and remain below split threshold.

### Candidate Spec: Sampled Implicit Promotion Matrix

Discovery purpose:
- Define the promotion matrix that maps sampled/implicit CSG route outcomes to implicit, subdivision, NURBS, B-spline, refusal, or non-CSG replacement.

Responsibilities:
- Functions/methods: promotion matrix builder; target selector; matrix verifier
- Data structures/models: promotion policy row; target family decision record
- Dependencies/services: CSG support matrix; patch family capability matrix
- Returns/outputs/signals: promotion decision; missing-target diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG policy matrix helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: implicit for volumetric predicate preservation; subdivision for bounded reconstructed surfaces
- Test strategy: matrix coverage for all 153 rows and target/refusal classes
- Data ownership: CSG owns promotion decision; target family owns payload
- Routes: sampled/implicit route to promotion selector
- Open questions / nuance discovered: NURBS/B-spline exactness thresholds
- Readiness blockers: none

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 20.5

Split decision:
- Review for split. Cohesive because this leaf only defines target selection.

### Candidate Spec: Promotion Provenance And Lossiness Records

Discovery purpose:
- Record source operands, route decisions, tolerances, sampling, and lossiness for promoted sampled/implicit CSG results.

Responsibilities:
- Functions/methods: provenance builder; lossiness recorder; tolerance normalizer
- Data structures/models: promotion provenance record; lossiness metadata record
- Dependencies/services: promotion matrix; route payload records
- Returns/outputs/signals: provenance payload; lossiness diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend surface metadata/provenance helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: none
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`, `src/impression/modeling/surface.py`
- Chosen defaults / parameters: every promoted result carries explicit source and lossiness metadata
- Test strategy: lossless, declared-tolerance, resampled, reconstructed, refused-metadata fixtures
- Data ownership: CSG owns route provenance; target family owns payload
- Routes: promotion selector to provenance builder to result patch
- Open questions / nuance discovered: metadata vocabulary for reconstruction types
- Readiness blockers: promotion matrix

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 0
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 18.5

Split decision:
- Keep cohesive. This leaf owns shared metadata only.

### Candidate Spec: Promotion Target Reconstruction Criteria

Discovery purpose:
- Define criteria for when sampled/implicit CSG outputs may promote to implicit, subdivision, NURBS, or B-spline result families.

Responsibilities:
- Functions/methods: target criteria evaluator; reconstruction budget checker; target refusal builder
- Data structures/models: target criteria record; reconstruction feasibility report
- Dependencies/services: promotion matrix; target family payload builders
- Returns/outputs/signals: target eligibility verdict; reconstruction refusal diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG promotion helpers and family payload checks
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: none
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`, `src/impression/modeling/surface.py`
- Chosen defaults / parameters: implicit and subdivision are default targets; spline targets require explicit criteria success
- Test strategy: implicit, subdivision, NURBS, B-spline eligibility and refusal fixtures
- Data ownership: target family owns payload criteria; CSG owns promotion selection
- Routes: promotion matrix to target criteria to route request
- Open questions / nuance discovered: spline fitting residual threshold
- Readiness blockers: promotion provenance records

Score:
- Functions/methods: 5
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 0
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 2
- Total: 23.5

Split decision:
- Review for split. Cohesive because this leaf only gates target eligibility.

### Candidate Spec: Promotion Persistence Coverage

Discovery purpose:
- Persist promoted sampled/implicit CSG results across `.impress` with source route, target family, provenance, and lossiness metadata.

Responsibilities:
- Functions/methods: promotion payload encoder; decoder; round-trip verifier
- Data structures/models: promotion persistence payload; target-family dispatch record
- Dependencies/services: target family codecs; provenance/lossiness records
- Returns/outputs/signals: round-trip result; payload refusal diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend `.impress` payload dispatch
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: none
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/io/impress.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters: promoted result serializes as target family plus promotion metadata
- Test strategy: implicit, subdivision, spline target round trips and malformed payload refusal
- Data ownership: `.impress` owns serialization; CSG owns semantic metadata
- Routes: promoted result to codec to restored SurfaceBody
- Open questions / nuance discovered: none
- Readiness blockers: target family payload records

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 0
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 18.5

Split decision:
- Keep cohesive. This leaf only covers promotion persistence.

### Candidate Spec: Promotion Fixture And No-Mesh Evidence

Discovery purpose:
- Add reference fixtures proving sampled/implicit promotion routes preserve surface truth and never use mesh source truth.

Responsibilities:
- Functions/methods: promotion fixture enumerator; no-mesh evidence collector; dirty evidence detector
- Data structures/models: promotion fixture row; no-mesh proof record; evidence report
- Dependencies/services: promotion routes; reference artifact lifecycle
- Returns/outputs/signals: clean evidence report; dirty evidence diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing evidence helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: none
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `tests/test_surface_csg.py`, `project/release-0.1.0a/reference-artifacts/`
- Chosen defaults / parameters: dirty artifacts never count as completion
- Test strategy: implicit, subdivision, spline promotion, refusal, no-mesh fixtures
- Data ownership: tests own evidence; CSG owns route behavior
- Routes: fixture builder to promotion route to evidence gate
- Open questions / nuance discovered: fixture count budget
- Readiness blockers: promotion persistence coverage

Score:
- Functions/methods: 3
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 0
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 2
- Total: 21.5

Split decision:
- Review for split. Cohesive because this leaf only proves promotion evidence.

### Candidate Spec: Representation Refusal And Non-CSG Replacement Contract

Discovery purpose:
- Make impossible sampled/implicit boolean outcomes supported explicit refusals or deliberate non-CSG replacement workflows instead of raw unsupported rows.

Responsibilities:
- Functions/methods: representability failure classifier; replacement workflow suggester; refusal diagnostic builder
- Data structures/models: representation refusal record; non-CSG replacement record; route diagnostic
- Dependencies/services: CSG planner; family representability checks; no-mesh-fallback evidence
- Returns/outputs/signals: supported refusal result; replacement workflow hint
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend CSG diagnostics and operation planner records
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`
- Chosen defaults / parameters: missing solver code is not a representation refusal
- Test strategy: heightmap overhang, displacement source mismatch, unsafe implicit field, deliberate non-CSG replacement
- Data ownership: CSG owns refusal result; family modules own failure facts
- Routes: representability check to supported refusal route
- Open questions / nuance discovered: replacement workflow vocabulary
- Readiness blockers: none

Score:
- Functions/methods: 6
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 22.5

Split decision:
- Review for split. Keep cohesive because this leaf owns the shared refusal contract.

### Candidate Spec: Sampled Implicit CSG Codec Coverage

Discovery purpose:
- Add `.impress` codec coverage for sampled/implicit CSG native, promoted, and refusal payload records.

Responsibilities:
- Functions/methods: payload encoder; decoder; codec coverage verifier
- Data structures/models: sampled/implicit CSG payload dispatch record; codec diagnostic
- Dependencies/services: implicit, heightmap, displacement, promotion, refusal payload records
- Returns/outputs/signals: round-trip result; unsupported payload diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing `.impress` codec dispatch
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: creates or updates surface-native CSG result payloads
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/io/impress.py`, `src/impression/modeling/csg.py`
- Chosen defaults / parameters: codec refuses unknown or unsafe route payloads
- Test strategy: native route, promoted route, refusal route, malformed payload round trips/refusals
- Data ownership: `.impress` owns serialization; route modules own semantics
- Routes: CSG result to payload codec to restored SurfaceBody
- Open questions / nuance discovered: none
- Readiness blockers: route payload records

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 20.5

Split decision:
- Review for split. Cohesive because this leaf owns shared codec dispatch.

### Candidate Spec: Sampled Implicit Reference Fixture Promotion

Discovery purpose:
- Promote clean reference fixtures for sampled/implicit CSG native, promoted, and refusal outcomes.

Responsibilities:
- Functions/methods: fixture promoter; fixture completeness verifier; fixture manifest updater
- Data structures/models: reference fixture row; fixture promotion report
- Dependencies/services: reference artifact lifecycle; route evidence reports
- Returns/outputs/signals: clean fixture set; missing fixture diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend reference artifact helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `tests/test_surface_csg.py`, `project/release-0.1.0a/reference-artifacts/`
- Chosen defaults / parameters: dirty or stale fixtures do not count as promoted
- Test strategy: native, promoted, refusal, unsafe, malformed fixture promotion tests
- Data ownership: tests own fixture evidence; CSG owns route behavior
- Routes: route execution to fixture promoter to evidence gate
- Open questions / nuance discovered: fixture naming convention for 153 rows
- Readiness blockers: route payload records

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 2
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 20.5

Split decision:
- Review for split. Cohesive because this leaf only promotes reference fixtures.

### Candidate Spec: Sampled Implicit No-Mesh-Fallback Evidence Gate

Discovery purpose:
- Add a no-hidden-mesh-fallback evidence gate for all sampled/implicit CSG routes and supported refusals.

Responsibilities:
- Functions/methods: no-mesh evidence collector; fallback detector; gate assertion
- Data structures/models: no-mesh proof record; fallback violation diagnostic
- Dependencies/services: CSG route registry; reference evidence helpers
- Returns/outputs/signals: pass/fail evidence gate; violation diagnostics
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend existing no-mesh fallback evidence helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `src/impression/modeling/csg.py`, `tests/test_surface_csg.py`
- Chosen defaults / parameters: sampling can create native payloads but cannot become mesh source truth
- Test strategy: native route, promotion route, refusal route, forbidden mesh attempt fixtures
- Data ownership: CSG owns route truth; tests own evidence
- Routes: route registry to evidence collector to completion gate
- Open questions / nuance discovered: none
- Readiness blockers: route registry rows

Score:
- Functions/methods: 3
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 2
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 19.5

Split decision:
- Keep cohesive. This is the shared no-mesh gate.

### Candidate Spec: Sampled Implicit Dirty Evidence Rejection

Discovery purpose:
- Reject dirty, stale, diagnostic-only, or under-evidenced sampled/implicit CSG artifacts from completion evidence.

Responsibilities:
- Functions/methods: dirty evidence detector; evidence state classifier; completion blocker builder
- Data structures/models: dirty evidence diagnostic; evidence state record
- Dependencies/services: reference artifact lifecycle; codec coverage; no-mesh evidence gate
- Returns/outputs/signals: dirty evidence report; completion blocker diagnostic
- UI surfaces/components: none
- UI fields/elements: none
- Reusable code plan: extend reference evidence helpers
- Database queries/tables/migrations: none
- Async/concurrency behavior: none
- Destructive/write behavior: none
- Security/privacy-sensitive behavior: validates unsafe or untrusted sampled/implicit payload states before execution
- Performance-sensitive behavior: bounded matrix, route, sampling, or fixture work
- Cross-screen reusable behavior: not applicable

Project readiness fields:
- Implementation owner/module: `tests/test_surface_csg.py`, `project/release-0.1.0a/reference-artifacts/`
- Chosen defaults / parameters: dirty evidence never counts as completion
- Test strategy: dirty, stale, missing, diagnostic-only, clean evidence fixtures
- Data ownership: tests own evidence state; architecture tracker owns completion posture
- Routes: artifact inventory to dirty detector to completion gate
- Open questions / nuance discovered: none
- Readiness blockers: reference fixture promotion

Score:
- Functions/methods: 2
- Data structures/models: 3
- Dependencies/services: 3
- Returns/outputs/signals: 2
- UI surfaces/components: 0
- UI fields/elements: 0
- Existing reusable code reused as-is: 0.5
- Adding code to an existing library/module: 2
- Creating a new reusable library/module: 0
- Database queries/tables/migrations: 0
- Async/concurrency behavior: 0
- Destructive/write behavior: 0
- Security/privacy-sensitive behavior: 1
- Performance-sensitive behavior: 1
- Cross-screen reusable behavior: 0
- Test scenarios/fixtures: 3
- Readiness blockers: 0
- Total: 17.5

Split decision:
- Keep cohesive. This leaf only rejects bad evidence.

## Change History

- 2026-05-30: Critically reviewed the specification manifest through five review/rescore/split loops and promoted the unsupported-row architecture into 29 final manifest leaves. Reason: broad sampled/implicit candidates hid route, representability, promotion, persistence, and evidence work.
- 2026-05-30: Created the unsupported-row implementation architecture and marked all 153 sampled/implicit CSG unsupported rows as architecture `In Progress`.
