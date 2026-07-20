# Surface Spec 159: Mesh Execution Inventory And Classification (v1.0)

## Overview

Create the authoritative inventory of mesh-producing code paths before writing migration specs that might otherwise miss a hidden executor.

## Backlink

- [Architecture: Mesh Execution Tessellation Boundary Architecture](../architecture/mesh-execution-tessellation-boundary-architecture.md)

## Scope

This specification promotes the manifest candidate `Mesh Execution Inventory And Classification` into a final implementation leaf. It covers the behavior, data ownership, reuse boundary, and verification expectations needed to implement this leaf without returning to the architecture document for hidden decisions.

## Implementation Artifact

- [Surface Spec 159 Implementation: Mesh Execution Inventory And Classification Report](surface-159-mesh-execution-inventory-report-v1_0.md)

## Responsibilities

- Functions/methods:
  - mesh-producing public API scanner
  - mesh-producing private helper scanner
  - classification report generator
- Data structures/models:
  - mesh path classification record
  - owner spec reference record
- Dependencies/services:
  - `rg`/static source search
  - existing modeling modules
- Returns/outputs/signals:
  - durable inventory table
  - per-symbol classification
  - owner/migration target
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: source tree and architecture docs
  - Additions to existing reusable library/module: none
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes documentation only
- Security/privacy-sensitive behavior:
  - none
- Performance-sensitive behavior:
  - bounded repository source scan
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `project/release-0.1.0a/specifications/`

Routes:

- architecture document to implementation specs

Reuse/extraction decision:

- reuse existing architecture tracker and source inventory; no code extraction

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- classify paths as `tessellation-boundary`, `mesh-consumer`,
    `legacy-compatibility`, `explicit-mesh-tool`, or `invalid-modeling-fallback`

Data ownership:

- architecture tracker owns the source list until final specs are written

## Behavior

The implementation must:

- satisfy every function, data-structure, dependency, and output responsibility listed above
- preserve the architecture boundary named in the backlink
- reject unsupported or ambiguous states with explicit diagnostics rather than silent fallback behavior
- keep mesh data outside canonical authored-surface state unless this spec explicitly names a tessellation, compatibility, or mesh-utility boundary
- expose only the public API surface needed by downstream specs and tests

## Constraints

- The implementation must remain deterministic for equivalent inputs.
- The implementation must keep metadata and stable identity behavior explicit when the leaf touches persisted or reusable surface state.
- The implementation must not introduce hidden mesh execution in authored modeling paths.
- The implementation must not broaden industry interchange, patch-family, or mesh compatibility scope beyond what this leaf names.

## Verification

Test strategy:

- documentation review plus static source-search reproducibility notes

Automated or review verification must prove that the implementation satisfies the manifest responsibilities and that failure modes are explicit. Where this spec touches serialization, verification must include round-trip or refusal coverage. Where this spec touches modeling output, verification must include surface-native output checks and no-hidden-mesh-fallback checks.

## Manifest Assessment

Score:

- Functions/methods: 3 x 2 = 6
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 2 x 1 = 2
- Returns/outputs/signals: 3 x 1 = 3
- UI surfaces/components: 0 x 2 = 0
- UI fields/elements: 0 x 1 = 0
- Existing reusable code reused as-is: 2 x 0.5 = 1
- Adding code to an existing library/module: 0 x 1 = 0
- Creating a new reusable library/module: 0 x 3 = 0
- Database queries/tables/migrations: 0 x 2 = 0
- Async/concurrency behavior: 0 x 3 = 0
- Destructive/write behavior: 1 x 3 = 3
- Security/privacy-sensitive behavior: 0 x 3 = 0
- Performance-sensitive behavior: 1 x 2 = 2
- Cross-screen reusable behavior: 0 x 2 = 0
- Readiness blockers: 0 x 2 = 0
- Total: 19

Split decision:

- Review for split.
- Cohesion reason: this is one discovery/inventory artifact; splitting before
  the first inventory would create disconnected partial truth.

Open questions / nuance resolved for implementation:

- Inventory must identify symbols, not only modules, or later specs will stay
  too vague.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- all manifest responsibilities are implemented or explicitly refused by the leaf
- owner/module, routes, data ownership, reuse, UI inventory, defaults, and test strategy are represented in code or verification artifacts
- related progression items can be checked without relying on unstated architecture assumptions
- downstream specs can cite this leaf instead of re-reading the manifest candidate
