# Surface Spec 241: .impress Implicit Payload Security Refusal Fixtures (v1.0)

## Overview

Define refusal diagnostics and tests for executable code, dynamic imports,
unknown field nodes, and over-budget implicit payloads in `.impress`.

## Backlink

- [Architecture: Full Surface Patch Family Architecture](../architecture/full-surface-patch-family-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Implicit Payload
Security Refusal Fixtures` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - executable-payload refusal
  - unknown-node refusal
  - field-tree budget refusal
- Data structures/models:
  - security diagnostic
  - unsafe implicit payload fixture
- Dependencies/services:
  - `.impress` reader
  - implicit field validator
- Returns/outputs/signals:
  - unsafe payload refusal
  - deterministic security diagnostic
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: implicit field validator once implemented
  - Additions to existing reusable library/module: `.impress` refusal fixture
    suite
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - reads invalid `.impress` fixture files
- Security/privacy-sensitive behavior:
  - rejects executable code, dynamic imports, unknown field nodes, and
    over-budget field trees
- Performance-sensitive behavior:
  - refusal path bounds field tree size before evaluation
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- future `.impress` reader and implicit payload tests

Routes:

- file reader to implicit field validator to refusal diagnostic

Reuse/extraction decision:

- reuse the reader/load-result refusal contract

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- unsafe implicit payloads refuse before runtime evaluation

Data ownership:

- reader owns security diagnostics
- evaluator never sees unsafe payloads

## Behavior

The implementation must:

- reject executable code strings, dynamic import nodes, unknown field node
  kinds, and over-budget field trees
- reject unsafe payloads before any runtime field evaluation
- report deterministic diagnostics including payload path, reason, and
  allow-list context
- prove the refusal path does not recover by mesh fallback or best-effort
  mutation

## Verification

Test strategy:

- executable-code fixture tests
- dynamic-import fixture tests
- unknown-node fixture tests
- over-budget fixture tests
- deterministic diagnostic snapshot tests

## Manifest Assessment

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
- Total: 21.5

Split decision:

- Review for split. Cohesion reason: this is one security refusal fixture suite
  after safe payload codec work was split out.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- unsafe implicit payload fixtures refuse before evaluation
- refusal diagnostics are deterministic and path-specific
- no unsafe payload is recovered as mesh or degraded implicit data
