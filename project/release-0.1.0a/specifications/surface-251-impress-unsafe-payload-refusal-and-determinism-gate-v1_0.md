# Surface Spec 251: .impress Unsafe Payload Refusal And Determinism Gate (v1.0)

## Overview

Prove malformed, unsupported, or unsafe `.impress` payloads refuse with
deterministic diagnostics and never recover by inventing mesh or executing
unsafe implicit payloads.

## Backlink

- [Architecture: Surface Body Completion Architecture](../architecture/surface-body-completion-architecture.md)

## Scope

This specification promotes the manifest candidate `.impress Unsafe Payload
Refusal And Determinism Gate` into a final implementation leaf.

## Responsibilities

- Functions/methods:
  - malformed payload refusal assertion
  - deterministic error reporter
- Data structures/models:
  - diagnostic metadata payload
  - invalid payload fixture record
- Dependencies/services:
  - `.impress` reader
  - implicit payload safety policy
  - deterministic writer
- Returns/outputs/signals:
  - invalid payload refusal diagnostic
  - deterministic error report
- UI surfaces/components:
  - not applicable
- UI fields/elements:
  - not applicable
- Reusable code plan:
  - Existing code reused as-is: reader/load result contract
  - Additions to existing reusable library/module: unsafe payload fixture set
  - New reusable library/module to create: none
- Database queries/tables/migrations:
  - none
- Async/concurrency behavior:
  - none
- Destructive/write behavior:
  - writes temporary invalid fixture files during tests
- Security/privacy-sensitive behavior:
  - rejects unsafe implicit and malformed payloads without recovery execution
- Performance-sensitive behavior:
  - deterministic error path bounded by payload size
- Cross-screen reusable behavior:
  - not applicable

## Implementation Boundary

Owner/module:

- `.impress` reader/load result modules and test fixtures

Routes:

- reader validation to load result to deterministic diagnostic

Reuse/extraction decision:

- extend reader/load-result contract tests

UI field/control inventory:

- not applicable

## Data And Defaults

Chosen defaults / parameters:

- invalid payloads refuse; no mesh recovery or best-effort mutation

Data ownership:

- `.impress` reader owns refusal diagnostics

## Behavior

The implementation must:

- refuse malformed payloads, unsupported families, unsafe implicit payloads, and
  unknown diagnostic metadata
- produce stable diagnostic ordering and stable diagnostic text keys
- prove refusal does not recover by mesh fallback or best-effort mutation
- keep unsafe payload fixtures separate from successful whole-store fixtures

## Verification

Test strategy:

- malformed payload tests
- unsafe implicit payload tests
- unsupported family payload tests
- deterministic error snapshot tests
- no mesh recovery assertions

## Manifest Assessment

Score:

- Functions/methods: 2 x 2 = 4
- Data structures/models: 2 x 1 = 2
- Dependencies/services: 3 x 1 = 3
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
- Total: 20.5

Split decision:

- Review for split. Cohesion reason: unsafe payload refusal and deterministic
  diagnostics are one reader/load-result gate.

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- unsafe payloads refuse before execution
- diagnostics are deterministic
- no invalid payload is recovered as mesh or mutated surface truth
