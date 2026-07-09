# Reference Review Spec 15: Fixture Discovery Integration (v1.0)

## Overview

Connect active reference fixture roots to source-record discovery without
loading models or relying on derived PNG/STL artifacts.

## Backlink

- [Reference Review Architecture](../architecture/reference-review-fixture-source-contract.md)

## Source Manifest

- This leaf is derived from the manifest candidate `Fixture Discovery Integration`.
- Manifest score: 21.5

## Scope

This specification covers:

- fixture root scanner
- fixture-to-source resolver
- discovery item
- discovery summary
- discovered fixture list
- skipped fixture diagnostic

## Behavior

This leaf must define:

- derived artifacts without source records are skipped with diagnostics
- reference roots to discovery items to queue model

## Constraints

- Async/concurrency behavior: discovery runs in bounded worker task
- Security/privacy-sensitive behavior: scan is restricted to configured
  reference roots
- Performance-sensitive behavior: incremental scan and bounded stat calls
- Cross-screen reusable behavior: discovery list feeds queue, release gate,
  and review reports

## Dependencies And Reuse

Dependencies/services:

- source record schema
- reference artifact root helpers

Reusable code plan:

- Existing code reused as-is: reference artifact root helpers
- Additions to existing reusable library/module: fixture discovery helper
- New reusable library/module to create: none

Implementation owner/module:

- future `source_registry.discovery`

## Data Ownership And Routes

Data ownership:

- discovery owns fixture list, not review decisions

Routes:

- reference roots to discovery items to queue model

## UI Contract

- Surface/component: none; queue rendering belongs to UI specs

## Test Strategy

- discovery tests for clean, dirty, missing source, duplicate fixture id,
  and non-reference-root fixtures

## Open Questions And Prerequisites

Open questions / nuance discovered:

- none

Prerequisites before implementation:

- none

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the Fixture Discovery Integration boundary is implemented as described
- the specified ownership, routing, and reuse boundaries are preserved
- the listed test strategy passes for this leaf
