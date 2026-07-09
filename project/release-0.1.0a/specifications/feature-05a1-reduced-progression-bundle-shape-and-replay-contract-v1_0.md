# Feature Spec 05A1: Reduced Progression Bundle Shape and Replay Contract (v1.0)

## Overview

This specification defines the output bundle shape for a reduced progression
result produced by control-station inference.

## Backlink

- [Feature Spec 05A: Reduced Progression Result Contract for Control-Station Inference (v1.0)](feature-05a-reduced-progression-result-contract-for-control-station-inference-v1_0.md)

## Scope

This specification covers:

- reduced progression bundle structure
- replayable reduced progression payload shape
- provenance fields attached to the reduced progression bundle itself

## Behavior

This leaf must define:

- what the reduced progression bundle contains
- which fields are required for deterministic replay
- which provenance fields belong to the bundle regardless of later diagnostics

## Constraints

- the bundle must remain inspectable and replayable
- bundle shape must not depend on optional reporting layers

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- reduced progression bundle shape is explicit
- replay contract is explicit
- bundle-local provenance is explicit
