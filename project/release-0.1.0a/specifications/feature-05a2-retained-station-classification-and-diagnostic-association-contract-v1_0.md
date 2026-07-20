# Feature Spec 05A2: Retained Station Classification and Diagnostic Association Contract (v1.0)

## Overview

This specification defines how retained station classes and supporting
diagnostic associations are recorded for control-station inference results.

## Backlink

- [Feature Spec 05A: Reduced Progression Result Contract for Control-Station Inference (v1.0)](feature-05a-reduced-progression-result-contract-for-control-station-inference-v1_0.md)

## Scope

This specification covers:

- retained topology-station classification
- retained hidden control-station classification
- association of supporting diagnostics to retained station records

## Behavior

This leaf must define:

- how retained station classes are recorded
- how diagnostic references are associated with retained station records
- how retained structure remains inspectable after reduction

## Constraints

- retained topology truth must stay explicit
- diagnostic association must not blur topology stations and hidden control
  stations

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- retained station-class recording is explicit
- diagnostic association contract is explicit
- retained structure remains inspectable
