# Feature Spec 05A: Reduced Progression Result Contract for Control-Station Inference (v1.0)

## Overview

This specification defines the durable output contract for control-station
inference.

## Backlink

- [Feature Spec 05: Control-Station Inference Program (v1.0)](feature-05-control-station-inference-program-v1_0.md)

## Scope

This specification covers:

- reduced progression result structure
- retained station-class recording and diagnostic association

## Behavior

This leaf must define:

- the child leaf that owns reduced progression result bundle shape and replay
  contract
- the child leaf that owns retained station-class recording and diagnostic
  association

## Constraints

- the result must remain inspectable and replayable
- retained topology truth must stay explicit

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Feature Spec 05A1: Reduced Progression Bundle Shape and Replay Contract](feature-05a1-reduced-progression-bundle-shape-and-replay-contract-v1_0.md)
- [Feature Spec 05A2: Retained Station Classification and Diagnostic Association Contract](feature-05a2-retained-station-classification-and-diagnostic-association-contract-v1_0.md)

## Acceptance

This specification is complete when:

- reduced progression bundle shape is explicit
- retained station-class recording and diagnostic association are explicit
