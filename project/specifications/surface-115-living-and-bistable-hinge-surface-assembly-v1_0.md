# Surface Spec 115: Living and Bistable Hinge Surface Assembly (v1.0)

## Overview

This specification defines surfaced living and bistable hinge generation.

## Backlink

- [Surface Spec 104: Surface-Native Hinge Replacement (v1.0)](surface-104-surface-native-hinge-replacement-v1_0.md)

## Scope

This specification covers:

- living hinge panels
- bistable hinge blanks
- surfaced cut/slot or flex-pattern assembly behavior

## Behavior

This branch must define:

- how living and bistable hinge geometry is expressed without mesh-first truth
- what surfaced output forms are returned
- how pattern and flex-region behavior remains deterministic

## Constraints

- flexible hinge generation must remain canonical before tessellation
- pattern semantics must be explicit enough for regression coverage
- public docs must include representative surfaced examples

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- living and bistable hinge builders have surfaced contracts
- representative fixtures are covered by verification
- documentation requirements are explicit

