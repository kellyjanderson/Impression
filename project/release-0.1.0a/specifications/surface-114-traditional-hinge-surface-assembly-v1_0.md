# Surface Spec 114: Traditional Hinge Surface Assembly (v1.0)

## Overview

This specification defines surfaced traditional hinge leaves and paired hinge assemblies.

## Backlink

- [Surface Spec 104: Surface-Native Hinge Replacement (v1.0)](surface-104-surface-native-hinge-replacement-v1_0.md)

## Scope

This specification covers:

- traditional hinge leaves
- paired barrel hinges
- surfaced pin and knuckle assembly behavior

## Behavior

This branch must define:

- how traditional hinge geometry is assembled from surfaced primitives and ops
- what public output form is returned for leaf and pair builders
- how opened/closed assembly states remain deterministic

## Constraints

- traditional hinge replacement must avoid mesh-first union/difference as primary truth
- assembly structure must remain explicit for preview/export and reference verification
- public hinge docs must include representative surfaced examples

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- traditional hinge builders have surfaced output contracts
- representative leaf and pair cases are covered by verification
- documentation requirements are explicit

