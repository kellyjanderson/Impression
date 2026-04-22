# Surface Spec 112: Surface Thread Convenience Builders (v1.0)

## Overview

This specification defines surfaced convenience builders for thread-derived modeling helpers.

## Backlink

- [Surface Spec 103: Surface-Native Threading Replacement (v1.0)](surface-103-surface-native-threading-replacement-v1_0.md)

## Scope

This specification covers:

- threaded rods
- nuts
- tapped-hole cutters
- runout relief helpers

## Behavior

This branch must define:

- how convenience thread builders terminate in surfaced outputs
- how helper composition avoids mesh-first assembly as canonical truth
- what body or collection forms are returned publicly

## Constraints

- convenience builders must compose from surfaced thread and primitive building blocks
- output semantics must remain explicit for preview/export and reference verification
- public usage docs must include representative examples

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- thread convenience builders have surfaced output contracts
- representative surfaced helpers are covered by verification
- documentation requirements are explicit

