# Feature Spec 04A: Internal Control-Station Representation and Provenance (v1.0)

## Overview

This specification defines the internal representation for hidden control
stations.

## Backlink

- [Feature Spec 04: Non-User-Facing Control Stations Program (v1.0)](feature-04-non-user-facing-control-stations-program-v1_0.md)

## Scope

This specification covers:

- hidden control-station records
- provenance metadata
- relationship to topology stations

## Behavior

This leaf must define:

- what a hidden control station records
- how provenance is carried
- how the representation stays distinct from topology-station truth

## Constraints

- hidden control-station representation must remain planner-owned
- provenance must remain durable enough for later diagnostics

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- hidden control-station representation is explicit
- provenance is explicit
- distinction from topology stations is explicit
