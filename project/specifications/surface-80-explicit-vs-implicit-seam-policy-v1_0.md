# Surface Spec 80: Explicit Versus Implicit Seam Representation Policy (v1.0)

## Overview

This specification defines whether seams exist as first-class kernel objects or
are implicit relationships derived from adjacency records.

## Backlink

Parent specification:

- [Surface Spec 27: Seam Identity and Ownership Policy (v1.0)](surface-27-seam-identity-ownership-policy-v1_0.md)

## Scope

This specification covers:

- seam object policy
- implicit seam alternative
- downstream consequences of the chosen policy

## Behavior

This branch must define:

- seams are explicit first-class kernel objects in v1
- implicit seam derivation is not the source of truth for v1
- downstream tessellation, adjacency, and watertightness work may depend on the
  explicit seam model

## Constraints

- the policy choice must be explicit
- downstream implications must be explicit
- the chosen policy must support required tessellation and continuity work

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- explicit versus implicit policy is explicit
- downstream consequences are explicit
- v1 support rationale is explicit

## Current Preferred Answer

For v1:

- seams are explicit kernel records
- patches participate through explicit oriented boundary-use records
- derived adjacency views may exist, but they must not replace seam truth
