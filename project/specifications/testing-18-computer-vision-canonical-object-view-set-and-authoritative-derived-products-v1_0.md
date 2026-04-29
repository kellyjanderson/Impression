# Testing Spec 18: Computer Vision Canonical Object-View Set and Authoritative Derived Products (v1.0)

## Overview

This specification defines the canonical object-view set and authoritative
derived product contract for object-view CV verification.

## Backlink

- [Testing Spec 06: Computer Vision Canonical Object-View Render Products and View-Space Verification (v1.0)](testing-06-computer-vision-canonical-object-view-render-products-and-view-space-verification-v1_0.md)

## Scope

This specification covers:

- the required canonical views
- stable view naming and ordering
- authoritative derived products such as silhouette, depth, or normals
- the diagnostic-only role of optional beauty renders

## Behavior

This leaf must define:

- which canonical views belong to the initial object-view lane
- which products are authoritative and which are diagnostic
- how view bundles are named and ordered reproducibly
- how object-view bundles stay comparable across repeated runs

## Constraints

- the object-view set must be declared rather than inferred
- authoritative products must stay distinct from beauty renders
- product naming and ordering must remain stable

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the initial canonical view set is explicit
- authoritative versus diagnostic products are explicit
- naming and ordering rules are explicit
- verification requirements are defined by its paired test specification
