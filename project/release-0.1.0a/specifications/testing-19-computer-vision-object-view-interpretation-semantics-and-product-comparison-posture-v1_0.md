# Testing Spec 19: Computer Vision Object-View Interpretation Semantics and Product Comparison Posture (v1.0)

## Overview

This specification defines how the object-view lane interprets declared
authoritative products after camera and view bundles are already stable.

## Backlink

- [Testing Spec 06: Computer Vision Canonical Object-View Render Products and View-Space Verification (v1.0)](testing-06-computer-vision-canonical-object-view-render-products-and-view-space-verification-v1_0.md)

## Scope

This specification covers:

- the semantic role of each authoritative product type
- comparison posture across repeated runs
- what object-view interpretation may prove
- what remains diagnostic-only

## Behavior

This leaf must define:

- how interpretation operates on declared products instead of inferred camera
  state
- which product types carry primary truth in the initial lane
- how product disagreement is surfaced
- how diagnostic products remain available without becoming silent proof

## Constraints

- object-view semantics must not absorb camera/framing validity work
- product comparison posture must stay explicit
- beauty renders must remain diagnostic unless another contract delegates more

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the initial object-view comparison posture is explicit
- primary truth products are explicit
- diagnostic-only boundaries are explicit
- verification requirements are defined by its paired test specification
