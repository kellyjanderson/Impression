# Testing Spec 06: Computer Vision Canonical Object-View Render Products and View-Space Verification (v1.0)

## Overview

This specification defines the parent branch for canonical object-view render
tooling and view-space interpretation used by CV-backed verification.

## Backlink

- [Testing Spec 01: Testing Tooling and Verification Program (v1.0)](testing-01-testing-tooling-and-verification-program-v1_0.md)

## Scope

This specification covers:

- canonical object-view sets such as front, side, top, and isometric
- required naming and ordering for view products
- preferred derived products such as silhouette, depth, and normal images
- the tooling contract for view-space interpretation over declared object views

## Behavior

This branch must define:

- which canonical views are required for a fixture that opts into object-view
  verification
- which derived products are authoritative versus diagnostic
- the tooling boundary between canonical render-product generation and
  downstream view-space interpretation
- how view-space verification operates on declared products instead of inferred
  camera pose
- how object-view products remain comparable across repeated runs

## Constraints

- object-view verification must depend on a declared view set rather than ad
  hoc renders
- view naming and ordering must be stable
- beauty renders remain diagnostic unless a stricter proof contract is declared
- this lane must not absorb camera/framing validity responsibilities that
  belong to the camera-contract leaf
- this leaf defines top-level CV tooling and facilitation work rather than a
  feature-level model-output requirement

## Refinement Status

Decomposed into child leaves.

## Child Specifications

- [Testing Spec 18: Computer Vision Canonical Object-View Set and Authoritative Derived Products](testing-18-computer-vision-canonical-object-view-set-and-authoritative-derived-products-v1_0.md)
- [Testing Spec 19: Computer Vision Object-View Interpretation Semantics and Product Comparison Posture](testing-19-computer-vision-object-view-interpretation-semantics-and-product-comparison-posture-v1_0.md)

## Acceptance

This specification is complete when:

- object-view product definition and object-view interpretation semantics are
  separated into honest executable child leaves
- the parent remains a container rather than an executable implementation leaf
- verification requirements are pushed down into the paired child test
  specifications
