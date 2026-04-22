# Testing Spec 13: Computer Vision Shared Harness Pipeline and Artifact Bundle Integration (v1.0)

## Overview

This specification defines the shared CV harness pipeline and how its artifact
bundles integrate with the reference-artifact lifecycle.

## Backlink

- [Testing Spec 02: Computer Vision Shared Fixture Contract and Harness Products (v1.0)](testing-02-computer-vision-shared-fixture-contract-and-harness-products-v1_0.md)

## Scope

This specification covers:

- shared harness layers such as builder, renderer, normalization,
  interpretation, and review publication
- deterministic artifact bundles emitted by a lane
- grouped completeness behavior for CV bundle products
- integration with top-level reference-artifact lifecycle rules

## Behavior

This leaf must define:

- the shared harness stages and their boundaries
- how a lane publishes deterministic artifact bundles for review
- when missing bundle members fail clearly instead of degrading silently
- how changed bundle meaning participates in baseline invalidation

## Constraints

- CV-backed verification must operate on deterministic harness products
- grouped bundle completeness must stay explicit
- shared harness plumbing must remain separate from lane-specific semantics

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the shared harness stages are explicit
- grouped bundle publication and completeness rules are explicit
- integration with reference-artifact lifecycle rules is explicit
- verification requirements are defined by its paired test specification
