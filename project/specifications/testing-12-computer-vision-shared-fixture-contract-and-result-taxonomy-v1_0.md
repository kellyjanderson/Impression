# Testing Spec 12: Computer Vision Shared Fixture Contract and Result Taxonomy (v1.0)

## Overview

This specification defines the shared declarative fixture contract and shared
result-taxonomy posture for CV-backed verification lanes.

## Backlink

- [Testing Spec 02: Computer Vision Shared Fixture Contract and Harness Products (v1.0)](testing-02-computer-vision-shared-fixture-contract-and-harness-products-v1_0.md)

## Scope

This specification covers:

- minimum shared fixture fields
- authoritative lane declaration
- expected result classes and pass/fail mapping
- shared positive, transformed, different, and unknown outcome posture

## Behavior

This leaf must define:

- which fields every CV-backed fixture must declare
- how lane-specific classes map into a shared result pattern
- how fixtures declare whether transformed outcomes are acceptable
- how ambiguity, unreadable, or unknown classes stay explicit

## Constraints

- CV fixtures must declare their truth contract rather than relying on
  inference
- unknown or unreadable outcomes must not silently pass
- lane-specific extensions must not bypass the shared taxonomy posture

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- every CV-backed fixture can be described through one durable shared contract
- the shared result taxonomy and pass/fail mapping are explicit
- uncertainty posture is explicit
- verification requirements are defined by its paired test specification
