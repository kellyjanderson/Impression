---
name: test-specifications-core
description: Create one durable paired test specification for each final feature leaf and define manual smoke, automated smoke, and automated acceptance expectations.
---

# Test Specifications Core

Test specifications define how a final feature leaf should be verified manually and automatically.

## Core Rule

There should be one test specification for each final feature leaf specification whose acceptance describes a durable feature contract.

Low-level support leaves do not automatically need standalone test specifications unless they are being treated as first-class feature leaves.

## Timing

When a feature specification is marked `final`, its test specification should be created in the same refinement pass.

Implementation should not start with feature behavior fully specified but verification shape still unwritten.

## Recommended Structure

Test specifications should usually include:

* overview
* backlink to the feature specification
* manual smoke check
* automated smoke tests
* automated acceptance tests
* optional implementation-facing notes

## Verification Emphasis

Manual guidance may stay light.

Automated guidance should carry more detail, especially around:

* smoke tests that fail quickly and clearly
* acceptance tests that prove the feature contract
* stable fixtures
* explicit regression cases

## Relationship To Code Tests

Test specifications do not replace automated tests in the codebase.

They define:

* what coverage should exist
* what manual fallback check should exist
* what fixtures, doubles, and observable outcomes matter
