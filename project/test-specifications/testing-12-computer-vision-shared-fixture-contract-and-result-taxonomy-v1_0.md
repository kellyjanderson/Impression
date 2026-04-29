# Testing Spec 12 Test: Computer Vision Shared Fixture Contract and Result Taxonomy

## Overview

This test specification defines verification for the shared declarative
fixture contract and result-taxonomy posture used by CV-backed lanes.

## Backlink

- [Testing Spec 12: Computer Vision Shared Fixture Contract and Result Taxonomy (v1.0)](../specifications/testing-12-computer-vision-shared-fixture-contract-and-result-taxonomy-v1_0.md)

## Automated Smoke Tests

- representative CV fixtures declare the required shared fields
- declared lane classes map into the shared result-taxonomy structure

## Automated Acceptance Tests

- fixtures explicitly declare pass/fail handling for transformed outcomes
- unknown, ambiguous, or unreadable outcomes remain explicit
- lane-specific classes do not bypass the shared taxonomy posture
- missing shared contract fields fail clearly
