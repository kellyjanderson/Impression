# Testing Architecture

## Overview

This document defines the top-level testing architecture for Impression.

Testing is a first-class system area.
It is not owned by any single feature trunk such as surface, loft, text, or
CSG.

The purpose of this architecture is to keep:

- test tooling
- verification harnesses
- baseline/reference lifecycle rules
- cross-feature diagnostic lanes
- test artifact publication and review behavior

in one top-level architecture branch.

## Scope

This architecture covers:

- reusable test tooling and facilitation infrastructure
- reference-image and reference-STL lifecycle architecture
- computer-vision verification architecture
- the boundary between authoritative proof lanes and diagnostic lanes
- the rule that feature trunks consume testing tools rather than owning testing
  tool structure

## Structural Rule

Testing tooling is its own architecture concern.

Feature branches may depend on testing tools and verification lanes, but they
should refer to them only as tools they use.

Feature trunks should not own:

- shared test harness structure
- reusable verification tool contracts
- cross-feature artifact lifecycle policy
- cross-feature CV lane structure

Those concerns belong in the testing architecture branch.

## Child Architecture Branches

The current major testing-architecture branches are:

- [Model Output Reference Verification](model-output-reference-verification.md)
- [Computer Vision Verification Architecture](computer-vision-verification-architecture.md)

These child architecture branches describe major testing subsystems.

Their implementation-facing specification work should refine the top-level
testing architecture/spec branch rather than being embedded inside unrelated
feature programs.

## Boundaries

Testing architecture owns:

- reusable verification contracts
- reusable tooling and facilitation work
- baseline and artifact lifecycle policy
- reusable diagnostic presentation lanes

Feature architecture owns:

- product and geometry behavior
- feature-specific correctness rules
- when and why a feature uses a testing tool

## Implications

- New testing tooling should refine from the testing architecture branch.
- Existing feature trunks may reference testing tools as dependencies or proof
  lanes, but should not replicate testing-tool structure.
- CV verification is a testing-architecture branch, not a feature branch.

