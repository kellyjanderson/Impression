# Surface Spec 106: Reference Artifact Regression Suite (v1.0)

## Overview

This specification defines the durable reference-artifact suite used to verify
surface-body and loft outputs through rendered images and STL files.

The project-wide definition for this behavior is documented in:

- [Model Output Reference Verification](../architecture/model-output-reference-verification.md)

## Backlink

- [Surface Spec 99: Surface-Native Replacement Program (v1.0)](surface-99-surface-native-replacement-program-v1_0.md)

## Scope

This specification covers:

- reference images
- reference STL files
- dirty versus clean artifact lifecycle
- artifact generation and comparison rules in automated tests
- the completeness requirement that model-outputting capabilities must carry a
  reference-artifact test
- optional canonical slice and silhouette-classification checks for fixtures
  whose correctness is not well described by a beauty render alone

## Behavior

This branch must define:

- stable artifact locations under `project/`
- first-run dirty-reference bootstrap flow
- clean-reference preference and promotion behavior
- automated checks that prove images and STL files are non-empty and model-related
- subsequent-run comparison behavior against dirty or clean references
- reference invalidation behavior when an existing fixture's test contract
  changes
- the requirement that model-outputting capabilities are incomplete without at
  least one durable named reference fixture
- how a fixture may add canonical slice verification with silhouette
  relationship classes such as same-shape, rotated-same-shape, and
  different-shape

## Constraints

- tests must not silently promote dirty artifacts to clean
- reference artifacts may detect change without claiming aesthetic correctness
- the project-specific process must be documented in `project/agents/`
- first-run bootstrap must create dirty image and STL references for new named
  fixtures
- subsequent runs must compare against clean when available, otherwise dirty
- when a fixture's test contract changes, the old dirty and clean references
  must be invalidated before a new dirty baseline is bootstrapped
- slice-based comparison lanes must define whether orientation mismatch is a
  failure or an allowed equivalence class

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- image and STL reference-artifact behavior is explicit
- first-run bootstrap and later comparison behavior are explicit
- fixture-contract invalidation behavior is explicit
- the model-output completeness requirement is explicit
- stronger local verification lanes such as canonical slice classification are
  explicit when used
- verification requirements are defined by its paired test specification
- project-specific reference-artifact documentation is explicit
