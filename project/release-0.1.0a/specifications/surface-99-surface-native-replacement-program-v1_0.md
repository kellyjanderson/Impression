# Surface Spec 99: Surface-Native Replacement Program (v1.0)

## Overview

This specification defines the tracked replacement program for deprecated
mesh-primary capabilities that still lack surface-first equivalents.

## Backlink

- [Surface-Native Capability Replacement Architecture](../architecture/surface-native-capability-replacement-architecture.md)

## Scope

This specification covers:

- the still-missing surface-first capability branches
- the requirement that each replacement branch has paired verification
- the requirement that each replacement branch includes durable documentation

## Behavior

This branch must define surface-first replacements for:

- drafting
- text
- booleans
- threading
- hinges
- heightfields and displacement
- reference artifacts and documentation rules needed to verify and deliver those replacements

## Constraints

- replacement branches must not terminate in mesh as primary truth
- each final replacement leaf must have a paired test specification
- each final replacement leaf must define required documentation updates

## Refinement Status

Decomposed into child branches.

This parent branch does not represent executable work directly.

## Child Specifications

- [Surface Spec 100: Surface-Native Drafting Replacement](surface-100-surface-native-drafting-replacement-v1_0.md)
- [Surface Spec 101: Surface-Native Text Replacement](surface-101-surface-native-text-replacement-v1_0.md)
- [Surface Spec 102: Surface-Body Boolean Replacement](surface-102-surface-body-boolean-replacement-v1_0.md)
- [Surface Spec 103: Surface-Native Threading Replacement](surface-103-surface-native-threading-replacement-v1_0.md)
- [Surface Spec 104: Surface-Native Hinge Replacement](surface-104-surface-native-hinge-replacement-v1_0.md)
- [Surface Spec 105: Surface-Native Heightfield and Displacement Replacement](surface-105-surface-native-heightfield-displacement-v1_0.md)
- [Surface Spec 106: Reference Artifact Regression Suite](surface-106-reference-artifact-regression-suite-v1_0.md)
- [Surface Spec 107: Documentation-First Delivery Requirements](surface-107-documentation-first-delivery-requirements-v1_0.md)

## Acceptance

This specification is complete when:

- every deprecated capability lacking a surface-first replacement is represented
  by a child branch that refines to final feature leaves
- paired verification leaves exist for the final feature leaves in those child
  branches
- documentation expectations are explicit across the replacement branch
