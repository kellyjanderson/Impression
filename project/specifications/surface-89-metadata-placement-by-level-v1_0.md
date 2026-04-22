# Surface Spec 89: Metadata Placement by Body, Shell, and Patch Level (v1.0)

## Overview

This specification defines which metadata belongs at the body, shell, and patch
levels.

## Backlink

Parent specification:

- [Surface Spec 30: Surface Metadata Placement Contract (v1.0)](surface-30-surface-metadata-placement-contract-v1_0.md)

## Scope

This specification covers:

- body-level metadata
- shell-level metadata
- patch-level metadata

## Behavior

This branch must define:

- which metadata fields are permitted at each structural level
- which fields are prohibited at each level
- what downstream systems may assume from metadata placement

## Constraints

- placement rules must be explicit
- placement must be deterministic
- downstream systems must not need to infer intended level heuristically

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- allowed metadata by level is explicit
- prohibited metadata by level is explicit
- downstream placement assumptions are explicit

