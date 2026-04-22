# Surface Spec 36: Export and Analysis Tessellation Policy Contract (v1.0)

## Overview

This specification defines the tessellation policy used for export-quality mesh
generation and analysis-oriented downstream consumers.

## Backlink

Parent specification:

- [Surface Spec 12: Preview / Export Tessellation Policy Split (v1.0)](surface-12-preview-export-tessellation-policy-v1_0.md)

## Scope

This specification covers:

- export tessellation policy
- analysis tessellation policy
- their shared or separate quality requirements

## Behavior

This branch must define:

- the default export tessellation policy
- whether analysis reuses export or defines a stricter dedicated mode
- the minimum quality contract required for STL and QA consumers

## Constraints

- export quality must be sufficient for artifact generation
- analysis policy must not be under-specified relative to QA needs
- both modes must remain derived from the same surface truth as preview

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
covers one downstream policy cluster with one quality contract.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- export defaults are explicit
- analysis policy reuse or separation is explicit
- minimum artifact/QA guarantees are explicit

