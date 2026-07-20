# Surface Spec 37: Cross-Mode Equivalence and Drift Bounds (v1.0)

## Overview

This specification defines the guarantees that prevent preview, export, and
analysis tessellation modes from drifting into separate geometric truths.

## Backlink

Parent specification:

- [Surface Spec 12: Preview / Export Tessellation Policy Split (v1.0)](surface-12-preview-export-tessellation-policy-v1_0.md)

## Scope

This specification covers:

- acceptable differences between tessellation modes
- prohibited representational drift
- validation expectations across modes

## Behavior

This branch must define:

- what “same surface truth” means across modes
- which differences are allowed between preview and export meshes
- which measures are used to detect unacceptable drift

## Constraints

- drift bounds must be testable
- allowed differences must not encode separate modeling results
- validation rules must be deterministic

## Refinement Status

Final.

This leaf is implementation-sized under the Specification Size Index because it
defines one equivalence contract with one verification concern.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- permitted cross-mode differences are explicit
- prohibited drift conditions are explicit
- cross-mode verification rules are explicit

