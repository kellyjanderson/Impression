# Surface Spec 12: Preview / Export Tessellation Policy Split (v1.0)

## Overview

This specification defines the branch responsible for the policy differences
between preview tessellation and export tessellation.

## Backlink

Parent specification:

- [Surface Spec 03: Tessellation Boundary and Rendering Contract (v1.0)](surface-03-tessellation-boundary-v1_0.md)

## Scope

This specification covers:

- preview tessellation policy
- export tessellation policy
- permitted differences between those modes
- guarantees that keep them tied to the same surface truth

## Behavior

This branch must define:

- what preview mode is allowed to trade off
- what export mode is required to preserve
- whether analysis uses preview, export, or a dedicated policy
- how the system prevents representational drift between consumer classes

## Constraints

- preview and export must share the same underlying surface truth
- policy differences must be explicit and testable
- export policy must remain suitable for STL-quality output
- preview policy must remain responsive without becoming a separate model path

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 35: Preview Tessellation Policy Contract (v1.0)](surface-35-preview-tessellation-policy-v1_0.md)
- [Surface Spec 36: Export and Analysis Tessellation Policy Contract (v1.0)](surface-36-export-analysis-tessellation-policy-v1_0.md)
- [Surface Spec 37: Cross-Mode Equivalence and Drift Bounds (v1.0)](surface-37-cross-mode-equivalence-drift-bounds-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- preview and export policies are separately defined
- the allowed differences between them are explicit
- the shared-surface-truth guarantee is explicit
