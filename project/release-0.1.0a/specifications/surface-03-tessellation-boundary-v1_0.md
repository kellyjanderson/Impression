# Surface Spec 03: Tessellation Boundary and Rendering Contract (v1.0)

## Overview

This specification defines the branch responsible for turning surface-native
geometry into deterministic meshes for preview, rendering, export, and mesh
analysis.

## Backlink

Parent specification:

- [Surface Spec 01: Surface-First Internal Model Program (v1.0)](surface-01-surface-first-internal-model-program-v1_0.md)

## Scope

This specification covers:

- tessellation requests and quality controls
- render/export tessellation policy
- seam-consistent patch tessellation
- watertightness expectations for valid surface bodies
- mesh production contracts for downstream consumers

## Behavior

The tessellation branch must define:

- how preview asks for tessellation
- how export asks for tessellation
- which tessellation settings are quality-facing versus tolerance-facing
- how adjacent patches share tessellated boundaries deterministically
- what mesh guarantees are expected from valid upstream surface bodies

## Constraints

- tessellation must be deterministic
- tessellation must not change modeled meaning
- preview and export may use different density/tolerance settings but must share
  the same surface truth
- watertightness rules must be explicit and testable
- tessellation must remain a boundary system, not a hidden new kernel

## Refinement Status

Decomposed into final child leaves.

This parent branch no longer requires another refinement round unless later
architecture changes invalidate the current split.

## Child Specifications

- [Surface Spec 11: Tessellation Request and Quality Contract (v1.0)](surface-11-tessellation-request-and-quality-contract-v1_0.md)
- [Surface Spec 12: Preview / Export Tessellation Policy Split (v1.0)](surface-12-preview-export-tessellation-policy-v1_0.md)
- [Surface Spec 13: Seam-Consistent Tessellation and Watertight Output Rules (v1.0)](surface-13-seam-consistent-tessellation-watertightness-v1_0.md)

## Acceptance

This branch is ready for implementation planning when:

- tessellation entry points are explicit
- deterministic guarantees are written as testable rules
- preview/export differences are bounded without creating representational drift
- the child branches define the tessellation boundary as final
  implementation-sized leaves
