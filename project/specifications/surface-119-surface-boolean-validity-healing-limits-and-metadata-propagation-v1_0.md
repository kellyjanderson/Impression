# Surface Spec 119: Surface Boolean Validity, Healing Limits, and Metadata Propagation (v1.0)

## Overview

This specification defines the post-reconstruction validity gate for surfaced
booleans, including bounded healing and deterministic metadata carry-forward.

## Backlink

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](surface-102-surface-body-boolean-replacement-v1_0.md)

## Scope

This specification covers:

- trim, seam, and shell validity checks after reconstruction
- bounded healing and canonical cleanup that is allowed
- unsupported or invalid result classification when healing is insufficient
- propagation of provenance and consumer metadata

## Behavior

This branch must define:

- what validity conditions must hold before a surfaced boolean result is accepted
- what healing is permitted versus forbidden
- how operation provenance and operand metadata propagate into the result

Allowed healing should be limited to deterministic normalization such as:

- zero-measure trim removal
- duplicate seam-use removal
- loop orientation repair
- seam/boundary-use canonical ordering

Disallowed healing includes:

- hidden mesh boolean fallback
- geometric warping to force closure
- collapsing materially distinct surfaced fragments into a fake success

## Constraints

- validity cleanup must not become hidden geometric repair or hidden mesh boolean fallback
- metadata propagation must be deterministic and operation-aware
- failure to satisfy validity rules must remain explicit to callers

## Refinement Status

Decomposed into final child leaves.

This parent branch does not yet represent executable work directly.

## Child Specifications

- [Surface Spec 134: Surface Boolean Deterministic Validity Gate and Bounded Cleanup](surface-134-surface-boolean-deterministic-validity-gate-and-bounded-cleanup-v1_0.md)
- [Surface Spec 135: Surface Boolean Metadata, Provenance, and Explicit Invalid-Result Posture](surface-135-surface-boolean-metadata-provenance-and-explicit-invalid-result-posture-v1_0.md)

## Acceptance

This specification is ready for implementation planning when:

- validity/cleanup and metadata/failure-posture concerns are separated into bounded final leaves
- the allowed-versus-forbidden healing boundary remains explicit across the child set
- paired verification leaves exist for the final children
