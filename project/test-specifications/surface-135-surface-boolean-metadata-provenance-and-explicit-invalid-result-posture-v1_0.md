# Surface Spec 135 Test: Surface Boolean Metadata, Provenance, and Explicit Invalid-Result Posture

## Overview

This test specification defines verification for surfaced boolean metadata
carry-forward, provenance, and explicit invalid-result posture.

## Backlink

- [Surface Spec 135: Surface Boolean Metadata, Provenance, and Explicit Invalid-Result Posture (v1.0)](../specifications/surface-135-surface-boolean-metadata-provenance-and-explicit-invalid-result-posture-v1_0.md)

## Automated Smoke Tests

- accepted surfaced boolean results carry explicit operation provenance
- invalid or unsupported surfaced boolean results remain explicit to callers

## Automated Acceptance Tests

- consumer metadata propagates deterministically from representative operands to accepted results
- provenance payloads remain aligned with the executed operation and operand identities
- invalid-result posture does not silently downgrade to mesh success
