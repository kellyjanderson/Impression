# Surface Spec 135: Surface Boolean Metadata, Provenance, and Explicit Invalid-Result Posture (v1.0)

## Overview

This specification defines deterministic metadata/provenance carry-forward and
the explicit caller-facing posture for invalid surfaced boolean results.

## Backlink

- [Surface Spec 119: Surface Boolean Validity, Healing Limits, and Metadata Propagation (v1.0)](surface-119-surface-boolean-validity-healing-limits-and-metadata-propagation-v1_0.md)

## Scope

This specification covers:

- propagation of operation provenance onto surfaced boolean results
- deterministic carry-forward of consumer metadata from operands to results
- explicit surfaced invalid-result posture when the validity gate cannot accept the result

## Behavior

This leaf must define:

- what provenance records are attached to accepted surfaced boolean results
- how operand metadata is carried forward without losing operation context
- how invalid or unsupported surfaced results are surfaced to callers when cleanup is insufficient

## Constraints

- metadata propagation must remain deterministic and operation-aware
- invalid-result posture must stay surfaced and explicit
- metadata carry-forward must not silently erase caller-visible information from accepted operands

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- result provenance and metadata carry-forward rules are explicit
- explicit invalid-result posture is defined for validity-gate failures
- the accepted-result metadata payload is explicit enough to satisfy downstream consumers
- verification requirements are defined by its paired test specification
