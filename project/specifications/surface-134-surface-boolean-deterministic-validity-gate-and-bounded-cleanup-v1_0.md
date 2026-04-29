# Surface Spec 134: Surface Boolean Deterministic Validity Gate and Bounded Cleanup (v1.0)

## Overview

This specification defines the deterministic post-reconstruction validity gate
and the bounded cleanup allowed for surfaced boolean results.

## Backlink

- [Surface Spec 119: Surface Boolean Validity, Healing Limits, and Metadata Propagation (v1.0)](surface-119-surface-boolean-validity-healing-limits-and-metadata-propagation-v1_0.md)

## Scope

This specification covers:

- trim, seam, shell, and boundary-use validity checks after reconstruction
- deterministic canonical cleanup that is allowed before a surfaced result is accepted
- the allowed-versus-forbidden healing boundary for the initial surfaced boolean lane

## Behavior

This leaf must define:

- what validity conditions must hold before a surfaced result is accepted
- what bounded cleanup is permitted without changing intended surfaced geometry
- what cleanup remains forbidden because it would hide geometric repair or hidden mesh fallback

## Constraints

- bounded cleanup must not become hidden geometric warping or hidden mesh boolean fallback
- cleanup must be deterministic for equal result topology and equal request state
- failure to satisfy the validity gate must remain explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- the deterministic validity gate is explicit
- the allowed bounded cleanup set is explicit
- the forbidden healing boundary remains explicit
- verification requirements are defined by its paired test specification
