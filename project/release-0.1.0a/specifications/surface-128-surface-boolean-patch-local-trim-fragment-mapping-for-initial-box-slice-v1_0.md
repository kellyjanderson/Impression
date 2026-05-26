# Surface Spec 128: Surface Boolean Patch-Local Trim-Fragment Mapping for Initial Box Slice (v1.0)

## Overview

This specification defines how bounded surfaced cut curves are mapped into
patch-local trim-space fragments for the initial box/box boolean slice.

## Backlink

- [Surface Spec 117: Surface Boolean Intersection, Classification, and Operand Splitting (v1.0)](surface-117-surface-boolean-intersection-classification-and-operand-splitting-v1_0.md)

## Scope

This specification covers:

- projecting discovered cut curves into the local parameter domains of affected patches
- trim-fragment endpoint ordering and orientation
- the canonical patch-local fragment record shape for the initial box slice

## Behavior

This leaf must define:

- how each discovered 3D cut curve yields per-patch trim-space fragments
- how fragment endpoints are ordered so downstream split logic is deterministic
- what patch-local payload is canonical before fragment classification or result reconstruction

## Constraints

- trim-fragment mapping must preserve the surfaced cut geometry rather than replacing it with mesh-owned edges
- fragment orientation and ordering must remain deterministic for equal operands and equal request state
- unsupported parameter-space mapping cases must remain explicit

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- per-patch trim-fragment output is explicit for the initial box slice
- endpoint ordering and orientation rules are explicit
- the canonical trim-fragment payload is explicit enough to drive downstream split records
- verification requirements are defined by its paired test specification
