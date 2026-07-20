# Surface Spec 69: Parameter Domain Normalization Policy (v1.0)

## Overview

This specification defines whether parameter domains are normalized and how
normalization relates to patch-family-specific native coordinates.

## Backlink

Parent specification:

- [Surface Spec 23: Patch Parameter-Domain Contract (v1.0)](surface-23-patch-parameter-domain-contract-v1_0.md)

## Scope

This specification covers:

- normalized versus native coordinate policy
- conversion requirements between them
- caller-visible parameter conventions

## Behavior

This branch must define:

- whether the kernel exposes normalized coordinates, native coordinates, or both
- how conversions are performed and exposed
- what callers may assume about coordinate ranges

## Constraints

- normalization policy must be explicit
- conversion behavior must be deterministic
- caller-visible coordinate conventions must remain stable

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- normalization policy is explicit
- conversion rules are explicit
- caller-visible coordinate conventions are explicit

