# Surface Spec 110: Surface Boolean Public API Migration and Reference Verification (v1.0)

## Overview

This specification defines the public migration boundary and regression evidence for surfaced boolean replacement.

## Backlink

- [Surface Spec 102: Surface-Body Boolean Replacement (v1.0)](surface-102-surface-body-boolean-replacement-v1_0.md)

## Scope

This specification covers:

- public `csg.py` migration posture
- compatibility posture for legacy mesh-first boolean callers
- regression evidence and docs required before promotion

## Behavior

This branch must define:

- when public boolean APIs gain `backend=\"surface\"` or equivalent surfaced entry
- what documentation and migration notes are required
- what reference fixtures must exist before the surfaced path is considered ready

## Constraints

- public promotion must not erase explicit legacy-mesh deprecation posture
- surfaced booleans must have durable examples and reference artifacts before completion
- rollback posture must remain explicit while legacy callers still exist

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- public boolean migration posture is explicit
- required docs and reference evidence are explicit
- verification requirements are defined by its paired test specification

