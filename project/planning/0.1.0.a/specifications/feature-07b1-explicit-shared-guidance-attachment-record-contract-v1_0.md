# Feature Spec 07B1: Explicit Shared Guidance Attachment Record Contract (v1.0)

## Overview

This specification defines the record contract for explicit shared guidance
attachment inside loft.

## Backlink

- [Feature Spec 07B: Explicit Shared Guidance Attachment and Consumption Rules (v1.0)](feature-07b-explicit-shared-guidance-attachment-and-consumption-rules-v1_0.md)

## Scope

This specification covers:

- explicit shared guidance input
- attachment records
- durable attachment metadata

## Behavior

This leaf must define:

- how explicit shared guidance is attached to loft
- what the attachment record contains
- how attachment metadata remains durable for later diagnostics and replay

## Constraints

- attachment identity must remain deterministic
- attachment records must remain separate from planner-consumption semantics

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- attachment record shape is explicit
- deterministic attachment identity is explicit
- durable attachment metadata expectations are explicit
