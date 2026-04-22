# Testing Spec 23: Computer Vision Diagnostic Panel Honesty Boundary and Proof Delegation Rules (v1.0)

## Overview

This specification defines what diagnostic panel artifacts may prove, what they
may never be treated as proving by default, and how a narrower proof role can
be delegated explicitly.

## Backlink

- [Testing Spec 08: Computer Vision Diagnostic Triptych and Panel-Region Presentation (v1.0)](testing-08-computer-vision-diagnostic-triptych-and-panel-region-presentation-v1_0.md)

## Scope

This specification covers:

- the default diagnostic-only posture
- honesty rules for independently rendered versus shared-scene panels
- explicit delegation from an authoritative proof lane
- failure-review versus truth-claim boundaries

## Behavior

This leaf must define:

- what diagnostic panels support during human review
- what they do not prove by default
- when another lane may delegate a narrow proof use to a panel artifact
- how the contract distinguishes shared-scene panels from stitched
  comparisons

## Constraints

- diagnostic panels must not silently become proof artifacts
- independently rendered panels must not imply shared-scale proof unless a
  stronger contract states that explicitly
- honesty posture must remain visible in fixture and harness documentation

## Refinement Status

Final.

## Child Specifications

None. Final leaf.

## Acceptance

This specification is complete when:

- diagnostic-only default posture is explicit
- proof delegation boundaries are explicit
- honesty rules for panel composition are explicit
- verification requirements are defined by its paired test specification
